"""
evaluation/eval_runner.py
==========================
Mini evaluation suite — Part 3 of the spec.

Runs the system on Q1, Q2, Q3 and scores each output on three dimensions:
  1. Completeness    (0-5): Does the answer cover all required sub-topics?
  2. Factual Precision (0-5): Are specific claims accurate and sourced?
  3. Attribution Quality (0-5): Are sources cited? Are conflicts flagged?

Scoring Rubrics (defined explicitly per spec requirement)
---------------------------------------------------------
COMPLETENESS (0-5):
  5 — All required sub-topics covered with appropriate depth
  4 — All major topics covered; minor gaps in one area
  3 — Most topics covered; one significant area missing
  2 — Partial coverage; two+ areas missing or very shallow
  1 — Only surface-level answer, major gaps
  0 — Does not address the question

FACTUAL PRECISION (0-5):
  5 — All specific facts checkable and accurate; no invented data
  4 — Nearly all facts accurate; 1-2 minor imprecisions
  3 — Generally accurate but some claims are vague or unverifiable
  2 — Multiple claims lack specificity; some appear inaccurate
  1 — Mostly vague; claims contradicted by known facts
  0 — Factually incorrect or completely unspecific

ATTRIBUTION QUALITY (0-5):
  5 — Specific regulations/studies/data sources cited; conflicts flagged
  4 — Most claims sourced; conflicts detected if present
  3 — Some sources cited but inconsistently; gaps in attribution
  2 — Minimal sourcing; sources are generic (e.g. "EU law")
  1 — Almost no sources; no conflict detection
  0 — No attribution whatsoever

Total max: 15 points per query.

Anti-over-decomposition scoring:
  For Q2 (narrow query): if the system used >2 agents, deduct 1 point
  from Completeness (over-decomposition adds noise, not value).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from models.schemas import (
    EvalDimension,
    EvalResult,
    EvalScore,
    QueryComplexity,
    ResearchTrace,
)
from evaluation.eval_queries import TEST_QUERIES, EvalQuery
from orchestrator.research_config import ResearchConfig
from orchestrator.research_orchestrator import ResearchOrchestrator
from providers.factory import get_provider

logger = logging.getLogger(__name__)

# Rubric strings for reporting
RUBRICS = {
    EvalDimension.COMPLETENESS: (
        "5=all topics covered with depth | 4=minor gap | 3=one significant area missing | "
        "2=two+ areas missing | 1=surface only | 0=not addressed"
    ),
    EvalDimension.FACTUAL_PRECISION: (
        "5=all facts accurate+checkable | 4=nearly all accurate | 3=mostly accurate but vague | "
        "2=multiple vague/inaccurate claims | 1=mostly vague | 0=incorrect/empty"
    ),
    EvalDimension.ATTRIBUTION_QUALITY: (
        "5=specific sources+conflicts flagged | 4=most claims sourced | "
        "3=some sources inconsistently | 2=minimal/generic sources | 1=almost none | 0=none"
    ),
}


class EvalRunner:
    """Runs Q1-Q3 and produces scored EvalResult objects.

    Parameters
    ----------
    provider:
        LLM provider name (default: anthropic).
    output_dir:
        Where to save eval_results.json.
    trace_dir:
        Where research traces are saved during eval.
    use_llm_scoring:
        If True, uses LLM to auto-score each dimension.
        If False, uses heuristic scoring from metadata.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        output_dir: str = "evals",
        trace_dir: str = "traces",
        use_llm_scoring: bool = True,
    ) -> None:
        self.provider_name = provider
        self.output_dir = output_dir
        self.trace_dir = trace_dir
        self.use_llm_scoring = use_llm_scoring
        self._provider = get_provider(provider)

    async def run_all(self) -> List[EvalResult]:
        """Run all three test queries and return scored results."""
        results: List[EvalResult] = []
        for eval_query in TEST_QUERIES:
            logger.info("[eval] Running %s: %s", eval_query.query_id, eval_query.query_text[:60])
            result = await self.run_query(eval_query)
            results.append(result)
            logger.info(
                "[eval] %s scored %.1f/15  (C=%.1f P=%.1f A=%.1f)",
                eval_query.query_id,
                result.total_score,
                *[s.score for s in result.scores],
            )

        await self._save(results)
        return results

    async def run_query(self, eval_query: EvalQuery) -> EvalResult:
        """Run a single eval query and score the result."""
        t0 = time.perf_counter()

        config = ResearchConfig(
            query=eval_query.query_text,
            provider=self.provider_name,
            token_budget=60_000,
            tokens_per_agent=3_500,
            max_subtasks=5,
            enable_synthesis=True,
            enable_retry=True,
            output_dir=self.trace_dir,
        )

        orchestrator = ResearchOrchestrator(config)

        try:
            trace = await orchestrator.run()
        except Exception as exc:
            logger.error("[eval] %s failed: %s", eval_query.query_id, exc)
            return self._error_result(eval_query, str(exc))

        latency = (time.perf_counter() - t0) * 1000
        agents_used = list({f.agent_id for f in trace.findings if not f.error})

        # Score the output
        if self.use_llm_scoring:
            scores = await self._llm_score(trace, eval_query)
        else:
            scores = self._heuristic_score(trace, eval_query)

        # Anti-over-decomposition penalty for Q2
        if eval_query.query_id == "Q2" and len(agents_used) > 2:
            for score in scores:
                if score.dimension == EvalDimension.COMPLETENESS:
                    original = score.score
                    score.score = max(0.0, score.score - 1.0)
                    score.justification = (
                        f"[Over-decomposition penalty: {len(agents_used)} agents used "
                        f"for a narrow question, expected ≤2. Score reduced from {original:.1f}] "
                        + score.justification
                    )

        total = sum(s.score for s in scores)

        return EvalResult(
            query_id=eval_query.query_id,
            query_text=eval_query.query_text,
            complexity_detected=trace.complexity,
            agents_used=agents_used,
            total_latency_ms=latency,
            tokens_used=trace.budget_summary.total_spent,
            scores=scores,
            total_score=round(total, 2),
            report_snippet=trace.report.executive_summary[:500],
        )

    # ------------------------------------------------------------------
    # LLM-based scoring
    # ------------------------------------------------------------------

    async def _llm_score(
        self, trace: ResearchTrace, eval_query: EvalQuery
    ) -> List[EvalScore]:
        """Use Claude to score the research report on all three dimensions."""
        report = trace.report

        scoring_prompt = f"""
You are evaluating a multi-agent research system's output against a scoring rubric.

ORIGINAL QUERY: {eval_query.query_text}

SYSTEM OUTPUT (executive summary):
{report.executive_summary}

KEY FINDINGS:
{chr(10).join(f"- {kt}" for kt in report.key_takeaways[:5])}

SOURCES CITED: {', '.join(report.sources_cited[:8]) if report.sources_cited else 'None'}

KNOWLEDGE GAPS FLAGGED: {', '.join(report.knowledge_gaps[:5]) if report.knowledge_gaps else 'None'}

CONFLICTS DETECTED: {len(report.conflicts_detected)} conflict(s)

AGENTS USED: {', '.join({f.agent_id for f in trace.findings if not f.error})}
COMPLEXITY DETECTED: {trace.complexity.value}

=== SCORING RUBRICS ===

COMPLETENESS (what must the answer cover):
{eval_query.completeness_notes}
Rubric: {RUBRICS[EvalDimension.COMPLETENESS]}

FACTUAL PRECISION (what specific facts should appear):
{eval_query.precision_notes}
Rubric: {RUBRICS[EvalDimension.FACTUAL_PRECISION]}

ATTRIBUTION QUALITY (what sources should be cited):
{eval_query.attribution_notes}
Rubric: {RUBRICS[EvalDimension.ATTRIBUTION_QUALITY]}

=== INSTRUCTIONS ===
Score the output on each dimension from 0.0 to 5.0 (decimals allowed).
Be critical — a score of 5 should be rare and require near-perfect coverage.

Respond ONLY with valid JSON:
{{
    "completeness": {{
        "score": <float 0-5>,
        "justification": "<2-3 sentences explaining the score>"
    }},
    "factual_precision": {{
        "score": <float 0-5>,
        "justification": "<2-3 sentences explaining the score>"
    }},
    "attribution": {{
        "score": <float 0-5>,
        "justification": "<2-3 sentences explaining the score>"
    }}
}}
"""

        try:
            response = await self._provider.complete(
                prompt=scoring_prompt,
                max_tokens=1500,
                temperature=0.1,
            )
            return self._parse_scores(response.text, eval_query.query_id)
        except Exception as exc:
            logger.warning("[eval] LLM scoring failed for %s: %s — using heuristics", eval_query.query_id, exc)
            return self._heuristic_score(trace, eval_query)

    def _parse_scores(self, raw: str, query_id: str) -> List[EvalScore]:
        """Parse LLM scoring response into EvalScore objects."""
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        data = json.loads(cleaned)

        dim_map = {
            "completeness": EvalDimension.COMPLETENESS,
            "factual_precision": EvalDimension.FACTUAL_PRECISION,
            "attribution": EvalDimension.ATTRIBUTION_QUALITY,
        }

        scores = []
        for key, dim in dim_map.items():
            dim_data = data.get(key, {})
            scores.append(EvalScore(
                query_id=query_id,
                dimension=dim,
                score=min(5.0, max(0.0, float(dim_data.get("score", 2.0)))),
                justification=dim_data.get("justification", "[No justification]"),
                rubric=RUBRICS[dim],
            ))
        return scores

    # ------------------------------------------------------------------
    # Heuristic scoring fallback
    # ------------------------------------------------------------------

    def _heuristic_score(
        self, trace: ResearchTrace, eval_query: EvalQuery
    ) -> List[EvalScore]:
        """Rules-based scoring when LLM scoring is unavailable."""
        report = trace.report
        valid_findings = [f for f in trace.findings if not f.error]
        n_agents = len({f.agent_id for f in valid_findings})
        has_sources = bool(report.sources_cited)
        has_gaps = bool(report.knowledge_gaps)
        has_conflicts_detected = bool(report.conflicts_detected)
        n_domains = len(report.findings_by_domain)
        completeness = report.completeness_score / 20  # 0-100 → 0-5

        # Completeness: based on completeness_score + agent coverage
        c_score = min(5.0, completeness * (n_agents / max(len(eval_query.domains_expected), 1)))

        # Precision: based on sources and confidence
        avg_conf = (
            sum(f.confidence for f in valid_findings) / len(valid_findings)
            if valid_findings else 0
        )
        p_score = min(5.0, (avg_conf / 100) * 5.0 * (0.7 + 0.3 * (1 if has_sources else 0)))

        # Attribution: based on sources cited
        a_score = min(5.0, (
            (2.0 if has_sources else 0) +
            (1.0 if len(report.sources_cited) >= 5 else 0.5) +
            (1.0 if has_conflicts_detected else 0) +
            (1.0 if has_gaps else 0)
        ))

        return [
            EvalScore(
                query_id=eval_query.query_id,
                dimension=EvalDimension.COMPLETENESS,
                score=round(c_score, 1),
                justification=f"Heuristic: {n_agents} agent(s), {n_domains} domain(s), completeness_score={report.completeness_score}",
                rubric=RUBRICS[EvalDimension.COMPLETENESS],
            ),
            EvalScore(
                query_id=eval_query.query_id,
                dimension=EvalDimension.FACTUAL_PRECISION,
                score=round(p_score, 1),
                justification=f"Heuristic: avg_confidence={avg_conf:.0f}/100, sources_cited={len(report.sources_cited)}",
                rubric=RUBRICS[EvalDimension.FACTUAL_PRECISION],
            ),
            EvalScore(
                query_id=eval_query.query_id,
                dimension=EvalDimension.ATTRIBUTION_QUALITY,
                score=round(a_score, 1),
                justification=f"Heuristic: {len(report.sources_cited)} sources, conflicts={len(report.conflicts_detected)}, gaps_flagged={len(report.knowledge_gaps)}",
                rubric=RUBRICS[EvalDimension.ATTRIBUTION_QUALITY],
            ),
        ]

    def _error_result(self, eval_query: EvalQuery, error: str) -> EvalResult:
        return EvalResult(
            query_id=eval_query.query_id,
            query_text=eval_query.query_text,
            complexity_detected=eval_query.complexity_expected,
            scores=[
                EvalScore(
                    query_id=eval_query.query_id,
                    dimension=d,
                    score=0.0,
                    justification=f"System error: {error[:100]}",
                    rubric=RUBRICS[d],
                )
                for d in EvalDimension
            ],
            total_score=0.0,
            report_snippet=f"[ERROR] {error}",
        )

    async def _save(self, results: List[EvalResult]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"eval_results_{ts}.json")
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "scoring_dimensions": {
                d.value: RUBRICS[d] for d in EvalDimension
            },
            "results": [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "complexity_detected": r.complexity_detected.value,
                    "agents_used": r.agents_used,
                    "total_latency_ms": round(r.total_latency_ms),
                    "tokens_used": r.tokens_used,
                    "scores": {
                        s.dimension.value: {
                            "score": s.score,
                            "justification": s.justification,
                        }
                        for s in r.scores
                    },
                    "total_score": r.total_score,
                    "report_snippet": r.report_snippet,
                }
                for r in results
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("[eval] Results saved → %s", path)
        print(f"\n[eval] Results saved → {path}")
