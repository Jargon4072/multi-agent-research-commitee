"""
agents/synthesizer.py
======================
ResearchSynthesizer — merges all sub-agent findings into a final ResearchReport.
This is Part 4 of the spec: synthesis + conflict resolution.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from models.schemas import (
    BudgetSummary,
    ConflictRecord,
    QueryComplexity,
    ResearchFinding,
    ResearchReport,
    SubtaskStatus,
)
from providers.base import LLMProvider
from agents.prompts import SYNTHESIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ResearchSynthesizer:
    """Calls the LLM with all research findings and produces a ResearchReport.

    Uses a higher-quality model (claude-sonnet if available) for synthesis
    since this is the most important output step.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def synthesize(
        self,
        query: str,
        findings: List[ResearchFinding],
        budget_summary: BudgetSummary,
        complexity: QueryComplexity = QueryComplexity.BROAD,
        max_tokens: int = 8192,
    ) -> ResearchReport:
        """Generate the final :class:`~models.schemas.ResearchReport`.

        1. Detect conflicts between findings.
        2. Build a rich synthesis prompt.
        3. Call the LLM (synthesis model if supported).
        4. Parse the response.
        5. Fall back to a rules-based report if parsing fails.
        """
        # Step 1: Detect conflicts between findings
        conflicts = self._detect_conflicts(findings)

        prompt = self._build_prompt(query, findings, conflicts)

        # Step 2: Try LLM synthesis (use synthesis model if provider supports it)
        try:
            call_kwargs: dict = dict(
                prompt=prompt,
                max_tokens=max_tokens,
                system=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.2,
            )
            # ClaudeProvider supports use_synthesis_model kwarg
            if hasattr(self.provider, "synthesis_model_name"):
                call_kwargs["use_synthesis_model"] = True

            response = await self.provider.complete(**call_kwargs)
            report = self._parse_response(
                response.text, query, findings, conflicts, budget_summary, complexity
            )
        except Exception as exc:
            logger.error("[synthesizer] LLM call failed: %s — building fallback report", exc)
            report = self._fallback_report(query, findings, conflicts, budget_summary, complexity)

        return report

    # ------------------------------------------------------------------
    # Conflict detection (Part 4)
    # ------------------------------------------------------------------

    def _detect_conflicts(self, findings: List[ResearchFinding]) -> List[ConflictRecord]:
        """Detect factual conflicts between findings by cross-checking conflicts_with fields."""
        conflicts: List[ConflictRecord] = []
        seen: set = set()

        for finding in findings:
            for other_subtask_id, conflict_desc in finding.conflicts_with.items():
                # Find the other finding
                other_findings = [
                    f for f in findings if f.subtask_id == other_subtask_id
                ]
                if not other_findings:
                    continue

                other = other_findings[0]
                pair_key = tuple(sorted([finding.subtask_id, other_subtask_id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                # Try to find a counter-claim from the other finding
                counter_claim = other.conflicts_with.get(finding.subtask_id, "")

                conflicts.append(ConflictRecord(
                    subtask_a=finding.subtask_id,
                    subtask_b=other_subtask_id,
                    dimension=self._extract_dimension(conflict_desc),
                    claim_a=conflict_desc,
                    claim_b=counter_claim or "[No counter-claim logged by other agent]",
                    resolved=False,
                ))

        return conflicts

    @staticmethod
    def _extract_dimension(conflict_desc: str) -> str:
        """Extract a short dimension label from a conflict description."""
        # Take first 60 chars as a proxy dimension name
        return conflict_desc[:60].rstrip(".,;") if conflict_desc else "unspecified"

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        findings: List[ResearchFinding],
        conflicts: List[ConflictRecord],
    ) -> str:
        lines = [
            f"ORIGINAL RESEARCH QUERY: {query}",
            "",
            "═" * 60,
            "SUB-AGENT FINDINGS",
            "═" * 60,
        ]

        for f in findings:
            if f.error:
                lines.append(f"\n[{f.agent_name}] ERROR: {f.error}")
                continue
            lines.append(
                f"\n[{f.agent_name} / {f.role.value}]  "
                f"subtask_id={f.subtask_id}  "
                f"confidence={f.confidence}/100  status={f.status.value}  "
                f"latency={f.latency_ms:.0f}ms"
            )
            lines.append(f"QUESTION: {f.query}")
            lines.append(f"ANSWER:\n{f.answer}")
            if f.key_points:
                lines.append("KEY POINTS:")
                for pt in f.key_points:
                    lines.append(f"  • {pt}")
            if f.knowledge_gaps:
                lines.append(f"GAPS: {'; '.join(f.knowledge_gaps)}")
            if f.sources_cited:
                lines.append(f"SOURCES: {'; '.join(f.sources_cited[:5])}")

        if conflicts:
            lines += ["", "DETECTED CONFLICTS", "-" * 50]
            for c in conflicts:
                lines.append(
                    f"  {c.subtask_a} vs {c.subtask_b} [{c.dimension}]:\n"
                    f"    A claims: {c.claim_a}\n"
                    f"    B claims: {c.claim_b}"
                )

        # Budget summary
        valid = [f for f in findings if not f.error]
        total_tokens = sum(f.tokens_used for f in valid)
        avg_conf = int(sum(f.confidence for f in valid) / len(valid)) if valid else 0
        lines += [
            "",
            f"STATS: {len(valid)}/{len(findings)} agents succeeded  |  "
            f"{total_tokens} tokens  |  avg confidence={avg_conf}/100",
            "",
            "═" * 60,
            "Produce the ResearchReport JSON exactly matching the schema.",
            "No markdown fences. No preamble.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        query: str,
        findings: List[ResearchFinding],
        conflicts: List[ConflictRecord],
        budget_summary: BudgetSummary,
        complexity: QueryComplexity,
    ) -> ResearchReport:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()

        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("[synthesizer] JSON parse failed: %s", exc)
            return self._fallback_report(query, findings, conflicts, budget_summary, complexity)

        # Consolidate sources across all findings
        all_sources = list({
            s for f in findings if not f.error for s in f.sources_cited
        })

        # Merge gaps from all findings with synthesis gaps
        all_gaps = list({
            g for f in findings if not f.error for g in f.knowledge_gaps
        })
        synthesis_gaps = data.get("knowledge_gaps", [])
        merged_gaps = list({*all_gaps, *synthesis_gaps})

        try:
            return ResearchReport(
                query=query,
                complexity=complexity,
                executive_summary=data.get("executive_summary", "[Synthesis missing]"),
                findings_by_domain=data.get("findings_by_domain", {}),
                structured_comparison=data.get("structured_comparison"),
                conflicts_detected=conflicts,
                knowledge_gaps=merged_gaps[:10],
                key_takeaways=data.get("key_takeaways", []),
                sources_cited=data.get("sources_cited", all_sources)[:20],
                completeness_score=int(data.get("completeness_score", 70)),
                budget_summary=budget_summary,
            )
        except Exception as exc:
            logger.warning("[synthesizer] model_validate failed: %s", exc)
            return self._fallback_report(query, findings, conflicts, budget_summary, complexity)

    # ------------------------------------------------------------------
    # Fallback report (no LLM — built from findings directly)
    # ------------------------------------------------------------------

    def _fallback_report(
        self,
        query: str,
        findings: List[ResearchFinding],
        conflicts: List[ConflictRecord],
        budget_summary: BudgetSummary,
        complexity: QueryComplexity,
    ) -> ResearchReport:
        valid = [f for f in findings if not f.error]
        all_sources = list({s for f in valid for s in f.sources_cited})
        all_gaps = list({g for f in valid for g in f.knowledge_gaps})

        # Build a simple executive summary from key points
        summary_parts = []
        for f in valid:
            if f.key_points:
                summary_parts.append(
                    f"**{f.agent_name}**: " + " ".join(f.key_points[:2])
                )
        executive_summary = (
            "\n\n".join(summary_parts)
            if summary_parts
            else "[Synthesis failed — see individual findings above]"
        )

        findings_by_domain = {
            f.role.value: f.answer[:800] for f in valid
        }

        key_takeaways = []
        for f in valid:
            key_takeaways.extend(f.key_points[:1])

        return ResearchReport(
            query=query,
            complexity=complexity,
            executive_summary=executive_summary,
            findings_by_domain=findings_by_domain,
            conflicts_detected=conflicts,
            knowledge_gaps=all_gaps[:8],
            key_takeaways=key_takeaways[:5],
            sources_cited=all_sources[:15],
            completeness_score=50,
            budget_summary=budget_summary,
        )
