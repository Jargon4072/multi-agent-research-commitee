"""
orchestrator/research_orchestrator.py
======================================
ResearchOrchestrator — decomposes queries, routes to sub-agents, synthesizes results.

Public surface
--------------
ResearchOrchestrator(config)
    .run(query)     → ResearchTrace  (batch, saves JSON)
    .stream(query)  → AsyncIterator[ResearchState]  (streaming)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Tuple

from agents.factory import build_all, build_synthesis_agent
from agents.base_agent import ResearchAgent
from models.schemas import (
    BudgetAllocation,
    BudgetSummary,
    QueryComplexity,
    ResearchFinding,
    ResearchReport,
    ResearchTrace,
    SubtaskStatus,
    SubtaskType,
    Subtask,
)
from orchestrator.research_config import ResearchConfig
from orchestrator.research_state import ResearchState
from providers.factory import get_provider
from agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_MAX_DECOMPOSE_RETRIES = 2

# Fallback: narrow query routing to single best agent
_NARROW_ROUTING_KEYWORDS: Dict[str, SubtaskType] = {
    "regulation": SubtaskType.POLICY,
    "directive": SubtaskType.POLICY,
    "law": SubtaskType.POLICY,
    "rule": SubtaskType.POLICY,
    "compliance": SubtaskType.POLICY,
    "recycling rate": SubtaskType.CHEMISTRY,
    "energy density": SubtaskType.CHEMISTRY,
    "lifecycle": SubtaskType.CHEMISTRY,
    "supply chain": SubtaskType.GEOGRAPHY,
    "market share": SubtaskType.GEOGRAPHY,
    "adoption": SubtaskType.GEOGRAPHY,
}


class ResearchOrchestrator:
    """Manages the full research pipeline: decompose → route → gather → synthesize.

    Parameters
    ----------
    config:
        A :class:`~orchestrator.research_config.ResearchConfig` instance.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self._provider = get_provider(config.provider)
        self._agents: Dict[str, ResearchAgent] = build_all(self._provider)
        self._synthesizer = build_synthesis_agent(self._provider)
        self._allocations: List[BudgetAllocation] = []
        self._total_tokens_used: int = 0
        self._retry_count: int = 0

        logger.info(
            "[orchestrator] Initialised with provider=%s agents=%s",
            config.provider, list(self._agents.keys()),
        )

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    async def run(self) -> ResearchTrace:
        """Execute the full research pipeline and return a :class:`ResearchTrace`."""
        query = self.config.query
        start_ms = time.perf_counter() * 1000

        logger.info("[orchestrator] Starting research run for query: %s", query[:80])

        # 1. Decompose query into subtasks
        subtasks, complexity = await self._decompose(query)
        logger.info(
            "[orchestrator] Complexity=%s  subtasks=%d: %s",
            complexity.value,
            len(subtasks),
            [s.subtask_id for s in subtasks],
        )

        # 2. Run sub-agents (with retry/fallback — Part 4)
        findings = await self._run_subtasks(subtasks)

        # 3. Synthesize
        budget_summary = self._build_budget_summary()
        report = await self._synthesizer.synthesize(
            query=query,
            findings=findings,
            budget_summary=budget_summary,
            complexity=complexity,
        )

        # 4. Build trace
        trace = ResearchTrace(
            query=query,
            complexity=complexity,
            subtasks=subtasks,
            findings=findings,
            report=report,
            budget_summary=budget_summary,
            config=self.config.as_dict(),
        )

        # 5. Auto-save
        await self._save(trace)
        return trace

    # ------------------------------------------------------------------
    # Streaming run
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[ResearchState]:
        """Same pipeline as run() but yields a ResearchState after each event.

        Events yielded:
        - "decomposed"        — after orchestrator produces subtask plan
        - "agent_started"     — before each sub-agent call
        - "agent_done"        — after each sub-agent completes
        - "agent_retry"       — when a retry is issued
        - "synthesis_started" — before final synthesis
        - "synthesis_done"    — after ResearchReport is built (is_final=True)
        """
        query = self.config.query

        # 1. Decompose
        subtasks, complexity = await self._decompose(query)
        yield ResearchState(
            event="decomposed",
            query=query,
            complexity=complexity,
            subtasks=subtasks,
        )

        # 2. Run subtasks sequentially for streaming (so we yield per-agent events)
        findings: List[ResearchFinding] = []
        prior_findings: List[ResearchFinding] = []

        for subtask in subtasks:
            agent = self._get_agent_for_subtask(subtask)
            if agent is None:
                logger.warning("[orchestrator] No agent for subtask %s", subtask.subtask_id)
                continue

            yield ResearchState(
                event="agent_started",
                query=query,
                complexity=complexity,
                current_agent=agent.config.display_name,
                current_subtask=subtask,
            )

            # First attempt
            finding = await self._run_single_subtask(subtask, agent, prior_findings, attempt=1)

            # Retry if insufficient — Part 4
            if (
                self.config.enable_retry
                and finding.status == SubtaskStatus.INSUFFICIENT
                and not finding.error
            ):
                self._retry_count += 1
                yield ResearchState(
                    event="agent_retry",
                    query=query,
                    complexity=complexity,
                    current_agent=agent.config.display_name,
                    current_subtask=subtask,
                    latest_finding=finding,
                )
                logger.info(
                    "[orchestrator] Retrying subtask=%s with retrieval fallback",
                    subtask.subtask_id,
                )
                # Fallback to retrieval agent (Part 4)
                fallback_agent = self._agents.get("retrieval", agent)
                rephrased_subtask = Subtask(
                    subtask_id=subtask.subtask_id + "_retry",
                    agent_type=SubtaskType.RETRIEVAL,
                    query=subtask.query,
                    context_hint=f"Primary agent ({agent.config.display_name}) returned INSUFFICIENT. "
                                 f"Please attempt this question using your cross-domain knowledge: {subtask.query}",
                )
                finding = await self._run_single_subtask(
                    rephrased_subtask, fallback_agent, prior_findings, attempt=2
                )
                finding.subtask_id = subtask.subtask_id  # re-link to original

            findings.append(finding)
            prior_findings.append(finding)

            yield ResearchState(
                event="agent_done",
                query=query,
                complexity=complexity,
                current_agent=agent.config.display_name,
                current_subtask=subtask,
                latest_finding=finding,
                all_findings=findings,
            )

        # 3. Synthesis
        yield ResearchState(
            event="synthesis_started",
            query=query,
            complexity=complexity,
            all_findings=findings,
        )

        budget_summary = self._build_budget_summary()
        report = await self._synthesizer.synthesize(
            query=query,
            findings=findings,
            budget_summary=budget_summary,
            complexity=complexity,
        )

        trace = ResearchTrace(
            query=query,
            complexity=complexity,
            subtasks=subtasks,
            findings=findings,
            report=report,
            budget_summary=budget_summary,
            config=self.config.as_dict(),
        )
        await self._save(trace)

        yield ResearchState(
            event="synthesis_done",
            query=query,
            complexity=complexity,
            all_findings=findings,
            final_report=report,
            is_final=True,
        )

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    async def _decompose(self, query: str) -> Tuple[List[Subtask], QueryComplexity]:
        """Use the LLM to decompose the query into subtasks.

        Implements smart scope detection to avoid over-decomposition.
        Falls back to heuristic routing if the LLM decomposition fails.
        """
        decompose_prompt = (
            f"Research query to analyze and decompose:\n\n{query}\n\n"
            "Analyze the query complexity and produce a decomposition plan."
        )

        for attempt in range(_MAX_DECOMPOSE_RETRIES):
            try:
                response = await self._provider.complete(
                    prompt=decompose_prompt,
                    max_tokens=2048,
                    system=ORCHESTRATOR_SYSTEM_PROMPT,
                    temperature=0.1,  # very deterministic for decomposition
                )
                subtasks, complexity = self._parse_decomposition(response.text)
                self._total_tokens_used += response.tokens_used

                # Enforce max subtask limit
                if len(subtasks) > self.config.max_subtasks:
                    logger.warning(
                        "[orchestrator] LLM produced %d subtasks, capping at %d",
                        len(subtasks), self.config.max_subtasks,
                    )
                    subtasks = subtasks[:self.config.max_subtasks]

                return subtasks, complexity

            except Exception as exc:
                logger.warning(
                    "[orchestrator] decomposition attempt %d failed: %s", attempt + 1, exc
                )

        # Fallback: heuristic decomposition
        logger.warning("[orchestrator] falling back to heuristic decomposition")
        return self._heuristic_decompose(query)

    def _parse_decomposition(
        self, raw: str
    ) -> Tuple[List[Subtask], QueryComplexity]:
        """Parse the orchestrator's JSON decomposition plan."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()

        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        data = json.loads(cleaned)

        # Parse complexity
        complexity_raw = data.get("complexity", "broad")
        try:
            complexity = QueryComplexity(complexity_raw)
        except ValueError:
            complexity = QueryComplexity.BROAD

        # Parse subtasks
        subtasks_raw = data.get("subtasks", [])
        subtasks: List[Subtask] = []
        for st in subtasks_raw:
            try:
                agent_type = SubtaskType(st.get("agent_type", "retrieval"))
            except ValueError:
                agent_type = SubtaskType.RETRIEVAL
            subtasks.append(Subtask(
                subtask_id=st.get("subtask_id", f"task_{len(subtasks)+1}"),
                agent_type=agent_type,
                query=st.get("query", ""),
                context_hint=st.get("context_hint", ""),
            ))

        if not subtasks:
            raise ValueError("No subtasks in decomposition response")

        return subtasks, complexity

    def _heuristic_decompose(self, query: str) -> Tuple[List[Subtask], QueryComplexity]:
        """Fallback: keyword-based routing for when LLM decomposition fails."""
        q_lower = query.lower()

        # Check for narrow query indicators
        narrow_indicators = [
            "what is", "when does", "when did", "what year", "which regulation",
            "define", "explain what", "how much", "what percentage",
        ]
        is_narrow = any(q_lower.startswith(ind) for ind in narrow_indicators)

        if is_narrow:
            # Route to single best agent
            for kw, role in _NARROW_ROUTING_KEYWORDS.items():
                if kw in q_lower:
                    return [
                        Subtask(
                            subtask_id="main_task",
                            agent_type=role,
                            query=query,
                        )
                    ], QueryComplexity.NARROW

            # Default narrow routing to policy (most common narrow questions)
            return [
                Subtask(subtask_id="main_task", agent_type=SubtaskType.POLICY, query=query)
            ], QueryComplexity.NARROW

        # Broad fallback: basic 3-agent decomposition
        subtasks = [
            Subtask(
                subtask_id="chemistry_env",
                agent_type=SubtaskType.CHEMISTRY,
                query=f"What is the environmental impact of lithium-ion vs solid-state batteries? Query context: {query}",
            ),
            Subtask(
                subtask_id="policy_regulatory",
                agent_type=SubtaskType.POLICY,
                query=f"What are the regulatory frameworks governing EV batteries? Query context: {query}",
            ),
            Subtask(
                subtask_id="geo_regional",
                agent_type=SubtaskType.GEOGRAPHY,
                query=f"How do regional factors (US, EU, China) affect EV battery landscape? Query context: {query}",
            ),
        ]
        return subtasks, QueryComplexity.BROAD

    # ------------------------------------------------------------------
    # Subtask execution
    # ------------------------------------------------------------------

    async def _run_subtasks(self, subtasks: List[Subtask]) -> List[ResearchFinding]:
        """Run all subtasks.

        For broad queries (3+ subtasks), runs in two waves:
        - Wave 1: primary specialists (chemistry, policy, geography) in parallel
        - Wave 2: comparison/retrieval agents with context from wave 1

        For narrow/moderate queries, runs everything in parallel.
        """
        if len(subtasks) <= 2:
            # Simple parallel execution
            return await self._run_wave(subtasks, prior_findings=[])

        # Split into specialist wave and aggregator wave
        specialist_types = {SubtaskType.CHEMISTRY, SubtaskType.POLICY, SubtaskType.GEOGRAPHY}
        wave1 = [s for s in subtasks if s.agent_type in specialist_types]
        wave2 = [s for s in subtasks if s.agent_type not in specialist_types]

        prior_findings = await self._run_wave(wave1, prior_findings=[])

        if wave2:
            wave2_findings = await self._run_wave(wave2, prior_findings=prior_findings)
            return prior_findings + wave2_findings

        return prior_findings

    async def _run_wave(
        self,
        subtasks: List[Subtask],
        prior_findings: List[ResearchFinding],
    ) -> List[ResearchFinding]:
        """Run a wave of subtasks concurrently."""
        tasks = []
        for subtask in subtasks:
            agent = self._get_agent_for_subtask(subtask)
            if agent:
                tasks.append(
                    self._run_single_subtask(subtask, agent, prior_findings, attempt=1)
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: List[ResearchFinding] = []
        for subtask, result in zip(subtasks, results):
            if isinstance(result, ResearchFinding):
                # Retry if needed (Part 4)
                if (
                    self.config.enable_retry
                    and result.status == SubtaskStatus.INSUFFICIENT
                ):
                    self._retry_count += 1
                    logger.info(
                        "[orchestrator] Retrying subtask=%s (INSUFFICIENT) with retrieval fallback",
                        subtask.subtask_id,
                    )
                    fallback_agent = self._agents.get("retrieval",
                        self._get_agent_for_subtask(subtask))
                    retry_subtask = Subtask(
                        subtask_id=subtask.subtask_id + "_retry",
                        agent_type=SubtaskType.RETRIEVAL,
                        query=subtask.query,
                        context_hint=f"Retry of INSUFFICIENT result for: {subtask.query}",
                    )
                    retry_result = await self._run_single_subtask(
                        retry_subtask, fallback_agent, prior_findings, attempt=2
                    )
                    retry_result.subtask_id = subtask.subtask_id
                    findings.append(retry_result)
                else:
                    findings.append(result)
            else:
                logger.error(
                    "[orchestrator] subtask %s raised: %s", subtask.subtask_id, result
                )
                # Create error finding
                agent = self._get_agent_for_subtask(subtask)
                if agent:
                    findings.append(agent._error_finding(
                        subtask_id=subtask.subtask_id,
                        query=subtask.query,
                        error=str(result),
                    ))

        return findings

    async def _run_single_subtask(
        self,
        subtask: Subtask,
        agent: ResearchAgent,
        prior_findings: List[ResearchFinding],
        attempt: int = 1,
    ) -> ResearchFinding:
        """Run a single subtask through its designated agent and log the call."""
        t0 = time.perf_counter()
        ts = datetime.utcnow()

        logger.info(
            "[%s][%s] START  subtask=%s attempt=%d  ts=%s",
            agent.config.id, agent.config.display_name,
            subtask.subtask_id, attempt,
            ts.isoformat(),
        )

        finding = await agent.research(
            subtask_id=subtask.subtask_id,
            query=subtask.query,
            context=prior_findings,
            max_tokens=self.config.tokens_per_agent,
            attempt=attempt,
        )

        latency = (time.perf_counter() - t0) * 1000
        self._total_tokens_used += finding.tokens_used

        self._allocations.append(BudgetAllocation(
            agent_id=agent.config.id,
            subtask_id=subtask.subtask_id,
            tokens_allocated=self.config.tokens_per_agent,
            tokens_used=finding.tokens_used,
        ))

        logger.info(
            "[%s][%s] DONE   subtask=%s status=%s confidence=%d tokens=%d latency=%.0fms",
            agent.config.id, agent.config.display_name,
            subtask.subtask_id,
            finding.status.value,
            finding.confidence,
            finding.tokens_used,
            latency,
        )

        return finding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_agent_for_subtask(self, subtask: Subtask) -> Optional[ResearchAgent]:
        """Return the agent for a subtask's agent_type, or None if not found."""
        agent = self._agents.get(subtask.agent_type.value)
        if agent is None:
            logger.warning(
                "[orchestrator] No agent registered for type %s", subtask.agent_type.value
            )
        return agent

    def _build_budget_summary(self) -> BudgetSummary:
        total_budget = self.config.token_budget
        total_spent = sum(a.tokens_used for a in self._allocations)
        remaining = max(0, total_budget - total_spent)
        pct = (total_spent / total_budget * 100) if total_budget > 0 else 0.0
        return BudgetSummary(
            total_budget=total_budget,
            total_spent=total_spent,
            remaining=remaining,
            utilization_pct=round(pct, 2),
            allocations=self._allocations,
            retry_count=self._retry_count,
        )

    async def _save(self, trace: ResearchTrace) -> None:
        """Persist trace as JSON to output_dir/{id}.json."""
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            path = os.path.join(self.config.output_dir, f"{trace.id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(trace.model_dump_json(indent=2))
            logger.info("[orchestrator] trace saved → %s", path)
        except Exception as exc:
            logger.error("[orchestrator] failed to save trace: %s", exc)
