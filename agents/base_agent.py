"""
agents/base_agent.py
====================
Base execution layer for every research sub-agent in the system.

Public surface
--------------
AgentConfig   – frozen dataclass carrying agent identity + system prompt
ResearchAgent – async research() method with prompt building, parse, and retry
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from models.schemas import ResearchFinding, SubtaskStatus, SubtaskType
from providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 2
_RETRY_SUFFIX: str = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY the JSON object described in the schema. No markdown fences."
)

# Required fields that every sub-agent response must contain
_REQUIRED_FIELDS = {"answer", "key_points", "confidence", "status"}


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Immutable identity + system prompt for one research sub-agent.

    Fields
    ------
    id:
        Stable machine identifier, e.g. ``"chemistry"``.
    name:
        Human display name, e.g. ``"Chemistry Expert"``.
    role:
        SubtaskType this agent handles.
    system_prompt:
        The full domain system prompt sent to the LLM.
    display_name:
        Rich display name, e.g. ``"Dr. Chen · Chemistry Expert"``.
    """

    id: str
    name: str
    role: SubtaskType
    system_prompt: str
    display_name: str


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Execution wrapper around one LLM provider call per research subtask.

    Concrete subclasses only need to call ``super().__init__(config, provider)``
    with the correct :class:`AgentConfig`.  All domain expertise lives in the
    system prompt.
    """

    def __init__(self, config: AgentConfig, provider: LLMProvider) -> None:
        self.config = config
        self.provider = provider
        self._id = config.id
        self._name = config.name
        self._role = config.role

        logger.debug(
            "[agent] Initialised %s (role=%s)", config.display_name, config.role.value
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research(
        self,
        subtask_id: str,
        query: str,
        context: List[ResearchFinding],
        max_tokens: int,
        attempt: int = 1,
    ) -> ResearchFinding:
        """Produce one :class:`~models.schemas.ResearchFinding` for the subtask.

        Parameters
        ----------
        subtask_id:
            Identifier linking this finding to its Subtask.
        query:
            The specific research question to answer.
        context:
            Findings from other agents already completed (may be empty).
        max_tokens:
            Hard token ceiling passed to the provider.
        attempt:
            Which attempt this is (1 = first, 2 = retry).
        """
        prompt = self._build_prompt(query, context)
        raw: str = ""
        last_error: Optional[str] = None
        tokens_used = 0
        latency_ms = 0.0

        for i in range(1 + _MAX_RETRIES):
            effective_prompt = prompt if i == 0 else prompt + _RETRY_SUFFIX
            t0 = time.perf_counter()

            try:
                response = await self.provider.complete(
                    prompt=effective_prompt,
                    max_tokens=max_tokens,
                    system=self.config.system_prompt,
                    temperature=0.4,   # moderate temp for balanced research
                )
                raw = response.text
                tokens_used = response.tokens_used
                latency_ms = response.latency_ms

            except Exception as exc:
                last_error = f"Provider error on attempt {i + 1}: {exc}"
                logger.warning(
                    "[agent:%s] provider error subtask=%s attempt=%d: %s",
                    self._id, subtask_id, i + 1, exc,
                )
                break

            finding = self._parse_response(
                raw=raw,
                subtask_id=subtask_id,
                query=query,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                attempt=attempt,
            )

            if finding.error is None:
                logger.info(
                    "[agent:%s] subtask=%s status=%s confidence=%d tokens=%d latency=%.0fms",
                    self._id, subtask_id,
                    finding.status.value, finding.confidence,
                    finding.tokens_used, finding.latency_ms,
                )
                return finding

            last_error = finding.error
            logger.warning(
                "[agent:%s] parse failure subtask=%s attempt=%d: %s",
                self._id, subtask_id, i + 1, last_error,
            )

        # All retries exhausted
        logger.error(
            "[agent:%s] all retries exhausted subtask=%s — returning error sentinel",
            self._id, subtask_id,
        )
        return self._error_finding(
            subtask_id=subtask_id,
            query=query,
            error=last_error or "Unknown error",
            attempt=attempt,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        context: List[ResearchFinding],
    ) -> str:
        """Build the user-turn prompt for this research agent.

        Context findings from other agents are injected so this agent
        can flag conflicts and avoid redundancy.
        """
        lines: List[str] = []

        lines.append(f"RESEARCH QUESTION: {query}")
        lines.append("")

        if context:
            lines.append("── Prior Findings From Other Agents ──────────────────────────")
            for f in context:
                if f.error:
                    continue  # skip failed findings
                lines.append(
                    f"[{f.agent_name} / {f.role.value}]  "
                    f"subtask_id={f.subtask_id}  "
                    f"confidence={f.confidence}/100"
                )
                for pt in f.key_points[:3]:
                    lines.append(f"  • {pt}")
                if f.knowledge_gaps:
                    lines.append(f"  Gaps: {'; '.join(f.knowledge_gaps[:2])}")
                lines.append("")
            lines.append("──────────────────────────────────────────────────────────────")
            lines.append("")
            lines.append(
                "Use the prior findings as context. If you see a factual conflict with "
                "another agent's claim, flag it in conflicts_with using that agent's "
                "subtask_id as the key."
            )
        else:
            lines.append(
                "This is the first research subtask — no prior findings yet. "
                "Answer based on your domain expertise."
            )

        lines.append("")
        lines.append(
            "Respond with ONLY a valid JSON object. "
            "No preamble. No markdown fences. No explanation outside the JSON."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        subtask_id: str,
        query: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        attempt: int = 1,
    ) -> ResearchFinding:
        """Parse raw LLM response into a :class:`ResearchFinding`."""
        # 1. Strip markdown fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()

        # 2. Extract first JSON object if embedded in prose
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)

        # 3. Repair truncated JSON
        def _repair(s: str) -> str:
            stack = []
            in_str = False
            escape = False
            for ch in s:
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch == "{":
                        stack.append("}")
                    elif ch == "[":
                        stack.append("]")
                    elif ch in ("}", "]"):
                        if stack and stack[-1] == ch:
                            stack.pop()
            return s + "".join(reversed(stack))

        try:
            payload: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                payload = json.loads(_repair(cleaned))
            except json.JSONDecodeError as exc:
                return self._error_finding(
                    subtask_id=subtask_id,
                    query=query,
                    error=f"JSON parse failed: {exc}. Raw (first 300): {raw[:300]}",
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    attempt=attempt,
                )

        # 4. Check required fields
        missing = _REQUIRED_FIELDS - set(payload.keys())
        if missing:
            return self._error_finding(
                subtask_id=subtask_id,
                query=query,
                error=f"Missing required fields: {missing}",
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                attempt=attempt,
            )

        # 5. Coerce status enum
        status_raw = payload.get("status", "COMPLETE")
        try:
            status = SubtaskStatus(status_raw)
        except ValueError:
            status = SubtaskStatus.PARTIAL

        # 6. Build ResearchFinding
        try:
            return ResearchFinding(
                subtask_id=subtask_id,
                agent_id=self._id,
                agent_name=self.config.display_name,
                role=self._role,
                query=query,
                answer=payload.get("answer", ""),
                key_points=payload.get("key_points", []),
                knowledge_gaps=payload.get("knowledge_gaps", []),
                conflicts_with=payload.get("conflicts_with", {}),
                confidence=int(payload.get("confidence", 70)),
                sources_cited=payload.get("sources_cited", []),
                status=status,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                attempt=attempt,
            )
        except Exception as exc:
            return self._error_finding(
                subtask_id=subtask_id,
                query=query,
                error=f"Model validation failed: {exc}",
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                attempt=attempt,
            )

    # ------------------------------------------------------------------
    # Error sentinel factory
    # ------------------------------------------------------------------

    def _error_finding(
        self,
        subtask_id: str,
        query: str,
        error: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        attempt: int = 1,
    ) -> ResearchFinding:
        """Return a safe default ResearchFinding carrying an error description."""
        return ResearchFinding(
            subtask_id=subtask_id,
            agent_id=self._id,
            agent_name=self.config.display_name,
            role=self._role,
            query=query,
            answer="[Agent failed to produce a valid response]",
            status=SubtaskStatus.INSUFFICIENT,
            confidence=0,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            error=error,
            attempt=attempt,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self._id!r}, role={self._role.value!r}, "
            f"provider={self.provider.provider_name!r})"
        )
