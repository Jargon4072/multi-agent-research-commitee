"""
orchestrator/research_config.py
================================
Configuration dataclass for the ResearchOrchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ResearchConfig:
    """Configuration for one research run.

    Fields
    ------
    query:
        The user's research question.
    provider:
        LLM provider name: ``"anthropic"`` (default) or ``"gemini"``.
    token_budget:
        Total token budget for the entire run (decomposition + agents + synthesis).
    tokens_per_agent:
        Max tokens allocated to each sub-agent call.
    max_subtasks:
        Hard cap on the number of subtasks the orchestrator can create.
        Prevents over-decomposition even if the LLM tries to produce more.
    enable_synthesis:
        If True, run the synthesis agent after all sub-agents complete.
    enable_retry:
        If True, retry INSUFFICIENT findings once using the retrieval fallback agent.
    output_dir:
        Directory where trace JSON files are saved.
    """

    query: str
    provider: str = "anthropic"
    token_budget: int = 80_000
    tokens_per_agent: int = 4_000
    max_subtasks: int = 5  # hard cap to prevent over-decomposition
    enable_synthesis: bool = True
    enable_retry: bool = True
    output_dir: str = "traces"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "token_budget": self.token_budget,
            "tokens_per_agent": self.tokens_per_agent,
            "max_subtasks": self.max_subtasks,
            "enable_synthesis": self.enable_synthesis,
            "enable_retry": self.enable_retry,
            "output_dir": self.output_dir,
        }
