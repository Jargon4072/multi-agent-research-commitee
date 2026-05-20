"""
orchestrator/research_state.py
================================
ResearchState — the streaming event payload yielded by ResearchOrchestrator.stream().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from models.schemas import (
    QueryComplexity,
    ResearchFinding,
    ResearchReport,
    Subtask,
)


@dataclass
class ResearchState:
    """Streaming event from the research orchestrator.

    Events
    ------
    ``decomposed``        — orchestrator produced subtask plan
    ``agent_started``     — a sub-agent is about to run
    ``agent_done``        — a sub-agent completed
    ``agent_retry``       — a retry is being issued for INSUFFICIENT result
    ``synthesis_started`` — final synthesis is starting
    ``synthesis_done``    — ResearchReport is ready (is_final=True)
    """

    event: str = ""
    query: str = ""
    complexity: Optional[QueryComplexity] = None
    subtasks: List[Subtask] = field(default_factory=list)
    current_agent: str = ""
    current_subtask: Optional[Subtask] = None
    latest_finding: Optional[ResearchFinding] = None
    all_findings: List[ResearchFinding] = field(default_factory=list)
    final_report: Optional[ResearchReport] = None
    is_final: bool = False
    tokens_used_so_far: int = 0
