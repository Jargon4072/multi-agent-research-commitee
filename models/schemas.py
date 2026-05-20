"""
models/schemas.py
=================
Pydantic v2 schemas for the Battery Research Multi-Agent System.

Models
------
1.  SubtaskStatus      — completion status of a sub-agent's research task
2.  SubtaskType        — which domain/role handles the subtask
3.  QueryComplexity    — orchestrator's scope assessment for a query
4.  Subtask            — a decomposed research task with routing metadata
5.  ResearchFinding    — one sub-agent's output for its assigned subtask
6.  ConflictRecord     — a tracked factual conflict between two findings
7.  BudgetAllocation   — per-agent token accounting (reused from original)
8.  BudgetSummary      — rolled-up token budget view (reused from original)
9.  ResearchReport     — final synthesized answer across all findings
10. ResearchTrace      — the full persisted record of a research run
11. EvalDimension      — scoring dimensions for the evaluation suite
12. EvalScore          — a single dimension score for a query
13. EvalResult         — all dimension scores + analysis for one query
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubtaskStatus(str, Enum):
    """Completion status returned by a sub-agent for its assigned subtask."""

    COMPLETE = "COMPLETE"         # Full, confident answer provided
    PARTIAL = "PARTIAL"           # Answer provided but with gaps noted
    INSUFFICIENT = "INSUFFICIENT" # Could not adequately answer the subtask
    CONFLICT = "CONFLICT"         # Answer contradicts another finding


class SubtaskType(str, Enum):
    """Domain/role that handles a particular subtask."""

    CHEMISTRY = "chemistry"       # Electrochemistry, battery science, lifecycle
    POLICY = "policy"             # Regulations, legislation, standards
    GEOGRAPHY = "geography"       # Regional differences, supply chain, geopolitics
    RETRIEVAL = "retrieval"       # Cross-domain fact synthesis, gap filling
    COMPARISON = "comparison"     # Structured head-to-head analysis
    SYNTHESIS = "synthesis"       # Final report generation (synthesis agent)


class QueryComplexity(str, Enum):
    """Orchestrator's assessment of how complex/broad a user query is."""

    NARROW = "narrow"     # Single domain, single fact → route to 1 agent
    MODERATE = "moderate" # 2-3 domains → route to 2-3 agents
    BROAD = "broad"       # Multi-domain, multi-region → full agent panel


# ---------------------------------------------------------------------------
# Model 1 — Subtask
# ---------------------------------------------------------------------------


class Subtask(BaseModel):
    """A decomposed research task produced by the orchestrator."""

    model_config = ConfigDict(frozen=False)

    subtask_id: str = Field(..., description="Short identifier, e.g. 'env_impact_li_ion'.")
    agent_type: SubtaskType = Field(..., description="Which specialized agent handles this.")
    query: str = Field(..., description="The specific research question for this sub-agent.")
    context_hint: str = Field(
        default="",
        description="Optional context or framing to help the agent focus.",
    )


# ---------------------------------------------------------------------------
# Model 2 — ResearchFinding
# ---------------------------------------------------------------------------


class ResearchFinding(BaseModel):
    """Output produced by a single research sub-agent for its assigned subtask."""

    model_config = ConfigDict(frozen=False)

    subtask_id: str = Field(..., description="Links this finding to its Subtask.")
    agent_id: str = Field(..., description="Stable machine identifier, e.g. 'chemistry'.")
    agent_name: str = Field(..., description="Human display name, e.g. 'Chemistry Expert'.")
    role: SubtaskType = Field(..., description="Functional role of this agent.")
    query: str = Field(..., description="The subtask query this finding answers.")
    answer: str = Field(..., description="The substantive research answer (markdown OK).")
    key_points: List[str] = Field(
        default_factory=list,
        description="3-5 bullet points distilling the most important facts.",
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list,
        description="Explicit gaps or uncertainties in this finding.",
    )
    conflicts_with: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of {subtask_id: description_of_conflict}.",
    )
    confidence: int = Field(
        default=70, ge=0, le=100,
        description="Agent's confidence in this answer (0-100).",
    )
    sources_cited: List[str] = Field(
        default_factory=list,
        description="Data sources, regulations, or documents referenced.",
    )
    status: SubtaskStatus = Field(
        default=SubtaskStatus.COMPLETE,
        description="Completion quality of this finding.",
    )
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = Field(default=None, description="Non-null if the agent failed.")
    attempt: int = Field(default=1, ge=1, description="Which attempt produced this finding (1 = first).")


# ---------------------------------------------------------------------------
# Model 3 — ConflictRecord
# ---------------------------------------------------------------------------


class ConflictRecord(BaseModel):
    """A factual conflict detected between two sub-agent findings."""

    model_config = ConfigDict(frozen=False)

    subtask_a: str = Field(..., description="subtask_id of the first finding.")
    subtask_b: str = Field(..., description="subtask_id of the second finding.")
    dimension: str = Field(..., description="What aspect they disagree on, e.g. 'recycling rate'.")
    claim_a: str = Field(..., description="What finding A claims.")
    claim_b: str = Field(..., description="What finding B claims.")
    resolved: bool = Field(default=False)
    resolution: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Models 4–5 — Budget (reused from original)
# ---------------------------------------------------------------------------


class BudgetAllocation(BaseModel):
    """Per-agent, per-subtask token budget record."""

    model_config = ConfigDict(frozen=False)

    agent_id: str
    subtask_id: str
    tokens_allocated: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BudgetSummary(BaseModel):
    """Rolled-up token budget view across all agents and subtasks."""

    model_config = ConfigDict(frozen=False)

    total_budget: int = Field(default=0, ge=0)
    total_spent: int = Field(default=0, ge=0)
    remaining: int = Field(default=0)
    utilization_pct: float = Field(default=0.0, ge=0.0)
    allocations: List[BudgetAllocation] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0, description="Number of agent retries issued.")


# ---------------------------------------------------------------------------
# Model 6 — ResearchReport
# ---------------------------------------------------------------------------


class ResearchReport(BaseModel):
    """The final synthesized answer produced after all sub-agent findings are gathered."""

    model_config = ConfigDict(frozen=False)

    query: str = Field(..., description="The original user query.")
    complexity: QueryComplexity = Field(
        default=QueryComplexity.BROAD,
        description="Orchestrator's scope assessment.",
    )
    executive_summary: str = Field(
        ..., description="2-3 paragraph synthesized answer to the original query."
    )
    findings_by_domain: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of domain name → narrative summary for that domain.",
    )
    structured_comparison: Optional[str] = Field(
        default=None,
        description="Optional structured comparison table or matrix (markdown).",
    )
    conflicts_detected: List[ConflictRecord] = Field(
        default_factory=list,
        description="Factual conflicts found across findings.",
    )
    knowledge_gaps: List[str] = Field(
        default_factory=list,
        description="Consolidated list of gaps from all sub-agents.",
    )
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="Top 5 most important conclusions.",
    )
    sources_cited: List[str] = Field(
        default_factory=list,
        description="All unique sources referenced across findings.",
    )
    completeness_score: int = Field(
        default=0, ge=0, le=100,
        description="Synthesis agent's self-assessment of answer completeness.",
    )
    budget_summary: BudgetSummary = Field(
        default_factory=BudgetSummary,
    )


# ---------------------------------------------------------------------------
# Model 7 — ResearchTrace
# ---------------------------------------------------------------------------


class ResearchTrace(BaseModel):
    """The full persisted record of a completed research run."""

    model_config = ConfigDict(frozen=False)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query: str = Field(..., description="The original user query.")
    complexity: QueryComplexity = Field(default=QueryComplexity.BROAD)
    subtasks: List[Subtask] = Field(default_factory=list)
    findings: List[ResearchFinding] = Field(default_factory=list)
    report: ResearchReport = Field(..., description="Final synthesized report.")
    budget_summary: BudgetSummary = Field(default_factory=BudgetSummary)
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Models 8-9 — Evaluation
# ---------------------------------------------------------------------------


class EvalDimension(str, Enum):
    """Scoring dimensions for the evaluation suite."""

    COMPLETENESS = "completeness"           # Covers all required sub-topics?
    FACTUAL_PRECISION = "factual_precision" # Are claims accurate and sourced?
    ATTRIBUTION_QUALITY = "attribution"     # Sources cited, conflicts flagged?


class EvalScore(BaseModel):
    """A single dimension score for one query evaluation."""

    model_config = ConfigDict(frozen=False)

    query_id: str = Field(..., description="e.g. 'Q1', 'Q2', 'Q3'.")
    dimension: EvalDimension
    score: float = Field(..., ge=0, le=5, description="Score 0-5.")
    justification: str = Field(..., description="Reasoning for the score.")
    rubric: str = Field(default="", description="The rubric used for this dimension.")


class EvalResult(BaseModel):
    """All dimension scores + analysis for one query."""

    model_config = ConfigDict(frozen=False)

    query_id: str
    query_text: str
    complexity_detected: QueryComplexity
    agents_used: List[str] = Field(default_factory=list)
    total_latency_ms: float = Field(default=0.0)
    tokens_used: int = Field(default=0)
    scores: List[EvalScore] = Field(default_factory=list)
    total_score: float = Field(default=0.0, description="Sum of all dimension scores (max 15).")
    analysis: str = Field(default="", description="~200-word analysis of what scores reveal.")
    report_snippet: str = Field(default="", description="First 500 chars of executive summary.")
