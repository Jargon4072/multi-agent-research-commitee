"""
evaluation/eval_queries.py
===========================
The three test queries for Part 3 of the evaluation suite.

Each query definition includes:
- query_id:   Short label (Q1, Q2, Q3)
- query_text: The actual question
- complexity_expected: What scope we EXPECT the orchestrator to classify
- domains_expected: Which agent types should be involved
- probe:      What the evaluators are probing (from the spec)
- eval_notes: Evaluation-specific notes for each scoring dimension
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models.schemas import QueryComplexity, SubtaskType


@dataclass
class EvalQuery:
    """Definition of one evaluation test query."""

    query_id: str
    query_text: str
    complexity_expected: QueryComplexity
    domains_expected: List[SubtaskType]
    probe: str                  # What this query tests (from spec)
    completeness_notes: str     # What a complete answer must cover
    precision_notes: str        # What specific facts should appear
    attribution_notes: str      # What sources should be cited


TEST_QUERIES: List[EvalQuery] = [
    EvalQuery(
        query_id="Q1",
        query_text=(
            "Compare the environmental impact and regulatory landscape of "
            "lithium-ion vs. solid-state batteries for electric vehicles "
            "across the US, EU, and China."
        ),
        complexity_expected=QueryComplexity.BROAD,
        domains_expected=[
            SubtaskType.CHEMISTRY,
            SubtaskType.POLICY,
            SubtaskType.GEOGRAPHY,
            SubtaskType.COMPARISON,
        ],
        probe="Multi-domain decomposition, synthesis quality",
        completeness_notes=(
            "Must cover: (1) Li-ion environmental impact (mining, manufacturing, recycling), "
            "(2) solid-state environmental impact / projections, "
            "(3) US regulatory framework (IRA, EPA), "
            "(4) EU regulatory framework (EU Battery Regulation 2023/1542), "
            "(5) China regulatory framework (NEV policy, GB/T standards), "
            "(6) regional comparison across at least US, EU, China."
        ),
        precision_notes=(
            "Should include: EU Battery Regulation adoption date (August 2023); "
            "IRA battery content requirements; "
            "Li-ion carbon footprint range (typically 60-200 kg CO2e/kWh manufactured); "
            "solid-state TRL (4-6); "
            "China's EV market share (~35% BEV/PHEV in 2023); "
            "lithium mining geographic concentration (Australia, Chile, China)."
        ),
        attribution_notes=(
            "Should cite: EU Regulation 2023/1542; "
            "IRA Section 30D; "
            "specific lifecycle assessment studies or IPCC/IEA data; "
            "MIIT or GB/T standards for China; "
            "should flag conflicts if any chemistry data contradicts policy claims."
        ),
    ),

    EvalQuery(
        query_id="Q2",
        query_text=(
            "What is the EU's current battery recycling regulation and when does it take effect?"
        ),
        complexity_expected=QueryComplexity.NARROW,
        domains_expected=[SubtaskType.POLICY],
        probe="Precision on a focused factual question — does the system over-decompose?",
        completeness_notes=(
            "The answer needs: (1) identify the regulation by name and number, "
            "(2) state the recycling-specific requirements, "
            "(3) give the effective dates / phase-in timeline. "
            "It does NOT need: general EV market overview, chemistry of recycling, "
            "comparison to other jurisdictions, or historical context."
        ),
        precision_notes=(
            "MUST include: Regulation (EU) 2023/1542 (the EU Battery Regulation); "
            "entry into force: August 17, 2023; "
            "recycled content requirements taking effect from 2030; "
            "collection targets for portable batteries (51% by 2028, 73% by 2030); "
            "LiB EV battery end-of-life provisions."
        ),
        attribution_notes=(
            "Must cite: EU Regulation 2023/1542 (Official Journal of the EU); "
            "ideally with specific Article references for recycled content and collection. "
            "No generic 'EU law' citations."
        ),
    ),

    EvalQuery(
        query_id="Q3",
        query_text="Are solid-state batteries actually better?",
        complexity_expected=QueryComplexity.MODERATE,
        domains_expected=[SubtaskType.CHEMISTRY, SubtaskType.COMPARISON],
        probe="Handling vague input — clarification, scope-setting, graceful degradation",
        completeness_notes=(
            "A good answer should: (1) acknowledge the ambiguity ('better' in what sense?), "
            "(2) compare across multiple dimensions (energy density, safety, longevity, "
            "cost, environmental impact, commercial readiness), "
            "(3) give a nuanced conclusion that avoids false certainty. "
            "Ideally surfaces the 'better for whom and when?' framing."
        ),
        precision_notes=(
            "Should include: solid-state theoretical energy density advantage "
            "(vs ~250-300 Wh/kg for NMC Li-ion); "
            "safety advantage (no liquid electrolyte = no thermal runaway in same mode); "
            "current cost disadvantage (solid-state manufacturing not scaled); "
            "TRL gap (solid-state TRL 4-6, Li-ion TRL 9); "
            "commercial availability timeline (optimistic: 2027-2030 for automotive)."
        ),
        attribution_notes=(
            "Should cite specific studies or industry reports on energy density claims; "
            "acknowledge uncertainty of projections for solid-state; "
            "flag that 'better' claims in marketing differ from peer-reviewed data."
        ),
    ),
]


def get_query(query_id: str) -> EvalQuery:
    """Return the EvalQuery with the given ID (e.g. 'Q1')."""
    for q in TEST_QUERIES:
        if q.query_id == query_id:
            return q
    raise ValueError(f"No test query with id {query_id!r}")
