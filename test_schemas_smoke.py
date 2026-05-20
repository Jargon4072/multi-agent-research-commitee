"""
test_schemas_smoke.py
=====================
Smoke tests for the research domain schemas.
Run with: python test_schemas_smoke.py
"""
import sys
from datetime import datetime

def test_research_finding():
    from models.schemas import ResearchFinding, SubtaskStatus, SubtaskType
    f = ResearchFinding(
        subtask_id="test_task",
        agent_id="chemistry",
        agent_name="Dr. Chen · Chemistry Expert",
        role=SubtaskType.CHEMISTRY,
        query="What is the carbon footprint of Li-ion batteries?",
        answer="Li-ion battery manufacturing emits approximately 60-200 kg CO2e per kWh.",
        key_points=["60-200 kg CO2e/kWh manufacturing", "Varies by grid carbon intensity"],
        confidence=75,
        status=SubtaskStatus.COMPLETE,
    )
    assert f.subtask_id == "test_task"
    assert f.confidence == 75
    assert f.status == SubtaskStatus.COMPLETE
    print("  ✓ ResearchFinding")

def test_research_report():
    from models.schemas import ResearchReport, BudgetSummary, QueryComplexity
    report = ResearchReport(
        query="Compare Li-ion vs solid-state batteries",
        complexity=QueryComplexity.BROAD,
        executive_summary="Solid-state batteries offer theoretical advantages but remain pre-commercial.",
        findings_by_domain={"chemistry": "Li-ion is mature; solid-state is TRL 4-6."},
        key_takeaways=["Li-ion dominates current market", "Solid-state targets 2027-2030"],
        completeness_score=80,
        budget_summary=BudgetSummary(total_budget=10000, total_spent=5000, remaining=5000),
    )
    assert report.completeness_score == 80
    print("  ✓ ResearchReport")

def test_research_trace():
    from models.schemas import ResearchTrace, ResearchReport, BudgetSummary, QueryComplexity
    trace = ResearchTrace(
        query="Test query",
        report=ResearchReport(
            query="Test query",
            executive_summary="Test summary.",
            completeness_score=70,
            budget_summary=BudgetSummary(),
        ),
        budget_summary=BudgetSummary(),
    )
    assert trace.id is not None
    print("  ✓ ResearchTrace")

def test_subtask():
    from models.schemas import Subtask, SubtaskType
    st = Subtask(
        subtask_id="env_impact",
        agent_type=SubtaskType.CHEMISTRY,
        query="What is the lifecycle environmental impact of Li-ion batteries?",
    )
    assert st.agent_type == SubtaskType.CHEMISTRY
    print("  ✓ Subtask")

def test_eval_score():
    from models.schemas import EvalScore, EvalDimension
    score = EvalScore(
        query_id="Q1",
        dimension=EvalDimension.COMPLETENESS,
        score=4.0,
        justification="Good coverage but missed one region.",
    )
    assert score.score == 4.0
    print("  ✓ EvalScore")

def test_provider_factory():
    """Verify provider factory registers both anthropic and gemini."""
    from providers.factory import _REGISTRY
    assert "anthropic" in _REGISTRY, "anthropic provider not registered"
    assert "gemini" in _REGISTRY, "gemini provider not registered"
    print("  ✓ Provider factory (anthropic + gemini registered)")

def test_agent_factory():
    """Verify agent factory can build agent configs."""
    from agents.factory import _AGENT_BY_ROLE
    from models.schemas import SubtaskType
    assert SubtaskType.CHEMISTRY.value in _AGENT_BY_ROLE
    assert SubtaskType.POLICY.value in _AGENT_BY_ROLE
    assert SubtaskType.GEOGRAPHY.value in _AGENT_BY_ROLE
    print("  ✓ Agent factory (chemistry, policy, geography registered)")

def test_orchestrator_config():
    from orchestrator.research_config import ResearchConfig
    config = ResearchConfig(query="Test query")
    assert config.max_subtasks == 5
    assert config.enable_retry is True
    assert config.provider == "anthropic"
    print("  ✓ ResearchConfig")

def test_eval_queries():
    from evaluation.eval_queries import TEST_QUERIES, get_query
    assert len(TEST_QUERIES) == 3
    q1 = get_query("Q1")
    assert "lithium-ion" in q1.query_text.lower()
    q2 = get_query("Q2")
    assert "recycling" in q2.query_text.lower()
    q3 = get_query("Q3")
    assert "solid-state" in q3.query_text.lower()
    print("  ✓ Eval queries (Q1, Q2, Q3 defined)")

def main():
    print("\n=== Battery Research System — Schema Smoke Tests ===\n")
    tests = [
        test_research_finding,
        test_research_report,
        test_research_trace,
        test_subtask,
        test_eval_score,
        test_provider_factory,
        test_agent_factory,
        test_orchestrator_config,
        test_eval_queries,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed  |  {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("  All smoke tests passed ✓")

if __name__ == "__main__":
    main()
