"""
cli/app.py
===========
Typer CLI for the Battery Research Multi-Agent System.

Commands
--------
research     Run a research query through the multi-agent pipeline.
eval         Run the Q1-Q3 evaluation suite and display scored results.
show-trace   Pretty-print a saved research trace JSON.
list-traces  List all saved research traces.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

app = typer.Typer(
    name="battery-research",
    help="Multi-agent research system for battery technology & EV policy questions.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# research command
# ---------------------------------------------------------------------------

@app.command()
def research(
    query: str = typer.Argument(..., help="The research question to investigate."),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider: anthropic (default) or gemini."),
    budget: int = typer.Option(80_000, "--budget", "-b", help="Total token budget."),
    tokens_per_agent: int = typer.Option(4_000, "--tokens-per-agent", "-t", help="Max tokens per sub-agent call."),
    max_subtasks: int = typer.Option(5, "--max-subtasks", "-s", help="Max subtasks (anti-over-decomposition cap)."),
    output_dir: str = typer.Option("traces", "--output-dir", "-o", help="Directory for trace JSON files."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Skip synthesis step."),
) -> None:
    """Run a multi-agent research query and display the structured report."""
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    asyncio.run(_run_research(
        query=query,
        provider=provider,
        budget=budget,
        tokens_per_agent=tokens_per_agent,
        max_subtasks=max_subtasks,
        output_dir=output_dir,
        enable_synthesis=not no_synthesis,
    ))


async def _run_research(
    query: str,
    provider: str,
    budget: int,
    tokens_per_agent: int,
    max_subtasks: int,
    output_dir: str,
    enable_synthesis: bool,
) -> None:
    from cli.display import ResearchDisplay
    from orchestrator.research_config import ResearchConfig
    from orchestrator.research_orchestrator import ResearchOrchestrator

    config = ResearchConfig(
        query=query,
        provider=provider,
        token_budget=budget,
        tokens_per_agent=tokens_per_agent,
        max_subtasks=max_subtasks,
        enable_synthesis=enable_synthesis,
        enable_retry=True,
        output_dir=output_dir,
    )

    display = ResearchDisplay(query=query)
    orchestrator = ResearchOrchestrator(config)
    start = time.time()
    final_report = None
    output_path = ""

    console.print(Panel(
        f"[bold cyan]Research Query[/bold cyan]\n{query}",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print(f"[dim]Provider: {provider}  |  Budget: {budget:,} tokens  |  Max subtasks: {max_subtasks}[/dim]\n")

    async for event in orchestrator.stream():
        if event.event == "decomposed":
            complexity_color = {"narrow": "green", "moderate": "yellow", "broad": "magenta"}.get(
                event.complexity.value if event.complexity else "broad", "white"
            )
            console.print(
                f"[bold]Scope:[/bold] [{complexity_color}]{event.complexity.value.upper() if event.complexity else '?'}[/{complexity_color}]"
                f"  →  [dim]{len(event.subtasks)} subtask(s)[/dim]"
            )
            for st in event.subtasks:
                console.print(f"  [dim cyan]• [{st.agent_type.value}][/dim cyan] {st.query[:70]}{'…' if len(st.query) > 70 else ''}")
            console.print()

        elif event.event == "agent_started":
            console.print(f"[bold yellow]▶[/bold yellow] [cyan]{event.current_agent}[/cyan] starting…")

        elif event.event == "agent_retry":
            console.print(
                f"  [bold red]↺[/bold red] Retrying [cyan]{event.current_agent}[/cyan] "
                f"(INSUFFICIENT) → fallback to retrieval agent"
            )

        elif event.event == "agent_done" and event.latest_finding:
            f = event.latest_finding
            if f.error:
                console.print(f"  [red]✗[/red] {f.agent_name}: ERROR — {f.error[:80]}")
            else:
                status_icon = {"COMPLETE": "✓", "PARTIAL": "~", "INSUFFICIENT": "!", "CONFLICT": "⚠"}.get(f.status.value, "?")
                status_color = {"COMPLETE": "green", "PARTIAL": "yellow", "INSUFFICIENT": "red", "CONFLICT": "magenta"}.get(f.status.value, "white")
                console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] [bold]{f.agent_name}[/bold] "
                    f"[dim]{f.status.value}[/dim]  conf={f.confidence}/100  "
                    f"[dim]{f.tokens_used} tokens  {f.latency_ms:.0f}ms[/dim]"
                )
                for pt in f.key_points[:2]:
                    console.print(f"    [dim]• {pt[:90]}{'…' if len(pt) > 90 else ''}[/dim]")
                if f.knowledge_gaps:
                    console.print(f"    [dim yellow]Gaps: {f.knowledge_gaps[0][:80]}[/dim yellow]")

        elif event.event == "synthesis_started":
            console.print("\n[bold magenta]⟳[/bold magenta] Synthesizing findings…")

        elif event.event == "synthesis_done" and event.final_report:
            final_report = event.final_report
            console.print("[bold green]✓ Synthesis complete[/bold green]\n")

    elapsed = time.time() - start

    # Find saved trace
    traces = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    output_path = str(traces[0]) if traces else output_dir

    if final_report:
        display.print_report(final_report, output_path, elapsed)
    else:
        console.print("[yellow]Report not available — check logs.[/yellow]")


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------

@app.command()
def eval(
    provider: str = typer.Option("anthropic", "--provider", "-p", help="LLM provider."),
    output_dir: str = typer.Option("evals", "--output-dir", "-o", help="Directory for eval results."),
    trace_dir: str = typer.Option("traces", "--trace-dir", help="Directory for trace JSON files."),
    query_id: Optional[str] = typer.Option(None, "--query", "-q", help="Run only Q1, Q2, or Q3."),
    no_llm_scoring: bool = typer.Option(False, "--no-llm-scoring", help="Use heuristic scoring instead of LLM."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the evaluation suite (Q1-Q3) and display scored results."""
    import logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    asyncio.run(_run_eval(
        provider=provider,
        output_dir=output_dir,
        trace_dir=trace_dir,
        query_id=query_id,
        use_llm_scoring=not no_llm_scoring,
    ))


async def _run_eval(
    provider: str,
    output_dir: str,
    trace_dir: str,
    query_id: Optional[str],
    use_llm_scoring: bool,
) -> None:
    from evaluation.eval_runner import EvalRunner
    from evaluation.eval_queries import TEST_QUERIES, get_query
    from models.schemas import EvalDimension

    runner = EvalRunner(
        provider=provider,
        output_dir=output_dir,
        trace_dir=trace_dir,
        use_llm_scoring=use_llm_scoring,
    )

    console.print(Panel(
        "[bold cyan]Battery Research Multi-Agent Evaluation Suite[/bold cyan]\n"
        "Scoring Q1 (broad), Q2 (narrow), Q3 (ambiguous) on three dimensions.\n"
        "[dim]Completeness · Factual Precision · Attribution Quality[/dim]",
        border_style="cyan",
    ))
    console.print(f"[dim]Provider: {provider}  |  LLM scoring: {use_llm_scoring}[/dim]\n")

    if query_id:
        try:
            q = get_query(query_id)
            queries = [q]
        except ValueError:
            console.print(f"[red]Unknown query id: {query_id}. Use Q1, Q2, or Q3.[/red]")
            raise typer.Exit(1)
    else:
        queries = TEST_QUERIES

    results = []
    for q in queries:
        console.print(f"[bold]Running {q.query_id}:[/bold] {q.query_text[:70]}…")
        result = await runner.run_query(q)
        results.append(result)

    # Display results table
    table = Table(title="Evaluation Results", box=box.ROUNDED, show_header=True)
    table.add_column("Query", style="cyan", width=5)
    table.add_column("Complexity", width=10)
    table.add_column("Agents", width=25)
    table.add_column("Completeness\n(0-5)", justify="center", width=14)
    table.add_column("Precision\n(0-5)", justify="center", width=12)
    table.add_column("Attribution\n(0-5)", justify="center", width=13)
    table.add_column("Total\n(0-15)", justify="center", style="bold", width=10)
    table.add_column("Latency", width=9)

    for r in results:
        score_map = {s.dimension.value: s.score for s in r.scores}
        total_color = "green" if r.total_score >= 11 else "yellow" if r.total_score >= 7 else "red"
        table.add_row(
            r.query_id,
            r.complexity_detected.value,
            ", ".join(r.agents_used[:3]) + ("…" if len(r.agents_used) > 3 else ""),
            f"{score_map.get('completeness', 0):.1f}",
            f"{score_map.get('factual_precision', 0):.1f}",
            f"{score_map.get('attribution', 0):.1f}",
            f"[{total_color}]{r.total_score:.1f}[/{total_color}]",
            f"{r.total_latency_ms/1000:.1f}s",
        )

    console.print(table)

    # Print justifications
    console.print("\n[bold]Score Justifications:[/bold]")
    for r in results:
        console.print(f"\n[bold cyan]{r.query_id}[/bold cyan]")
        for s in r.scores:
            console.print(f"  [bold]{s.dimension.value.replace('_', ' ').title()}[/bold]: {s.score:.1f}/5")
            console.print(f"    {s.justification}")

    # Save results
    await runner._save(results)


# ---------------------------------------------------------------------------
# show-trace command
# ---------------------------------------------------------------------------

@app.command(name="show-trace")
def show_trace(
    path: str = typer.Argument(..., help="Path to a trace JSON file."),
) -> None:
    """Pretty-print a saved research trace."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    from models.schemas import ResearchTrace
    try:
        trace = ResearchTrace.model_validate(data)
    except Exception as exc:
        typer.echo(f"Invalid trace file: {exc}", err=True)
        raise typer.Exit(1)

    from cli.display import ResearchDisplay
    display = ResearchDisplay(query=trace.query)
    display.print_report(trace.report, path, elapsed=0.0)
    display.print_findings(trace.findings)


# ---------------------------------------------------------------------------
# list-traces command
# ---------------------------------------------------------------------------

@app.command(name="list-traces")
def list_traces(
    output_dir: str = typer.Option("traces", "--output-dir", "-o", help="Traces directory."),
) -> None:
    """List all saved research traces."""
    traces = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        console.print("[dim]No traces found.[/dim]")
        return

    table = Table(title=f"Saved Traces ({output_dir})", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Query", width=50)
    table.add_column("Complexity", width=10)
    table.add_column("Agents", width=8, justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Date")

    for p in traces:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            query = d.get("query", "—")[:50]
            complexity = d.get("complexity", "—")
            n_findings = len(d.get("findings", []))
            tokens = d.get("budget_summary", {}).get("total_spent", "—")
            created = d.get("created_at", "—")[:16]
            table.add_row(p.name, query, complexity, str(n_findings), str(tokens), created)
        except Exception:
            table.add_row(p.name, "[red]unreadable[/red]", "—", "—", "—", "—")

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
