"""
cli/display.py
==============
ResearchDisplay — Rich terminal output for the research system.
"""
from __future__ import annotations

from typing import List, Optional

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from models.schemas import ResearchFinding, ResearchReport, SubtaskStatus

console = Console()


class ResearchDisplay:
    """Handles Rich console rendering for the research pipeline."""

    def __init__(self, query: str) -> None:
        self.query = query

    def print_report(
        self,
        report: ResearchReport,
        output_path: str,
        elapsed: float,
    ) -> None:
        """Print the final ResearchReport in a structured format."""
        console.print(Rule("[bold cyan]Research Report[/bold cyan]", style="cyan"))
        console.print()

        # Executive summary
        console.print(Panel(
            Markdown(report.executive_summary),
            title="[bold]Executive Summary[/bold]",
            border_style="cyan",
            padding=(1, 2),
        ))

        # Findings by domain
        if report.findings_by_domain:
            console.print(Rule("[bold]Findings by Domain[/bold]", style="dim"))
            for domain, text in report.findings_by_domain.items():
                domain_color = {
                    "chemistry": "green",
                    "policy": "blue",
                    "geography": "yellow",
                    "retrieval": "magenta",
                    "comparison": "cyan",
                    "synthesis": "white",
                }.get(domain.lower(), "white")
                console.print(Panel(
                    Markdown(text[:1200] + ("…" if len(text) > 1200 else "")),
                    title=f"[{domain_color}]{domain.upper()}[/{domain_color}]",
                    border_style=domain_color,
                    padding=(0, 1),
                ))
                console.print()

        # Structured comparison
        if report.structured_comparison:
            console.print(Rule("[bold]Structured Comparison[/bold]", style="dim"))
            console.print(Markdown(report.structured_comparison))
            console.print()

        # Key takeaways
        if report.key_takeaways:
            table = Table(title="Key Takeaways", box=box.SIMPLE, show_header=False)
            table.add_column("", style="bold yellow", width=3)
            table.add_column("")
            for i, kt in enumerate(report.key_takeaways, 1):
                table.add_row(str(i), kt)
            console.print(table)
            console.print()

        # Conflicts detected
        if report.conflicts_detected:
            console.print(Rule("[bold red]Conflicts Detected[/bold red]", style="red"))
            for c in report.conflicts_detected:
                console.print(
                    f"  [red]⚠[/red] [{c.subtask_a}] vs [{c.subtask_b}]: {c.dimension}"
                )
                console.print(f"    A: {c.claim_a[:120]}")
                console.print(f"    B: {c.claim_b[:120]}")
            console.print()

        # Knowledge gaps
        if report.knowledge_gaps:
            console.print("[dim]Knowledge Gaps:[/dim]")
            for gap in report.knowledge_gaps[:5]:
                console.print(f"  [dim yellow]• {gap}[/dim yellow]")
            console.print()

        # Sources
        if report.sources_cited:
            console.print("[dim]Sources Referenced:[/dim]")
            for src in report.sources_cited[:10]:
                console.print(f"  [dim]- {src}[/dim]")
            console.print()

        # Footer
        console.print(Rule(style="dim"))
        console.print(
            f"[dim]Completeness: {report.completeness_score}/100  |  "
            f"Elapsed: {elapsed:.1f}s  |  "
            f"Trace: {output_path}[/dim]"
        )

    def print_findings(self, findings: List[ResearchFinding]) -> None:
        """Print a summary table of all sub-agent findings."""
        if not findings:
            return

        table = Table(title="Sub-Agent Findings", box=box.ROUNDED)
        table.add_column("Agent", style="cyan")
        table.add_column("Subtask ID", style="dim")
        table.add_column("Status", width=14)
        table.add_column("Conf", justify="center", width=6)
        table.add_column("Tokens", justify="right", width=8)
        table.add_column("Latency", justify="right", width=9)
        table.add_column("Attempt", justify="center", width=8)

        status_styles = {
            SubtaskStatus.COMPLETE: "green",
            SubtaskStatus.PARTIAL: "yellow",
            SubtaskStatus.INSUFFICIENT: "red",
            SubtaskStatus.CONFLICT: "magenta",
        }

        for f in findings:
            style = status_styles.get(f.status, "white")
            table.add_row(
                f.agent_name,
                f.subtask_id,
                f"[{style}]{f.status.value}[/{style}]",
                str(f.confidence) if not f.error else "[red]ERR[/red]",
                str(f.tokens_used),
                f"{f.latency_ms:.0f}ms",
                str(f.attempt),
            )

        console.print(table)
