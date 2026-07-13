"""Rich terminal rendering of an XRayReport. Statistics only, by design."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from token_xray import __version__
from token_xray.analysis.aggregates import MetricResult, XRayReport

_METRIC_TITLES = {
    "spend_by_model_day": "Spend & tokens by model / day",
    "token_histograms": "Token distributions (per request)",
    "duplicate_prompts": "Duplicate prompts",
    "model_mix": "Model mix by price tier",
    "error_rate": "Errors",
    "temporal_patterns": "Temporal patterns",
    "long_context_tail": "Long-context tail",
}


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".") if value < 1 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _rows_table(rows: list[dict]) -> Table:
    table = Table(show_header=True, header_style="bold")
    if not rows:
        return table
    for column in rows[0]:
        table.add_column(str(column))
    for row in rows:
        table.add_row(*(_fmt(v) for v in row.values()))
    return table


def _dict_table(data: dict) -> Table:
    table = Table(show_header=False)
    table.add_column("key", style="dim")
    table.add_column("value")
    for key, value in data.items():
        if isinstance(value, dict):
            table.add_row(key, "  ".join(f"{k}={_fmt(v)}" for k, v in value.items()))
        elif isinstance(value, list):
            inner = _rows_table(value) if value and isinstance(value[0], dict) else _fmt(value)
            table.add_row(key, inner)  # type: ignore[arg-type]
        else:
            table.add_row(key, _fmt(value))
    return table


def _metric_renderable(metric: MetricResult):
    if metric.status != "computed":
        return f"[dim]{metric.unavailable_reason}[/dim]"
    value = metric.value
    if isinstance(value, list):
        return _rows_table(value)
    if isinstance(value, dict):
        return _dict_table(value)
    return _fmt(value)


def render_terminal(report: XRayReport, console: Optional[Console] = None) -> None:
    console = console or Console()
    totals = (
        f"format: [bold]{report.source_format}[/bold]    "
        f"requests: [bold]{report.total_requests:,}[/bold]    "
        f"input tokens: [bold]{report.total_input_tokens:,}[/bold]    "
        f"output tokens: [bold]{report.total_output_tokens:,}[/bold]    "
        f"cost (USD): [bold]{_fmt(report.total_cost_usd)}[/bold]"
    )
    console.print(Panel(totals, title=f"token-xray v{__version__}", subtitle="aggregates only — no prompt text"))
    for name, metric in report.metrics.items():
        console.print(Panel(_metric_renderable(metric), title=_METRIC_TITLES.get(name, name)))
