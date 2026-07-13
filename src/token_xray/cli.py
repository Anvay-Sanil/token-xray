"""token-xray command-line interface.

One command: ``analyze``. Reads a supported export file, prints an aggregate
report, and optionally writes JSON/HTML artifacts. Fully offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from token_xray.analysis.aggregates import compute_report
from token_xray.parsers import UnknownFormatError, parse, supported_formats
from token_xray.report import export_json, render_html, render_terminal

app = typer.Typer(
    name="token-xray",
    help="Local-first LLM spend analyzer. Zero network calls; aggregates only.",
    add_completion=False,
)

_err_console = Console(stderr=True)


@app.callback()
def _main() -> None:
    """Local-first LLM spend analyzer. Zero network calls; aggregates only."""


def run_analysis(
    file: Path,
    *,
    fmt: Optional[str] = None,
    json_out: Optional[Path] = None,
    html_out: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    """Parse, analyze, and render. Shared by the CLI command and tests."""
    parsed = parse(file, fmt=fmt)
    report = compute_report(parsed)
    if not quiet:
        render_terminal(report)
    if json_out is not None:
        export_json(report, json_out)
    if html_out is not None:
        render_html(report, html_out)


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to a usage/log export file."),
    fmt: Optional[str] = typer.Option(
        None,
        "--format",
        help=f"Force a specific input format ({', '.join(supported_formats())}). Auto-detected by default.",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Write an aggregates-only JSON report to this path."
    ),
    html_out: Optional[Path] = typer.Option(
        None, "--html", help="Write a single-file HTML report to this path."
    ),
) -> None:
    """Analyze an LLM usage export and print aggregate statistics."""
    if not file.exists():
        _err_console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=2)
    try:
        run_analysis(file, fmt=fmt, json_out=json_out, html_out=html_out)
    except UnknownFormatError as exc:
        _err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if json_out is not None:
        typer.echo(f"JSON report written to {json_out} (aggregates only - review before sharing).")
    if html_out is not None:
        typer.echo(f"HTML report written to {html_out}.")


if __name__ == "__main__":
    app()
