"""Single-file HTML report. Inline CSS only — no external assets, no scripts.

The artifact must stay as offline as the tool: nothing in it may reference a
remote URL, so it renders identically on an air-gapped machine.
"""

from __future__ import annotations

import html as html_mod
from pathlib import Path

from token_xray import __version__
from token_xray.analysis.aggregates import MetricResult, XRayReport

_CSS = """
body { font-family: ui-monospace, Consolas, monospace; margin: 2rem auto; max-width: 60rem;
       color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .85rem; }
th, td { border: 1px solid #ccc; padding: .3rem .5rem; text-align: left; }
th { background: #eee; }
.unavailable { color: #888; font-style: italic; }
.totals { background: #eee; padding: .8rem 1rem; border: 1px solid #ccc; }
footer { margin-top: 2rem; color: #888; font-size: .8rem; }
"""

_TITLES = {
    "spend_by_model_day": "Spend & tokens by model / day",
    "token_histograms": "Token distributions (per request)",
    "duplicate_prompts": "Duplicate prompts",
    "model_mix": "Model mix by price tier",
    "error_rate": "Errors",
    "temporal_patterns": "Temporal patterns",
    "long_context_tail": "Long-context tail",
}


def _esc(value: object) -> str:
    return html_mod.escape("—" if value is None else str(value))


def _rows_html(rows: list[dict]) -> str:
    if not rows:
        return "<p>(no rows)</p>"
    head = "".join(f"<th>{_esc(c)}</th>" for c in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row.values()) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _value_html(value: object) -> str:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return _rows_html(value)
    if isinstance(value, dict):
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_value_html(v) if isinstance(v, (dict, list)) else _esc(v)}</td></tr>"
            for k, v in value.items()
        )
        return f"<table><tbody>{rows}</tbody></table>"
    return f"<p>{_esc(value)}</p>"


def _metric_html(name: str, metric: MetricResult) -> str:
    title = _esc(_TITLES.get(name, name))
    if metric.status != "computed":
        return f"<h2>{title}</h2><p class='unavailable'>{_esc(metric.unavailable_reason)}</p>"
    return f"<h2>{title}</h2>{_value_html(metric.value)}"


def render_html(report: XRayReport, path: Path) -> None:
    sections = "".join(_metric_html(name, metric) for name, metric in report.metrics.items())
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>token-xray report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>token-xray v{_esc(__version__)}</h1>
<div class="totals">
format: <b>{_esc(report.source_format)}</b> &middot;
requests: <b>{report.total_requests:,}</b> &middot;
input tokens: <b>{report.total_input_tokens:,}</b> &middot;
output tokens: <b>{report.total_output_tokens:,}</b> &middot;
cost (USD): <b>{_esc(report.total_cost_usd)}</b>
</div>
{sections}
<footer>Generated locally by token-xray. Aggregates only — no prompt text.</footer>
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")
