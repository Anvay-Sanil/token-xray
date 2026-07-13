"""Aggregate metric computation and the metric-availability matrix.

Every metric is returned as a ``MetricResult`` whose status is either
``computed`` or ``unavailable`` (with the fixed reason string). Billing-level
exports lack per-request rows and prompt text, so several metrics simply cannot
be derived from them; those are labelled, never estimated.

This module reports statistics only. It contains no interpretation, no
suggestion strings, and no cost projections of any kind.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from token_xray.analysis.duplication import duplicate_stats
from token_xray.models import NormalizedRecord, ParsedExport

UNAVAILABLE_REASON = "unavailable from this export format"

# Heuristic price-tier patterns, checked in order (cheap first so that e.g.
# "gpt-4o-mini" matches its "mini" suffix before the broader "gpt-4" rule).
# Patterns are regexes: "mini"/"o1"/"o3" carry a boundary guard so they cannot
# match inside a longer word ("mini" inside "geMINI" was a real bug — see the
# tier regression test). Unmatched models are reported as "unknown", never guessed.
_TIER_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("cheap", tuple(re.compile(p) for p in (
        r"(?<![a-z])mini", "nano", r"(?<![a-z])lite(?![a-z])", "haiku", "flash", r"3\.5-turbo", "35-turbo",
    ))),
    ("frontier", tuple(re.compile(p) for p in (
        "opus", r"(?<![a-z0-9])o1(?![a-z])", r"(?<![a-z0-9])o3(?![a-z])", r"gpt-4\.5", "gpt-5", "ultra",
    ))),
    ("mid", tuple(re.compile(p) for p in (
        "gpt-4", "sonnet", r"gemini-1\.5-pro", "gemini-pro", "mistral-large",
    ))),
)


@dataclass(frozen=True)
class MetricResult:
    status: str  # "computed" | "unavailable"
    value: Optional[Any] = None
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"status": self.status}
        if self.status == "computed":
            d["value"] = self.value
        else:
            d["reason"] = self.unavailable_reason
        return d


@dataclass(frozen=True)
class XRayReport:
    source_format: str
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Optional[float]
    metrics: dict[str, MetricResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_format": self.source_format,
            "totals": {
                "requests": self.total_requests,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": self.total_cost_usd,
            },
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
        }


def _unavailable() -> MetricResult:
    return MetricResult(status="unavailable", unavailable_reason=UNAVAILABLE_REASON)


def _computed(value: Any) -> MetricResult:
    return MetricResult(status="computed", value=value)


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, math.ceil(p / 100 * len(sorted_values)) - 1))
    return sorted_values[index]


def _day(record: NormalizedRecord) -> str:
    return record.timestamp.date().isoformat() if record.timestamp else "unknown"


def _tier(model: Optional[str]) -> str:
    name = (model or "").lower()
    for tier, patterns in _TIER_PATTERNS:
        if any(p.search(name) for p in patterns):
            return tier
    return "unknown"


def _spend_by_model_day(records: tuple[NormalizedRecord, ...]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": None}
    )
    for r in records:
        g = grouped[(r.model or "unknown", _day(r))]
        g["n_requests"] += r.n_requests
        g["input_tokens"] += r.input_tokens or 0
        g["output_tokens"] += r.output_tokens or 0
        if r.cost_usd is not None:
            g["cost_usd"] = (g["cost_usd"] or 0.0) + r.cost_usd
    return [
        {"model": model, "day": day, **values}
        for (model, day), values in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]


def _histogram(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else 0,
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1] if ordered else 0,
        "mean": round(sum(ordered) / len(ordered), 1) if ordered else 0,
    }


def _model_mix(records: tuple[NormalizedRecord, ...]) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(lambda: {"models": set(), "n_requests": 0, "input_tokens": 0})
    total_requests = sum(r.n_requests for r in records) or 1
    for r in records:
        g = grouped[_tier(r.model)]
        g["models"].add(r.model or "unknown")
        g["n_requests"] += r.n_requests
        g["input_tokens"] += r.input_tokens or 0
    return [
        {
            "tier": tier,
            "models": sorted(values["models"]),
            "n_requests": values["n_requests"],
            "share_of_requests": round(values["n_requests"] / total_requests, 4),
            "input_tokens": values["input_tokens"],
        }
        for tier, values in sorted(grouped.items())
    ]


def _temporal(records: tuple[NormalizedRecord, ...], per_request: bool) -> dict:
    by_day: dict[str, dict] = defaultdict(lambda: {"n_requests": 0, "cost_usd": None})
    by_hour: dict[int, int] = defaultdict(int)
    for r in records:
        if r.timestamp is None:
            continue
        g = by_day[_day(r)]
        g["n_requests"] += r.n_requests
        if r.cost_usd is not None:
            g["cost_usd"] = (g["cost_usd"] or 0.0) + r.cost_usd
        if per_request:
            by_hour[r.timestamp.hour] += r.n_requests
    result: dict[str, Any] = {
        "by_day": [{"day": day, **values} for day, values in sorted(by_day.items())]
    }
    if per_request:
        result["by_hour"] = [{"hour": h, "n_requests": n} for h, n in sorted(by_hour.items())]
    return result


def _long_context_tail(input_tokens: list[int]) -> dict:
    ordered = sorted(input_tokens)
    total = sum(ordered) or 1
    top_1pct_count = max(1, len(ordered) // 100)
    top_share = sum(ordered[-top_1pct_count:]) / total
    return {
        "p95_input_tokens": _percentile(ordered, 95),
        "p99_input_tokens": _percentile(ordered, 99),
        "max_input_tokens": ordered[-1] if ordered else 0,
        "requests_over_32k_input": sum(1 for v in ordered if v > 32_000),
        "share_of_input_tokens_in_top_1pct_requests": round(top_share, 4),
    }


def compute_report(parsed: ParsedExport) -> XRayReport:
    """Compute all metrics for a parsed export, honoring its capability matrix."""
    caps, records = parsed.capabilities, parsed.records

    input_values = [r.input_tokens for r in records if r.input_tokens is not None]
    output_values = [r.output_tokens for r in records if r.output_tokens is not None]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]

    metrics: dict[str, MetricResult] = {}

    metrics["spend_by_model_day"] = (
        _computed(_spend_by_model_day(records)) if (caps.has_tokens or caps.has_cost) else _unavailable()
    )

    metrics["token_histograms"] = (
        _computed({"input": _histogram(input_values), "output": _histogram(output_values)})
        if (caps.per_request and caps.has_tokens)
        else _unavailable()
    )

    if caps.has_prompt_text:
        dup = duplicate_stats(records)
        metrics["duplicate_prompts"] = _computed(
            {
                "analyzed_records": dup.analyzed_records,
                "exact_duplicate_records": dup.exact_duplicate_records,
                "exact_duplicate_rate": round(dup.exact_duplicate_rate, 4),
                "near_duplicate_records": dup.near_duplicate_records,
                "near_duplicate_rate": round(dup.near_duplicate_rate, 4),
            }
        )
    else:
        metrics["duplicate_prompts"] = _unavailable()

    metrics["model_mix"] = _computed(_model_mix(records)) if records else _unavailable()

    if caps.per_request:
        total = sum(r.n_requests for r in records) or 1
        errors = sum(r.n_requests for r in records if r.is_error)
        metrics["error_rate"] = _computed(
            {"total_requests": total, "error_requests": errors, "error_rate": round(errors / total, 4)}
        )
    else:
        metrics["error_rate"] = _unavailable()

    metrics["temporal_patterns"] = (
        _computed(_temporal(records, caps.per_request)) if caps.has_timestamp else _unavailable()
    )

    metrics["long_context_tail"] = (
        _computed(_long_context_tail(input_values))
        if (caps.per_request and caps.has_tokens and input_values)
        else _unavailable()
    )

    return XRayReport(
        source_format=parsed.source_format,
        total_requests=sum(r.n_requests for r in records),
        total_input_tokens=sum(input_values),
        total_output_tokens=sum(output_values),
        total_cost_usd=round(sum(costs), 6) if costs else None,
        metrics=metrics,
    )
