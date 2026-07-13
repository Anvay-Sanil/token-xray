"""Tests for the aggregate computations and the availability matrix.

Amendment 2: every metric is either computed or explicitly labelled
"unavailable from this export format" — never guessed, never accompanied by a
recommendation (F7 guard).
"""

from __future__ import annotations

from token_xray.analysis.aggregates import compute_report
from token_xray.parsers import parse

UNAVAILABLE = "unavailable from this export format"


def test_per_request_export_computes_everything(fixtures_dir):
    report = compute_report(parse(fixtures_dir / "litellm_proxy_sample.jsonl"))

    assert report.source_format == "litellm_jsonl"
    assert report.total_requests == 14

    spend = report.metrics["spend_by_model_day"]
    assert spend.status == "computed"
    assert any(row["model"] == "gpt-4o" for row in spend.value)
    assert all("cost_usd" in row for row in spend.value)

    hist = report.metrics["token_histograms"]
    assert hist.status == "computed"
    assert hist.value["input"]["count"] == 14
    assert hist.value["input"]["p50"] > 0
    assert hist.value["input"]["max"] >= 45000

    dup = report.metrics["duplicate_prompts"]
    assert dup.status == "computed"
    assert dup.value["exact_duplicate_records"] >= 4

    mix = report.metrics["model_mix"]
    assert mix.status == "computed"
    tiers = {row["tier"] for row in mix.value}
    assert tiers <= {"frontier", "mid", "cheap", "unknown"}

    err = report.metrics["error_rate"]
    assert err.status == "computed"
    assert err.value["error_requests"] == 2

    temporal = report.metrics["temporal_patterns"]
    assert temporal.status == "computed"
    assert len(temporal.value["by_day"]) == 3

    tail = report.metrics["long_context_tail"]
    assert tail.status == "computed"
    assert tail.value["p99_input_tokens"] >= 40000


def test_billing_export_labels_unavailable_metrics(fixtures_dir):
    report = compute_report(parse(fixtures_dir / "openai_usage_sample.csv"))

    # Computable from billing aggregates:
    assert report.metrics["temporal_patterns"].status == "computed"
    assert report.metrics["model_mix"].status == "computed"

    # Not computable — must be labelled, not guessed:
    for name in ("duplicate_prompts", "token_histograms", "error_rate", "long_context_tail"):
        metric = report.metrics[name]
        assert metric.status == "unavailable"
        assert metric.unavailable_reason == UNAVAILABLE
        assert metric.value is None

    # OpenAI activity export has tokens but no cost column:
    spend = report.metrics["spend_by_model_day"]
    assert spend.status == "computed"
    assert all(row["cost_usd"] is None for row in spend.value)
    assert all(row["input_tokens"] > 0 for row in spend.value)


def test_anthropic_export_has_cost_but_no_request_level_metrics(fixtures_dir):
    report = compute_report(parse(fixtures_dir / "anthropic_console_sample.csv"))
    spend = report.metrics["spend_by_model_day"]
    assert spend.status == "computed"
    assert any(row["cost_usd"] and row["cost_usd"] > 0 for row in spend.value)
    assert report.metrics["duplicate_prompts"].status == "unavailable"


def test_totals_are_summed(fixtures_dir):
    report = compute_report(parse(fixtures_dir / "anthropic_console_sample.csv"))
    assert report.total_requests == 4  # one aggregate row each
    assert report.total_cost_usd is not None and report.total_cost_usd > 10
    assert report.total_input_tokens > 9_000_000


def test_report_contains_no_advice_strings(fixtures_dir):
    """F7 guard: aggregate output must never contain recommendation language."""
    report = compute_report(parse(fixtures_dir / "litellm_proxy_sample.jsonl"))
    text = repr(report.to_dict()).lower()
    for banned in ("recommend", "should ", "consider ", "savings", "switch to", "migrate"):
        assert banned not in text


def test_tier_classification_survives_substring_collisions():
    """P1 regression (review 2026-07-14): 'mini' must not match inside 'geMINI'."""
    from token_xray.analysis.aggregates import _tier

    assert _tier("gemini-1.5-pro") == "mid"
    assert _tier("gemini-pro") == "mid"
    assert _tier("gemini-2.0-flash") == "cheap"  # flash IS the cheap gemini tier
    assert _tier("gemini-ultra") == "frontier"
    assert _tier("gpt-4o-mini") == "cheap"
    assert _tier("gpt-4o") == "mid"
    assert _tier("o1-preview") == "frontier"
    assert _tier("claude-3-haiku-20240307") == "cheap"
    assert _tier("claude-3-opus-20240229") == "frontier"
    assert _tier("gpt-3.5-turbo") == "cheap"
    assert _tier("some-custom-model") == "unknown"
    assert _tier(None) == "unknown"
