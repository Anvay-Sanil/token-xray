"""Parser + auto-detection tests, driven by the committed sample fixtures.

Encodes Amendment 2: billing-level exports (OpenAI, Anthropic) carry no prompt
text and no per-request rows, so their parsers must never emit a prompt hash and
must report per_request=False.
"""

from __future__ import annotations

import pytest

from token_xray.parsers import (
    UnknownFormatError,
    detect,
    parse,
    supported_formats,
)


def test_litellm_jsonl_parses(fixtures_dir):
    pe = parse(fixtures_dir / "litellm_proxy_sample.jsonl")
    assert pe.source_format == "litellm_jsonl"
    assert pe.record_count == 14
    assert pe.capabilities.per_request
    assert pe.capabilities.has_prompt_text
    assert any(r.prompt_sketch is not None for r in pe.records)
    assert sum(1 for r in pe.records if r.is_error) == 2
    assert any((r.input_tokens or 0) > 40000 for r in pe.records)  # long-context tail row


def test_helicone_csv_parses(fixtures_dir):
    pe = parse(fixtures_dir / "helicone_sample.csv")
    assert pe.source_format == "helicone"
    assert pe.record_count == 10
    assert pe.capabilities.per_request
    assert pe.capabilities.has_prompt_text
    assert any(r.prompt_sketch is not None for r in pe.records)
    assert sum(1 for r in pe.records if r.is_error) == 2  # 429 + 500


def test_openai_csv_parses(fixtures_dir):
    pe = parse(fixtures_dir / "openai_usage_sample.csv")
    assert pe.source_format == "openai_csv"
    assert pe.record_count == 5
    assert not pe.capabilities.per_request
    assert not pe.capabilities.has_prompt_text
    assert not pe.capabilities.has_cost  # activity export carries no cost column
    assert all(r.prompt_hash is None for r in pe.records)
    assert all((r.input_tokens or 0) > 0 for r in pe.records)
    assert all(r.n_requests >= 1 for r in pe.records)


def test_anthropic_csv_parses(fixtures_dir):
    pe = parse(fixtures_dir / "anthropic_console_sample.csv")
    assert pe.source_format == "anthropic_csv"
    assert pe.record_count == 4
    assert not pe.capabilities.per_request
    assert pe.capabilities.has_cost
    assert all(r.cost_usd is not None for r in pe.records)
    assert all(r.prompt_hash is None for r in pe.records)


def test_detect_selects_correct_parser(fixtures_dir):
    assert detect(fixtures_dir / "openai_usage_sample.csv").name == "openai_csv"
    assert detect(fixtures_dir / "anthropic_console_sample.csv").name == "anthropic_csv"
    assert detect(fixtures_dir / "helicone_sample.csv").name == "helicone"
    assert detect(fixtures_dir / "litellm_proxy_sample.jsonl").name == "litellm_jsonl"


def test_format_override(fixtures_dir):
    pe = parse(fixtures_dir / "openai_usage_sample.csv", fmt="openai_csv")
    assert pe.source_format == "openai_csv"


def test_unknown_format_raises_and_names_supported(tmp_path):
    junk = tmp_path / "mystery.txt"
    junk.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    with pytest.raises(UnknownFormatError) as exc:
        parse(junk)
    message = str(exc.value)
    for name in supported_formats():
        assert name in message


def test_unknown_forced_format_raises(fixtures_dir):
    with pytest.raises(UnknownFormatError):
        parse(fixtures_dir / "openai_usage_sample.csv", fmt="not_a_real_format")


def test_malformed_jsonl_line_raises_clean_parse_error(fixtures_dir, tmp_path):
    """P1 regression (review 2026-07-14): a corrupt mid-file line must produce an
    explicit ParseError naming the line — never a raw traceback."""
    import pytest
    from token_xray.parsers import ParseError, UnknownFormatError

    good = (fixtures_dir / "litellm_proxy_sample.jsonl").read_text(encoding="utf-8").splitlines()
    broken = tmp_path / "truncated.jsonl"
    broken.write_text(good[0] + "\n" + good[1] + "\n" + '{"model": "gpt-4o", "usage": {"promp\n', encoding="utf-8")

    with pytest.raises(ParseError) as excinfo:
        parse(broken)
    assert "line 3" in str(excinfo.value)
    assert str(broken) in str(excinfo.value)
    # detection succeeded, so this must NOT be reported as an unknown format
    assert not isinstance(excinfo.value, UnknownFormatError)


def test_openai_activity_real_2025_schema(fixtures_dir):
    """Regression from real-export validation (2026-07-14, real Apr-2025 export):
    epoch timestamps must parse and num_requests must be read — defaulting each
    aggregate row to 1 request silently under-reports totals (P0 class)."""
    pe = parse(fixtures_dir / "openai_activity_real_schema.csv")
    assert pe.source_format == "openai_csv"
    assert pe.capabilities.has_timestamp

    assert sum(r.n_requests for r in pe.records) == 65  # 20+30+10+5, NOT 4
    days = {r.timestamp.date().isoformat() for r in pe.records if r.timestamp}
    assert days == {"2025-04-23", "2025-04-24"}
    assert sum(r.input_tokens or 0 for r in pe.records) == 8000
    assert sum(r.output_tokens or 0 for r in pe.records) == 1150


def test_parse_ts_accepts_unix_epoch():
    from token_xray.parsers._util import parse_ts

    assert parse_ts("1745366400").date().isoformat() == "2025-04-23"  # seconds
    assert parse_ts(1745366400).date().isoformat() == "2025-04-23"  # int input
    assert parse_ts("1745366400000").date().isoformat() == "2025-04-23"  # milliseconds
    assert parse_ts("2025-04-23T00:00:00Z").date().isoformat() == "2025-04-23"  # ISO still works
    assert parse_ts("20250423") is not None  # basic ISO date, not epoch


def test_anthropic_usage_real_2026_schema(fixtures_dir):
    """Regression from real-export validation (2026-07-14, real Jul-2026 export
    from platform.claude.com/usage): input tokens are split across four cache
    columns and must be summed; model column is model_version; timestamps are
    hourly. Neither real Anthropic format was even DETECTED before this fix."""
    pe = parse(fixtures_dir / "anthropic_usage_real_schema.csv")
    assert pe.source_format == "anthropic_csv"
    assert pe.capabilities.has_tokens
    assert pe.capabilities.has_timestamp
    assert not pe.capabilities.has_cost  # usage export carries no cost column

    assert sum(r.input_tokens or 0 for r in pe.records) == 296 + (208367 + 1000 + 50000) + 8
    assert sum(r.output_tokens or 0 for r in pe.records) == 27 + 39423 + 1
    days = {r.timestamp.date().isoformat() for r in pe.records if r.timestamp}
    assert days == {"2026-07-03", "2026-07-04"}
    assert {r.model for r in pe.records} == {"claude-opus-4-8", "claude-haiku-4-5"}


def test_anthropic_cost_real_2026_schema(fixtures_dir):
    """Real cost export (claude_api_cost_*.csv): long format, one row per
    model x token_type with cost_usd, display-style model names, no tokens."""
    pe = parse(fixtures_dir / "anthropic_cost_real_schema.csv")
    assert pe.source_format == "anthropic_csv"
    assert pe.capabilities.has_cost
    assert not pe.capabilities.has_tokens

    assert sum(r.cost_usd or 0 for r in pe.records) == pytest.approx(14.00)
    assert {r.model for r in pe.records} == {"Claude Haiku 4.5", "Claude Opus 4.8"}
