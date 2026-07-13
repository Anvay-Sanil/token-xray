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
