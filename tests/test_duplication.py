"""Tests for near/exact duplicate analysis over MinHash signatures.

Amendment 1: duplication analysis consumes only signatures and hashes stored on
records at ingest — there is no prompt text left to inspect by this stage.
"""

from __future__ import annotations

from token_xray.analysis.duplication import duplicate_stats
from token_xray.parsers import parse
from token_xray.parsers.base import build_record


def _rec(prompt: str):
    return build_record(
        source_format="litellm_jsonl",
        timestamp=None,
        model="m",
        route=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        prompt_text=prompt,
    )


def test_exact_duplicates_detected():
    records = [_rec("identical prompt text here"), _rec("identical prompt text here"), _rec("something entirely different about databases")]
    stats = duplicate_stats(records)
    assert stats.analyzed_records == 3
    assert stats.exact_duplicate_records == 2
    assert stats.exact_duplicate_rate == 2 / 3


def test_near_duplicates_detected():
    a = "please summarize the following customer support ticket into one concise sentence for the agent"
    b = "please summarize the following customer support ticket into one concise paragraph for the agent"
    c = "write a python function that reverses a singly linked list in place"
    stats = duplicate_stats([_rec(a), _rec(b), _rec(c)])
    assert stats.near_duplicate_records >= 2
    assert 0 < stats.near_duplicate_rate <= 1


def test_no_duplicates_yields_zero_rates():
    stats = duplicate_stats(
        [
            _rec("translate this legal clause into french"),
            _rec("write a haiku about mountains in winter"),
            _rec("explain the difference between tcp and udp"),
        ]
    )
    assert stats.exact_duplicate_records == 0
    assert stats.near_duplicate_records == 0


def test_records_without_signatures_are_skipped():
    no_sig = build_record(
        source_format="openai_csv",
        timestamp=None,
        model="m",
        route=None,
        input_tokens=10,
        output_tokens=5,
        cost_usd=None,
    )
    stats = duplicate_stats([no_sig, no_sig])
    assert stats.analyzed_records == 0
    assert stats.near_duplicate_rate == 0.0


def test_fixture_has_nonzero_duplicate_rate(fixtures_dir):
    pe = parse(fixtures_dir / "litellm_proxy_sample.jsonl")
    stats = duplicate_stats(pe.records)
    assert stats.analyzed_records == 14
    assert stats.exact_duplicate_records >= 4  # fixture contains verbatim repeats
    assert stats.near_duplicate_records >= stats.exact_duplicate_records
