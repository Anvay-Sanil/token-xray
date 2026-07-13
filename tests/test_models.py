"""Tests for the normalized record model and the build_record ingest helper.

build_record is where Amendment 1 lives: sketch + hash are computed at ingest
and the raw prompt text is dropped on the spot, never reaching a stored field.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from token_xray.models import FormatCapabilities, NormalizedRecord, ParsedExport
from token_xray.parsers.base import build_record


def test_build_record_computes_signature_and_hash_from_prompt():
    rec = build_record(
        source_format="litellm_jsonl",
        timestamp=None,
        model="gpt-4o",
        route="chat",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
        prompt_text="Please summarize this support ticket for the agent",
    )
    assert isinstance(rec, NormalizedRecord)
    assert rec.prompt_hash is not None and len(rec.prompt_hash) == 64
    assert rec.prompt_sketch is not None and 0 < len(rec.prompt_sketch) <= 128


def test_build_record_never_retains_raw_prompt_text():
    secret = "my customer's SSN is 123-45-6789 and email is jane@example.com"
    rec = build_record(
        source_format="litellm_jsonl",
        timestamp=None,
        model="gpt-4o",
        route="chat",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        prompt_text=secret,
    )
    for f in fields(rec):
        val = str(getattr(rec, f.name))
        assert secret not in val
        assert "123-45-6789" not in val
        assert "jane@example.com" not in val


def test_build_record_without_prompt_has_no_signature():
    rec = build_record(
        source_format="openai_csv",
        timestamp=None,
        model="gpt-4o",
        route=None,
        input_tokens=100,
        output_tokens=0,
        cost_usd=0.2,
        n_requests=42,
    )
    assert rec.prompt_hash is None
    assert rec.prompt_sketch is None
    assert rec.n_requests == 42


def test_build_record_blank_prompt_has_no_signature():
    rec = build_record(
        source_format="helicone",
        timestamp=None,
        model="gpt-4o",
        route=None,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        prompt_text="   \n  ",
    )
    assert rec.prompt_hash is None
    assert rec.prompt_sketch is None


def test_record_is_immutable():
    rec = build_record(
        source_format="openai_csv",
        timestamp=None,
        model="m",
        route=None,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        rec.model = "other"  # type: ignore[misc]


def test_parsed_export_counts_records():
    caps = FormatCapabilities(
        per_request=True,
        has_prompt_text=True,
        has_cost=True,
        has_tokens=True,
        has_timestamp=True,
    )
    rec = build_record(
        source_format="litellm_jsonl",
        timestamp=None,
        model="m",
        route=None,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
    )
    pe = ParsedExport(source_format="litellm_jsonl", capabilities=caps, records=(rec,))
    assert pe.record_count == 1
