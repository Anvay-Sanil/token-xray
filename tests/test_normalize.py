"""Tests for prompt normalization, hashing, and MinHash signatures.

These functions are the privacy boundary: they turn raw prompt text into
irreversible aggregates (a salted hash + a MinHash signature) so the raw text
can be dropped immediately at ingest.
"""

from __future__ import annotations

from token_xray.analysis import normalize as N


def test_normalize_lowercases_and_collapses_whitespace():
    assert N.normalize_prompt("  Hello   WORLD\n\tfoo ") == "hello world foo"


def test_normalize_empty_returns_empty():
    assert N.normalize_prompt("   \n  ") == ""


def test_prompt_hash_is_hex_and_deterministic():
    a = N.prompt_hash("hello world")
    b = N.prompt_hash("hello world")
    assert a == b
    assert len(a) == 64
    int(a, 16)  # valid hex


def test_prompt_hash_differs_for_different_text():
    assert N.prompt_hash("hello world") != N.prompt_hash("goodbye world")


def test_minhash_signature_has_fixed_length():
    sig = N.minhash_signature("the quick brown fox jumps over the lazy dog")
    assert len(sig) == N.NUM_PERM
    assert all(isinstance(x, int) for x in sig)


def test_identical_text_yields_identical_signature():
    text = "summarize the following support ticket for the customer"
    assert N.minhash_signature(text) == N.minhash_signature(text)


def test_similar_text_has_high_estimated_jaccard():
    # The two prompts differ by a single word in a long sentence -> high Jaccard,
    # so the MinHash estimate stays comfortably above the threshold (not flaky).
    a = N.minhash_signature(
        "please summarize the following customer support ticket into a single "
        "concise sentence for the on call agent to read quickly today"
    )
    b = N.minhash_signature(
        "please summarize the following customer support ticket into a single "
        "concise paragraph for the on call agent to read quickly today"
    )
    est = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    assert est > 0.5


def test_different_text_has_low_estimated_jaccard():
    a = N.minhash_signature("translate this legal contract clause into french")
    b = N.minhash_signature("write a python function to reverse a linked list")
    est = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    assert est < 0.2
