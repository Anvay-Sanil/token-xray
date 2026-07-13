"""Tests for prompt normalization, hashing, and the bottom-k sketch.

These functions are the privacy boundary: they turn raw prompt text into
irreversible aggregates (a salted hash + a bottom-k shingle sketch) so the raw
text can be dropped immediately at ingest.

Sketch accuracy contract (stated in the README): for prompts with at most
SKETCH_K shingles the sketch contains every shingle hash, so the Jaccard
estimate is exact; beyond that it degrades gracefully (~1/sqrt(k) error).
"""

from __future__ import annotations

import random

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


def test_sketch_is_sorted_and_bounded():
    sketch = N.bottom_k_sketch("the quick brown fox jumps over the lazy dog")
    assert list(sketch) == sorted(sketch)
    assert len(sketch) <= N.SKETCH_K
    assert all(isinstance(x, int) for x in sketch)


def test_short_prompt_sketch_holds_all_shingles():
    # 9 words -> 7 three-word shingles -> sketch is the complete hash set.
    sketch = N.bottom_k_sketch("please summarize this support ticket for the agent now")
    assert len(sketch) == 7


def test_identical_text_yields_identical_sketch():
    text = "summarize the following support ticket for the customer"
    assert N.bottom_k_sketch(text) == N.bottom_k_sketch(text)


def test_exact_jaccard_for_short_prompts():
    a_text = "please summarize the following customer support ticket into one concise sentence for the agent"
    b_text = "please summarize the following customer support ticket into one concise paragraph for the agent"
    a, b = N.bottom_k_sketch(a_text), N.bottom_k_sketch(b_text)
    est = N.estimate_jaccard(a, b)
    # Both prompts are far under SKETCH_K shingles, so the estimate is EXACT.
    sa, sb = set(N._shingles(N.normalize_prompt(a_text))), set(N._shingles(N.normalize_prompt(b_text)))
    exact = len({N._hash64(s) for s in sa} & {N._hash64(s) for s in sb}) / len(
        {N._hash64(s) for s in sa} | {N._hash64(s) for s in sb}
    )
    assert est == exact
    assert est > 0.5


def test_disjoint_prompts_estimate_zero():
    a = N.bottom_k_sketch("translate this legal contract clause into french")
    b = N.bottom_k_sketch("write a python function to reverse a linked list")
    assert N.estimate_jaccard(a, b) == 0.0


def test_estimate_error_bound_on_long_texts():
    """Property test: sketch estimate vs exact Jaccard on synthetic long texts."""
    rng = random.Random(42)
    vocabulary = [f"word{i}" for i in range(500)]
    for _ in range(20):
        base = [rng.choice(vocabulary) for _ in range(400)]  # >> SKETCH_K shingles
        variant = list(base)
        for _ in range(rng.randint(10, 120)):  # random word substitutions
            variant[rng.randrange(len(variant))] = rng.choice(vocabulary)
        a_text, b_text = " ".join(base), " ".join(variant)

        a_shingles = {N._hash64(s) for s in N._shingles(N.normalize_prompt(a_text))}
        b_shingles = {N._hash64(s) for s in N._shingles(N.normalize_prompt(b_text))}
        exact = len(a_shingles & b_shingles) / len(a_shingles | b_shingles)

        est = N.estimate_jaccard(N.bottom_k_sketch(a_text), N.bottom_k_sketch(b_text))
        # k=128 -> standard error ~ sqrt(J(1-J)/k) <= 0.045; assert a 3-sigma bound.
        assert abs(est - exact) <= 0.14, f"estimate {est} vs exact {exact}"
