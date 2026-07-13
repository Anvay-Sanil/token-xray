"""Turn raw prompt text into irreversible aggregates. Stdlib only.

This module is the privacy boundary. A parser calls it once at ingest to obtain
a salted hash (for exact-duplicate grouping) and a bottom-k shingle sketch (for
near-duplicate detection), after which the raw prompt text is discarded and
never stored on any record or in any output artifact.

Sketch accuracy: the sketch keeps the k smallest 64-bit shingle hashes. For
prompts with at most k shingles it therefore contains the complete shingle set
and the Jaccard estimate is exact; for longer texts the k-minimum-values
estimator applies with standard error ~sqrt(J(1-J)/k) (k=128 -> under ~0.05).
"""

from __future__ import annotations

import hashlib
import re

# Sketch size: number of smallest shingle hashes retained per prompt.
SKETCH_K = 128

# Namespacing salt for the exact-duplicate hash. A hash is already one-way; the
# salt prevents trivial dictionary lookups of short prompts. It is constant so
# identical prompts hash identically within and across runs (needed for grouping).
_SALT = "token-xray/v0"

# blake2b personalization for shingle hashing (max 16 bytes).
_PERSON = b"token-xray/sk"

# Shingle width (in words). Short prompts fall back to a single shingle.
_SHINGLE_K = 3

_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """Lowercase, collapse all whitespace runs to single spaces, and strip."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def prompt_hash(normalized: str) -> str:
    """Return a salted SHA-256 hex digest of already-normalized prompt text."""
    return hashlib.sha256((_SALT + normalized).encode("utf-8")).hexdigest()


def _shingles(normalized: str, k: int = _SHINGLE_K) -> list[str]:
    words = normalized.split()
    if not words:
        return []
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _hash64(shingle: str) -> int:
    """Deterministic 64-bit hash of one shingle (blake2b, personalized)."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big")


def bottom_k_sketch(normalized: str, k: int = SKETCH_K) -> tuple[int, ...]:
    """Return the k smallest distinct shingle hashes, sorted ascending.

    Identical text yields an identical sketch. Two sketches estimate the
    Jaccard similarity of their shingle sets via ``estimate_jaccard``.
    """
    hashes = {_hash64(s) for s in _shingles(normalized)}
    return tuple(sorted(hashes)[:k])


def estimate_jaccard(a: tuple[int, ...], b: tuple[int, ...], k: int = SKETCH_K) -> float:
    """Estimate Jaccard similarity of two shingle sets from their sketches.

    Exact when both inputs have fewer than k shingles (the sketches are then the
    complete hash sets); otherwise the standard k-minimum-values estimate over
    the k smallest values of the union.
    """
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    if len(a) < k and len(b) < k:
        return len(set_a & set_b) / len(set_a | set_b)
    union_bottom = sorted(set_a | set_b)[:k]
    shared = sum(1 for v in union_bottom if v in set_a and v in set_b)
    return shared / len(union_bottom)
