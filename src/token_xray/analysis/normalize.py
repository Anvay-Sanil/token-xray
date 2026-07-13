"""Turn raw prompt text into irreversible aggregates.

This module is the privacy boundary. A parser calls it once at ingest to obtain
a salted hash (for exact-duplicate grouping) and a MinHash signature (for
near-duplicate detection), after which the raw prompt text is discarded and
never stored on any record or in any output artifact.
"""

from __future__ import annotations

import hashlib
import re

from datasketch import MinHash

# Number of MinHash permutations. Fixed so signatures are comparable across
# records within a run. datasketch's default is 128.
NUM_PERM = 128

# Namespacing salt for the exact-duplicate hash. A hash is already one-way; the
# salt prevents trivial dictionary lookups of short prompts. It is constant so
# identical prompts hash identically within and across runs (needed for grouping).
_SALT = "token-xray/v0"

# Shingle width (in words) for MinHash. Short prompts fall back to a single shingle.
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


def minhash_signature(normalized: str, num_perm: int = NUM_PERM) -> tuple[int, ...]:
    """Return a MinHash signature (tuple of ints) for already-normalized text.

    Identical text yields an identical signature; the fraction of equal positions
    between two signatures estimates the Jaccard similarity of their shingle sets.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(normalized):
        m.update(shingle.encode("utf-8"))
    return tuple(int(x) for x in m.hashvalues)
