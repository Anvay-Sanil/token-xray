"""Exact and near-duplicate prompt analysis over bottom-k sketches. Stdlib only.

Operates exclusively on the ``prompt_hash`` / ``prompt_sketch`` fields that
parsers computed at ingest — raw prompt text no longer exists by this stage.

Pairing is sub-quadratic by construction (never all-pairs):

1. Records are grouped by exact prompt hash; near-duplicate search runs over
   unique prompts only (exact duplicates inherit the flag by definition).
2. Candidate generation uses prefix filtering: each unique prompt is indexed
   under its smallest ``_PREFIX`` sketch values. Two prompts with Jaccard >=
   0.5 share smallest-values with overwhelming probability, so recall stays
   high while the index stays linear in the number of unique prompts.
3. Every candidate pair is verified with the sketch Jaccard estimate before it
   counts. Caps on bucket size, gathered candidates, and verification attempts
   bound the worst case; they can miss pairs in adversarial distributions,
   which is acceptable for an aggregate rate and documented in the README.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from token_xray.analysis.normalize import estimate_jaccard
from token_xray.models import NormalizedRecord

# Verified estimated-Jaccard threshold for two prompts to count as near-duplicates.
# 0.5 on 3-word shingles corresponds to templated prompts differing by a word or
# two — the pattern that matters for duplicate-rate reporting.
NEAR_DUP_THRESHOLD = 0.5

_PREFIX = 16  # smallest sketch values indexed/queried per unique prompt
_BUCKET_CAP = 200  # stored postings per indexed value
_CANDIDATE_CAP = 64  # candidates gathered per query
_VERIFY_CAP = 16  # verification attempts per query


@dataclass(frozen=True)
class DuplicateStats:
    """Duplicate counts over the records that carried prompt sketches."""

    analyzed_records: int
    exact_duplicate_records: int
    near_duplicate_records: int

    @property
    def exact_duplicate_rate(self) -> float:
        return self.exact_duplicate_records / self.analyzed_records if self.analyzed_records else 0.0

    @property
    def near_duplicate_rate(self) -> float:
        return self.near_duplicate_records / self.analyzed_records if self.analyzed_records else 0.0


def duplicate_stats(records: Iterable[NormalizedRecord]) -> DuplicateStats:
    """Compute exact- and near-duplicate counts from record sketches.

    A record is an exact duplicate if its prompt hash appears more than once; a
    near duplicate if its prompt has a verified sketch-Jaccard >=
    ``NEAR_DUP_THRESHOLD`` with any other prompt. Exact duplicates are by
    definition also near duplicates.
    """
    with_sketch = [r for r in records if r.prompt_sketch and r.prompt_hash]
    if not with_sketch:
        return DuplicateStats(0, 0, 0)

    # --- stage 1: exact groups --------------------------------------------
    group_sizes: Counter[str] = Counter(r.prompt_hash for r in with_sketch)  # type: ignore[misc]
    sketches: dict[str, tuple[int, ...]] = {}
    for record in with_sketch:
        sketches.setdefault(record.prompt_hash, record.prompt_sketch)  # type: ignore[arg-type]
    unique_hashes = list(sketches)

    exact_records = sum(size for size in group_sizes.values() if size > 1)

    # --- stage 2: prefix-filter index over unique prompts ------------------
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, prompt_hash in enumerate(unique_hashes):
        for value in sketches[prompt_hash][:_PREFIX]:
            bucket = buckets[value]
            if len(bucket) < _BUCKET_CAP:
                bucket.append(idx)

    # --- stage 3: gather candidates, verify, count -------------------------
    near_records = 0
    for idx, prompt_hash in enumerate(unique_hashes):
        size = group_sizes[prompt_hash]
        if size > 1:  # exact-duplicate group: near by definition
            near_records += size
            continue

        sketch = sketches[prompt_hash]
        gathered: Counter[int] = Counter()
        for value in sketch[:_PREFIX]:
            for other in buckets.get(value, ()):
                if other != idx:
                    gathered[other] += 1
            if len(gathered) >= _CANDIDATE_CAP:
                break

        for other, _shared in gathered.most_common(_VERIFY_CAP):
            if estimate_jaccard(sketch, sketches[unique_hashes[other]]) >= NEAR_DUP_THRESHOLD:
                near_records += 1
                break

    return DuplicateStats(
        analyzed_records=len(with_sketch),
        exact_duplicate_records=exact_records,
        near_duplicate_records=near_records,
    )
