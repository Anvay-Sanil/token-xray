"""Exact and near-duplicate prompt analysis over stored signatures.

Operates exclusively on the ``prompt_hash`` / ``minhash_signature`` fields that
parsers computed at ingest — raw prompt text no longer exists by this stage.

Detection is two-stage: an LSH index at a low threshold generates candidate
pairs with high recall, then every candidate is verified against the estimated
Jaccard similarity computed directly from the stored signatures. Only verified
pairs count, so the reported rate is not inflated by LSH banding noise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from datasketch import MinHash, MinHashLSH

from token_xray.analysis.normalize import NUM_PERM
from token_xray.models import NormalizedRecord

# Verified estimated-Jaccard threshold for two prompts to count as near-duplicates.
# 0.5 on 3-word shingles corresponds to templated prompts differing by a word or
# two — the pattern that matters for duplicate-rate reporting.
NEAR_DUP_THRESHOLD = 0.5

# LSH banding threshold for candidate generation only. Kept below the decision
# threshold so borderline pairs still surface as candidates for verification.
_LSH_CANDIDATE_THRESHOLD = 0.4


@dataclass(frozen=True)
class DuplicateStats:
    """Duplicate counts over the records that carried prompt signatures."""

    analyzed_records: int
    exact_duplicate_records: int
    near_duplicate_records: int

    @property
    def exact_duplicate_rate(self) -> float:
        return self.exact_duplicate_records / self.analyzed_records if self.analyzed_records else 0.0

    @property
    def near_duplicate_rate(self) -> float:
        return self.near_duplicate_records / self.analyzed_records if self.analyzed_records else 0.0


def _rehydrate(signature: Sequence[int]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    m.hashvalues[:] = list(signature)
    return m


def _estimated_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def duplicate_stats(records: Iterable[NormalizedRecord]) -> DuplicateStats:
    """Compute exact- and near-duplicate counts from record signatures.

    A record is an exact duplicate if its prompt hash appears more than once; a
    near duplicate if at least one other record's signature has a verified
    estimated Jaccard >= ``NEAR_DUP_THRESHOLD``. Exact duplicates are by
    definition also near duplicates.
    """
    with_sig = [r for r in records if r.minhash_signature is not None and r.prompt_hash is not None]
    if not with_sig:
        return DuplicateStats(0, 0, 0)

    hash_counts = Counter(r.prompt_hash for r in with_sig)
    exact_flags = [hash_counts[r.prompt_hash] > 1 for r in with_sig]

    lsh = MinHashLSH(threshold=_LSH_CANDIDATE_THRESHOLD, num_perm=NUM_PERM)
    minhashes = [_rehydrate(r.minhash_signature) for r in with_sig]  # type: ignore[arg-type]
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    near_flags = list(exact_flags)  # exact dupes always count as near dupes
    for i, (record, m) in enumerate(zip(with_sig, minhashes)):
        if near_flags[i]:
            continue
        for key in lsh.query(m):
            j = int(key)
            if j == i:
                continue
            candidate = with_sig[j].minhash_signature
            assert record.minhash_signature is not None and candidate is not None
            if _estimated_jaccard(record.minhash_signature, candidate) >= NEAR_DUP_THRESHOLD:
                near_flags[i] = True
                break

    return DuplicateStats(
        analyzed_records=len(with_sig),
        exact_duplicate_records=sum(exact_flags),
        near_duplicate_records=sum(near_flags),
    )
