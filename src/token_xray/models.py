"""Core immutable data structures shared across parsers, analysis, and report.

A ``NormalizedRecord`` is the single unified row shape every parser produces.
It holds aggregates only: token counts, cost, and — when prompt text was
available at ingest — an irreversible hash and MinHash signature. It never
holds raw prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class NormalizedRecord:
    """One usage row, normalized across all supported export formats.

    ``n_requests`` is 1 for per-request logs and may be larger for billing-level
    exports where a single row already aggregates many calls.
    """

    source_format: str
    timestamp: Optional[datetime]
    model: Optional[str]
    route: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    is_error: bool = False
    status: Optional[str] = None
    n_requests: int = 1
    prompt_hash: Optional[str] = None
    minhash_signature: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class FormatCapabilities:
    """What a given export format can and cannot tell us.

    Drives the availability matrix (Amendment 2): metrics that need capabilities
    a format lacks are labelled unavailable rather than guessed.
    """

    per_request: bool
    has_prompt_text: bool
    has_cost: bool
    has_tokens: bool
    has_timestamp: bool


@dataclass(frozen=True)
class ParsedExport:
    """The result of parsing one export file."""

    source_format: str
    capabilities: FormatCapabilities
    records: tuple[NormalizedRecord, ...]

    @property
    def record_count(self) -> int:
        return len(self.records)
