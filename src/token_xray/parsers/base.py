"""Parser contract and the shared record-building helper.

``build_record`` is the one place where raw prompt text is allowed to exist,
and only transiently: it derives the hash + bottom-k sketch and lets the text
go out of scope. Every parser MUST create records through this helper so the
"text never persists" guarantee holds in exactly one auditable location.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from token_xray.analysis import normalize as _normalize
from token_xray.models import FormatCapabilities, NormalizedRecord, ParsedExport


class ParseError(ValueError):
    """A file was recognized but could not be parsed cleanly.

    Raised with a human-readable message (file + line/row where possible) so the
    CLI can fail with a clear explanation instead of a traceback.
    """


def build_record(
    *,
    source_format: str,
    timestamp: Optional[datetime],
    model: Optional[str],
    route: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cost_usd: Optional[float],
    is_error: bool = False,
    status: Optional[str] = None,
    n_requests: int = 1,
    prompt_text: Optional[str] = None,
) -> NormalizedRecord:
    """Build a NormalizedRecord, converting any prompt text to aggregates.

    ``prompt_text`` is read here and nowhere stored: if it is non-blank we keep a
    salted hash and a bottom-k sketch; the string itself is dropped on return.
    """
    prompt_hash: Optional[str] = None
    prompt_sketch: Optional[tuple[int, ...]] = None

    if prompt_text:
        normalized = _normalize.normalize_prompt(prompt_text)
        if normalized:
            prompt_hash = _normalize.prompt_hash(normalized)
            prompt_sketch = _normalize.bottom_k_sketch(normalized)

    return NormalizedRecord(
        source_format=source_format,
        timestamp=timestamp,
        model=model,
        route=route,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        is_error=is_error,
        status=status,
        n_requests=n_requests,
        prompt_hash=prompt_hash,
        prompt_sketch=prompt_sketch,
    )


@runtime_checkable
class Parser(Protocol):
    """Contract implemented by every format parser."""

    name: str
    capabilities: FormatCapabilities

    def can_parse(self, path: Path, head: str) -> bool:
        """Return True if this parser recognizes the file from a text preview."""
        ...

    def parse(self, path: Path) -> ParsedExport:
        """Parse the whole file into a ParsedExport."""
        ...
