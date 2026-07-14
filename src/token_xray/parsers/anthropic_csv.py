"""Parser for Anthropic Console CSV exports (billing aggregate, no prompt text).

Two real export variants ship from platform.claude.com (validated against
public real-world files, 2026-07):

- usage export (``claude_api_tokens_*.csv``): hourly/daily rows per model with
  input tokens split across four cache columns (summed here) and output tokens;
  carries no cost column.
- cost export (``claude_api_cost_*.csv``): long-format rows per model and
  token_type with ``cost_usd``; carries no token counts.

Both share the ``usage_date_utc`` column used for detection. Neither carries
request counts or prompt text, so per-request metrics are unavailable
(Amendment 2). Legacy/simplified column names remain tolerated.
"""

from __future__ import annotations

from pathlib import Path

from token_xray.models import FormatCapabilities, ParsedExport
from token_xray.parsers._util import csv_header, parse_ts, pick, read_csv_rows, to_float, to_int
from token_xray.parsers.base import build_record

# Input tokens in the usage export are split by cache behavior; total input
# volume is their sum.
_CACHE_INPUT_COLS = (
    "usage_input_tokens_no_cache",
    "usage_input_tokens_cache_write_5m",
    "usage_input_tokens_cache_write_1h",
    "usage_input_tokens_cache_read",
)


def _input_tokens(row: dict) -> int | None:
    parts = [to_int(pick(row, col)) for col in _CACHE_INPUT_COLS]
    if any(p is not None for p in parts):
        return sum(p or 0 for p in parts)
    return to_int(pick(row, "input_tokens", "uncached_input_tokens", "input"))


class AnthropicCsvParser:
    name = "anthropic_csv"

    def can_parse(self, path: Path, head: str) -> bool:
        cols = csv_header(head)
        if "usage_date_utc" in cols:  # both real console export variants
            return True
        return "input_tokens" in cols and "output_tokens" in cols

    def parse(self, path: Path) -> ParsedExport:
        records = []
        for row in read_csv_rows(Path(path)):
            records.append(
                build_record(
                    source_format=self.name,
                    timestamp=parse_ts(pick(row, "usage_date_utc", "date", "usage_date", "timestamp")),
                    model=pick(row, "model_version", "model"),
                    route=pick(row, "workspace"),
                    input_tokens=_input_tokens(row),
                    output_tokens=to_int(pick(row, "usage_output_tokens", "output_tokens", "output")),
                    cost_usd=to_float(pick(row, "cost_usd", "cost", "amount_usd", "amount")),
                    n_requests=to_int(pick(row, "requests", "n_requests")) or 1,
                    prompt_text=None,
                )
            )

        caps = FormatCapabilities(
            per_request=False,
            has_prompt_text=False,
            has_cost=any(r.cost_usd is not None for r in records),
            has_tokens=any(r.input_tokens is not None or r.output_tokens is not None for r in records),
            has_timestamp=any(r.timestamp is not None for r in records),
        )
        return ParsedExport(source_format=self.name, capabilities=caps, records=tuple(records))
