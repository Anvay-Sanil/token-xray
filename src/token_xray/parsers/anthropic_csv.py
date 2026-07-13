"""Parser for Anthropic Console usage CSV exports (billing aggregate, no prompt text).

Reports input/output tokens and cost per model/day. No individual requests and
no prompt text, so per-request metrics are unavailable (Amendment 2).
"""

from __future__ import annotations

from pathlib import Path

from token_xray.models import FormatCapabilities, ParsedExport
from token_xray.parsers._util import csv_header, parse_ts, pick, read_csv_rows, to_float, to_int
from token_xray.parsers.base import build_record


class AnthropicCsvParser:
    name = "anthropic_csv"

    def can_parse(self, path: Path, head: str) -> bool:
        cols = csv_header(head)
        return "input_tokens" in cols and "output_tokens" in cols

    def parse(self, path: Path) -> ParsedExport:
        records = []
        for row in read_csv_rows(Path(path)):
            records.append(
                build_record(
                    source_format=self.name,
                    timestamp=parse_ts(pick(row, "date", "usage_date", "timestamp")),
                    model=pick(row, "model"),
                    route=pick(row, "workspace"),
                    input_tokens=to_int(pick(row, "input_tokens", "uncached_input_tokens", "input")),
                    output_tokens=to_int(pick(row, "output_tokens", "output")),
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
