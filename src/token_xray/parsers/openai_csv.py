"""Parser for OpenAI usage/activity CSV exports (billing aggregate, no prompt text).

These exports report token totals and request counts per model/day but carry no
cost column and no individual requests. Per Amendment 2, that means cost and all
per-request metrics are simply unavailable from this format.
"""

from __future__ import annotations

from pathlib import Path

from token_xray.models import FormatCapabilities, ParsedExport
from token_xray.parsers._util import csv_header, parse_ts, pick, read_csv_rows, to_float, to_int
from token_xray.parsers.base import build_record


class OpenAICsvParser:
    name = "openai_csv"

    def can_parse(self, path: Path, head: str) -> bool:
        cols = csv_header(head)
        if any("n_context_tokens" in c for c in cols):
            return True
        return "n_requests" in cols and "n_generated_tokens_total" in cols

    def parse(self, path: Path) -> ParsedExport:
        records = []
        for row in read_csv_rows(Path(path)):
            records.append(
                build_record(
                    source_format=self.name,
                    timestamp=parse_ts(pick(row, "timestamp", "date", "start_time")),
                    model=pick(row, "model"),
                    route=pick(row, "operation"),
                    input_tokens=to_int(
                        pick(row, "n_context_tokens_total", "n_context_tokens", "context_tokens", "input_tokens")
                    ),
                    output_tokens=to_int(
                        pick(row, "n_generated_tokens_total", "n_generated_tokens", "generated_tokens", "output_tokens")
                    ),
                    cost_usd=to_float(pick(row, "cost", "cost_usd", "amount", "amount_in_usd")),
                    n_requests=to_int(pick(row, "n_requests", "num_model_requests", "requests")) or 1,
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
