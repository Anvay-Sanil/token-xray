"""Parser for Helicone exports (CSV; prompt text lives in a request_body JSON cell)."""

from __future__ import annotations

import json
from pathlib import Path

from token_xray.models import FormatCapabilities, ParsedExport
from token_xray.parsers._util import (
    csv_header,
    messages_to_text,
    parse_ts,
    pick,
    read_csv_rows,
    to_float,
    to_int,
)
from token_xray.parsers.base import build_record


class HeliconeParser:
    name = "helicone"

    def can_parse(self, path: Path, head: str) -> bool:
        cols = csv_header(head)
        return "request_id" in cols and ("request_body" in cols or "prompt_tokens" in cols)

    @staticmethod
    def _is_http_error(status: object) -> bool:
        if status is None:
            return False
        try:
            return int(float(str(status))) >= 400
        except ValueError:
            return str(status).lower() in {"error", "failed", "failure"}

    @staticmethod
    def _prompt_from_body(row: dict) -> str | None:
        body = pick(row, "request_body", "body")
        if body:
            try:
                text = messages_to_text(json.loads(body).get("messages"))
                if text:
                    return text
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass
        return pick(row, "prompt")

    def parse(self, path: Path) -> ParsedExport:
        records = []
        for row in read_csv_rows(Path(path)):
            status = pick(row, "status", "response_status", "status_code")
            records.append(
                build_record(
                    source_format=self.name,
                    timestamp=parse_ts(pick(row, "created_at", "created", "request_created_at", "timestamp")),
                    model=pick(row, "model"),
                    route=pick(row, "path", "route"),
                    input_tokens=to_int(pick(row, "prompt_tokens", "input_tokens")),
                    output_tokens=to_int(pick(row, "completion_tokens", "output_tokens")),
                    cost_usd=to_float(pick(row, "cost", "cost_usd", "total_cost")),
                    is_error=self._is_http_error(status),
                    status=str(status) if status is not None else None,
                    prompt_text=self._prompt_from_body(row),
                )
            )

        caps = FormatCapabilities(
            per_request=True,
            has_prompt_text=any(r.prompt_hash is not None for r in records),
            has_cost=any(r.cost_usd is not None for r in records),
            has_tokens=any(r.input_tokens is not None or r.output_tokens is not None for r in records),
            has_timestamp=any(r.timestamp is not None for r in records),
        )
        return ParsedExport(source_format=self.name, capabilities=caps, records=tuple(records))
