"""Parser for LiteLLM proxy request logs (JSONL, one JSON object per line)."""

from __future__ import annotations

import json
from pathlib import Path

from token_xray.models import FormatCapabilities, ParsedExport
from token_xray.parsers._util import messages_to_text, parse_ts
from token_xray.parsers.base import build_record

_HINT_KEYS = ("call_type", "usage", "response_cost", "litellm_call_id", "messages")
_ERROR_STATUSES = {"failure", "error", "failed"}


class LiteLLMJsonlParser:
    name = "litellm_jsonl"

    def can_parse(self, path: Path, head: str) -> bool:
        first = next((ln for ln in head.splitlines() if ln.strip()), "")
        try:
            obj = json.loads(first)
        except (json.JSONDecodeError, ValueError):
            return False
        return isinstance(obj, dict) and any(k in obj for k in _HINT_KEYS)

    def parse(self, path: Path) -> ParsedExport:
        records = []
        with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                usage = obj.get("usage") or {}
                status = obj.get("status")
                is_error = bool(obj.get("exception")) or (
                    status is not None and str(status).lower() in _ERROR_STATUSES
                )
                prompt_text = messages_to_text(obj.get("messages")) or obj.get("prompt") or obj.get("input")
                records.append(
                    build_record(
                        source_format=self.name,
                        timestamp=parse_ts(obj.get("startTime") or obj.get("timestamp") or obj.get("start_time")),
                        model=obj.get("model"),
                        route=obj.get("call_type") or obj.get("endpoint"),
                        input_tokens=usage.get("prompt_tokens", obj.get("prompt_tokens")),
                        output_tokens=usage.get("completion_tokens", obj.get("completion_tokens")),
                        cost_usd=obj.get("response_cost", obj.get("cost")),
                        is_error=is_error,
                        status=str(status) if status is not None else None,
                        prompt_text=prompt_text if isinstance(prompt_text, str) else None,
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
