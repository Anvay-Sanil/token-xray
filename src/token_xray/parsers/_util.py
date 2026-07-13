"""Shared, dependency-light helpers for the format parsers.

Deliberately tolerant: real exports drift in column names and types, so lookups
are case-insensitive with aliases and numeric coercion never raises.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


def read_head(path: Path, n_bytes: int = 4096) -> str:
    """Read a small text preview used only for format detection."""
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        return f.read(n_bytes)


def csv_header(head: str) -> list[str]:
    """Return lowercased column names from a CSV preview, or [] if it looks like JSON."""
    lines = [ln for ln in head.splitlines() if ln.strip()]
    if not lines or lines[0].lstrip().startswith("{"):
        return []
    return [c.strip().strip('"').lower() for c in lines[0].split(",")]


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def pick(row: dict, *aliases: str) -> Optional[str]:
    """Case-insensitive lookup returning the first non-empty aliased value."""
    low = {str(k).lower(): v for k, v in row.items()}
    for alias in aliases:
        value = low.get(alias.lower())
        if value is not None and str(value).strip() != "":
            return value
    return None


def to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").lstrip("$")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_ts(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


def messages_to_text(messages: object) -> Optional[str]:
    """Flatten an OpenAI/Anthropic-style messages list to plain text, or None."""
    if not isinstance(messages, list):
        return None
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    joined = "\n".join(p for p in parts if p)
    return joined or None
