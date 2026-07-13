"""Format parsers and auto-detection.

A tiny registry: detection tries each parser against a small text preview of the
file, in order, and the first match wins. An explicit ``fmt`` overrides detection.
"""

from __future__ import annotations

from pathlib import Path

from token_xray.models import ParsedExport
from token_xray.parsers._util import read_head
from token_xray.parsers.anthropic_csv import AnthropicCsvParser
from token_xray.parsers.base import Parser
from token_xray.parsers.helicone import HeliconeParser
from token_xray.parsers.litellm_jsonl import LiteLLMJsonlParser
from token_xray.parsers.openai_csv import OpenAICsvParser

# Order matters: JSONL (LiteLLM) is checked first, then the CSV formats whose
# header signatures are mutually exclusive.
_PARSERS: tuple[Parser, ...] = (
    LiteLLMJsonlParser(),
    HeliconeParser(),
    OpenAICsvParser(),
    AnthropicCsvParser(),
)


class UnknownFormatError(ValueError):
    """Raised when a file matches no known export format."""


def supported_formats() -> list[str]:
    return [p.name for p in _PARSERS]


def detect(path: str | Path) -> Parser:
    """Return the parser that recognizes ``path``, or raise UnknownFormatError."""
    path = Path(path)
    head = read_head(path)
    for parser in _PARSERS:
        if parser.can_parse(path, head):
            return parser
    raise UnknownFormatError(
        f"Could not detect the format of '{path}'. "
        f"Supported formats: {', '.join(supported_formats())}."
    )


def parse(path: str | Path, fmt: str | None = None) -> ParsedExport:
    """Parse ``path`` into a ParsedExport, auto-detecting the format unless ``fmt`` is given."""
    path = Path(path)
    if fmt is not None:
        for parser in _PARSERS:
            if parser.name == fmt:
                return parser.parse(path)
        raise UnknownFormatError(
            f"Unknown format '{fmt}'. Supported formats: {', '.join(supported_formats())}."
        )
    return detect(path).parse(path)


__all__ = ["UnknownFormatError", "detect", "parse", "supported_formats"]
