"""Write the human-reviewable JSON artifact (aggregates only)."""

from __future__ import annotations

import json
from pathlib import Path

from token_xray import __version__
from token_xray.analysis.aggregates import XRayReport


def export_json(report: XRayReport, path: Path) -> None:
    """Write ``xray_report.json``: tool metadata + the aggregate report.

    The file is small and meant to be read by a human before it is shared.
    """
    payload = {
        "tool": "token-xray",
        "version": __version__,
        "privacy": "aggregates only; contains no prompt text",
        "report": report.to_dict(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
