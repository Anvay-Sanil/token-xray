"""Report renderers: terminal (rich), JSON export, and single-file HTML.

All renderers consume the aggregate-only XRayReport. None of them can access
prompt text because none exists anywhere upstream.
"""

from token_xray.report.html import render_html
from token_xray.report.json_export import export_json
from token_xray.report.terminal import render_terminal

__all__ = ["export_json", "render_html", "render_terminal"]
