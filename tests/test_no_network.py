"""Zero-network guarantee, end to end.

conftest.py already blocks sockets for every test; this module proves two
things explicitly:

1. the guard itself is live (any socket use raises), and
2. the ENTIRE pipeline — parse, analyze, render terminal, write JSON and HTML —
   completes with sockets blocked, for every supported format.

If any dependency or future code path tries to phone home, these tests fail.
"""

from __future__ import annotations

import socket

import pytest

from token_xray.cli import run_analysis

ALL_FIXTURES = [
    "openai_usage_sample.csv",
    "anthropic_console_sample.csv",
    "litellm_proxy_sample.jsonl",
    "helicone_sample.csv",
]


def test_socket_guard_is_active():
    with pytest.raises(RuntimeError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 443))


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_full_pipeline_runs_offline(fixture_name, fixtures_dir, tmp_path):
    """Parse + analyze + render every output artifact with sockets blocked."""
    json_path = tmp_path / "xray_report.json"
    html_path = tmp_path / "xray_report.html"
    run_analysis(
        fixtures_dir / fixture_name,
        fmt=None,
        json_out=json_path,
        html_out=html_path,
        quiet=True,
    )
    assert json_path.exists() and json_path.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0
