"""Privacy guarantee: no prompt text in any output artifact.

The fixtures for per-request formats contain known prompt strings. These tests
run the full pipeline and assert that no fragment of any of those prompts —
nor any message structure that could carry them — appears in the JSON or HTML
artifacts.
"""

from __future__ import annotations

import json

from token_xray.cli import run_analysis

# Distinctive fragments of every prompt present in the per-request fixtures.
PROMPT_FRAGMENTS = [
    "customer support ticket",
    "sentiment of this product review",
    "English to French",
    "Fibonacci",
    "payment obligation",
]


def _artifacts_for(path, tmp_path):
    json_path = tmp_path / "out.json"
    html_path = tmp_path / "out.html"
    run_analysis(path, fmt=None, json_out=json_path, html_out=html_path, quiet=True)
    return json_path.read_text(encoding="utf-8"), html_path.read_text(encoding="utf-8")


def test_litellm_artifacts_contain_no_prompt_text(fixtures_dir, tmp_path):
    json_text, html_text = _artifacts_for(fixtures_dir / "litellm_proxy_sample.jsonl", tmp_path)
    for fragment in PROMPT_FRAGMENTS:
        assert fragment.lower() not in json_text.lower()
        assert fragment.lower() not in html_text.lower()
    assert '"messages"' not in json_text


def test_helicone_artifacts_contain_no_prompt_text(fixtures_dir, tmp_path):
    json_text, html_text = _artifacts_for(fixtures_dir / "helicone_sample.csv", tmp_path)
    for fragment in PROMPT_FRAGMENTS:
        assert fragment.lower() not in json_text.lower()
        assert fragment.lower() not in html_text.lower()


def test_json_is_valid_and_aggregates_only(fixtures_dir, tmp_path):
    json_text, _ = _artifacts_for(fixtures_dir / "litellm_proxy_sample.jsonl", tmp_path)
    data = json.loads(json_text)
    assert data["tool"] == "token-xray"
    assert set(data["report"].keys()) == {"source_format", "totals", "metrics"}
    # No value anywhere in the report should be a suspiciously long string
    # (aggregates are numbers, short labels, and small lists).
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            assert len(node) < 200
    walk(data)
