"""CLI behavior tests (Typer runner — no subprocess, so the socket guard stays active)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from token_xray.cli import app

runner = CliRunner()


def _all_output(result) -> str:
    """stdout + stderr regardless of how this Click version splits them."""
    try:
        return result.output + result.stderr
    except ValueError:  # older Click mixes stderr into output already
        return result.output


def test_analyze_prints_report(fixtures_dir):
    result = runner.invoke(app, ["analyze", str(fixtures_dir / "litellm_proxy_sample.jsonl")])
    assert result.exit_code == 0
    assert "token-xray" in result.output
    assert "litellm_jsonl" in result.output


def test_analyze_writes_json(fixtures_dir, tmp_path):
    out = tmp_path / "r.json"
    result = runner.invoke(
        app,
        ["analyze", str(fixtures_dir / "anthropic_console_sample.csv"), "--json-out", str(out)],
    )
    assert result.exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["report"]["source_format"] == "anthropic_csv"


def test_analyze_writes_html(fixtures_dir, tmp_path):
    out = tmp_path / "r.html"
    result = runner.invoke(
        app,
        ["analyze", str(fixtures_dir / "litellm_proxy_sample.jsonl"), "--html", str(out)],
    )
    assert result.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "http://" not in html and "https://" not in html  # no external assets, ever


def test_unknown_file_fails_with_supported_formats(tmp_path):
    junk = tmp_path / "mystery.txt"
    junk.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    result = runner.invoke(app, ["analyze", str(junk)])
    assert result.exit_code != 0
    for name in ("openai_csv", "anthropic_csv", "litellm_jsonl", "helicone"):
        assert name in _all_output(result)


def test_missing_file_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.csv")])
    assert result.exit_code != 0


def test_billing_export_report_shows_unavailable_label(fixtures_dir):
    result = runner.invoke(app, ["analyze", str(fixtures_dir / "openai_usage_sample.csv")])
    assert result.exit_code == 0
    assert "unavailable from this export format" in result.output


def test_terminal_output_contains_no_advice(fixtures_dir):
    """F7 guard: the rendered report must contain statistics only."""
    result = runner.invoke(app, ["analyze", str(fixtures_dir / "litellm_proxy_sample.jsonl")])
    lowered = result.output.lower()
    for banned in ("recommend", "you should", "consider ", "savings", "switch to", "migrate"):
        assert banned not in lowered


def test_malformed_file_exits_cleanly_without_traceback(fixtures_dir, tmp_path):
    good = (fixtures_dir / "litellm_proxy_sample.jsonl").read_text(encoding="utf-8").splitlines()
    broken = tmp_path / "truncated.jsonl"
    broken.write_text(good[0] + "\n" + '{"model": "gpt-4o", "usa\n', encoding="utf-8")

    result = runner.invoke(app, ["analyze", str(broken)])
    assert result.exit_code == 2
    out = _all_output(result)
    assert "line 2" in out
    assert "Traceback" not in out
