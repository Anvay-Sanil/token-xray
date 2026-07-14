"""Fixture hygiene: committed fixtures must be provably synthetic.

Real exports contain emails, org IDs, and key fragments. Fixtures modeled on
real schemas keep the real HEADER but must carry none of those value shapes —
enforced here so a future fixture refresh can't accidentally commit a real
file (standing condition from Artifact 1 sign-off, 2026-07-14).
"""

from __future__ import annotations

BANNED_PATTERNS = ("@", "org-", "sk-")


def test_fixtures_contain_no_identifying_patterns(fixtures_dir):
    fixture_files = sorted(p for p in fixtures_dir.iterdir() if p.is_file())
    assert fixture_files, "fixture directory is empty?"
    for path in fixture_files:
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            assert pattern not in text, (
                f"{path.name} contains banned pattern {pattern!r} - fixtures "
                f"must be synthetic (no emails, org IDs, or key-like strings)"
            )
