"""History hygiene: the guarantee covers every commit, not just the working tree.

Standing DoD condition (Anvay, 2026-07-14, post history-rewrite): no email,
org-ID, or key-like string may exist in ANY revision. The single allowed
exception class is RFC-2606 ``@example.com`` addresses, which the privacy test
uses as deliberately fake PII.

Skipped outside a git checkout (e.g. installed from a wheel or sdist).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|org-[A-Za-z0-9]{8,}|sk-[A-Za-z0-9.]{2,}"
_ALLOWED = re.compile(r"^[A-Za-z0-9._%+-]+@example\.com$")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="requires a git checkout",
)
def test_no_identifying_strings_in_any_revision():
    revisions = _git("rev-list", "--all").split()
    assert revisions, "no revisions found?"

    hits: set[str] = set()
    for start in range(0, len(revisions), 50):  # chunk to stay under cmdline limits
        chunk = revisions[start : start + 50]
        out = _git("grep", "-I", "-h", "-o", "-E", PATTERN, *chunk, "--")
        hits.update(line.strip() for line in out.splitlines() if line.strip())

    offenders = sorted(h for h in hits if not _ALLOWED.fullmatch(h))
    assert not offenders, (
        f"identifying strings found in git history: {offenders} - "
        f"rewrite history before publishing (see plan/notes.md, 2026-07-14)"
    )
