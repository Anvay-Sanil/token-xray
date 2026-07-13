"""Shared test harness.

The autouse ``_no_network`` fixture blocks all socket access for every test.
This is the enforcement mechanism behind token-xray's zero-network guarantee:
if any code path (ours or a dependency's) tries to open a connection during a
test, the test fails loudly instead of silently phoning home.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkBlockedError(RuntimeError):
    """Raised when code attempts network access while the guard is active."""


def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NetworkBlockedError(
        "Network access is blocked: token-xray must run fully offline."
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Disable every common socket entry point for the duration of a test."""
    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
