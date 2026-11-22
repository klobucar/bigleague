"""Test fixtures: force BIGLEAGUE_NO_CACHE on import, reset shared state per test."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

# Set before any bigleague import — cache module reads this at construction time.
os.environ.setdefault("BIGLEAGUE_NO_CACHE", "1")

import pytest
from bigleague import http as _http
from bigleague import players as _players


@pytest.fixture(autouse=True)
async def _reset_http_client() -> AsyncIterator[None]:
    """Drop the shared async client before each test so respx mocks bind cleanly."""
    await _http.close_client()
    yield
    await _http.close_client()


@pytest.fixture(autouse=True)
def _reset_register() -> Iterator[None]:
    """The player register is memoized per process; forget it between tests."""
    _players._forget_register()
    yield
    _players._forget_register()
