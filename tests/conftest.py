"""Test fixtures: force BIGLEAGUE_NO_CACHE on import, reset httpx client per test."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

# Set before any bigleague import — cache module reads this at construction time.
os.environ.setdefault("BIGLEAGUE_NO_CACHE", "1")

import pytest
from bigleague import http as _http


@pytest.fixture(autouse=True)
async def _reset_http_client() -> AsyncIterator[None]:
    """Drop the shared async client before each test so respx mocks bind cleanly."""
    await _http.close_client()
    yield
    await _http.close_client()
