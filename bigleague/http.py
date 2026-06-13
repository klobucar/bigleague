"""Shared async HTTP client + retry helpers used by savant.py and mlb.py.

A single `httpx.AsyncClient` is reused across the process; closing or
recreating it per request would defeat HTTP/2 connection reuse and cost
several hundred ms per call. The client is created lazily so importing
this module from a non-async context is free.

`SAVANT_SEMAPHORE` caps concurrent Savant fetches at 5 — Savant 429s
above that on the bare CSV endpoint, and we'd rather slow ourselves
down than get banned.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
SAVANT_SEMAPHORE = asyncio.Semaphore(5)

_MLB_BASE = "https://statsapi.mlb.com/api/v1"


async def get_client() -> httpx.AsyncClient:
    """Return the shared client, recreating it if its event loop changed.

    A cached client's pooled connections are bound to the loop it was
    created on; reusing it across `asyncio.run(...)` calls raises
    `RuntimeError: Event loop is closed` (which the retry helpers don't
    catch). If the stored loop isn't the current running loop — or the
    client is closed — discard it and start fresh.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is not None and (_client.is_closed or _client_loop is not loop):
        old, old_loop = _client, _client_loop
        _client = None
        _client_loop = None
        if not old.is_closed and old_loop is not None and not old_loop.is_closed():
            # Best-effort close on the loop that owns the connections; if
            # that loop is gone we just drop the reference.
            with contextlib.suppress(RuntimeError):
                asyncio.run_coroutine_threadsafe(old.aclose(), old_loop)
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": _BROWSER_UA},
        )
        _client_loop = loop
    return _client


async def close_client() -> None:
    """Close the shared client. Safe to call multiple times."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        # A client bound to a loop that no longer exists can't be closed
        # cleanly — suppress and drop the reference.
        with contextlib.suppress(RuntimeError):
            await _client.aclose()
    _client = None
    _client_loop = None


async def savant_get(url: str, params: dict[str, Any]) -> httpx.Response:
    """GET against baseballsavant.mlb.com with semaphore + retries.

    Retries 429 and 5xx with exponential backoff (3 attempts total).
    Transport errors retry too. 4xx (other than 429) bubbles up.
    """
    client = await get_client()
    async with SAVANT_SEMAPHORE:
        for attempt in range(3):
            try:
                resp = await client.get(url, params=params)
                if (resp.status_code == 429 or resp.status_code >= 500) and attempt < 2:
                    await asyncio.sleep(0.75 * (2**attempt))
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TransportError:
                if attempt < 2:
                    await asyncio.sleep(0.75 * (2**attempt))
                    continue
                raise
    raise RuntimeError("unreachable")


async def mlb_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET a /api/v1 path on statsapi.mlb.com and return parsed JSON.

    Retries 5xx and TransportError with exponential backoff.
    """
    client = await get_client()
    url = f"{_MLB_BASE}{path}"
    for attempt in range(3):
        try:
            resp = await client.get(url, params=params)
            if 500 <= resp.status_code < 600 and attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data
        except httpx.TransportError:
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            raise
    raise RuntimeError("unreachable")
