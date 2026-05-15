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
from typing import Any

import httpx

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_client: httpx.AsyncClient | None = None
SAVANT_SEMAPHORE = asyncio.Semaphore(5)

_MLB_BASE = "https://statsapi.mlb.com/api/v1"


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": _BROWSER_UA},
        )
    return _client


async def close_client() -> None:
    """Close the shared client. Safe to call multiple times."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


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
