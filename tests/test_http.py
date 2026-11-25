"""Retry and client-lifecycle behavior of the shared HTTP helpers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from bigleague import http

_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
_MLB_URL = "https://statsapi.mlb.com/api/v1/teams"

_real_sleep = asyncio.sleep


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of actually waiting them out."""
    delays: list[float] = []

    async def _fake_sleep(delay: float, *args: object) -> None:
        delays.append(delay)
        await _real_sleep(0)

    monkeypatch.setattr(http.asyncio, "sleep", _fake_sleep)
    return delays


@respx.mock
async def test_savant_get_retries_429_with_exponential_backoff(sleeps: list[float]) -> None:
    route = respx.get(_SAVANT_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(503), httpx.Response(200, text="ok")]
    )
    resp = await http.savant_get(_SAVANT_URL, {})
    assert resp.text == "ok"
    assert route.call_count == 3
    assert sleeps == [0.75, 1.5]


@respx.mock
async def test_savant_get_gives_up_after_three_attempts(sleeps: list[float]) -> None:
    route = respx.get(_SAVANT_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError):
        await http.savant_get(_SAVANT_URL, {})
    assert route.call_count == 3
    assert len(sleeps) == 2  # no sleep after the final attempt


@respx.mock
async def test_savant_get_does_not_retry_plain_4xx(sleeps: list[float]) -> None:
    route = respx.get(_SAVANT_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        await http.savant_get(_SAVANT_URL, {})
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
async def test_savant_get_retries_transport_errors_then_raises(sleeps: list[float]) -> None:
    route = respx.get(_SAVANT_URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(httpx.ConnectError):
        await http.savant_get(_SAVANT_URL, {})
    assert route.call_count == 3
    assert sleeps == [0.75, 1.5]


@respx.mock
async def test_savant_get_recovers_from_transient_transport_error(sleeps: list[float]) -> None:
    respx.get(_SAVANT_URL).mock(
        side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, text="ok")]
    )
    resp = await http.savant_get(_SAVANT_URL, {})
    assert resp.text == "ok"
    assert sleeps == [0.75]


async def test_savant_semaphore_caps_concurrency(sleeps: list[float]) -> None:
    in_flight = 0
    peak = 0

    async def _slow(_request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await _real_sleep(0.01)
        in_flight -= 1
        return httpx.Response(200)

    with respx.mock:
        respx.get(_SAVANT_URL).mock(side_effect=_slow)
        await asyncio.gather(*(http.savant_get(_SAVANT_URL, {}) for _ in range(12)))
    assert peak == 5


@respx.mock
async def test_mlb_get_does_not_retry_4xx(sleeps: list[float]) -> None:
    route = respx.get(f"{_MLB_URL}").mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        await http.mlb_get("/teams")
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
async def test_mlb_get_retries_transport_errors_then_raises(sleeps: list[float]) -> None:
    route = respx.get(f"{_MLB_URL}").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        await http.mlb_get("/teams")
    assert route.call_count == 3
    assert sleeps == [0.5, 1.0]


async def test_get_client_is_reused_within_a_loop() -> None:
    a = await http.get_client()
    b = await http.get_client()
    assert a is b


async def test_get_client_replaces_a_closed_client() -> None:
    a = await http.get_client()
    await a.aclose()
    b = await http.get_client()
    assert b is not a
    assert not b.is_closed


def test_get_client_replaces_client_from_a_dead_loop() -> None:
    """A client cached by one asyncio.run must not leak into the next."""
    first = asyncio.run(http.get_client())
    second = asyncio.run(http.get_client())
    assert second is not first
    asyncio.run(http.close_client())


async def test_close_client_is_idempotent() -> None:
    await http.get_client()
    await http.close_client()
    await http.close_client()
    assert http._client is None
