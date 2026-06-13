from __future__ import annotations

from datetime import date, timedelta

import httpx
import polars as pl
import pytest
import respx
from bigleague import savant
from bigleague.cache import StatcastCache
from bigleague.schema import STATCAST_SCHEMA

_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def _fake_csv_body(rows: int = 1) -> bytes:
    """Minimal CSV that exercises the schema (just batter/pitcher/game_pk)."""
    header = ",".join(STATCAST_SCHEMA.keys())
    sample = dict.fromkeys(STATCAST_SCHEMA, "")
    sample.update(
        {
            "batter": "1",
            "pitcher": "2",
            "game_pk": "3",
            "game_date": "2024-04-01",
            "game_year": "2024",
            "balls": "0",
            "strikes": "0",
            "outs_when_up": "0",
            "inning": "1",
            "at_bat_number": "1",
            "pitch_number": "1",
            "home_team": "NYY",
            "away_team": "BOS",
            "home_score": "0",
            "away_score": "0",
            "bat_score": "0",
            "fld_score": "0",
            "post_away_score": "0",
            "post_home_score": "0",
            "post_bat_score": "0",
            "post_fld_score": "0",
        }
    )
    row_str = ",".join(sample[k] for k in STATCAST_SCHEMA)
    body_lines = [header] + [row_str] * rows
    return "\n".join(body_lines).encode("utf-8")


def test_chunks_splits_into_seven_day_windows() -> None:
    chunks = savant._chunks(date(2024, 4, 1), date(2024, 4, 21))
    assert len(chunks) == 3
    assert chunks[0] == (date(2024, 4, 1), date(2024, 4, 7))
    assert chunks[1] == (date(2024, 4, 8), date(2024, 4, 14))
    assert chunks[2] == (date(2024, 4, 15), date(2024, 4, 21))


def test_chunks_handles_short_range() -> None:
    chunks = savant._chunks(date(2024, 4, 1), date(2024, 4, 3))
    assert chunks == [(date(2024, 4, 1), date(2024, 4, 3))]


def test_player_chunks_fully_historical_is_one_chunk() -> None:
    # 2020 is well in the past for any conceivable test run date.
    chunks = savant._player_chunks(date(2020, 4, 1), date(2020, 9, 30))
    assert chunks == [(date(2020, 4, 1), date(2020, 9, 30))]


def test_player_chunks_fully_future_is_one_chunk() -> None:
    today = date.today()
    start = today + timedelta(days=1)
    end = today + timedelta(days=30)
    assert savant._player_chunks(start, end) == [(start, end)]


def test_player_chunks_splits_across_today() -> None:
    today = date.today()
    start = today - timedelta(days=30)
    end = today + timedelta(days=2)
    chunks = savant._player_chunks(start, end)
    assert chunks == [(start, today - timedelta(days=1)), (today, end)]


@respx.mock
async def test_statcast_omits_player_params_when_no_filter() -> None:
    route = respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    await savant.statcast(date(2024, 4, 1), date(2024, 4, 2))
    sent = route.calls[0].request
    assert "batters_lookup[]" not in sent.url.params
    assert "pitchers_lookup[]" not in sent.url.params
    assert "player_type" not in sent.url.params


@respx.mock
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("nyy", "NYY"),  # case-folded (and dedupes the cache scope)
        ("CHW", "CWS"),  # alias Savant rejects → its real code
        ("WSN", "WSH"),
        ("ARI", "AZ"),
    ],
)
async def test_statcast_normalizes_team_filter(given: str, expected: str) -> None:
    """The team filter is normalized to Savant's codes before it goes out:
    Savant returns an empty frame for aliases like CHW/KCR/SDP/WSN."""
    route = respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    await savant.statcast(date(2024, 4, 1), date(2024, 4, 2), team=given)
    assert route.calls[0].request.url.params["team"] == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("oak", "ATH"),  # alias map applies after case-folding
        ("NYY", "NYY"),  # unknown-to-the-map codes pass through unchanged
        (" tb ", "TB"),  # whitespace stripped
    ],
)
def test_normalize_team_accepts_plausible_codes(given: str, expected: str) -> None:
    assert savant.normalize_team(given) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "../../tmp/pwn",  # path traversal — became a cache filename verbatim
        "a/b",
        "",
        "TOOLONGCODE",
        "NY1",
    ],
)
def test_normalize_team_rejects_non_team_shaped_input(bad: str) -> None:
    """Anything that isn't 2-5 uppercase letters after normalization must
    raise: the result feeds the parquet cache filename."""
    with pytest.raises(ValueError, match="invalid team abbreviation"):
        savant.normalize_team(bad)


@respx.mock
async def test_statcast_batter_includes_player_params() -> None:
    route = respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    await savant.statcast_batter(660271, date(2024, 4, 1), date(2024, 4, 2))
    sent = route.calls[0].request
    # Savant ignores `player_id` here — must be batters_lookup[].
    assert sent.url.params["batters_lookup[]"] == "660271"
    assert sent.url.params["player_type"] == "batter"


@respx.mock
async def test_statcast_pitcher_includes_player_params() -> None:
    route = respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    await savant.statcast_pitcher(660271, date(2024, 4, 1), date(2024, 4, 2))
    sent = route.calls[0].request
    assert sent.url.params["pitchers_lookup[]"] == "660271"
    assert sent.url.params["player_type"] == "pitcher"


@respx.mock
async def test_empty_response_returns_empty_dataframe() -> None:
    respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=b""))
    df = await savant.statcast(date(2024, 4, 1), date(2024, 4, 2))
    assert df.height == 0
    # Schema preserved on empty results
    assert set(df.columns) >= {"batter", "pitcher", "game_date"}


@respx.mock
async def test_cached_chunk_skips_http(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A previously cached chunk must not trigger a second HTTP call."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setattr(savant, "cache", StatcastCache(tmp_path))

    # Pre-seed cache for the chunk we'll query — use a historical date
    # so the right-edge guard doesn't refuse to write.
    start, end = date(2020, 4, 1), date(2020, 4, 7)
    key = savant._cache_key("statcast", "all", start, end)
    savant.cache.set(key, pl.DataFrame({"x": [1]}))

    route = respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    df = await savant.statcast(start, end)
    assert route.call_count == 0
    assert df.height == 1


@respx.mock
async def test_future_chunk_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Chunks whose end_date >= today must not be persisted."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setattr(savant, "cache", StatcastCache(tmp_path))

    today = date.today()
    start = today
    end = today  # end_date is today -> should NOT cache
    respx.get(_CSV_URL).mock(return_value=httpx.Response(200, content=_fake_csv_body()))
    await savant.statcast(start, end)
    # Cache dir must remain empty
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.skipif(
    "BIGLEAGUE_NETWORK_TESTS" not in __import__("os").environ,
    reason="set BIGLEAGUE_NETWORK_TESTS=1 to run live Savant calls",
)
async def test_statcast_batter_filter_hits_one_player_on_real_savant() -> None:
    """Real-network guard: Savant must actually filter to the requested batter.

    Catches the silent-cap bug where the wrong param key (e.g. `player_id`
    instead of `batters_lookup[]`) returns ~25k unfiltered rows per chunk.
    """
    # Ohtani, first week of April 2025 — a historical window with a known
    # ceiling well under Savant's 25k row cap for a single batter.
    df = await savant.statcast_batter(660271, date(2025, 4, 1), date(2025, 4, 7))
    assert df.height > 0, "expected some pitches for Ohtani that week"
    assert df.height < 1000, (
        f"got {df.height} rows — Savant's batter filter isn't being applied "
        f"and we're pulling league-wide data."
    )
    assert df["batter"].unique().to_list() == [660271]


async def test_semaphore_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SAVANT_SEMAPHORE actually bounds in-flight requests at 5."""
    import asyncio

    from bigleague import http as http_mod

    in_flight = 0
    high_water = 0

    async def _fake_get(self, url, params=None, **kw):
        nonlocal in_flight, high_water
        in_flight += 1
        high_water = max(high_water, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, content=_fake_csv_body(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    # Force fresh semaphore + client
    http_mod.SAVANT_SEMAPHORE = asyncio.Semaphore(5)
    await http_mod.close_client()

    # Request 60 days = 9 chunks > the 5-permit limit.
    await savant.statcast(date(2024, 4, 1), date(2024, 5, 30))
    assert high_water <= 5
