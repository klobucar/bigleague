from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest
import respx
from bigleague import savant, stats
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


def test_player_chunks_single_historical_season_is_one_chunk() -> None:
    # 2020 is well in the past for any conceivable test run date.
    chunks = savant._player_chunks(date(2020, 4, 1), date(2020, 9, 30))
    assert chunks == [(date(2020, 4, 1), date(2020, 9, 30))]


def test_player_chunks_splits_multi_year_history_at_year_boundaries() -> None:
    """Regression: a multi-season span used to go out as ONE request, which
    Savant silently truncates at ~25k rows — and the truncated frame then got
    cached. Calendar-year chunks stay well under the cap and cache stably."""
    chunks = savant._player_chunks(date(2019, 6, 15), date(2021, 8, 1))
    assert chunks == [
        (date(2019, 6, 15), date(2019, 12, 31)),
        (date(2020, 1, 1), date(2020, 12, 31)),
        (date(2021, 1, 1), date(2021, 8, 1)),
    ]


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
    # The today-onward portion is always its own chunk (stays uncached)...
    assert chunks[-1] == (today, end)
    # ...and the historical portion covers [start, yesterday].
    assert chunks[0][0] == start
    assert chunks[-2][1] == today - timedelta(days=1)


def test_cache_key_contains_schema_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parquet cached under an old StatcastPitch schema must never mix with
    freshly fetched frames; the fingerprint retires old keys on schema change."""
    start, end = date(2024, 4, 1), date(2024, 4, 7)
    key = savant._cache_key("statcast", "all", start, end)
    assert key.endswith(f"_{savant._SCHEMA_FINGERPRINT}")
    monkeypatch.setattr(savant, "_SCHEMA_FINGERPRINT", "0" * 8)
    assert savant._cache_key("statcast", "all", start, end) != key


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


_PCT_URL = "https://baseballsavant.mlb.com/leaderboard/percentile-rankings"


@respx.mock
@pytest.mark.parametrize("fn_name", ["batter_percentile_ranks", "pitcher_percentile_ranks"])
async def test_percentile_ranks_filter_to_requested_player(fn_name: str) -> None:
    """Regression: player_id was accepted but ignored — the full leaderboard
    came back and callers had to filter it themselves."""
    csv = b"player_id,xwoba,k_percent\n1,90,80\n2,50,40\n"
    respx.get(_PCT_URL).mock(return_value=httpx.Response(200, content=csv))
    df = await getattr(savant, fn_name)(1, 2024)
    assert df.height == 1
    assert df["player_id"].to_list() == [1]
    assert df["xwoba"].to_list() == [90]


@respx.mock
async def test_percentile_ranks_without_id_column_return_full_frame() -> None:
    """If Savant renames the id column we degrade to the old behavior
    (full frame) instead of erroring or silently dropping all rows."""
    csv = b"mlbam_id,xwoba\n1,90\n2,50\n"
    respx.get(_PCT_URL).mock(return_value=httpx.Response(200, content=csv))
    df = await savant.batter_percentile_ranks(1, 2024)
    assert df.height == 2


def test_running_splits_is_deprecated_alias_for_sprint_speed() -> None:
    import bigleague

    assert savant.running_splits is savant.sprint_speed
    assert bigleague.sprint_speed is savant.sprint_speed
    assert bigleague.running_splits is savant.sprint_speed


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


def test_get_client_returns_fresh_client_per_event_loop() -> None:
    """Regression: a client cached across `asyncio.run(...)` calls keeps
    pooled connections bound to the first (closed) loop, so the next call
    dies with 'RuntimeError: Event loop is closed'."""
    import asyncio

    from bigleague import http as http_mod

    c1 = asyncio.run(http_mod.get_client())
    c2 = asyncio.run(http_mod.get_client())
    assert c2 is not c1

    async def _same_loop_reuses() -> bool:
        a = await http_mod.get_client()
        b = await http_mod.get_client()
        return a is b

    assert asyncio.run(_same_loop_reuses())
    asyncio.run(http_mod.close_client())


# ── pitch arsenal ────────────────────────────────────────────────────────────
#
# The fixtures are real rows from Savant (2026 season, fetched live), trimmed to
# five pitchers with their exact bytes: UTF-8 BOM, quoted ids, "" for pitches
# not thrown on the `n_` board. An earlier version of these tests fed hand-built
# frames with `avg_speed` columns to `parse_pitch_arsenal`; the endpoint it
# actually used never returns such a column, so the pair always returned [].

_FIXTURES = Path(__file__).parent / "fixtures" / "savant"
_ARSENALS_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals"
_ARSENAL_STATS_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
_ARSENAL_BOARDS = {
    "avg_speed": "pitch_arsenals_avg_speed.csv",
    "avg_spin": "pitch_arsenals_avg_spin.csv",
    "n_": "pitch_arsenals_n.csv",
}
SKUBAL = 669373
WALDRON = 663362  # throws a knuckleball
VAUGHN = 683734  # a position player: one pitch, 100% knuckleball
FRIED = 608331  # not on the min=100 run-value board at all


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _mock_arsenals() -> respx.Route:
    """Serve the three `pitch-arsenals` boards, picked by the `type` param."""

    def _by_type(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_fixture(_ARSENAL_BOARDS[request.url.params["type"]]))

    return respx.get(_ARSENALS_URL).mock(side_effect=_by_type)


def test_arsenal_fixtures_have_the_layout_the_endpoint_really_returns() -> None:
    """Guard the fixtures themselves: they must keep Savant's real columns."""
    speed = pl.read_csv(_FIXTURES / _ARSENAL_BOARDS["avg_speed"])
    assert speed.columns == [
        "last_name, first_name",
        "pitcher",
        *(f"{p}_avg_speed" for p in ["ff", "si", "fc", "sl", "ch", "cu", "fs", "kn", "st", "sv"]),
    ]
    assert pl.read_csv(_FIXTURES / _ARSENAL_BOARDS["n_"]).schema["n_ff"] == pl.String
    assert (_FIXTURES / _ARSENAL_BOARDS["n_"]).read_bytes().startswith(b"\xef\xbb\xbf")  # BOM
    runvalue = pl.read_csv(_FIXTURES / "pitch_arsenal_stats_pitcher.csv", ignore_errors=True)
    assert "avg_speed" not in runvalue.columns  # why the old pairing returned nothing


@respx.mock
async def test_pitcher_pitch_arsenal_returns_velocity_spin_and_usage() -> None:
    _mock_arsenals()
    df = await savant.pitcher_pitch_arsenal(SKUBAL, 2026)

    assert df.columns == ["pitch_type", "pitch_name", "avg_speed", "avg_spin", "pitch_usage"]
    # Skubal, 2026, as read off Baseball Savant.
    assert df.select("pitch_type", "avg_speed", "avg_spin", "pitch_usage").rows() == [
        ("FF", 96.8, 2332.0, 38.0),
        ("SI", 96.7, 2193.0, 17.5),
        ("SL", 89.7, 2203.0, 13.9),
        ("CH", 87.7, 1847.0, 25.6),
        ("CU", 80.4, 2547.0, 5.0),
    ]
    assert df["pitch_name"].to_list() == [
        "4-Seam Fastball",
        "Sinker",
        "Slider",
        "Changeup",
        "Curveball",
    ]


@respx.mock
async def test_parse_pitch_arsenal_of_the_real_boards_is_correct_and_usage_sorted() -> None:
    """The regression: pitcher_pitch_arsenal + parse_pitch_arsenal used to
    return [] for every pitcher and season."""
    _mock_arsenals()
    out = stats.parse_pitch_arsenal(await savant.pitcher_pitch_arsenal(SKUBAL, 2026))

    assert out == [
        {"pitch": "FF", "mph": 96.8, "spin": 2332.0, "usage_pct": 38.0},
        {"pitch": "CH", "mph": 87.7, "spin": 1847.0, "usage_pct": 25.6},
        {"pitch": "SI", "mph": 96.7, "spin": 2193.0, "usage_pct": 17.5},
        {"pitch": "SL", "mph": 89.7, "spin": 2203.0, "usage_pct": 13.9},
        {"pitch": "CU", "mph": 80.4, "spin": 2547.0, "usage_pct": 5.0},
    ]


@respx.mock
async def test_pitcher_pitch_arsenal_requests_three_boards_with_a_low_minimum() -> None:
    route = _mock_arsenals()
    await savant.pitcher_pitch_arsenal(SKUBAL, 2026)

    assert route.call_count == 3
    sent = [dict(c.request.url.params) for c in route.calls]
    assert {p["type"] for p in sent} == {"avg_speed", "avg_spin", "n_"}
    for p in sent:
        # min=10, not 100: the old board's minimum dropped Skubal's SL and CU.
        assert (p["year"], p["min"], p["hand"], p["csv"]) == ("2026", "10", "", "true")


@respx.mock
async def test_pitcher_pitch_arsenal_min_pitches_is_passed_through() -> None:
    route = _mock_arsenals()
    await savant.pitcher_pitch_arsenal(SKUBAL, 2026, min_pitches=50)
    assert {c.request.url.params["min"] for c in route.calls} == {"50"}


@respx.mock
async def test_arsenal_handles_a_knuckleball_and_blank_usage_cells() -> None:
    _mock_arsenals()
    out = stats.parse_pitch_arsenal(await savant.pitcher_pitch_arsenal(WALDRON, 2026))
    by_pitch = {p["pitch"]: p for p in out}

    assert set(by_pitch) == {"FF", "SI", "FC", "KN", "ST"}  # the "" cells are pitches not thrown
    assert out[0]["pitch"] == "KN"  # his most-used pitch
    assert by_pitch["KN"]["usage_pct"] == 30.1


@respx.mock
async def test_arsenal_for_a_one_pitch_pitcher() -> None:
    _mock_arsenals()
    out = stats.parse_pitch_arsenal(await savant.pitcher_pitch_arsenal(VAUGHN, 2026))
    assert [(p["pitch"], p["usage_pct"]) for p in out] == [("KN", 100.0)]


@respx.mock
async def test_arsenal_for_a_pitcher_not_on_the_board_is_empty_but_typed() -> None:
    _mock_arsenals()
    df = await savant.pitcher_pitch_arsenal(1, 2026)
    assert df.is_empty()
    assert df.columns == ["pitch_type", "pitch_name", "avg_speed", "avg_spin", "pitch_usage"]
    assert stats.parse_pitch_arsenal(df) == []


@respx.mock
async def test_arsenal_tolerates_empty_boards() -> None:
    respx.get(_ARSENALS_URL).mock(return_value=httpx.Response(200, content=b""))
    df = await savant.pitcher_pitch_arsenal(SKUBAL, 2026)
    assert df.is_empty()


@respx.mock
async def test_arsenal_includes_pitchers_the_run_value_board_leaves_out() -> None:
    """Fried has no row on the min=100 run-value board, but throws seven pitches."""
    _mock_arsenals()
    respx.get(_ARSENAL_STATS_URL).mock(
        return_value=httpx.Response(200, content=_fixture("pitch_arsenal_stats_pitcher.csv"))
    )
    assert (await savant.pitcher_pitch_arsenal_stats(FRIED, 2026)).is_empty()
    assert (await savant.pitcher_pitch_arsenal(FRIED, 2026)).height == 7


@respx.mock
async def test_pitcher_pitch_arsenal_stats_is_the_run_value_board() -> None:
    route = respx.get(_ARSENAL_STATS_URL).mock(
        return_value=httpx.Response(200, content=_fixture("pitch_arsenal_stats_pitcher.csv"))
    )
    df = await savant.pitcher_pitch_arsenal_stats(SKUBAL, 2026)

    params = dict(route.calls.last.request.url.params)
    assert (params["type"], params["year"], params["min"], params["csv"]) == (
        "pitcher",
        "2026",
        "100",
        "true",
    )
    assert df["pitch_type"].to_list() == ["FF", "CH", "SI"]  # only 3 clear min=100
    assert "run_value_per_100" in df.columns
    assert "avg_speed" not in df.columns
    # ...which is why parse_pitch_arsenal can't read it:
    assert stats.parse_pitch_arsenal(df) == []


def test_pitcher_pitch_arsenal_stats_is_exported() -> None:
    import bigleague

    assert bigleague.pitcher_pitch_arsenal_stats is savant.pitcher_pitch_arsenal_stats
    assert "pitcher_pitch_arsenal_stats" in bigleague.__all__
