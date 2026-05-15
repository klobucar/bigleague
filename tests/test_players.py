from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import polars as pl
import pytest
from bigleague import players


@pytest.fixture
def stub_register(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pl.DataFrame:
    """Stand in a synthetic Chadwick frame so tests don't hit GitHub."""
    df = pl.DataFrame(
        {
            "name_last": ["Judge", "Trout", "Trout", "Smith"],
            "name_first": ["Aaron", "Mike", "Michael", "Will"],
            "key_mlbam": [592450, 545361, 545361, None],
            "key_fangraphs": [15640, 10155, 10155, None],
            "key_bbref": ["judgeaa01", "troutmi01", "troutmi01", None],
            "mlb_played_first": [2016, 2011, 2011, None],
            "mlb_played_last": [2024, 2024, 2024, None],
        },
        schema_overrides={
            "key_mlbam": pl.Int64,
            "key_fangraphs": pl.Int64,
            "mlb_played_first": pl.Int64,
            "mlb_played_last": pl.Int64,
        },
    )
    cache_path = tmp_path / "chadwick_people.parquet"
    df.write_parquet(cache_path)
    monkeypatch.setattr(players, "_REGISTER_PATH", cache_path)
    return df


async def test_lookup_case_insensitive(stub_register) -> None:
    res = await players.lookup("judge", "aaron")
    assert res.height == 1
    assert res["key_mlbam"][0] == 592450


async def test_lookup_fuzzy_prefix_match(stub_register) -> None:
    # "Mi" prefix matches both Mike and Michael
    res = await players.lookup("Trout", "Mi", fuzzy=True)
    assert res.height == 2


async def test_lookup_exact_first_wins_over_fuzzy(stub_register) -> None:
    res = await players.lookup("Trout", "Mike", fuzzy=True)
    assert res.height == 1
    assert res["name_first"][0] == "Mike"


async def test_resolve_id_returns_mlbam(stub_register) -> None:
    rid = await players.resolve_id("Aaron", "Judge")
    assert rid == 592450


async def test_resolve_id_returns_none_when_no_mlbam(stub_register) -> None:
    rid = await players.resolve_id("Will", "Smith")  # has null key_mlbam
    assert rid is None


async def test_reverse_lookup_filters_by_id(stub_register) -> None:
    res = await players.reverse_lookup([592450])
    assert res.height == 1
    assert res["name_last"][0] == "Judge"


async def test_mlb_years_extends_for_active_players(
    stub_register, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend "today" is 2026 — Trout's last_active should jump from 2024 to 2026
    class _FakeDate:
        @staticmethod
        def today() -> _date:
            return _date(2026, 5, 1)

    monkeypatch.setattr(players, "_dt", type("M", (), {"date": _FakeDate}))

    debut, last = await players.mlb_years("Mike", "Trout")
    assert debut == 2011
    assert last == 2026


async def test_mlb_years_falls_back_when_register_missing(
    stub_register, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeDate:
        @staticmethod
        def today() -> _date:
            return _date(2026, 5, 1)

    monkeypatch.setattr(players, "_dt", type("M", (), {"date": _FakeDate}))

    debut, last = await players.mlb_years("Unknown", "Player")
    assert debut == 2023  # cy - 3
    assert last == 2026


async def test_register_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """First call should hit HTTP, write parquet, and not re-download next time."""
    monkeypatch.setattr(players, "_REGISTER_PATH", tmp_path / "register.parquet")

    csv = b"name_last,name_first,key_mlbam,key_fangraphs,key_bbref,mlb_played_first,mlb_played_last\nJudge,Aaron,592450,15640,judgeaa01,2016,2024\n"
    call_count = 0

    class _FakeResp:
        content = csv

        def raise_for_status(self) -> None:
            return

    class _FakeClient:
        async def get(self, *_a, **_kw):
            nonlocal call_count
            call_count += 1
            return _FakeResp()

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(players, "get_client", _fake_get_client)

    # Force re-read by clearing the polars cache import
    df1 = await players._load_register()
    df2 = await players._load_register()  # should hit parquet, not re-download
    assert call_count == 1
    assert df1.height == df2.height == 1
