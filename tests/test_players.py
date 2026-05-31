from __future__ import annotations

from collections.abc import Callable
from datetime import date as _date
from pathlib import Path

import polars as pl
import pytest
from bigleague import players


@pytest.fixture
def stub_register(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pl.DataFrame:
    """Stand in a synthetic Chadwick frame so tests don't hit GitHub.

    Points BIGLEAGUE_CACHE_DIR at tmp_path and clears the global no-cache
    shield so `_load_register` reads the pre-written parquet from disk.
    """
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
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
    df.write_parquet(tmp_path / "chadwick_people.parquet")
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


_REGISTER_HEADER = (
    b"name_last,name_first,key_mlbam,key_fangraphs,key_bbref,mlb_played_first,mlb_played_last\n"
)
# One data row, carried by a single shard; the other shards are header-only.
_REGISTER_ROW = b"Judge,Aaron,592450,15640,judgeaa01,2016,2024\n"


def _mock_register_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Patch get_client to serve the sharded register; return a get counter.

    The register is fetched as 16 shards, so one full load == 16 gets. Only
    `people-0.csv` carries the data row; the rest are header-only, so a
    completed load concats to a single player.
    """
    call_count = 0

    class _FakeResp:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return

    class _FakeClient:
        async def get(self, url, *_a, **_kw):
            nonlocal call_count
            call_count += 1
            body = (
                _REGISTER_HEADER + _REGISTER_ROW
                if str(url).endswith("people-0.csv")
                else _REGISTER_HEADER
            )
            return _FakeResp(body)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(players, "get_client", _fake_get_client)
    return lambda: call_count


def _parquet_files(d: Path) -> list[Path]:
    return list(d.glob("*.parquet"))


async def test_register_cache_dir_env_writes_under_tmp_and_reads_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BIGLEAGUE_CACHE_DIR routes the register parquet under tmp_path, not
    ~/.bigleague, and a second load reads it from there without re-fetching."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    calls = _mock_register_http(monkeypatch)
    n_shards = len(players._REGISTER_SHARDS)

    df1 = await players._load_register()
    assert _parquet_files(tmp_path) == [tmp_path / "chadwick_people.parquet"]  # under tmp_path
    assert players._register_path() == tmp_path / "chadwick_people.parquet"

    df2 = await players._load_register()  # hits parquet, no re-download
    assert calls() == n_shards  # one full fetch only
    assert df1.height == df2.height == 1


async def test_register_no_cache_never_writes_and_always_refetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    calls = _mock_register_http(monkeypatch)
    n_shards = len(players._REGISTER_SHARDS)

    await players._load_register()
    await players._load_register()

    assert calls() == 2 * n_shards  # every load refetches all shards
    assert _parquet_files(tmp_path) == []  # nothing persisted


def test_register_path_defaults_to_home_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither env var set, the register lives under ~/.bigleague/cache."""
    monkeypatch.delenv("BIGLEAGUE_CACHE_DIR", raising=False)
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    expected = Path("~/.bigleague/cache/chadwick_people.parquet").expanduser()
    assert players._register_path() == expected
