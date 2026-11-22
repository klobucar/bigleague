from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import date as _date
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from bigleague import players


@pytest.fixture
def stub_register(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pl.DataFrame:
    """Stand in a synthetic (already pruned) register so tests don't hit GitHub.

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
            "key_fangraphs": [15640, 10155, 10155, 7777],
            "key_bbref": ["judgeaa01", "troutmi01", "troutmi01", "smithwi99"],
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
    df.write_parquet(tmp_path / players._REGISTER_FILENAME)
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
    rid = await players.resolve_id("Will", "Smith")  # FanGraphs/bbref ids, null key_mlbam
    assert rid is None


async def test_lookup_returns_rows_without_mlbam(stub_register) -> None:
    """Rows that only carry FanGraphs/bbref ids must stay visible to `lookup`."""
    res = await players.lookup("Smith", "Will")
    assert res.height == 1
    assert res["key_bbref"][0] == "smithwi99"


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


# ── fetching the register ────────────────────────────────────────────────────

_REGISTER_HEADER = (
    b"name_last,name_first,key_mlbam,key_fangraphs,key_bbref,mlb_played_first,mlb_played_last\n"
)
# One data row, carried by a single shard; the other shards are header-only.
_REGISTER_ROW = b"Judge,Aaron,592450,15640,judgeaa01,2016,2024\n"


class _Counter:
    """How many shard GETs the fake client has served."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> int:
        return self.n


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    body_for: Callable[[str], bytes],
    *,
    piece: int = 7,
    fail_first: int = 0,
) -> _Counter:
    """Patch `get_client` to stream the register shards from `body_for`.

    Responses are delivered `piece` bytes at a time, so the fetch has to
    reassemble records that straddle chunk boundaries. The first `fail_first`
    requests raise, to exercise retry after a failed load.
    """
    counter = _Counter()

    class _FakeResp:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return

        async def aiter_bytes(self) -> AsyncIterator[bytes]:
            for i in range(0, len(self.content), piece):
                yield self.content[i : i + piece]

    class _Stream:
        def __init__(self, resp: _FakeResp) -> None:
            self._resp = resp

        async def __aenter__(self) -> _FakeResp:
            return self._resp

        async def __aexit__(self, *_exc: object) -> None:
            return None

    class _FakeClient:
        def stream(self, _method: str, url: object, *_a: object, **_kw: object) -> _Stream:
            counter.n += 1
            if counter.n <= fail_first:
                raise RuntimeError("boom")
            return _Stream(_FakeResp(body_for(str(url).rsplit("/", 1)[-1])))

    async def _fake_get_client() -> _FakeClient:
        return _FakeClient()

    monkeypatch.setattr(players, "get_client", _fake_get_client)
    return counter


def _one_player(shard: str) -> bytes:
    """`people-0.csv` carries one player; every other shard is header-only."""
    return _REGISTER_HEADER + _REGISTER_ROW if shard == "people-0.csv" else _REGISTER_HEADER


def _parquet_files(d: Path) -> list[Path]:
    return sorted(d.glob("*.parquet"))


async def test_register_cache_dir_env_writes_under_tmp_and_reads_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BIGLEAGUE_CACHE_DIR routes the register parquet under tmp_path, not
    ~/.bigleague, and a later process reads it from there without re-fetching."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    calls = _serve(monkeypatch, _one_player)
    n_shards = len(players._REGISTER_SHARDS)

    df1 = await players._load_register()
    assert _parquet_files(tmp_path) == [tmp_path / "chadwick_people.v2.parquet"]  # under tmp_path
    assert players._register_path() == tmp_path / "chadwick_people.v2.parquet"
    assert calls() == n_shards

    players._forget_register()  # a "new process"
    df2 = await players._load_register()  # hits parquet, no re-download
    assert calls() == n_shards  # one full fetch only
    assert df1.height == df2.height == 1


async def test_register_no_cache_never_writes_and_fetches_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NO_CACHE: nothing is read from or written to disk. The register is
    still held in memory, so a second lookup in the same process doesn't
    download it all again (the old behavior grew RSS by ~70 MB per call)."""
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    calls = _serve(monkeypatch, _one_player)
    n_shards = len(players._REGISTER_SHARDS)

    await players._load_register()
    await players._load_register()

    assert calls() == n_shards  # fetched once, then served from memory
    assert _parquet_files(tmp_path) == []  # nothing persisted

    players._forget_register()  # a "new process" fetches again; still no disk
    await players._load_register()
    assert calls() == 2 * n_shards
    assert _parquet_files(tmp_path) == []


async def test_register_no_cache_ignores_existing_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With NO_CACHE a parquet already on disk is not read."""
    pl.DataFrame({"name_last": ["Stale"]}).write_parquet(tmp_path / players._REGISTER_FILENAME)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    calls = _serve(monkeypatch, _one_player)

    df = await players._load_register()
    assert df["name_last"].to_list() == ["Judge"]
    assert calls() == len(players._REGISTER_SHARDS)


def test_register_path_defaults_to_home_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither env var set, the register lives under ~/.bigleague/cache."""
    monkeypatch.delenv("BIGLEAGUE_CACHE_DIR", raising=False)
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    expected = Path("~/.bigleague/cache/chadwick_people.v2.parquet").expanduser()
    assert players._register_path() == expected


# ── loaded once per process ──────────────────────────────────────────────────


async def test_register_is_loaded_once_and_reused(
    stub_register, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every lookup after the first reuses the in-memory frame: no parquet
    read per call (the old code re-read all 516k rows every time)."""
    reads = 0
    real_read = pl.read_parquet

    def _counting_read(*a: Any, **kw: Any) -> pl.DataFrame:
        nonlocal reads
        reads += 1
        return real_read(*a, **kw)

    monkeypatch.setattr(players.pl, "read_parquet", _counting_read)

    await players.resolve_id("Aaron", "Judge")
    await players.lookup("Trout")
    await players.reverse_lookup([592450])
    await players.mlb_years("Mike", "Trout")

    assert reads == 1
    first = await players._load_register()
    assert await players._load_register() is first  # same object, not a copy


async def test_concurrent_first_calls_share_one_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    calls = _serve(monkeypatch, _one_player)

    ids = await asyncio.gather(*(players.resolve_id("Aaron", "Judge") for _ in range(25)))

    assert ids == [592450] * 25
    assert calls() == len(players._REGISTER_SHARDS)  # one load, not 25


async def test_failed_load_is_retried_by_the_next_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    _serve(monkeypatch, _one_player, fail_first=1)

    with pytest.raises(RuntimeError, match="boom"):
        await players.resolve_id("Aaron", "Judge")
    assert _parquet_files(tmp_path) == []  # nothing half-written

    assert await players.resolve_id("Aaron", "Judge") == 592450  # recovered


async def test_one_cancelled_caller_does_not_abort_the_shared_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    calls = _serve(monkeypatch, _one_player)

    doomed = asyncio.create_task(players.resolve_id("Aaron", "Judge"))
    survivor = asyncio.create_task(players.resolve_id("Aaron", "Judge"))
    await asyncio.sleep(0)  # let both start waiting on the load
    doomed.cancel()

    assert await survivor == 592450
    assert calls() == len(players._REGISTER_SHARDS)


def test_memo_survives_across_event_loops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A bot calling asyncio.run() repeatedly must not reload per run."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    calls = _serve(monkeypatch, _one_player)

    assert asyncio.run(players.resolve_id("Aaron", "Judge")) == 592450
    assert asyncio.run(players.resolve_id("Aaron", "Judge")) == 592450
    assert calls() == len(players._REGISTER_SHARDS)


# ── what is kept ─────────────────────────────────────────────────────────────

_PRUNE_CSV = (
    _REGISTER_HEADER
    + b"Mlbam,Only,111,,,,\n"  # key_mlbam only
    + b"Fg,Only,,222,,,\n"  # key_fangraphs only
    + b"Bbref,Only,,,bbrefon01,,\n"  # key_bbref only
    + b"Years,Only,,,,1932,1935\n"  # MLB years only
    + b"Bare,Name,,,,,\n"  # nothing usable: dropped
    + b",Nobody,,,,,\n"  # no last name, no ids: dropped
)


async def test_pruning_keeps_any_id_or_mlb_year_and_drops_bare_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    _serve(monkeypatch, lambda s: _PRUNE_CSV if s == "people-0.csv" else _REGISTER_HEADER)

    df = await players._load_register()
    assert df["name_last"].to_list() == ["Mlbam", "Fg", "Bbref", "Years"]
    assert (await players.lookup("Bare")).is_empty()
    assert (await players.lookup("Fg")).height == 1  # reachable via lookup()
    assert await players.resolve_id("Only", "Fg") is None  # but has no MLBAM id
    assert await players.resolve_id("Only", "Mlbam") == 111


async def test_register_keeps_stable_dtypes_even_for_header_only_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    _serve(monkeypatch, _one_player)

    df = await players._load_register()
    assert df.schema == pl.Schema(
        {
            "name_last": pl.String,
            "name_first": pl.String,
            "key_mlbam": pl.Int64,
            "key_fangraphs": pl.Int64,
            "key_bbref": pl.String,
            "mlb_played_first": pl.Int64,
            "mlb_played_last": pl.Int64,
        }
    )


async def test_shard_order_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows keep the order of the full register, so which namesake `resolve_id`
    and `mlb_years` pick first doesn't change."""
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    _serve(
        monkeypatch,
        lambda s: _REGISTER_HEADER + f"Same,{s[7]},{ord(s[7])},,,,\n".encode(),
    )
    df = await players._load_register()
    assert df["name_first"].to_list() == list("0123456789abcdef")


# ── streamed, block-wise parsing ─────────────────────────────────────────────

# Quoted fields with commas, doubled quotes, and an embedded newline: a block
# cut must never land inside one of them.
_TRICKY = (
    _REGISTER_HEADER
    + b'"Smith, Jr.",Will,1,10,smitwi01,2000,2001\n'
    + b'"O""Neil",Pat,2,20,oneipa01,1990,1991\n'
    + b'"Multi\nLine",Ann,3,30,multan01,1980,1981\n'
    + b"Plain,Bob,4,40,plaibo01,1970,1971\n"
    + b"Tail,Cy,5,50,tailcy01,1960,1961"  # no trailing newline
)


@pytest.mark.parametrize("block", [1, 16, 40, 10_000])
async def test_block_parsing_matches_whole_shard_parsing(
    monkeypatch: pytest.MonkeyPatch, block: int
) -> None:
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    monkeypatch.setattr(players, "_BLOCK_BYTES", block)
    _serve(monkeypatch, lambda s: _TRICKY if s == "people-0.csv" else _REGISTER_HEADER, piece=5)

    df = await players._load_register()
    expected = players._parse_block(_REGISTER_HEADER, _TRICKY.split(b"\n", 1)[1])
    assert df.equals(expected)
    assert df["name_last"].to_list() == ["Smith, Jr.", 'O"Neil', "Multi\nLine", "Plain", "Tail"]


def test_cut_point_never_lands_inside_a_quoted_field() -> None:
    buf = b'a,1\n"two\nlines",2\nb,3\n'
    cut = players._cut_point(buf)
    assert buf[:cut] == buf  # whole thing: every newline ends a record here
    inside = b'a,1\n"two\nlines'
    assert players._cut_point(inside) == len(b"a,1\n")  # the embedded newline is skipped
    assert players._cut_point(b"no newline yet") == 0


async def test_empty_shard_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIGLEAGUE_NO_CACHE", "1")
    _serve(monkeypatch, lambda s: b"")
    with pytest.raises(ValueError, match="empty"):
        await players._load_register()


# ── bigleague 0.1.0 caches and corrupt files ─────────────────────────────────


def _legacy_register() -> pl.DataFrame:
    """What 0.1.0 cached: the FULL register, bare-name rows and all."""
    return pl.DataFrame(
        {
            "name_last": ["Judge", "Bare", "Trout", "Fgonly"],
            "name_first": ["Aaron", "Name", "Mike", "Pat"],
            "key_mlbam": [592450, None, 545361, None],
            "key_fangraphs": [15640, None, 10155, 4242],
            "key_bbref": ["judgeaa01", None, "troutmi01", None],
            "mlb_played_first": [2016, None, 2011, None],
            "mlb_played_last": [2024, None, 2024, None],
        },
        schema_overrides={
            "key_mlbam": pl.Int64,
            "key_fangraphs": pl.Int64,
            "mlb_played_first": pl.Int64,
            "mlb_played_last": pl.Int64,
        },
    )


async def test_cache_written_by_0_1_0_is_pruned_and_rewritten_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    legacy = tmp_path / "chadwick_people.parquet"
    _legacy_register().write_parquet(legacy)
    calls = _serve(monkeypatch, _one_player)

    df = await players._load_register()

    assert calls() == 0  # no network: derived from the old cache
    assert df["name_last"].to_list() == ["Judge", "Trout", "Fgonly"]  # bare name dropped
    assert (tmp_path / "chadwick_people.v2.parquet").exists()  # rewritten in the new format
    assert legacy.exists()  # the old file is left alone (a 0.1.0 install may share the dir)

    players._forget_register()  # next process: reads the new file, not the old one
    assert (await players._load_register()).equals(df)
    assert calls() == 0


async def test_legacy_cache_in_read_only_dir_still_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    _legacy_register().write_parquet(tmp_path / "chadwick_people.parquet")

    def _no_write(*_a: object, **_kw: object) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(players, "_write_register_sync", _no_write)
    df = await players._load_register()
    assert df.height == 3


async def test_corrupt_cache_is_discarded_and_refetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)
    monkeypatch.setenv("BIGLEAGUE_CACHE_DIR", str(tmp_path))
    bad = tmp_path / players._REGISTER_FILENAME
    bad.write_bytes(b"not a parquet file")
    calls = _serve(monkeypatch, _one_player)

    df = await players._load_register()

    assert df.height == 1
    assert calls() == len(players._REGISTER_SHARDS)
    assert pl.read_parquet(bad).height == 1  # replaced with a good file
