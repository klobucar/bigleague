from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from bigleague.cache import ClassicalFilter, StatcastCache

# bouquet is an optional extra; bloom-specific behaviour can't be exercised
# (or even asserted on) when it isn't installed.
requires_bloom = pytest.mark.skipif(
    ClassicalFilter is None, reason="bouquet not installed (install the `bloom` extra)"
)


@pytest.fixture(autouse=True)
def _enable_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test override of the global no-cache shield."""
    monkeypatch.delenv("BIGLEAGUE_NO_CACHE", raising=False)


def test_no_cache_env_bypasses_disk(tmp_path: Path) -> None:
    os.environ["BIGLEAGUE_NO_CACHE"] = "1"
    try:
        c = StatcastCache(tmp_path)
        c.set("k", pl.DataFrame({"a": [1]}))
        assert c.get("k") is None
        assert not list(tmp_path.glob("*.parquet"))
    finally:
        del os.environ["BIGLEAGUE_NO_CACHE"]


@requires_bloom
def test_definite_miss_skips_disk_check(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    with patch.object(Path, "exists", autospec=True) as mock_exists:
        result = c.get("never_seen_key")
    assert result is None
    mock_exists.assert_not_called()


def test_set_then_get_round_trips(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    c.set("k", df)
    got = c.get("k")
    assert got is not None
    assert got.equals(df)


@requires_bloom
def test_seed_bloom_does_not_read_parquet(tmp_path: Path) -> None:
    # Pre-populate the cache dir
    c1 = StatcastCache(tmp_path)
    c1.set("a", pl.DataFrame({"x": [1]}))
    # New cache instance — should seed bloom from filenames only
    with patch("bigleague.cache.pl.read_parquet") as mock_read:
        c2 = StatcastCache(tmp_path)
    mock_read.assert_not_called()
    assert c2.bloom is not None
    assert b"a" in c2.bloom


def test_clear_removes_files_and_resets_bloom(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    c.set("a", pl.DataFrame({"x": [1]}))
    c.set("b", pl.DataFrame({"x": [2]}))
    c.clear()
    assert not list(tmp_path.glob("*.parquet"))
    if ClassicalFilter is not None:
        assert c.bloom is not None
        assert b"a" not in c.bloom
        assert b"b" not in c.bloom
    else:
        assert c.bloom is None


def test_size_returns_total_parquet_bytes(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    assert c.size() == 0
    c.set("a", pl.DataFrame({"x": [1, 2, 3]}))
    expected = sum(f.stat().st_size for f in tmp_path.glob("*.parquet"))
    assert c.size() == expected
    assert c.size() > 0


@requires_bloom
def test_false_positive_falls_through_to_disk_miss(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    assert c.bloom is not None
    c.bloom.insert(b"ghost")  # bloom says yes but no file exists
    assert c.get("ghost") is None


def test_set_rejects_key_escaping_cache_dir(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    with pytest.raises(ValueError, match="escapes cache dir"):
        c.set("../escape", pl.DataFrame({"x": [1]}))
    assert not (tmp_path.parent / "escape.parquet").exists()


def test_get_rejects_key_escaping_cache_dir(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    with pytest.raises(ValueError, match="escapes cache dir"):
        c.get("../../etc/secret")


def test_normal_keys_still_round_trip_after_containment_check(tmp_path: Path) -> None:
    c = StatcastCache(tmp_path)
    df = pl.DataFrame({"a": [1]})
    c.set("statcast_NYY_2024-04-01_2024-04-07", df)
    got = c.get("statcast_NYY_2024-04-01_2024-04-07")
    assert got is not None
    assert got.equals(df)


def test_cache_works_without_bloom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With bouquet absent, the cache still round-trips — just no negative
    shortcut, so every lookup falls through to a real disk check."""
    # `bigleague.__init__` rebinds the name `cache` to the StatcastCache
    # instance, shadowing the submodule — reach it via sys.modules.
    import sys

    monkeypatch.setattr(sys.modules["bigleague.cache"], "ClassicalFilter", None)
    c = StatcastCache(tmp_path)
    assert c.bloom is None
    df = pl.DataFrame({"a": [1, 2, 3]})
    c.set("k", df)
    got = c.get("k")
    assert got is not None
    assert got.equals(df)
    assert c.get("missing") is None
