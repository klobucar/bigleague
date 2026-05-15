"""Parquet cache for Statcast chunks, optionally bloom-gated.

With the `bloom` extra installed, a bouquet filter answers "have I ever
cached this key?" in O(1): on a definite miss we skip the
`Path.exists()` call entirely; on a hit (or false positive) we fall
through to the disk check. Filenames seed the bloom on init — we never
read parquet contents during seeding, which would make the first call
to a warm-cache process unnecessarily slow.

Without the extra, every `get()` does a `Path.exists()` syscall — fine
for small caches, just no negative-lookup shortcut. Install with
`uv sync --extra bloom`.

`BIGLEAGUE_NO_CACHE=1` disables both reads and writes for tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

try:
    from bouquet import ClassicalFilter
except ImportError:
    ClassicalFilter = None  # type: ignore[assignment,misc]

_DEFAULT_CACHE_DIR = Path("~/.bigleague/cache").expanduser()


class StatcastCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        env_dir = os.environ.get("BIGLEAGUE_CACHE_DIR")
        self.cache_dir = (
            cache_dir
            or (Path(env_dir).expanduser() if env_dir else None)
            or _DEFAULT_CACHE_DIR
        )
        self.bloom: ClassicalFilter | None = (
            ClassicalFilter.from_false_pos(0.01, 10_000) if ClassicalFilter else None
        )
        # Skip filesystem setup entirely when caching is disabled — the
        # default cache_dir may not be writable in some environments.
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.bloom is not None:
            self._seed_bloom()

    def _seed_bloom(self) -> None:
        """Populate the bloom from filenames on disk. No parquet reads."""
        assert self.bloom is not None
        for f in self.cache_dir.glob("*.parquet"):
            self.bloom.insert(f.stem)

    def get(self, key: str) -> pl.DataFrame | None:
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return None
        if self.bloom is not None and key not in self.bloom:
            return None
        path = self.cache_dir / f"{key}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def set(self, key: str, df: pl.DataFrame) -> None:
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return
        path = self.cache_dir / f"{key}.parquet"
        df.write_parquet(path)
        if self.bloom is not None:
            self.bloom.insert(key)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        if self.bloom is not None:
            self.bloom = ClassicalFilter.from_false_pos(0.01, 10_000)

    def size(self) -> int:
        """Total bytes used by the on-disk parquet cache."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))


cache = StatcastCache()
