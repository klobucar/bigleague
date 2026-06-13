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
    # bouquet is a Rust extension with no type stubs; ty can't see its exports.
    from bouquet import ClassicalFilter  # ty: ignore[unresolved-import]
except ImportError:
    ClassicalFilter = None  # type: ignore[assignment,misc]

_DEFAULT_CACHE_DIR = Path("~/.bigleague/cache").expanduser()


def resolve_cache_dir() -> Path:
    """The directory all of bigleague's on-disk caches live under.

    `BIGLEAGUE_CACHE_DIR` (expanduser) when set, else `~/.bigleague/cache`.
    Resolved at call time so the env var is honored regardless of import
    order, and shared by StatcastCache and the Chadwick register so a single
    env var configures every cache.
    """
    env_dir = os.environ.get("BIGLEAGUE_CACHE_DIR")
    return Path(env_dir).expanduser() if env_dir else _DEFAULT_CACHE_DIR


class StatcastCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or resolve_cache_dir()
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

    def _key_path(self, key: str) -> Path:
        """Resolve a cache key to its parquet path, refusing escapes.

        Keys come from upstream-validated inputs, but defense in depth:
        a key containing path separators (or `..`) must never resolve
        outside cache_dir.
        """
        path = self.cache_dir / f"{key}.parquet"
        if not path.resolve().is_relative_to(self.cache_dir.resolve()):
            msg = f"cache key escapes cache dir: {key!r}"
            raise ValueError(msg)
        return path

    def _seed_bloom(self) -> None:
        """Populate the bloom from filenames on disk. No parquet reads."""
        assert self.bloom is not None
        for f in self.cache_dir.glob("*.parquet"):
            self.bloom.insert(f.stem)

    def get(self, key: str) -> pl.DataFrame | None:
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return None
        path = self._key_path(key)
        if self.bloom is not None and key not in self.bloom:
            return None
        if not path.exists():
            return None
        try:
            return pl.read_parquet(path)
        except (pl.exceptions.PolarsError, OSError):
            # Corrupt/truncated parquet (e.g. a crash mid-write before the
            # atomic-rename pattern existed): drop it and treat as a miss
            # so the cache self-heals instead of raising forever.
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, df: pl.DataFrame) -> None:
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return
        path = self._key_path(key)
        # Write to a temp file in the same directory, then atomically rename
        # over the final path, so a crash mid-write can never leave a
        # truncated parquet that poisons every later get(). The temp path is
        # derived from the containment-validated final path, so it can't
        # escape cache_dir either.
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(path)
        if self.bloom is not None:
            self.bloom.insert(key)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        if self.bloom is not None and ClassicalFilter is not None:
            self.bloom = ClassicalFilter.from_false_pos(0.01, 10_000)

    def size(self) -> int:
        """Total bytes used by the on-disk parquet cache."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))


cache = StatcastCache()
