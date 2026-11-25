"""Parquet cache for Statcast chunks, optionally bloom-gated.

With the `bloom` extra installed, a bouquet filter answers "have I ever
cached this key?" in O(1): on a definite miss we skip the
`Path.exists()` call entirely; on a hit (or false positive) we fall
through to the disk check. Filenames seed the bloom lazily on first
real use (never at import) — we never read parquet contents during
seeding, which would make the first call to a warm-cache process
unnecessarily slow.

Without the extra, every `get()` does a `Path.exists()` syscall — fine
for small caches, just no negative-lookup shortcut. Install with
`uv sync --extra bloom`.

`BIGLEAGUE_NO_CACHE=1` disables both reads and writes for tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl

try:
    # Distribution `bouquet-bloom`, import name `bouquet`. AtomicFilter (not
    # ClassicalFilter) because the module-level `cache` singleton is read and
    # written from worker threads (asyncio.to_thread): ClassicalFilter and
    # BlockedFilter are pinned to the thread that created them and panic when
    # another thread touches them. AtomicFilter is thread-safe and has the
    # same insert / `in` surface.
    from bouquet import AtomicFilter  # ty: ignore[import-not-found]
except ImportError:
    AtomicFilter = None  # type: ignore[assignment,misc]

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


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write `df` to `path` so readers only ever see a complete file.

    Writes to a temp file in the same directory, then atomically renames it
    over `path`, so a crash mid-write can never leave a truncated parquet
    that poisons every later read. The temp name is unique per writer: two
    threads (or processes) caching the same key would otherwise share one
    temp path, and one writer's rename could publish the other's half-written
    bytes or fail outright because the file was already renamed away. The
    temp file is removed if the write fails.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class StatcastCache:
    """All filesystem work is lazy: __init__ touches nothing on disk, so
    importing the module (which constructs the global `cache`) is side-effect
    free, and `BIGLEAGUE_NO_CACHE` is honored per-call rather than being
    frozen at construction time."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or resolve_cache_dir()
        self.bloom: AtomicFilter | None = (
            AtomicFilter.from_false_pos(0.01, 10_000) if AtomicFilter else None
        )
        self._bloom_seeded = False

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

    def _ensure_bloom_seeded(self) -> None:
        """Populate the bloom from filenames on disk, once, on first real
        use. No parquet reads — contents never matter for membership."""
        if self.bloom is None or self._bloom_seeded:
            return
        # One batched call (releases the GIL) rather than a call per file.
        self.bloom.insert_many([f.stem for f in self.cache_dir.glob("*.parquet")])
        self._bloom_seeded = True

    def get(self, key: str) -> pl.DataFrame | None:
        if os.environ.get("BIGLEAGUE_NO_CACHE"):
            return None
        path = self._key_path(key)
        self._ensure_bloom_seeded()
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
        # mkdir-on-demand: the dir is created at write time, not import time,
        # so NO_CACHE flipping between import and first write can't strand
        # writes in a never-created directory.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_bloom_seeded()
        # The temp file lives next to the containment-validated final path,
        # so it can't escape cache_dir either.
        atomic_write_parquet(df, path)
        if self.bloom is not None:
            self.bloom.insert(key)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        if self.bloom is not None:
            self.bloom.clear()

    def size(self) -> int:
        """Total bytes used by the on-disk parquet cache."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))


cache = StatcastCache()
