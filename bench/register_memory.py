"""Resident-memory benchmark for player register lookups.

Measures the RSS of a process that only resolves player ids, in four
scenarios, each in a fresh subprocess so numbers aren't polluted:

    cold    empty cache dir: the first lookup downloads the register
    warm    cache already written by this version: lookups read it from disk
    legacy  cache holds a bigleague 0.1.0 parquet (the full, unpruned
            register): checks old caches keep working
    nocache BIGLEAGUE_NO_CACHE=1: every first lookup downloads, nothing is
            written to disk

RSS is read with `ps` after importing bigleague and after each lookup.

Usage:
    uv run python bench/register_memory.py
    uv run python bench/register_memory.py --lookups 8

Needs network for the `cold`, `legacy` and `nocache` scenarios. Not run in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# Distinct names, so a lookup that cached results by name couldn't hide a
# per-call cost.
PLAYERS = [
    ("Aaron", "Judge"),
    ("Mike", "Trout"),
    ("Shohei", "Ohtani"),
    ("Tarik", "Skubal"),
    ("Gerrit", "Cole"),
    ("Paul", "Skenes"),
    ("Juan", "Soto"),
    ("Mookie", "Betts"),
]


def _rss_mb() -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip()) / 1024


# ── worker ─────────────────────────────────────────────────────────────────


def _worker(lookups: int) -> None:
    import asyncio
    import time

    import bigleague as bl
    import polars as pl  # noqa: F401  (counted in the "imported" baseline)

    points: list[tuple[str, float, float]] = [("imported", _rss_mb(), 0.0)]

    async def run() -> None:
        for first, last in PLAYERS[:lookups]:
            t0 = time.perf_counter()
            pid = await bl.resolve_id(first, last)
            dt = time.perf_counter() - t0
            assert pid is not None, f"{first} {last} not found"
            points.append((f"{first[0]}. {last}", _rss_mb(), dt))

    asyncio.run(run())
    print(json.dumps(points))


# ── driver ─────────────────────────────────────────────────────────────────


def _spawn(cache_dir: Path, lookups: int, *, no_cache: bool) -> list[tuple[str, float, float]]:
    env = os.environ.copy()
    # `python bench/x.py` puts bench/ on sys.path, not the project root.
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["BIGLEAGUE_CACHE_DIR"] = str(cache_dir)
    if no_cache:
        env["BIGLEAGUE_NO_CACHE"] = "1"
    else:
        env.pop("BIGLEAGUE_NO_CACHE", None)
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", "--lookups", str(lookups)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _print(title: str, points: list[tuple[str, float, float]]) -> None:
    print(f"\n{title}")
    print(f"  {'point':<14} {'RSS MB':>8} {'call s':>8}")
    for label, rss, dt in points:
        call = f"{dt:8.2f}" if dt else f"{'':>8}"
        print(f"  {label:<14} {rss:8.0f} {call}")
    peak = max(p[1] for p in points)
    print(f"  {'max':<14} {peak:8.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--lookups", type=int, default=5)
    ap.add_argument(
        "--legacy-parquet",
        type=Path,
        help="a bigleague 0.1.0 chadwick_people.parquet for the `legacy` scenario "
        "(default: build one from the cold run if this checkout still writes that format)",
    )
    args = ap.parse_args()
    if args.worker:
        _worker(args.lookups)
        return

    with tempfile.TemporaryDirectory(prefix="bl-register-bench-") as tmp:
        root = Path(tmp)
        cold_dir = root / "cache"
        _print(
            "cold (empty cache, downloads the register)",
            _spawn(cold_dir, args.lookups, no_cache=False),
        )
        _print(
            "warm (cache written by the cold run)", _spawn(cold_dir, args.lookups, no_cache=False)
        )
        _print("nocache (BIGLEAGUE_NO_CACHE=1)", _spawn(root / "none", args.lookups, no_cache=True))

        legacy = args.legacy_parquet
        if legacy is not None:
            legacy_dir = root / "legacy"
            legacy_dir.mkdir()
            shutil.copy(legacy, legacy_dir / "chadwick_people.parquet")
            _print(
                "legacy (cache written by bigleague 0.1.0)",
                _spawn(legacy_dir, args.lookups, no_cache=False),
            )


if __name__ == "__main__":
    main()
