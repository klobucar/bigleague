"""RAM + time benchmark: bigleague vs pybaseball vs pandas+pyarrow.

Each library runs in an isolated subprocess so RSS isn't polluted across
imports. Workers report tracemalloc peak heap, ru_maxrss, wall time, and
in-memory frame size.

Usage:
    python bench/compare_pybaseball.py
    python bench/compare_pybaseball.py --player 660271 --year 2025

Requires network. Not run in CI. Install missing deps with:
    min add pybaseball pandas pyarrow httpx
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

DEFAULT_PLAYER = 660271  # Ohtani
DEFAULT_YEAR = 2025
SCENARIOS = ["month_cold", "month_warm", "season", "hotzones"]
LIBS = ["bigleague", "pybaseball", "pandas_pyarrow"]


# ── workers ────────────────────────────────────────────────────────────────


def _bigleague_fetch(player: int, start: str, end: str):
    import asyncio
    from datetime import date as _d

    import bigleague as bl

    s = _d.fromisoformat(start)
    e = _d.fromisoformat(end)
    return asyncio.run(bl.statcast_batter(player, s, e))


def worker_bigleague(scenario: str, player: int, year: int) -> dict:
    import bigleague as bl

    if scenario in ("month_cold", "month_warm"):
        df = _bigleague_fetch(player, f"{year}-04-01", f"{year}-04-30")
    elif scenario == "season":
        df = _bigleague_fetch(player, f"{year}-04-01", f"{year}-09-30")
    elif scenario == "hotzones":
        df = _bigleague_fetch(player, f"{year}-04-01", f"{year}-04-30")
        bl.hot_zones_grid(df)
    return {"rows": df.height, "frame_bytes": df.estimated_size()}


def worker_pybaseball(scenario: str, player: int, year: int) -> dict:
    from pybaseball import statcast_batter  # ty: ignore[import-not-found]

    if scenario in ("month_cold", "month_warm"):
        df = statcast_batter(f"{year}-04-01", f"{year}-04-30", player)
    elif scenario == "season":
        df = statcast_batter(f"{year}-04-01", f"{year}-09-30", player)
    elif scenario == "hotzones":
        df = statcast_batter(f"{year}-04-01", f"{year}-04-30", player)
        _pandas_hot_zones(df)
    return {"rows": len(df), "frame_bytes": int(df.memory_usage(deep=True).sum())}


def worker_pandas_pyarrow(scenario: str, player: int, year: int) -> dict:
    import io

    import httpx
    import pandas as pd  # ty: ignore[import-not-found]

    def fetch(start: str, end: str):
        # Same endpoint as bigleague (savant.py:25-26) but un-chunked like
        # pybaseball. `batters_lookup[]` is the param Savant actually honors
        # for a single-batter filter — `player_id` is silently ignored and
        # returns league-wide data capped at 25k rows.
        r = httpx.get(
            "https://baseballsavant.mlb.com/statcast_search/csv",
            params={
                "all": "true",
                "type": "details",
                "batters_lookup[]": str(player),
                "player_type": "batter",
                "game_date_gt": start,
                "game_date_lt": end,
            },
            timeout=120,
        )
        r.raise_for_status()
        return pd.read_csv(
            io.BytesIO(r.content),
            engine="pyarrow",
            dtype_backend="pyarrow",
        )

    if scenario in ("month_cold", "month_warm"):
        df = fetch(f"{year}-04-01", f"{year}-04-30")
    elif scenario == "season":
        df = fetch(f"{year}-04-01", f"{year}-09-30")
    elif scenario == "hotzones":
        df = fetch(f"{year}-04-01", f"{year}-04-30")
        _pandas_hot_zones(df)
    return {"rows": len(df), "frame_bytes": int(df.memory_usage(deep=True).sum())}


def _pandas_hot_zones(df) -> dict[int, float]:
    """Mirror of bigleague.hot_zones_grid for pandas frames."""
    from bigleague.stats import AB_EVENTS, HIT_EVENTS

    out: dict[int, float] = {}
    sub = df[df["zone"].between(1, 9)]
    for z in range(1, 10):
        zd = sub[sub["zone"] == z]
        ab = zd["events"].isin(AB_EVENTS).sum()
        hits = zd["events"].isin(HIT_EVENTS).sum()
        out[z] = float(hits / ab) if ab else 0.0
    return out


# ── driver ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _spawn(lib: str, scenario: str, player: int, year: int, no_cache: bool) -> dict:
    env = os.environ.copy()
    # `python bench/x.py` puts bench/ on sys.path, not the project root,
    # so bigleague's source tree isn't importable without help.
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # Both libs try to mkdir under $HOME at import / first-use. Redirect to /tmp.
    env.setdefault("PYBASEBALL_CACHE", "/tmp/pybaseball_cache_bench")
    env.setdefault("BIGLEAGUE_CACHE_DIR", "/tmp/bigleague_cache_bench")
    if no_cache:
        env["BIGLEAGUE_NO_CACHE"] = "1"
    else:
        env.pop("BIGLEAGUE_NO_CACHE", None)
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            lib,
            "--scenario",
            scenario,
            "--player",
            str(player),
            "--year",
            str(year),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "lib": lib,
            "scenario": scenario,
            "error": (proc.stderr or proc.stdout)[-400:].strip(),
        }
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if s.startswith("{"):
            return json.loads(s)
    return {"lib": lib, "scenario": scenario, "error": "no json on stdout"}


def _fmt_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _rss_bytes(ru_maxrss: int) -> int:
    # macOS reports bytes; Linux reports kilobytes.
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def run_one_worker(args: argparse.Namespace) -> None:
    tracemalloc.start()
    t0 = time.monotonic()
    fn = {
        "bigleague": worker_bigleague,
        "pybaseball": worker_pybaseball,
        "pandas_pyarrow": worker_pandas_pyarrow,
    }[args.worker]
    result = fn(args.scenario, args.player, args.year)
    wall = time.monotonic() - t0
    _, peak_heap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        json.dumps(
            {
                "lib": args.worker,
                "scenario": args.scenario,
                "rows": result["rows"],
                "frame_bytes": result["frame_bytes"],
                "peak_heap_bytes": peak_heap,
                "rss_bytes": _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "wall_s": round(wall, 3),
            }
        )
    )


def run_driver(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        # month_warm requires a pre-run to populate bigleague's cache.
        # We do that pre-run silently for bigleague only; pybaseball and
        # pandas+pyarrow have no cache so the "warm" run is just a re-fetch.
        if scenario == "month_warm":
            _spawn("bigleague", "month_warm", args.player, args.year, no_cache=False)
        for lib in LIBS:
            no_cache = scenario in ("month_cold", "season")
            print(f"  → {lib} / {scenario} ...", file=sys.stderr, flush=True)
            rows.append(_spawn(lib, scenario, args.player, args.year, no_cache))

    cols = ("lib", "scenario", "rows", "frame", "heap", "rss", "wall")
    widths = (16, 14, 8, 10, 10, 10, 9)
    print()
    print(
        "".join(
            c.rjust(w) if i else c.ljust(w)
            for i, (c, w) in enumerate(zip(cols, widths, strict=True))
        )
    )
    print("-" * sum(widths))
    for r in rows:
        if "error" in r:
            print(f"{r['lib']:<16}{r['scenario']:<14}  ERR: {r['error'][:80]}")
            continue
        line = (
            f"{r['lib']:<16}"
            f"{r['scenario']:<14}"
            f"{r['rows']:>8}"
            f"{_fmt_bytes(r['frame_bytes']):>10}"
            f"{_fmt_bytes(r['peak_heap_bytes']):>10}"
            f"{_fmt_bytes(r['rss_bytes']):>10}"
            f"{r['wall_s']:>8.2f}s"
        )
        print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--player", type=int, default=DEFAULT_PLAYER)
    p.add_argument("--year", type=int, default=DEFAULT_YEAR)
    p.add_argument("--worker", choices=LIBS)
    p.add_argument("--scenario", choices=SCENARIOS)
    args = p.parse_args()
    if args.worker:
        run_one_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
