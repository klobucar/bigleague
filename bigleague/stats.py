"""Pure polars computation. No I/O, no async, no side effects.

All inputs are `pl.DataFrame` (typically from `bigleague.savant`); all
outputs are dicts or scalars. Tests can construct synthetic frames
directly without mocking HTTP.

Event-string sets (`AB_EVENTS`, `HIT_EVENTS`) are module-level so they
aren't rebuilt on every call.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl

# Statcast at-bat-resolving event strings.
AB_EVENTS: frozenset[str] = frozenset(
    {
        "single",
        "double",
        "triple",
        "home_run",
        "field_out",
        "strikeout",
        "strikeout_double_play",
        "double_play",
        "grounded_into_double_play",
        "field_error",
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "other_out",
    }
)

HIT_EVENTS: frozenset[str] = frozenset({"single", "double", "triple", "home_run"})


def season_range(year: int) -> tuple[date, date]:
    """Return (Mar 1, Nov 30) of the given year — full MLB regular season."""
    return date(year, 3, 1), date(year, 11, 30)


# ---------------------------------------------------------------------------
# Internal: scalar formatter shared with fangraphs.aggregate_career.
# ---------------------------------------------------------------------------


def _fmt(val: object) -> str:
    """Format a stat value FanGraphs-style: 3 decimals if < 10, else 1."""
    if val is None:
        return "—"
    if isinstance(val, float):
        if val != val:  # NaN
            return "—"
        return f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
    return str(val)


# ---------------------------------------------------------------------------
# Head-to-head matchup
# ---------------------------------------------------------------------------


def matchup_stats(df: pl.DataFrame, pitcher_id: int) -> dict[str, int | float]:
    """Compute pa/ab/hits/strikeouts/AVG against a specific pitcher.

    `df` must already be batter-side Statcast data (i.e. has a `pitcher`
    column identifying who threw each pitch).
    """
    matchup = df.filter(pl.col("pitcher") == pitcher_id)
    if matchup.is_empty():
        raise ValueError(f"No matchup rows for pitcher_id={pitcher_id}")

    abs_df = matchup.filter(pl.col("events").is_in(AB_EVENTS))
    hits_df = abs_df.filter(pl.col("events").is_in(HIT_EVENTS))
    strikeouts = matchup.filter(
        pl.col("events").is_in(["strikeout", "strikeout_double_play"])
    ).height
    ab = abs_df.height
    h = hits_df.height
    return {
        # The frame is pitch-level; only the final pitch of a PA carries a
        # non-null `events`, so counting rows would count pitches, not PAs.
        "pa": matchup.filter(pl.col("events").is_not_null()).height,
        "ab": ab,
        "hits": h,
        "strikeouts": strikeouts,
        "batting_avg": (h / ab) if ab > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Hot / cold rolling splits
# ---------------------------------------------------------------------------


def hot_cold_batter(df: pl.DataFrame) -> dict[str, int | float]:
    """Compute PA/AB/H/BB and AVG/OBP/SLG/OPS from batter Statcast data."""
    abs_df = df.filter(pl.col("events").is_in(AB_EVENTS))
    hits_df = abs_df.filter(pl.col("events").is_in(HIT_EVENTS))
    bb = df.filter(pl.col("events") == "walk").height
    ab = abs_df.height
    h = hits_df.height
    pa = ab + bb

    tb = (
        abs_df.filter(pl.col("events") == "single").height
        + 2 * abs_df.filter(pl.col("events") == "double").height
        + 3 * abs_df.filter(pl.col("events") == "triple").height
        + 4 * abs_df.filter(pl.col("events") == "home_run").height
    )

    avg = h / ab if ab > 0 else 0.0
    obp = (h + bb) / pa if pa > 0 else 0.0
    slg = tb / ab if ab > 0 else 0.0
    return {
        "pa": pa,
        "ab": ab,
        "h": h,
        "bb": bb,
        "avg": round(avg, 3),
        "obp": round(obp, 3),
        "slg": round(slg, 3),
        "ops": round(obp + slg, 3),
    }


def hot_cold_pitcher(df: pl.DataFrame) -> dict[str, int]:
    """Compute pitches/K/BB/HR-allowed/PA-faced from pitcher Statcast data."""
    pitches = df.height
    k = df.filter(pl.col("events").is_in(["strikeout", "strikeout_double_play"])).height
    bb = df.filter(pl.col("events") == "walk").height
    hr = df.filter(pl.col("events") == "home_run").height
    pa = df.filter(pl.col("events").is_not_null()).height
    return {
        "pitches": pitches,
        "k": k,
        "bb": bb,
        "hr_allowed": hr,
        "pa_faced": pa,
    }


# ---------------------------------------------------------------------------
# Hot zones grid (3x3 Statcast zones 1-9)
# ---------------------------------------------------------------------------


def _zone_ab_hits(df: pl.DataFrame) -> dict[int, tuple[int, int]]:
    """One group_by pass over zones 1-9: `{zone: (ab_count, hit_count)}`.

    Drops null zones and out-of-zone (10-14) pitches first. Always returns
    exactly keys 1..9; zones absent from the data get (0, 0).
    """
    needed = df.select(["zone", "events"]).drop_nulls("zone")
    needed = needed.filter(pl.col("zone").is_between(1, 9))
    agg = needed.group_by("zone").agg(
        pl.col("events").is_in(AB_EVENTS).sum().alias("ab"),
        pl.col("events").is_in(HIT_EVENTS).sum().alias("hits"),
    )
    result: dict[int, tuple[int, int]] = dict.fromkeys(range(1, 10), (0, 0))
    for zone, ab, hits in agg.iter_rows():
        result[int(zone)] = (int(ab), int(hits))
    return result


def hot_zones_grid(df: pl.DataFrame) -> dict[int, float]:
    """Return batting average per Statcast zone for zones 1-9.

    Drops null zones and out-of-zone (10-14) pitches before computing.
    Always returns exactly keys 1..9; zones with zero ABs return 0.0.
    """
    return {z: (h / ab) if ab > 0 else 0.0 for z, (ab, h) in _zone_ab_hits(df).items()}


def hot_zones_ab_counts(df: pl.DataFrame) -> dict[int, int]:
    """Return at-bat counts per Statcast zone for zones 1-9.

    Companion to `hot_zones_grid`: a zone average of 0.0 is ambiguous
    (no at-bats vs. hitless at-bats); the AB counts disambiguate.
    """
    return {z: ab for z, (ab, _h) in _zone_ab_hits(df).items()}


# ---------------------------------------------------------------------------
# Pitch arsenal (from savant.pitcher_pitch_arsenal leaderboard)
# ---------------------------------------------------------------------------


def parse_pitch_arsenal(df: pl.DataFrame) -> list[dict[str, str | float]]:
    """Format a per-pitch arsenal frame as a usage-sorted list.

    Expects `pitch_type`, optionally `pitch_name`, `avg_speed`, `avg_spin`,
    `pitch_usage`, and a `pitches` count column. True usage % is computed
    from raw pitch counts so it sums to 100 even if pybaseball-style
    rounded `pitch_usage` doesn't.
    """
    if df.is_empty():
        return []

    has_pitches = "pitches" in df.columns
    has_speed = "avg_speed" in df.columns
    has_spin = "avg_spin" in df.columns
    has_usage = "pitch_usage" in df.columns

    total_pitches = df["pitches"].cast(pl.Float64).sum() if has_pitches else 0.0

    out: list[dict[str, str | float]] = []
    for row in df.iter_rows(named=True):
        mph = row.get("avg_speed") if has_speed else None
        if mph is None:
            continue
        spin = row.get("avg_spin") if has_spin else 0.0
        if total_pitches and has_pitches and row.get("pitches") is not None:
            usage_pct = (float(row["pitches"]) / float(total_pitches)) * 100.0
        elif has_usage and row.get("pitch_usage") is not None:
            usage_pct = float(row["pitch_usage"])
        else:
            continue
        out.append(
            {
                "pitch": str(row.get("pitch_type") or "").upper(),
                "mph": float(mph),
                "spin": float(spin or 0.0),
                "usage_pct": usage_pct,
            }
        )

    out.sort(key=lambda x: x["usage_pct"], reverse=True)  # type: ignore[arg-type, return-value]
    return out


# ---------------------------------------------------------------------------
# Exit velocity (from savant.batter_exit_velo leaderboard)
# ---------------------------------------------------------------------------


def parse_exit_velo(df: pl.DataFrame, player_id: int) -> dict[str, str]:
    """Format exit-velocity row for one player. Raises if player not present."""
    row = df.filter(pl.col("player_id") == player_id)
    if row.is_empty():
        raise ValueError(f"No exit-velocity data for player_id={player_id}.")
    r = row.row(0, named=True)

    def _f(key: str, fmt: str) -> str:
        v = r.get(key)
        return fmt.format(v) if v is not None else "—"

    # Savant statcast-leaderboard column semantics (easy to mislabel):
    # `ev95percent` is Hard Hit % (EV >= 95 mph); `anglesweetspotpercent`
    # is Sweet Spot % (LA 8-32°); `brl_pa` is barrels-per-PA (a rate),
    # while `barrels` is the raw count.
    return {
        "Avg Exit Velo": _f("avg_hit_speed", "{:.1f} mph"),
        "Max Exit Velo": _f("max_hit_speed", "{:.1f} mph"),
        "Avg Launch Angle": _f("avg_hit_angle", "{:.1f}°"),
        "Barrel %": _f("brl_percent", "{:.1f}%"),
        "Hard Hit %": _f("ev95percent", "{:.1f}%"),
        "Barrels": str(int(r.get("barrels") or 0)),
        "BBE": str(int(r.get("attempts") or r.get("pa") or 0)),
    }


# ---------------------------------------------------------------------------
# Percentile ranks (from savant.{batter,pitcher}_percentile_ranks)
# ---------------------------------------------------------------------------

_PITCHER_PERCENTILE_COLS: dict[str, str] = {
    "hard_hit_percent": "Hard Hit%",
    "k_percent": "K%",
    "bb_percent": "BB%",
    "xwoba": "xwOBA",
    "xera": "xERA",
    "fastball_avg_speed": "FB Velo",
    "spin_rate_percent": "Spin Rate",
}

_BATTER_PERCENTILE_COLS: dict[str, str] = {
    "exit_velocity_avg": "Exit Velo",
    "hard_hit_percent": "Hard Hit%",
    "k_percent": "K%",
    "bb_percent": "BB%",
    "xwoba": "xwOBA",
    "xba": "xBA",
    "xslg": "xSLG",
    "sprint_speed": "Sprint Speed",
}


def parse_percentile_ranks(
    df: pl.DataFrame,
    player_id: int,
    player_type: Literal["batter", "pitcher"],
) -> dict[str, str]:
    """Format selected percentile-rank columns as `{label: 'NN th percentile'}`.

    Tries multiple ID column names because Savant inconsistently names this
    column across leaderboard endpoints.
    """
    cols = _PITCHER_PERCENTILE_COLS if player_type == "pitcher" else _BATTER_PERCENTILE_COLS

    row: pl.DataFrame | None = None
    for id_col in ("player_id", "pitcher_id", "batter_id", "mlbam_id"):
        if id_col in df.columns:
            cand = df.filter(pl.col(id_col) == player_id)
            if not cand.is_empty():
                row = cand
                break
    if row is None or row.is_empty():
        raise ValueError(f"No percentile data for player_id={player_id}.")

    r = row.row(0, named=True)
    out: dict[str, str] = {}
    for raw_col, label in cols.items():
        v = r.get(raw_col)
        if v is not None:
            out[label] = f"{int(v)} th percentile"
    return out


# ---------------------------------------------------------------------------
# Expected stats (xBA, xSLG, xwOBA, etc.)
# ---------------------------------------------------------------------------


def parse_expected_stats(df: pl.DataFrame, player_id: int) -> dict[str, str]:
    """Format expected-stat row for one player. Raises if player not present."""
    row = df.filter(pl.col("player_id") == player_id)
    if row.is_empty():
        raise ValueError(f"No expected-stats data for player_id={player_id}.")
    r = row.row(0, named=True)

    out: dict[str, str] = {}
    label_map = {
        "ba": "BA",
        "est_ba": "xBA",
        "slg": "SLG",
        "est_slg": "xSLG",
        "woba": "wOBA",
        "est_woba": "xwOBA",
        "era": "ERA",
        "est_era": "xERA",
        "fip": "FIP",
        "est_fip": "xFIP",
    }
    for key, label in label_map.items():
        v = r.get(key)
        if v is not None:
            try:
                out[label] = f"{float(v):.3f}"
            except (TypeError, ValueError):
                out[label] = str(v)
    return out


# ---------------------------------------------------------------------------
# Career aggregation (FanGraphs season frames -> totals dict)
# ---------------------------------------------------------------------------


def _ip_display_to_innings(ip: float) -> float:
    """Convert display IP (x.1 = 1/3, x.2 = 2/3 of an inning) to true innings.

    FanGraphs (like every baseball source) renders partial innings as .1/.2,
    so the decimals must not be summed or weighted as-is: 200.1 + 100.2 is
    301.0 innings, not 300.3.
    """
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3.0


def _innings_to_ip_display(innings: float) -> float:
    """Convert true innings back to the .1/.2 display convention."""
    whole, outs = divmod(round(innings * 3), 3)
    return whole + outs / 10.0


def aggregate_career(
    frames: list[pl.DataFrame],
    first: str,
    last: str,
    years: list[int],
    player_type: Literal["pitcher", "batter"],
) -> dict[str, str | dict[str, str]]:
    """Combine per-year FanGraphs frames into a career summary dict.

    IP-weighted averages for pitcher rate stats; PA-weighted for batter
    rate stats; sums for counting stats. Frames must already be filtered
    to the target player and tagged with a `_year` column.
    """
    if not frames:
        raise ValueError(f"No FanGraphs data for {first} {last} in {years[0]}-{years[-1]}.")

    combined = pl.concat(frames, how="vertical_relaxed")
    target = f"{first.lower()} {last.lower()}"
    rows = combined.filter(pl.col("Name").str.to_lowercase() == target)
    if rows.is_empty():
        raise ValueError(f"No FanGraphs data for {first} {last} in {years[0]}-{years[-1]}.")

    seasons = rows["_year"].n_unique()
    totals: dict[str, str] = {"Seasons": str(seasons)}

    # WAR is a cumulative counting stat — career WAR is the sum of season
    # WARs, never an IP/PA-weighted average.
    if player_type == "pitcher":
        for col in ("W", "L", "SV", "G", "GS", "WAR"):
            if col in rows.columns:
                totals[col] = _fmt(rows[col].cast(pl.Float64).sum())
        if "IP" in rows.columns:
            # Display IP uses the .1/.2 = 1/3 / 2/3 convention; convert to
            # true innings before summing/weighting, and back for output.
            ip = (
                rows["IP"]
                .cast(pl.Float64)
                .map_elements(_ip_display_to_innings, return_dtype=pl.Float64)
            )
            ip_sum = float(ip.sum() or 0.0)
            totals["IP"] = _fmt(_innings_to_ip_display(ip_sum))
            for col in ("ERA", "FIP", "WHIP", "K/9", "BB/9"):
                if col in rows.columns:
                    if ip_sum > 0:
                        weighted_sum = float((rows[col].cast(pl.Float64) * ip).sum() or 0.0)
                        totals[col] = _fmt(weighted_sum / ip_sum)
                    else:
                        totals[col] = _fmt(None)  # rates are undefined at 0 IP
    else:
        for col in ("G", "PA", "AB", "H", "HR", "RBI", "SB", "BB", "WAR"):
            if col in rows.columns:
                totals[col] = _fmt(rows[col].cast(pl.Float64).sum())
        if "PA" in rows.columns:
            pa = rows["PA"].cast(pl.Float64)
            pa_sum = float(pa.sum() or 1.0)
            for col in ("AVG", "OBP", "SLG", "OPS", "wRC+"):
                if col in rows.columns:
                    weighted_sum = float((rows[col].cast(pl.Float64) * pa).sum() or 0.0)
                    totals[col] = _fmt(weighted_sum / pa_sum)

    span = f"{years[0]}-{years[-1]}"
    return {"type": player_type, "team": f"Career ({span})", "stats": totals}
