from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from bigleague import stats


def _make_batter_df(events: list[str | None], pitchers: list[int] | None = None) -> pl.DataFrame:
    n = len(events)
    return pl.DataFrame(
        {
            "events": events,
            "pitcher": pitchers if pitchers is not None else [1] * n,
        }
    )


def test_season_range_returns_mar1_nov30() -> None:
    start, end = stats.season_range(2024)
    assert start == date(2024, 3, 1)
    assert end == date(2024, 11, 30)


def test_ab_events_contain_hit_events() -> None:
    assert stats.HIT_EVENTS <= stats.AB_EVENTS


def test_matchup_stats_filters_by_pitcher() -> None:
    df = _make_batter_df(
        ["single", "field_out", "home_run", "strikeout"],
        pitchers=[10, 10, 20, 10],
    )
    out = stats.matchup_stats(df, 10)
    assert out["pa"] == 3
    assert out["ab"] == 3
    assert out["hits"] == 1
    assert out["strikeouts"] == 1
    assert out["batting_avg"] == pytest.approx(1 / 3)


def test_matchup_stats_pa_counts_resolved_events_not_pitches() -> None:
    """Regression: PA was `matchup.height`, which counts pitches — the frame
    is pitch-level and only the final pitch of a PA has a non-null `events`."""
    df = _make_batter_df(
        [None, None, "single", None, "strikeout"],
        pitchers=[10] * 5,
    )
    out = stats.matchup_stats(df, 10)
    assert out["pa"] == 2  # two resolved PAs, not five pitches
    assert out["ab"] == 2
    assert out["hits"] == 1


def test_matchup_stats_raises_for_missing_pitcher() -> None:
    df = _make_batter_df(["single"], [10])
    with pytest.raises(ValueError, match="No matchup rows"):
        stats.matchup_stats(df, 99)


def test_hot_cold_batter_basic() -> None:
    df = _make_batter_df(["single", "double", "home_run", "field_out", "strikeout", "walk", "walk"])
    out = stats.hot_cold_batter(df)
    # 5 ABs (single, double, hr, out, k), 3 H, 2 BB, PA=7
    assert out["ab"] == 5
    assert out["h"] == 3
    assert out["bb"] == 2
    assert out["pa"] == 7
    # AVG = 3/5 = .600, OBP = 5/7 ≈ .714, SLG = (1+2+4)/5 = 1.400
    assert out["avg"] == pytest.approx(0.600)
    assert out["slg"] == pytest.approx(1.400)


def test_hot_cold_pitcher_counts_events() -> None:
    df = _make_batter_df(["strikeout", "strikeout_double_play", "walk", "home_run", "single", None])
    out = stats.hot_cold_pitcher(df)
    assert out["pitches"] == 6
    assert out["k"] == 2
    assert out["bb"] == 1
    assert out["hr_allowed"] == 1
    assert out["pa_faced"] == 5  # 5 non-null events


def test_hot_zones_grid_returns_keys_1_through_9() -> None:
    df = pl.DataFrame(
        {
            "zone": [1, 1, 5, 5, 5, 9, None, 12],
            "events": [
                "single",
                "field_out",
                "home_run",
                "double",
                "field_out",
                "strikeout",
                "single",
                "single",
            ],
        }
    )
    grid = stats.hot_zones_grid(df)
    assert set(grid.keys()) == set(range(1, 10))
    # Zone 1: 1 hit / 2 abs = .500
    assert grid[1] == pytest.approx(0.5)
    # Zone 5: 2 hits / 3 abs ≈ .667
    assert grid[5] == pytest.approx(2 / 3)
    # Zone 9: 0 hits / 1 ab = 0.0
    assert grid[9] == 0.0
    # Zones with no data
    assert grid[2] == 0.0


def test_hot_zones_ab_counts_matches_grid_zones() -> None:
    df = pl.DataFrame(
        {
            "zone": [1, 1, 5, 9],
            "events": ["single", "field_out", "ball", "walk"],
        }
    )
    counts = stats.hot_zones_ab_counts(df)
    assert set(counts.keys()) == set(range(1, 10))
    assert counts[1] == 2
    # Pitches without an AB-resolving event don't count as at-bats.
    assert counts[5] == 0
    assert counts[9] == 0


def test_parse_pitch_arsenal_sorts_by_usage() -> None:
    df = pl.DataFrame(
        {
            "pitch_type": ["FF", "SL", "CH"],
            "avg_speed": [95.0, 84.0, 86.0],
            "avg_spin": [2400, 2600, 1800],
            "pitches": [600, 300, 100],
        }
    )
    out = stats.parse_pitch_arsenal(df)
    assert [p["pitch"] for p in out] == ["FF", "SL", "CH"]
    usages = [float(p["usage_pct"]) for p in out]  # type: ignore[arg-type]
    assert usages[0] > usages[1] > usages[2]
    assert out[0]["mph"] == 95.0


def test_parse_exit_velo_filters_by_player() -> None:
    df = pl.DataFrame(
        {
            "player_id": [1, 2],
            "avg_hit_speed": [92.5, 88.0],
            "max_hit_speed": [115.0, 110.0],
            "avg_hit_angle": [12.5, 10.0],
            "brl_percent": [10.0, 5.0],
            "ev95percent": [48.0, 38.0],  # Hard Hit % (EV >= 95 mph)
            "anglesweetspotpercent": [40.0, 30.0],  # Sweet Spot %, NOT Hard Hit %
            "brl_pa": [7.5, 3.2],  # barrels per PA — a rate, not a count
            "barrels": [25, 12],
            "attempts": [400, 300],
        }
    )
    out = stats.parse_exit_velo(df, 1)
    assert "92.5" in out["Avg Exit Velo"]
    assert out["BBE"] == "400"
    # Regression: Hard Hit % must come from ev95percent (anglesweetspotpercent
    # is Sweet Spot %), and Barrels from the raw `barrels` count (brl_pa is a
    # rate per PA).
    assert out["Hard Hit %"] == "48.0%"
    assert out["Barrels"] == "25"
    assert out["Barrel %"] == "10.0%"


def test_parse_percentile_ranks_pitcher() -> None:
    df = pl.DataFrame(
        {
            "player_id": [1],
            "k_percent": [78],
            "bb_percent": [55],
            "xwoba": [82],
        }
    )
    out = stats.parse_percentile_ranks(df, 1, "pitcher")
    assert out["K%"] == "78 th percentile"
    assert out["xwOBA"] == "82 th percentile"


def test_aggregate_career_pitcher_ip_weights_rate_stats() -> None:
    f1 = pl.DataFrame(
        {
            "Name": ["Test Player"],
            "_year": [2020],
            "IP": [200.0],
            "ERA": [3.00],
            "W": [15],
        }
    )
    f2 = pl.DataFrame(
        {
            "Name": ["Test Player"],
            "_year": [2021],
            "IP": [100.0],
            "ERA": [6.00],
            "W": [5],
        }
    )
    out = stats.aggregate_career([f1, f2], "Test", "Player", [2020, 2021], "pitcher")
    assert out["type"] == "pitcher"
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    # IP-weighted ERA = (3*200 + 6*100) / 300 = 4.00
    assert stats_block["ERA"] == "4.000"
    # Sum of W = 20
    assert stats_block["W"] == "20.0"


def test_aggregate_career_zero_ip_reports_zero_not_one() -> None:
    """Regression: `ip.sum() or 1.0` fabricated 1.0 IP for a 0-IP career."""
    f1 = pl.DataFrame({"Name": ["Test Player"], "_year": [2020], "IP": [0.0], "ERA": [0.0]})
    out = stats.aggregate_career([f1], "Test", "Player", [2020], "pitcher")
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    assert stats_block["IP"] == "0.000"
    # Rate stats are undefined over 0 IP — never a fabricated 0.000 ERA.
    assert stats_block["ERA"] == "—"


def test_aggregate_career_sums_ip_in_thirds() -> None:
    """Display IP uses .1/.2 for 1/3 and 2/3 innings: 200.1 + 100.2 is
    301.0 innings, not 300.3."""
    f1 = pl.DataFrame({"Name": ["Test Player"], "_year": [2020], "IP": [200.1], "ERA": [3.00]})
    f2 = pl.DataFrame({"Name": ["Test Player"], "_year": [2021], "IP": [100.2], "ERA": [6.00]})
    out = stats.aggregate_career([f1, f2], "Test", "Player", [2020, 2021], "pitcher")
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    assert stats_block["IP"] == "301.0"
    # ERA weights by true innings: (3*200⅓ + 6*100⅔) / 301 ≈ 4.003
    assert stats_block["ERA"] == "4.003"


def test_ip_display_helpers_round_trip() -> None:
    assert stats._ip_display_to_innings(200.1) == pytest.approx(200 + 1 / 3)
    assert stats._ip_display_to_innings(100.2) == pytest.approx(100 + 2 / 3)
    assert stats._innings_to_ip_display(301.0) == 301.0
    assert stats._innings_to_ip_display(10 + 2 / 3) == 10.2


def test_aggregate_career_sums_pitcher_war() -> None:
    """Regression: WAR was IP-weight-averaged; it's cumulative and must sum."""
    f1 = pl.DataFrame({"Name": ["Test Player"], "_year": [2020], "IP": [200.0], "WAR": [6.0]})
    f2 = pl.DataFrame({"Name": ["Test Player"], "_year": [2021], "IP": [100.0], "WAR": [1.5]})
    out = stats.aggregate_career([f1, f2], "Test", "Player", [2020, 2021], "pitcher")
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    # 6.0 + 1.5 = 7.5 — not the IP-weighted average (4.5)
    assert stats_block["WAR"] == "7.500"


def test_aggregate_career_sums_batter_war_but_weights_rates() -> None:
    f1 = pl.DataFrame(
        {"Name": ["Test Player"], "_year": [2020], "PA": [600.0], "WAR": [5.0], "AVG": [0.300]}
    )
    f2 = pl.DataFrame(
        {"Name": ["Test Player"], "_year": [2021], "PA": [200.0], "WAR": [1.0], "AVG": [0.200]}
    )
    out = stats.aggregate_career([f1, f2], "Test", "Player", [2020, 2021], "batter")
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    # WAR sums: 5 + 1 = 6 — not the PA-weighted average (4.0)
    assert stats_block["WAR"] == "6.000"
    # Rate stats stay PA-weighted: (0.3*600 + 0.2*200) / 800 = 0.275
    assert stats_block["AVG"] == "0.275"


# ---------------------------------------------------------------------------
# Single-pass counting: must match a naive one-filter-per-stat reference.
# ---------------------------------------------------------------------------

_ALL_EVENTS: list[str | None] = [
    *sorted(stats.AB_EVENTS),
    "walk",
    "hit_by_pitch",
    "sac_fly",
    None,
    None,
    None,
]


def _random_events_df(seed: int, n: int = 2_000) -> pl.DataFrame:
    import random

    rng = random.Random(seed)
    return _make_batter_df(
        [rng.choice(_ALL_EVENTS) for _ in range(n)],
        [rng.randint(1, 4) for _ in range(n)],
    )


def _naive(df: pl.DataFrame, pred: pl.Expr) -> int:
    return df.filter(pred).height


@pytest.mark.parametrize("seed", range(5))
def test_hot_cold_batter_matches_naive_reference(seed: int) -> None:
    df = _random_events_df(seed)
    ev = pl.col("events")
    ab = _naive(df, ev.is_in(stats.AB_EVENTS))
    h = _naive(df, ev.is_in(stats.HIT_EVENTS))
    bb = _naive(df, ev == "walk")
    tb = sum(
        mult * _naive(df, ev == name)
        for mult, name in ((1, "single"), (2, "double"), (3, "triple"), (4, "home_run"))
    )
    got = stats.hot_cold_batter(df)
    assert (got["ab"], got["h"], got["bb"], got["pa"]) == (ab, h, bb, ab + bb)
    assert got["slg"] == round(tb / ab, 3)
    assert got["obp"] == round((h + bb) / (ab + bb), 3)


@pytest.mark.parametrize("seed", range(5))
def test_hot_cold_pitcher_matches_naive_reference(seed: int) -> None:
    df = _random_events_df(seed)
    ev = pl.col("events")
    assert stats.hot_cold_pitcher(df) == {
        "pitches": df.height,
        "k": _naive(df, ev.is_in(["strikeout", "strikeout_double_play"])),
        "bb": _naive(df, ev == "walk"),
        "hr_allowed": _naive(df, ev == "home_run"),
        "pa_faced": _naive(df, ev.is_not_null()),
    }


@pytest.mark.parametrize("seed", range(5))
def test_matchup_stats_matches_naive_reference(seed: int) -> None:
    df = _random_events_df(seed)
    m = df.filter(pl.col("pitcher") == 2)
    ev = pl.col("events")
    ab = _naive(m, ev.is_in(stats.AB_EVENTS))
    h = _naive(m, ev.is_in(stats.HIT_EVENTS))
    assert stats.matchup_stats(df, 2) == {
        "pa": _naive(m, ev.is_not_null()),
        "ab": ab,
        "hits": h,
        "strikeouts": _naive(m, ev.is_in(["strikeout", "strikeout_double_play"])),
        "batting_avg": h / ab,
    }


def test_hot_cold_counts_return_python_ints() -> None:
    """Callers format these; numpy/polars scalars would leak into JSON etc."""
    out = stats.hot_cold_batter(_make_batter_df(["single", None, "walk"]))
    assert all(type(out[k]) is int for k in ("pa", "ab", "h", "bb"))
    out_p = stats.hot_cold_pitcher(_make_batter_df(["strikeout"]))
    assert all(type(v) is int for v in out_p.values())


def test_hot_cold_on_empty_frame_is_all_zero() -> None:
    empty = pl.DataFrame(schema={"events": pl.String, "pitcher": pl.Int64})
    assert stats.hot_cold_batter(empty) == {
        "pa": 0, "ab": 0, "h": 0, "bb": 0, "avg": 0.0, "obp": 0.0, "slg": 0.0, "ops": 0.0,
    }  # fmt: skip
    assert stats.hot_cold_pitcher(empty) == {
        "pitches": 0, "k": 0, "bb": 0, "hr_allowed": 0, "pa_faced": 0,
    }  # fmt: skip


def test_hot_cold_on_all_null_events() -> None:
    df = _make_batter_df([None, None, None])
    assert stats.hot_cold_batter(df)["pa"] == 0
    assert stats.hot_cold_pitcher(df) == {
        "pitches": 3, "k": 0, "bb": 0, "hr_allowed": 0, "pa_faced": 0,
    }  # fmt: skip


def test_aggregate_career_skips_null_ip_seasons() -> None:
    frames = [
        pl.DataFrame({"Name": ["A B"], "_year": [2020], "IP": [10.1], "ERA": [3.0]}),
        pl.DataFrame(
            {"Name": ["A B"], "_year": [2021], "IP": [None], "ERA": [9.0]},
            schema_overrides={"IP": pl.Float64},
        ),
    ]
    out = stats.aggregate_career(frames, "a", "b", [2020, 2021], "pitcher")
    block = out["stats"]
    assert isinstance(block, dict)
    assert block["IP"] == "10.1"
    assert block["ERA"] == "3.000"


# ---------------------------------------------------------------------------
# Parser edge cases
# ---------------------------------------------------------------------------


def test_parse_pitch_arsenal_falls_back_to_reported_usage() -> None:
    df = pl.DataFrame(
        {
            "pitch_type": ["ff", "sl", "ch"],
            "avg_speed": [95.0, 86.0, None],
            "avg_spin": [2300.0, None, 1700.0],
            "pitch_usage": [55.0, None, 20.0],
        }
    )
    out = stats.parse_pitch_arsenal(df)
    # SL has no usage at all and CH has no speed: both dropped.
    assert out == [{"pitch": "FF", "mph": 95.0, "spin": 2300.0, "usage_pct": 55.0}]


def test_parse_pitch_arsenal_empty() -> None:
    assert stats.parse_pitch_arsenal(pl.DataFrame()) == []


def test_parse_exit_velo_missing_player_raises() -> None:
    with pytest.raises(ValueError, match="player_id=9"):
        stats.parse_exit_velo(pl.DataFrame({"player_id": [1]}), 9)


def test_parse_percentile_ranks_tries_alternate_id_columns() -> None:
    df = pl.DataFrame({"mlbam_id": [7], "xba": [88], "k_percent": [None]})
    assert stats.parse_percentile_ranks(df, 7, "batter") == {"xBA": "88 th percentile"}


def test_parse_percentile_ranks_missing_player_raises() -> None:
    with pytest.raises(ValueError, match="No percentile data"):
        stats.parse_percentile_ranks(pl.DataFrame({"player_id": [1]}), 2, "pitcher")


def test_parse_expected_stats_formats_and_tolerates_strings() -> None:
    df = pl.DataFrame({"player_id": [1], "est_ba": [0.2814], "era": ["n/a"]})
    assert stats.parse_expected_stats(df, 1) == {"xBA": "0.281", "ERA": "n/a"}


def test_parse_expected_stats_missing_player_raises() -> None:
    with pytest.raises(ValueError, match="expected-stats"):
        stats.parse_expected_stats(pl.DataFrame({"player_id": [1]}), 2)
