"""Async wrappers around the official MLB Stats API (statsapi.mlb.com).

Team ID and abbreviation tables are hand-curated; preserve them verbatim.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

from bigleague.http import mlb_get

_log = logging.getLogger("bigleague")

TEAM_IDS: dict[str, int] = {
    "ARI": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHC": 112,
    "CHW": 145,
    "CWS": 145,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KCR": 118,
    "KC": 118,
    "LAA": 108,
    "LAD": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYM": 121,
    "NYY": 147,
    "OAK": 133,
    "PHI": 143,
    "PIT": 134,
    "SDP": 135,
    "SD": 135,
    "SFG": 137,
    "SF": 137,
    "SEA": 136,
    "STL": 138,
    "TBR": 139,
    "TB": 139,
    "TEX": 140,
    "TOR": 141,
    "WSN": 120,
    "WAS": 120,
}

POS_NAMES: dict[str, str] = {
    "P": "SP/RP",
    "SP": "SP",
    "RP": "RP",
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "LF",
    "CF": "CF",
    "RF": "RF",
    "OF": "OF",
    "DH": "DH",
}

# Canonical 3-letter abbreviations — preferred when multiple aliases map
# to the same MLB team ID (e.g. SDP/SD both -> 135; canonical is SD,
# which matches Statcast and Baseball Reference usage).
_CANONICAL_ABBRS: frozenset[str] = frozenset(
    {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
        "HOU", "KC",  "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
        "PHI", "PIT", "SD",  "SF",  "SEA", "STL", "TB",  "TEX", "TOR", "WSN",
    }
)  # fmt: skip
_TEAM_ID_TO_ABBR: dict[int, str] = {
    tid: abbr for abbr, tid in TEAM_IDS.items() if abbr in _CANONICAL_ABBRS
}


def _team_id(abbrev: str) -> int:
    tid = TEAM_IDS.get(abbrev.upper())
    if tid is None:
        valid = ", ".join(sorted({k for k in TEAM_IDS if len(k) <= 3}))
        raise ValueError(f"Unknown team abbreviation {abbrev!r}. Valid: {valid}")
    return tid


# ---------------------------------------------------------------------------
# Roster + injuries + transactions
# ---------------------------------------------------------------------------


async def roster(team: str) -> list[dict[str, str]]:
    """Active 26-man roster: [{name, pos, status}].

    Raises ValueError if the API returns no roster rows for the team.
    """
    tid = _team_id(team)
    data = await mlb_get(f"/teams/{tid}/roster", {"rosterType": "active"})
    rows = data.get("roster", [])
    if not rows:
        raise ValueError(f"No active roster data for {team}.")
    return [
        {
            "name": p.get("person", {}).get("fullName", "Unknown"),
            "pos": p.get("position", {}).get("abbreviation", "?"),
            "status": p.get("status", {}).get("description", "Active"),
        }
        for p in rows
    ]


async def injuries(team: str) -> list[dict[str, str]]:
    """Players currently on the IL: [{name, pos, note}].

    The legacy `rosterType=injuries` endpoint was broken silently for
    most of 2025 — it now returns the active roster. Pull the 40-man
    instead and filter on D* status codes (D7/D10/D15/D60).
    """
    tid = _team_id(team)
    data = await mlb_get(f"/teams/{tid}/roster", {"rosterType": "40Man"})
    rows = data.get("roster", [])

    out: list[dict[str, str]] = []
    for p in rows:
        status = p.get("status", {}) or {}
        code = status.get("code", "")
        if not code.startswith("D"):
            continue
        stint = status.get("description", "").replace("Injured ", "").strip()
        injury = (p.get("note") or "").strip()
        if stint and injury:
            note = f"{stint} IL — {injury}"
        elif stint:
            note = f"{stint} IL"
        else:
            note = injury or "—"
        out.append(
            {
                "name": p.get("person", {}).get("fullName", "Unknown"),
                "pos": p.get("position", {}).get("abbreviation", "?"),
                "note": note,
            }
        )
    return out


async def transactions(team: str, days: int = 7) -> list[dict[str, str]]:
    """Recent roster transactions, newest first: [{date, desc}].

    Raises ValueError if the team has no transactions in the window.
    """
    tid = _team_id(team)
    today = date.today()
    start = today - timedelta(days=days)
    data = await mlb_get(
        "/transactions",
        {
            "teamId": tid,
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": today.strftime("%Y-%m-%d"),
        },
    )
    txs = data.get("transactions", [])
    if not txs:
        raise ValueError(f"No transactions for {team} in the last {days} days.")
    return [
        {
            "date": t.get("date", "?")[:10],
            "desc": t.get("description", "—"),
        }
        for t in sorted(txs, key=lambda x: x.get("date", ""), reverse=True)
    ]


# ---------------------------------------------------------------------------
# Schedule: live, next, recent
# ---------------------------------------------------------------------------


async def live_scores(team: str) -> list[dict[str, Any]]:
    """Today's games for a team with linescore details.

    Raises ValueError if the team has no game today (e.g. an off day).
    """
    tid = _team_id(team)
    today = date.today().strftime("%Y-%m-%d")
    data = await mlb_get(
        "/schedule",
        {
            "sportId": 1,
            "date": today,
            "teamId": tid,
            "hydrate": "linescore,probablePitcher(note)",
        },
    )
    dates = data.get("dates", [])
    if not dates:
        raise ValueError(f"{team.upper()} has no game today ({today}).")

    games: list[dict[str, Any]] = []
    for game in dates[0].get("games", []):
        status = game.get("status", {}).get("detailedState", "?")
        home = game["teams"]["home"]
        away = game["teams"]["away"]
        linescore = game.get("linescore", {})
        inning = linescore.get("currentInningOrdinal", "")
        inning_h = linescore.get("inningHalf", "")
        outs = linescore.get("outs", 0)
        games.append(
            {
                "game_pk": game.get("gamePk"),
                "home_team": home["team"]["name"],
                "away_team": away["team"]["name"],
                "home_score": home.get("score", 0),
                "away_score": away.get("score", 0),
                "status": status,
                "inning": f"{inning_h} {inning}".strip() if inning else status,
                "outs": outs,
                "start_time": game.get("gameDate"),
            }
        )
    return games


async def next_game(team: str) -> dict[str, str]:
    """Next scheduled game with probable pitchers within 14 days.

    Raises ValueError if no game is scheduled in the next 14 days
    (e.g. the offseason).
    """
    tid = _team_id(team)
    today = date.today()
    end = today + timedelta(days=14)
    data = await mlb_get(
        "/schedule",
        {
            "sportId": 1,
            "startDate": today.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "teamId": tid,
            "hydrate": "probablePitcher",
        },
    )
    for day in data.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status not in ("Preview", "Scheduled"):
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            game_time = game.get("gameDate", "")
            return {
                "date": day["date"],
                "time": game_time[11:16] + " UTC" if len(game_time) > 16 else "TBD",
                "home_team": home["team"]["name"],
                "away_team": away["team"]["name"],
                "home_probable": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "away_probable": away.get("probablePitcher", {}).get("fullName", "TBD"),
            }
    raise ValueError(f"No upcoming games found for {team} in the next 14 days.")


async def recent_results(team: str, n: int = 10, lookback_days: int = 30) -> list[dict[str, Any]]:
    """Last N completed regular-season games, oldest -> newest.

    `result` is "W"/"L"/"P" — P covers postponed/suspended (kept so the
    caller can render them as a neutral marker rather than dropping).
    """
    tid = _team_id(team)
    today = date.today()
    start = today - timedelta(days=lookback_days)
    data = await mlb_get(
        "/schedule",
        {
            "sportId": 1,
            "teamId": tid,
            "gameType": "R",
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": today.strftime("%Y-%m-%d"),
        },
    )

    out: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            status = (game.get("status") or {}).get("abstractGameState", "")
            detailed = (game.get("status") or {}).get("detailedState", "")
            home = (game.get("teams") or {}).get("home") or {}
            away = (game.get("teams") or {}).get("away") or {}
            home_id = (home.get("team") or {}).get("id")
            is_home = home_id == tid
            our = home if is_home else away
            opp = away if is_home else home

            if status == "Final":
                won = bool(our.get("isWinner"))
                result = "W" if won else "L"
                score = f"{our.get('score', 0)}-{opp.get('score', 0)}"
            elif "Postponed" in detailed or "Suspended" in detailed:
                result = "P"
                score = ""
            else:
                continue

            out.append(
                {
                    "date": day.get("date", ""),
                    "is_home": is_home,
                    "result": result,
                    "opponent": (opp.get("team") or {}).get("abbreviation")
                    or (opp.get("team") or {}).get("name", "?"),
                    "score": score,
                }
            )

    out.sort(key=lambda g: g["date"])
    return out[-n:]


async def schedule(team: str, year: int, game_type: str = "R") -> list[dict[str, Any]]:
    """A team's full-season schedule with results and a running record.

    Games come back in play order (doubleheaders by start time). Each row:

    - `date`, `start_time` (UTC ISO timestamp), `game_pk`
    - `is_home`, `opponent` (canonical abbreviation, or the team name when
      the id is unknown)
    - `status`: "W"/"L" for final games, "P" for postponed or suspended,
      "S" for games not yet played
    - `score`: "ours-theirs" for final games, otherwise ""
    - `record`: the team's "W-L" after that game (carried forward over
      postponed and scheduled rows)

    `game_type` is the MLB Stats API code: "R" regular season (default),
    "S" spring training, "P" postseason.

    Raises ValueError if the API returns no games (e.g. a future season
    whose schedule hasn't been published).
    """
    tid = _team_id(team)
    data = await mlb_get(
        "/schedule",
        {"sportId": 1, "teamId": tid, "season": year, "gameType": game_type},
    )

    rows: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            state = game.get("status") or {}
            abstract = state.get("abstractGameState", "")
            detailed = state.get("detailedState", "")
            home = (game.get("teams") or {}).get("home") or {}
            away = (game.get("teams") or {}).get("away") or {}
            is_home = (home.get("team") or {}).get("id") == tid
            ours, theirs = (home, away) if is_home else (away, home)
            opp_team = theirs.get("team") or {}

            if abstract == "Final":
                our_runs, their_runs = ours.get("score", 0), theirs.get("score", 0)
                won = ours.get("isWinner")
                if won is None:
                    won = our_runs > their_runs
                status = "W" if won else "L"
                score = f"{our_runs}-{their_runs}"
            elif "Postponed" in detailed or "Suspended" in detailed:
                status, score = "P", ""
            else:
                status, score = "S", ""

            rows.append(
                {
                    "date": day.get("date", ""),
                    "start_time": game.get("gameDate", ""),
                    "game_pk": game.get("gamePk"),
                    "is_home": is_home,
                    "opponent": _TEAM_ID_TO_ABBR.get(opp_team.get("id"))
                    or opp_team.get("name", "?"),
                    "status": status,
                    "score": score,
                }
            )

    if not rows:
        raise ValueError(f"No {game_type!r} schedule for {team.upper()} in {year}.")

    rows.sort(key=lambda g: (g["date"], g["start_time"]))
    wins = losses = 0
    for row in rows:
        wins += row["status"] == "W"
        losses += row["status"] == "L"
        row["record"] = f"{wins}-{losses}"
    return rows


# ---------------------------------------------------------------------------
# Players (autocomplete) + standings
# ---------------------------------------------------------------------------


async def players(season: int) -> list[dict[str, Any]]:
    """Full MLB player roster for a season: [{id, first, last, team, position}]."""
    data = await mlb_get("/sports/1/players", {"season": season})
    out: list[dict[str, Any]] = []
    for p in data.get("people", []):
        team_id = (p.get("currentTeam") or {}).get("id")
        team_abbr = _TEAM_ID_TO_ABBR.get(team_id, "") if team_id else ""
        out.append(
            {
                "id": p.get("id"),
                # useName is the name the player actually goes by ("Jazz");
                # firstName is the legal name ("Jasrado") and is always
                # populated, so it must be the fallback, not the preference.
                "first": (p.get("useName") or p.get("firstName") or "").strip(),
                "last": (p.get("lastName") or "").strip(),
                "team": team_abbr,
                "position": (p.get("primaryPosition") or {}).get("abbreviation", ""),
            }
        )
    return out


async def players_multi_season(seasons: list[int]) -> list[dict[str, Any]]:
    """Fetch players for multiple seasons concurrently; log per-season failures.

    Results keep the order of `seasons`; a failed season contributes nothing.
    """
    results = await asyncio.gather(*(players(s) for s in seasons), return_exceptions=True)
    combined: list[dict[str, Any]] = []
    for season, result in zip(seasons, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result  # never swallow cancellation / KeyboardInterrupt
            _log.warning("players(%d) failed: %s", season, result)
            continue
        combined.extend(result)
    return combined


async def standings(year: int) -> list[dict[str, Any]]:
    """MLB division standings: list of {division, teams: [{name, w, l, gb, pct}]}.

    Raises ValueError if the API has no standings records for the year.
    """
    data = await mlb_get(
        "/standings",
        {
            "leagueId": "103,104",
            "season": year,
            "standingsType": "regularSeason",
            "hydrate": "team,division",
        },
    )
    records = data.get("records", [])
    if not records:
        raise ValueError(f"No standings data for {year}.")

    out: list[dict[str, Any]] = []
    for div_record in records:
        div = (div_record.get("division") or {}).get("nameShort") or (
            div_record.get("division") or {}
        ).get("name", "Unknown")
        teams = [
            {
                "name": (tr.get("team") or {}).get("name", "?"),
                "w": tr.get("wins", 0),
                "l": tr.get("losses", 0),
                "gb": tr.get("gamesBack", "-"),
                "pct": tr.get("winningPercentage", "—"),
            }
            for tr in div_record.get("teamRecords", [])
        ]
        out.append({"division": div, "teams": teams})
    return out
