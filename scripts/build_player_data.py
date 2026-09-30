#!/usr/bin/env python3
"""Build data/players.json for the player-prop calculator from nflverse data.

Run by a daily GitHub Actions job, or by hand:

    python3 scripts/build_player_data.py

Standard library only. It downloads weekly player stats, rosters and injury
reports, then works out each player's recent per-game averages and how much
each defense allows to each position compared with an average defense.
"""

import csv
import io
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "players.json")

POSITIONS = ("QB", "RB", "WR", "TE")

# key: (label, stat columns added together, kind, positions offered)
# kind decides the distribution: "yards" is lognormal, "count" negative binomial,
# "td" Poisson. TD props are scaled by the team's projected points instead of
# the opponent's defense, since the team projection already includes the defense.
STATS = {
    "pass_yds": ("Passing yards", ("passing_yards",), "yards", ("QB",)),
    "pass_att": ("Pass attempts", ("attempts",), "count", ("QB",)),
    "pass_cmp": ("Completions", ("completions",), "count", ("QB",)),
    "pass_td": ("Passing TDs", ("passing_tds",), "td", ("QB",)),
    "pass_int": ("Interceptions", ("passing_interceptions",), "count", ("QB",)),
    "rush_yds": ("Rushing yards", ("rushing_yards",), "yards", ("QB", "RB", "WR")),
    "rush_att": ("Rush attempts", ("carries",), "count", ("QB", "RB")),
    "rec_yds": ("Receiving yards", ("receiving_yards",), "yards", ("RB", "WR", "TE")),
    "rec": ("Receptions", ("receptions",), "count", ("RB", "WR", "TE")),
    "rush_rec_yds": ("Rush + rec yards", ("rushing_yards", "receiving_yards"), "yards", ("RB", "WR", "TE")),
    "anytime_td": ("Anytime TD", ("rushing_tds", "receiving_tds"), "td", ("QB", "RB", "WR", "TE")),
}

# Player averages: each older game counts DECAY times as much as the one after
# it, and last season's games count PRIOR_WEIGHT as much again.
DECAY = 0.9
PRIOR_WEIGHT = 0.3
MAX_GAMES = 20

# Defense factors: last season counts DEF_PRIOR_WEIGHT, and DEF_SHRINK phantom
# league-average games pull small samples toward 1.0. They're only used for the
# quarterback's passing stats: in the 2025 backtest they made rushing and
# receiving projections slightly worse.
DEF_PRIOR_WEIGHT = 0.3
DEF_SHRINK = 4
DEFENSE_STATS = ("pass_yds", "pass_att", "pass_cmp", "pass_int")

# How far a player's actual result varies around his projection. Yards use a
# coefficient of variation (sd / mean); counts use variance / mean. Estimated by
# scripts/backtest_props.py, averaged over the 2024 and 2025 seasons.
SPREAD = {
    "pass_yds": {"QB": 0.35},
    "pass_att": {"QB": 3.1},
    "pass_cmp": {"QB": 2.25},
    "pass_int": {"QB": 1.0},
    "rush_yds": {"QB": 0.64, "RB": 0.58, "WR": 1.2},
    "rush_att": {"QB": 1.5, "RB": 2.25},
    "rec_yds": {"RB": 0.66, "WR": 0.63, "TE": 0.65},
    "rec": {"RB": 1.13, "WR": 1.22, "TE": 1.26},
    "rush_rec_yds": {"RB": 0.5, "WR": 0.6, "TE": 0.6},
}


PRACTICE_LABELS = {
    "Did Not Participate In Practice": "Did not practice",
    "Limited Participation in Practice": "Limited practice",
}


def fetch_text(url: str) -> str:
    """Download a URL, falling back to curl when Python has no SSL certificates."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except OSError:
        return subprocess.run(["curl", "-fsSL", "--max-time", "120", url],
                              capture_output=True, text=True, check=True).stdout


def read_csv(url: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(fetch_text(url))))


def num(row: dict, col: str) -> float:
    v = row.get(col, "")
    return float(v) if v not in ("", "NA") else 0.0


def stat_value(row: dict, key: str) -> float:
    return sum(num(row, c) for c in STATS[key][1])


def game_weight(index: int, row_season: int, season: int) -> float:
    return DECAY ** index * (1.0 if row_season == season else PRIOR_WEIGHT)


def player_averages(history: list[dict], season: int) -> tuple[dict, float]:
    """Weighted per-game averages for every stat. history is newest first."""
    history = history[:MAX_GAMES]
    weights = [game_weight(i, int(r["season"]), season) for i, r in enumerate(history)]
    total = sum(weights)
    if not total:
        return {}, 0.0
    return ({k: sum(w * stat_value(r, k) for w, r in zip(weights, history)) / total for k in STATS},
            total)


def defense_factors(rows: list[dict], season: int) -> dict:
    """factors[defense][stat][position]: what the defense allows per game to that
    position, relative to the league average, shrunk toward 1."""
    allowed: dict = {}   # (season, week, defense) -> position -> stat -> total
    for r in rows:
        key = (int(r["season"]), int(r["week"]), r["opponent_team"])
        pos_totals = allowed.setdefault(key, {})
        if r["position"] in POSITIONS:
            totals = pos_totals.setdefault(r["position"], dict.fromkeys(STATS, 0.0))
            for k in STATS:
                totals[k] += stat_value(r, k)

    # League averages come from this season once it has games, else last season.
    avg_season = season if any(s == season for s, _, _ in allowed) else season - 1
    league: dict = {}
    for (s, _, _), pos_totals in allowed.items():
        if s != avg_season:
            continue
        for pos in POSITIONS:
            for k in STATS:
                league.setdefault((k, pos), []).append(pos_totals.get(pos, {}).get(k, 0.0))
    league_avg = {kp: sum(v) / len(v) for kp, v in league.items() if v}

    factors: dict = {}
    sums: dict = {}
    for (s, _, d), pos_totals in allowed.items():
        w = 1.0 if s == season else DEF_PRIOR_WEIGHT
        for pos in POSITIONS:
            for k in STATS:
                acc = sums.setdefault((d, k, pos), [0.0, 0.0])
                acc[0] += w * pos_totals.get(pos, {}).get(k, 0.0)
                acc[1] += w
    for (d, k, pos), (total, weight) in sums.items():
        avg = league_avg.get((k, pos), 0.0)
        if avg <= 0 or pos not in STATS[k][3] or k not in DEFENSE_STATS:
            continue
        f = (total + DEF_SHRINK * avg) / ((weight + DEF_SHRINK) * avg)
        factors.setdefault(d, {}).setdefault(k, {})[pos] = round(f, 3)
    return factors


def team_points(games: list[dict], season: int) -> dict:
    """Each team's points per game this season, with last season at PRIOR_WEIGHT."""
    sums: dict = {}
    for g in games:
        if g["home_score"] in ("", "NA") or int(g["season"]) < season - 1:
            continue
        w = 1.0 if int(g["season"]) == season else PRIOR_WEIGHT
        for team, pts in ((g["home_team"], g["home_score"]), (g["away_team"], g["away_score"])):
            acc = sums.setdefault(team, [0.0, 0.0])
            acc[0] += w * float(pts)
            acc[1] += w
    return {t: round(total / weight, 2) for t, (total, weight) in sums.items()}


def build() -> dict:
    games = read_csv(GAMES_URL)
    season = max(int(g["season"]) for g in games if g["home_score"] not in ("", "NA"))

    stats = [r for y in (season - 1, season) for r in read_csv(f"{RELEASES}/stats_player/stats_player_week_{y}.csv")]
    stats = [r for r in stats if r["season_type"] in ("REG", "POST")]
    stats.sort(key=lambda r: (int(r["season"]), int(r["week"])), reverse=True)

    roster = read_csv(f"{RELEASES}/rosters/roster_{season}.csv")
    latest_week = max(int(r["week"]) for r in roster if r["week"] not in ("", "NA"))
    roster = {r["gsis_id"]: r for r in roster
              if r["week"] not in ("", "NA") and int(r["week"]) == latest_week and r["position"] in POSITIONS}

    try:
        injuries = read_csv(f"{RELEASES}/injuries/injuries_{season}.csv")
    except (OSError, subprocess.CalledProcessError):
        injuries = []
    # Only this week's report counts: last week's statuses are stale. Before the
    # game-day status comes out (usually Friday), fall back to practice participation.
    inj_week = max((int(r["week"]) for r in injuries), default=0)
    this_week = [r for r in injuries if int(r["week"]) == inj_week]
    injury = {}
    for r in this_week:
        status = r["report_status"] or PRACTICE_LABELS.get(r["practice_status"], "")
        if status:
            injury[r["gsis_id"]] = {**r, "report_status": status}
    injury_teams = sorted({r["team"] for r in this_week})

    history: dict = {}
    for r in stats:
        history.setdefault(r["player_id"], []).append(r)

    players = []
    for pid, info in roster.items():
        if info["status"] not in ("ACT", "RES") or pid not in history:
            continue
        pos = info["position"]
        avgs, n_eff = player_averages(history[pid], season)
        if not avgs:
            continue
        usage = avgs["pass_att"] + avgs["rush_att"] + 1.5 * avgs["rec"]
        if usage < 1:
            continue
        recent = history[pid][:5]
        inj = injury.get(pid)
        status = "IR" if info["status"] == "RES" else (inj["report_status"] if inj else "")
        players.append({
            "id": pid,
            "name": info["full_name"],
            "team": info["team"],
            "pos": pos,
            "status": status,
            "injury": (inj or {}).get("report_primary_injury") or (inj or {}).get("practice_primary_injury", ""),
            "games": len(history[pid][:MAX_GAMES]),
            "games_this_season": sum(1 for r in history[pid] if int(r["season"]) == season),
            "weight": round(n_eff, 2),
            "usage": round(usage, 1),
            "avg": {k: round(avgs[k], 3) for k, v in STATS.items() if pos in v[3]},
            "recent": [{"week": int(r["week"]), "season": int(r["season"]), "opp": r["opponent_team"],
                        **{k: stat_value(r, k) for k, v in STATS.items() if pos in v[3]}} for r in recent],
        })
    # Within each team: by position, then most-used first, with injured reserve last.
    players.sort(key=lambda p: (p["team"], POSITIONS.index(p["pos"]), p["status"] == "IR", -p["usage"]))

    return {
        "season": season,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stats_through_week": max((int(r["week"]) for r in stats if int(r["season"]) == season), default=0),
        "injury_week": inj_week,
        "injury_teams": injury_teams,
        "stats": {k: {"label": v[0], "kind": v[2], "positions": list(v[3])} for k, v in STATS.items()},
        "spread": SPREAD,
        "team_ppg": team_points(games, season),
        "defense": defense_factors(stats, season),
        "players": players,
    }


def main():
    data = build()
    # Leave the file alone when only the timestamp would change, so the daily
    # job doesn't commit identical data.
    try:
        with open(OUT_PATH, encoding="utf-8") as f:
            old = json.load(f)
        if {**old, "updated": None} == {**data, "updated": None}:
            print("No changes.", file=sys.stderr)
            return
    except (OSError, ValueError):
        pass
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote {len(data['players'])} players, stats through week {data['stats_through_week']} "
          f"of {data['season']}, injury report week {data['injury_week']}.", file=sys.stderr)


if __name__ == "__main__":
    main()
