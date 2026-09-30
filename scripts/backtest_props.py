#!/usr/bin/env python3
"""Backtest the player-prop model on a past season, week by week.

    python3 scripts/backtest_props.py [SEASON]

Before each week it rebuilds player averages and defense factors from earlier
games only, then compares its projections with what happened. It prints how
accurate the projections were, how widely results varied around them (the
SPREAD settings in build_player_data.py), and whether the over/under
probabilities were calibrated.
"""

import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_player_data as b  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import football_odds as fo  # noqa: E402

# Minimum projection for a player-game to count, roughly the players sportsbooks
# offer props on.
MIN_PROJECTION = {"pass_yds": 150, "pass_att": 20, "pass_cmp": 12, "pass_td": 0.8, "pass_int": 0.4,
                  "rush_yds": 25, "rush_att": 8, "rec_yds": 25, "rec": 2.5, "rush_rec_yds": 35,
                  "anytime_td": 0.25}


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    stats = [r for y in (season - 1, season)
             for r in b.read_csv(f"{b.RELEASES}/stats_player/stats_player_week_{y}.csv")
             if r["season_type"] in ("REG", "POST") and r["position"] in b.POSITIONS]
    stats.sort(key=lambda r: (int(r["season"]), int(r["week"])), reverse=True)
    games = [g for g in b.read_csv(b.GAMES_URL) if g["home_score"] not in ("", "NA")]


    results = {}   # (stat, pos) -> list of (projection, actual, naive)
    for week in range(4, 19):
        def before(r):
            return int(r["season"]) < season or (int(r["season"]) == season and int(r["week"]) < week)
        past = [r for r in stats if before(r)]
        factors = b.defense_factors(past, season)
        past_games = [g for g in games if before(g)]
        ratings = fo.Ratings(past_games + [g for g in games if int(g["season"]) == season and int(g["week"]) == week])
        ppg = b.team_points(past_games, season)
        history = {}
        for r in past:
            history.setdefault(r["player_id"], []).append(r)
        week_games = {g["home_team"]: g for g in games if int(g["season"]) == season and int(g["week"]) == week}
        week_games.update({g["away_team"]: g for g in week_games.values()})

        for r in stats:
            if int(r["season"]) != season or int(r["week"]) != week or r["season_type"] != "REG":
                continue
            h = history.get(r["player_id"], [])
            if sum(1 for x in h if int(x["season"]) == season) < 2 or len(h) < 4:
                continue
            avgs, _ = b.player_averages(h, season)
            this_season = [x for x in h if int(x["season"]) == season]
            pos, team, opp = r["position"], r["team"], r["opponent_team"]
            g = week_games.get(team)
            if not g:
                continue
            hp, ap = ratings.project(g["home_team"], g["away_team"], g["location"] == "Neutral")
            team_proj = hp if g["home_team"] == team else ap
            for k, (_, _, kind, positions) in b.STATS.items():
                if pos not in positions:
                    continue
                if kind == "td":
                    proj = avgs[k] * team_proj / ppg.get(team, team_proj)
                else:
                    proj = avgs[k] * factors.get(opp, {}).get(k, {}).get(pos, 1.0)
                if proj < MIN_PROJECTION[k]:
                    continue
                naive = st.mean(b.stat_value(x, k) for x in this_season)
                results.setdefault((k, pos), []).append((proj, b.stat_value(r, k), naive, avgs[k]))

    print(f"Backtest of {season}, weeks 4-18\n")
    print(f"{'Stat':<14}{'Pos':<5}{'n':>6}{'MAE model':>11}{'MAE no opp':>12}{'MAE naive':>11}{'Spread':>9}")
    for (k, pos), rows in sorted(results.items()):
        if len(rows) < 100:
            continue
        kind = b.STATS[k][2]
        mae = st.mean(abs(p - a) for p, a, _, _ in rows)
        mae_base = st.mean(abs(base - a) for _, a, _, base in rows)
        mae_naive = st.mean(abs(n - a) for _, a, n, _ in rows)
        if kind == "yards":   # coefficient of variation of actual around projection
            spread = math.sqrt(sum((a - p) ** 2 for p, a, _, _ in rows) / sum(p * p for p, _, _, _ in rows))
        else:                 # variance / mean of actual around projection
            spread = sum((a - p) ** 2 for p, a, _, _ in rows) / sum(p for p, _, _, _ in rows)
        print(f"{k:<14}{pos:<5}{len(rows):>6}{mae:>11.2f}{mae_base:>12.2f}{mae_naive:>11.2f}{spread:>9.2f}")

    # Calibration: using each player's season average as a stand-in for the
    # sportsbook line, compare predicted and actual over rates.
    print("\nCalibration: predicted chance of going over vs how often it happened")
    buckets = {}
    for (k, pos), rows in results.items():
        spread = b.SPREAD.get(k, {}).get(pos)
        for p, a, n, _ in rows:
            line = math.floor(n) + 0.5
            over = fo.prop_probabilities(b.STATS[k][2], p, spread, line)[0]
            key = min(int(over * 10), 9)
            acc = buckets.setdefault(key, [0, 0, 0.0])
            acc[0] += 1
            acc[1] += a > line
            acc[2] += over
    for key in sorted(buckets):
        n, hits, pred = buckets[key]
        print(f"  predicted {key * 10:>2}-{key * 10 + 10}%: n={n:>5}  predicted avg {pred / n:6.1%}  actual {hits / n:6.1%}")


if __name__ == "__main__":
    main()
