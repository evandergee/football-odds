#!/usr/bin/env python3
"""NFL odds toolkit: odds conversion, bookmaker vig, team ratings and game predictions.

Standard library only. Examples:

    python3 football_odds.py convert -110
    python3 football_odds.py vig -110 -110
    python3 football_odds.py teams
    python3 football_odds.py week
    python3 football_odds.py game BUF KC
    python3 football_odds.py game bills chiefs --ml +140 -165
    python3 football_odds.py predict 24.5 21.5 --spread -3 --total 44.5
"""

import argparse
import csv
import io
import math
import os
import ssl
import subprocess
import sys
import urllib.request
from fractions import Fraction

# NFL standard deviations, in points, of the final margin and of the total around
# closing betting lines, 2023-2025 seasons. Both can be overridden on the command line.
MARGIN_SD = 13.0
TOTAL_SD = 13.0

# Team rating settings, chosen by backtesting the 2024 and 2025 seasons.
DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
HOME_FIELD = 1.5     # points
PRIOR_WEIGHT = 0.2   # weight of last season's games
SHRINK = 6           # phantom average games added to every team

# Edge bands for the red / yellow / green ratings. A positive edge below
# CAUTION_MIN is within the model's error; one above CAUTION_MAX usually means
# the model is missing news, since sportsbooks are more accurate than it is.
CAUTION_MIN = 0.03
CAUTION_MAX = 0.10

TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def nickname(abbr: str) -> str:
    return TEAMS.get(abbr, abbr).split()[-1]


def find_team(text: str) -> str:
    """Accept an abbreviation (KC), nickname (Chiefs), city or full name."""
    t = text.strip().lower()
    for abbr, name in TEAMS.items():
        if t in (abbr.lower(), name.lower(), nickname(abbr).lower(), name.lower().rsplit(" ", 1)[0]):
            return abbr
    raise SystemExit(f"Unknown team '{text}'. Use an abbreviation like KC or a name like Chiefs.")


# ------------------------------------------------------------------ colour

class Colour:
    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    codes = {"good": "32", "caution": "33", "bad": "31", "dim": "2"}

    @classmethod
    def wrap(cls, text: str, kind: str) -> str:
        return f"\033[{cls.codes[kind]}m{text}\033[0m" if cls.enabled and kind in cls.codes else text


def edge_rating(edge: float) -> str:
    if edge < 0:
        return "bad"
    if edge < CAUTION_MIN or edge > CAUTION_MAX:
        return "caution"
    return "good"


# ---------------------------------------------------------------- conversion

def to_decimal(odds: str) -> float:
    """Parse American (-110 / +150 / 150), decimal (1.91) or fractional (10/11) odds."""
    s = str(odds).strip()
    if "/" in s:
        num, den = s.split("/")
        return 1 + float(num) / float(den)
    value = float(s)
    if s.startswith(("+", "-")) or abs(value) >= 100:
        if abs(value) < 100:
            raise ValueError(f"American odds must be at least 100 either way, got {odds}")
        return 1 + value / 100 if value > 0 else 1 + 100 / abs(value)
    if value <= 1:
        raise ValueError(f"decimal odds must be greater than 1, got {odds}")
    return value


def to_american(decimal: float) -> str:
    # Rounding first means a 50% price shows as +100 on both sides.
    if decimal >= 2 or round(100 / (decimal - 1)) == 100:
        return f"+{round((decimal - 1) * 100)}"
    return f"-{round(100 / (decimal - 1))}"


def to_fractional(decimal: float) -> str:
    frac = Fraction(decimal - 1).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def fair_american(prob: float) -> str:
    return to_american(1 / prob) if 0 < prob < 1 else "-"


# ----------------------------------------------------------------------- vig

def remove_vig(decimals: list[float]) -> tuple[float, list[float]]:
    """Return the bookmaker's vig and the fair (vig-free) probabilities.

    Uses the proportional method: each implied probability is scaled so they sum to 1.
    """
    implied = [1 / d for d in decimals]
    book = sum(implied)
    return book - 1, [p / book for p in implied]


# --------------------------------------------------------------- game model

def normal_cdf(x: float, mean: float, sd: float) -> float:
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))


def discrete_normal(mean: float, sd: float, lo: int, hi: int) -> dict[int, float]:
    """Probability of each whole-number result, from a normal curve rounded to integers."""
    return {k: normal_cdf(k + 0.5, mean, sd) - normal_cdf(k - 0.5, mean, sd) for k in range(lo, hi + 1)}


def margin_distribution(home_pts: float, away_pts: float, sd: float = MARGIN_SD) -> dict[int, float]:
    """P(home margin = k). Games level after regulation go to overtime, which is
    usually decided by a field goal, so that probability is split between +3 and -3."""
    dist = discrete_normal(home_pts - away_pts, sd, -100, 100)
    tie = dist.pop(0)
    dist[3] += tie / 2
    dist[-3] += tie / 2
    return dist


def total_distribution(home_pts: float, away_pts: float, sd: float = TOTAL_SD) -> dict[int, float]:
    return discrete_normal(home_pts + away_pts, sd, 0, 150)


def line_probabilities(dist: dict[int, float], line: float) -> tuple[float, float, float]:
    """(P(result > line), P(result == line), P(result < line))."""
    above = sum(p for k, p in dist.items() if k > line)
    push = dist.get(int(line), 0.0) if line == int(line) else 0.0
    return above, push, 1 - above - push


def kelly_fraction(win: float, loss: float, decimal: float) -> float:
    """Kelly stake as a fraction of bankroll, allowing for pushes; 0 with no edge."""
    b = decimal - 1
    if win + loss == 0:
        return 0.0
    return max(0.0, (b * win - loss) / (b * (win + loss)))


# ------------------------------------------------------------- team ratings

def fetch_text(url: str) -> str:
    """Download a URL. Falls back to the system curl when Python has no SSL
    certificates, which is common with python.org installs on macOS."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except (ssl.SSLError, urllib.error.URLError) as err:
        try:
            return subprocess.run(["curl", "-fsSL", "--max-time", "60", url],
                                  capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit(f"Couldn't download NFL results ({err}). Check your internet connection, "
                             "or download games.csv yourself and pass it with --data.")


def load_games(path: str | None) -> list[dict]:
    if path:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return list(csv.DictReader(io.StringIO(fetch_text(DATA_URL))))


def is_played(game: dict) -> bool:
    return game["home_score"] not in ("", "NA")


class Ratings:
    """Offense and defense ratings in points relative to an average team.

    Points for = league average + own offense - opponent defense + home field share,
    solved by repeated averaging. SHRINK adds phantom average games so that teams
    with few games stay close to average.
    """

    def __init__(self, games: list[dict]):
        done = [g for g in games if is_played(g)]
        self.season = max(int(g["season"]) for g in done)
        current = [g for g in done if int(g["season"]) == self.season]
        self.games_this_season = len(current)
        reg_weeks = [int(g["week"]) for g in current if g["game_type"] == "REG"]
        self.last_week = max(reg_weeks) if reg_weeks else 0
        self._fit([g for g in done if int(g["season"]) >= self.season - 1])

        future = [g for g in games if int(g["season"]) == self.season and not is_played(g)]
        self.next_week = min((int(g["week"]) for g in future), default=None)
        self.upcoming = [g for g in future if int(g["week"]) == self.next_week]

    def _fit(self, games: list[dict]):
        mine, theirs = {}, {}
        total = weight = 0.0
        for g in games:
            w = 1.0 if int(g["season"]) == self.season else PRIOR_WEIGHT
            h = 0.0 if g["location"] == "Neutral" else HOME_FIELD / 2
            for team, opp, pts, hh in ((g["home_team"], g["away_team"], int(g["home_score"]), h),
                                       (g["away_team"], g["home_team"], int(g["away_score"]), -h)):
                mine.setdefault(team, []).append((opp, pts, hh, w))
                theirs.setdefault(opp, []).append((team, pts, hh, w))
                total += pts * w
                weight += w
        self.avg = total / weight
        off: dict[str, float] = {}
        dfn: dict[str, float] = {}
        for _ in range(40):
            off = {t: sum(w * (pts - self.avg - hh + dfn.get(o, 0.0)) for o, pts, hh, w in obs)
                   / (sum(x[3] for x in obs) + SHRINK) for t, obs in mine.items()}
            dfn = {t: sum(w * (self.avg + off.get(a, 0.0) + hh - pts) for a, pts, hh, w in obs)
                   / (sum(x[3] for x in obs) + SHRINK) for t, obs in theirs.items()}
        self.off, self.dfn = off, dfn

    def project(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        h = 0.0 if neutral else HOME_FIELD / 2
        return (self.avg + self.off.get(home, 0.0) - self.dfn.get(away, 0.0) + h,
                self.avg + self.off.get(away, 0.0) - self.dfn.get(home, 0.0) - h)

    def find_upcoming(self, home: str, away: str) -> dict | None:
        return next((g for g in self.upcoming if g["home_team"] == home and g["away_team"] == away), None)

    def describe(self) -> str:
        text = f"Ratings use {self.games_this_season} games from {self.season}"
        if self.last_week:
            text += f" (through week {self.last_week})"
        return text + f" plus {self.season - 1} results at {PRIOR_WEIGHT:.0%} weight."


# ----------------------------------------------------------------------- CLI

def half_point(x: float) -> float:
    return round(x * 2) / 2 + 0.0   # + 0.0 turns -0.0 into 0.0


def fmt_line(x: float) -> str:
    return f"{x:+g}" if x else "PK"


def signed(value: str) -> str:
    return f"+{value}" if value and not value.startswith("-") else value


def print_prediction(home: str, away: str, home_pts: float, away_pts: float, spread: float | None,
                     total_line: float | None, ml=None, spread_odds=None, total_odds=None,
                     margin_sd: float = MARGIN_SD, total_sd: float = TOTAL_SD):
    """Print moneyline, spread and total probabilities, plus edges against any bookmaker
    odds given. ml and spread_odds are (home, away) pairs; total_odds is (over, under)."""
    margin = margin_distribution(home_pts, away_pts, margin_sd)
    total = total_distribution(home_pts, away_pts, total_sd)
    proj_margin = home_pts - away_pts
    proj_total = home_pts + away_pts
    if spread is None:
        spread = half_point(-proj_margin)
    if total_line is None:
        total_line = half_point(proj_total)

    print(f"Projected score: {away} {away_pts:.1f}, {home} {home_pts:.1f}")
    print(f"Fair spread: {home} {fmt_line(half_point(-proj_margin))} · fair total: {half_point(proj_total):g}\n")

    # Home covers a spread of L when margin + L > 0, i.e. margin > -L.
    home_ml, _, away_ml = line_probabilities(margin, 0)
    cover, spread_push, no_cover = line_probabilities(margin, -spread)
    over, total_push, under = line_probabilities(total, total_line)

    rows = [
        (f"{away} moneyline", away_ml, 0.0, home_ml, ml, 1),
        (f"{home} moneyline", home_ml, 0.0, away_ml, ml, 0),
        (f"{away} {fmt_line(-spread)}", no_cover, spread_push, cover, spread_odds, 1),
        (f"{home} {fmt_line(spread)}", cover, spread_push, no_cover, spread_odds, 0),
        (f"Over {total_line:g}", over, total_push, under, total_odds, 0),
        (f"Under {total_line:g}", under, total_push, over, total_odds, 1),
    ]

    has_book = any(r[4] and r[4][r[5]] for r in rows)
    header = f"{'Market':<22}{'Win':>8}{'Push':>7}{'Fair':>7}"
    if has_book:
        header += f"{'Book':>7}{'Edge':>9}{'Kelly':>8}  Rating"
    print(header)
    for name, win, push, loss, book, idx in rows:
        # Fair odds ignore pushes, since a push returns the stake.
        line = f"{name:<22}{win:>8.1%}{(f'{push:.1%}' if push else '-'):>7}{fair_american(win / (win + loss)):>7}"
        if book and book[idx]:
            d = to_decimal(book[idx])
            edge = win * (d - 1) - loss
            rating = edge_rating(edge)
            line += f"{to_american(d):>7}{edge:>+9.1%}{kelly_fraction(win, loss, d):>8.1%}  {rating}"
            line = Colour.wrap(line, rating)
        print(line)

    if has_book:
        print(Colour.wrap(f"\nRatings: bad = negative edge · caution = edge under {CAUTION_MIN:.0%}, or over "
                          f"{CAUTION_MAX:.0%}, which usually means the model is missing news · "
                          f"good = {CAUTION_MIN:.0%} to {CAUTION_MAX:.0%}.", "dim"))


def cmd_convert(args):
    d = to_decimal(args.odds)
    print(f"American:    {to_american(d)}")
    print(f"Decimal:     {d:.3f}")
    print(f"Fractional:  {to_fractional(d)}")
    print(f"Implied %:   {1 / d:.2%}")


def cmd_vig(args):
    decimals = [to_decimal(o) for o in args.odds]
    vig, fair = remove_vig(decimals)
    print(f"Bookmaker vig: {vig:.2%}\n")
    print(f"{'Outcome':<9}{'Odds':>7}{'Implied':>10}{'Fair %':>9}{'Fair odds':>11}")
    for i, (d, p) in enumerate(zip(decimals, fair), start=1):
        print(f"{'Side ' + str(i):<9}{to_american(d):>7}{1 / d:>10.2%}{p:>9.2%}{fair_american(p):>11}")


def cmd_predict(args):
    print_prediction(args.home, args.away, args.home_pts, args.away_pts, args.spread, args.total,
                     args.ml, args.spread_odds, args.total_odds, args.margin_sd, args.total_sd)


def cmd_teams(args):
    r = Ratings(load_games(args.data))
    print(r.describe())
    print(f"League average: {r.avg:.1f} points per team per game\n")
    print(f"{'Team':<24}{'Offense':>9}{'Defense':>9}{'Overall':>9}")
    for t in sorted(r.off, key=lambda t: r.off[t] + r.dfn.get(t, 0), reverse=True):
        print(f"{TEAMS.get(t, t):<24}{r.off[t]:>+9.1f}{r.dfn.get(t, 0):>+9.1f}{r.off[t] + r.dfn.get(t, 0):>+9.1f}")
    print("\nPoints above or below an average team. Higher is better for both offense and defense.")


def cmd_week(args):
    r = Ratings(load_games(args.data))
    if not r.upcoming:
        raise SystemExit("No upcoming games in the data yet.")
    print(f"Week {r.next_week}, {r.season}. {r.describe()}\n")
    print(f"{'Game':<14}{'Day':<11}{'Model spread':>16}{'Book spread':>16}{'Model total':>13}{'Book total':>12}")
    for g in r.upcoming:
        home, away = g["home_team"], g["away_team"]
        hp, ap = r.project(home, away, g["location"] == "Neutral")
        game = f"{away} @ {home}" + (" *" if g["location"] == "Neutral" else "")
        book_spread = f"{home} {fmt_line(-float(g['spread_line']))}" if g["spread_line"] not in ("", "NA") else "-"
        book_total = g["total_line"] if g["total_line"] not in ("", "NA") else "-"
        print(f"{game:<14}{g['weekday'][:3] + ' ' + g['gameday'][5:]:<11}{home + ' ' + fmt_line(half_point(ap - hp)):>16}"
              f"{book_spread:>16}{half_point(hp + ap):>13g}{book_total:>12}")
    print("\n* neutral site. Spreads are the home team's line. "
          "For a full breakdown of one game, run: python3 football_odds.py game AWAY HOME")


def cmd_game(args):
    away, home = find_team(args.away_team), find_team(args.home_team)
    if away == home:
        raise SystemExit("Pick two different teams.")
    r = Ratings(load_games(args.data))
    listed = r.find_upcoming(home, away)
    neutral = args.neutral or bool(listed and listed["location"] == "Neutral")
    home_pts, away_pts = r.project(home, away, neutral)
    if args.home_pts is not None:
        home_pts = args.home_pts
    if args.away_pts is not None:
        away_pts = args.away_pts

    spread, total, ml, spread_odds, total_odds = args.spread, args.total, args.ml, args.spread_odds, args.total_odds
    if listed and not args.no_book:
        def book(key):
            return signed(listed[key]) if listed[key] not in ("", "NA") else None
        if spread is None and book("spread_line"):
            spread = -float(listed["spread_line"])   # nflverse's line is positive when home is favoured
        if total is None and book("total_line"):
            total = float(listed["total_line"])
        ml = ml or (book("home_moneyline"), book("away_moneyline"))
        spread_odds = spread_odds or (book("home_spread_odds"), book("away_spread_odds"))
        total_odds = total_odds or (book("over_odds"), book("under_odds"))

    print(f"{TEAMS[away]} @ {TEAMS[home]}" + (" (neutral site)" if neutral else ""))
    print(r.describe())
    for t in (away, home):
        print(f"  {nickname(t)}: offense {r.off.get(t, 0):+.1f}, defense {r.dfn.get(t, 0):+.1f} points vs average")
    if listed and not args.no_book:
        print(f"Sportsbook lines loaded for {listed['weekday']} {listed['gameday']}.")
    print()
    print_prediction(nickname(home), nickname(away), home_pts, away_pts, spread, total,
                     ml, spread_odds, total_odds, args.margin_sd, args.total_sd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-color", action="store_true", help="turn off coloured output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert odds between American, decimal and fractional")
    p.add_argument("odds", help="e.g. -110, +150, 1.91, 10/11")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("vig", help="bookmaker vig and fair odds for a market")
    p.add_argument("odds", nargs="+", help="odds for every side, e.g. -110 -110")
    p.set_defaults(func=cmd_vig)

    def add_lines(p):
        p.add_argument("--spread", type=float, help="home team's spread, e.g. -3.5 when favoured")
        p.add_argument("--total", type=float, help="total points line")
        p.add_argument("--ml", nargs=2, metavar=("HOME", "AWAY"), help="bookmaker moneyline odds")
        p.add_argument("--spread-odds", nargs=2, metavar=("HOME", "AWAY"), help="bookmaker spread odds, e.g. -110 -110")
        p.add_argument("--total-odds", nargs=2, metavar=("OVER", "UNDER"), help="bookmaker total odds, e.g. -110 -110")
        p.add_argument("--margin-sd", type=float, default=MARGIN_SD, help=f"margin standard deviation (default {MARGIN_SD:g})")
        p.add_argument("--total-sd", type=float, default=TOTAL_SD, help=f"total standard deviation (default {TOTAL_SD:g})")

    def add_data(p):
        p.add_argument("--data", help="use a local copy of nflverse's games.csv instead of downloading it")

    p = sub.add_parser("predict", help="moneyline, spread and total from projected points you supply")
    p.add_argument("home_pts", type=float, help="projected points for the home team")
    p.add_argument("away_pts", type=float, help="projected points for the away team")
    p.add_argument("--home", default="Home", help="home team name")
    p.add_argument("--away", default="Away", help="away team name")
    add_lines(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("teams", help="offense and defense ratings for every team")
    add_data(p)
    p.set_defaults(func=cmd_teams)

    p = sub.add_parser("week", help="this week's games: model lines next to sportsbook lines")
    add_data(p)
    p.set_defaults(func=cmd_week)

    p = sub.add_parser("game", help="full prediction for one matchup, using team ratings")
    p.add_argument("away_team", help="away team, e.g. BUF or Bills")
    p.add_argument("home_team", help="home team, e.g. KC or Chiefs")
    p.add_argument("--neutral", action="store_true", help="neutral-site game (no home-field edge)")
    p.add_argument("--home-pts", type=float, help="override the home team's projected points")
    p.add_argument("--away-pts", type=float, help="override the away team's projected points")
    p.add_argument("--no-book", action="store_true", help="don't load sportsbook lines for this week's games")
    add_lines(p)
    add_data(p)
    p.set_defaults(func=cmd_game)

    args = parser.parse_args()
    if args.no_color:
        Colour.enabled = False
    args.func(args)


if __name__ == "__main__":
    main()
