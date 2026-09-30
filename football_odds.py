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
    python3 football_odds.py props mahomes
    python3 football_odds.py props "josh allen" pass_yds 249.5 --odds -115 -105
"""

import argparse
import csv
import io
import json
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
TEAM_SD = 9.0        # one team's points around its projection

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


# ---------------------------------------------------------------- prop model

def negbin_pmf(k: int, mean: float, dispersion: float) -> float:
    """P(X = k) for a count with the given mean and variance/mean ratio.
    A ratio of 1 or less is treated as Poisson."""
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    if dispersion <= 1.0001:
        return math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1))
    r = mean / (dispersion - 1)
    p = r / (r + mean)
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) + r * math.log(p) + k * math.log(1 - p))


def prop_probabilities(kind: str, mean: float, spread: float | None, line: float) -> tuple[float, float, float]:
    """(P(over), P(push), P(under)) for a player prop.

    kind "yards" uses a lognormal curve with coefficient of variation `spread`,
    "count" a negative binomial with variance/mean `spread`, and "td" a Poisson.
    Results are whole numbers, so only a whole-number line can push.
    """
    whole = line == int(line)
    if kind == "yards":
        if mean <= 0:
            return 0.0, 0.0, 1.0
        s2 = math.log(1 + spread ** 2)
        mu, sd = math.log(mean) - s2 / 2, math.sqrt(s2)

        def above(x):   # P(result > x) on the continuous curve
            return 1.0 if x <= 0 else 1 - normal_cdf(math.log(x), mu, sd)
        over = above(math.floor(line) + 0.5)
        push = above(line - 0.5) - over if whole else 0.0
        return over, push, 1 - over - push
    dispersion = 1.0 if kind == "td" else spread
    below = sum(negbin_pmf(k, mean, dispersion) for k in range(0, math.floor(line) + (0 if whole else 1)))
    push = negbin_pmf(int(line), mean, dispersion) if whole else 0.0
    return 1 - below - push, push, below


def fair_prop_line(kind: str, mean: float, spread: float | None) -> float:
    """The half-point line closest to a 50/50 split."""
    if kind == "yards":
        s2 = math.log(1 + spread ** 2)
        return math.floor(math.exp(math.log(max(mean, 0.01)) - s2 / 2)) + 0.5
    best, gap = 0.5, 1.0
    for n in range(400):
        g = abs(prop_probabilities(kind, mean, spread, n + 0.5)[0] - 0.5)
        if g >= gap:
            break
        best, gap = n + 0.5, g
    return best


def kelly_text(win: float, loss: float, decimal: float, rating: str) -> str:
    """Kelly stake, shown only for good-price bets: for thin or suspiciously big
    edges it can suggest reckless stakes, especially on heavy favourites."""
    return f"{kelly_fraction(win, loss, decimal):.1%}" if rating == "good" else "-"


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


# -------------------------------------------------------------- player props

PLAYERS_URL = "https://evandergee.github.io/football-odds/data/players.json"

# Props in the order each position is usually bet, and the minimum average a
# player needs for a prop to be shown (a receiver's rushing yards usually aren't).
PROP_ORDER = {
    "QB": ["pass_yds", "pass_td", "pass_cmp", "pass_att", "pass_int", "rush_yds", "rush_att", "anytime_td"],
    "RB": ["rush_yds", "rush_att", "rush_rec_yds", "rec_yds", "rec", "anytime_td"],
    "WR": ["rec_yds", "rec", "rush_rec_yds", "rush_yds", "anytime_td"],
    "TE": ["rec_yds", "rec", "rush_rec_yds", "anytime_td"],
}
MIN_AVERAGE = {"rush_yds": 5, "rush_att": 1, "rec_yds": 5, "rec": 0.5}
PROP_ALIASES = {
    "passing-yards": "pass_yds", "pass-yards": "pass_yds", "passing-tds": "pass_td", "pass-tds": "pass_td",
    "completions": "pass_cmp", "attempts": "pass_att", "pass-attempts": "pass_att", "interceptions": "pass_int",
    "ints": "pass_int", "rushing-yards": "rush_yds", "rush-yards": "rush_yds", "carries": "rush_att",
    "rush-attempts": "rush_att", "receiving-yards": "rec_yds", "rec-yards": "rec_yds", "receptions": "rec",
    "catches": "rec", "rush-rec-yards": "rush_rec_yds", "scrimmage-yards": "rush_rec_yds", "td": "anytime_td",
    "anytime-td": "anytime_td", "touchdown": "anytime_td",
}


def load_players(path: str | None) -> dict:
    """Player data: a file given with --players, else the published copy (updated
    several times a day), else the copy in this folder."""
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(fetch_text(PLAYERS_URL))
    except SystemExit:
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "players.json")
        if not os.path.exists(local):
            raise
        print(Colour.wrap("Couldn't download the latest player data, so using the local copy.", "caution"))
        with open(local, encoding="utf-8") as f:
            return json.load(f)


def find_player(players: list[dict], text: str) -> dict:
    t = text.strip().lower()
    exact = [p for p in players if p["name"].lower() == t]
    matches = exact or [p for p in players if t in p["name"].lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No player matching '{text}'. Try part of the name, like 'mahomes'.")
    listing = "\n".join(f"  {p['name']} ({p['pos']}, {p['team']})" for p in matches[:15])
    raise SystemExit(f"More than one player matches '{text}':\n{listing}\nUse more of the name.")


def props_for(p: dict) -> list[str]:
    return [s for s in PROP_ORDER[p["pos"]] if s in p["avg"] and p["avg"][s] >= MIN_AVERAGE.get(s, 0)]


def project_prop(p: dict, stat: str, pdata: dict, ratings: "Ratings", game: dict | None) -> tuple[float, str]:
    """(projected mean, explanation). Passing props are adjusted for the opponent's
    pass defense; touchdown props for the team's projected points."""
    base = p["avg"][stat]
    if not game:
        return base, ""
    home, away = game["home_team"], game["away_team"]
    opp = away if p["team"] == home else home
    if pdata["stats"][stat]["kind"] == "td":
        ppg = pdata["team_ppg"].get(p["team"])
        if not ppg:
            return base, ""
        hp, ap = ratings.project(home, away, game["location"] == "Neutral")
        proj = hp if p["team"] == home else ap
        return base * proj / ppg, (f"{nickname(p['team'])} projected {proj:.1f} points vs their {ppg:.1f} average, "
                                   f"so touchdown chances are scaled by {proj / ppg - 1:+.0%}.")
    f = pdata["defense"].get(opp, {}).get(stat, {}).get(p["pos"])
    if not f:
        return base, ""
    return base * f, f"{TEAMS.get(opp, opp)} allow {f - 1:+.0%} {pdata['stats'][stat]['label'].lower()} vs an average defense."


def player_warnings(p: dict, pdata: dict) -> list[tuple[str, str]]:
    injury = f" ({p['injury'].lower()})" if p.get("injury") else ""
    w = []
    if p["status"] == "IR":
        w.append(("bad", f"{p['name']} is on injured reserve and won't play."))
    elif p["status"] == "Out":
        w.append(("bad", f"{p['name']} is listed as Out{injury} and won't play."))
    elif p["status"] == "Doubtful":
        w.append(("bad", f"{p['name']} is listed as Doubtful{injury} and probably won't play."))
    elif p["status"] == "Questionable":
        w.append(("caution", f"{p['name']} is listed as Questionable{injury}."))
    elif p["status"]:
        w.append(("caution", f"{p['name']}: {p['status'].lower()} this week{injury}. Game status usually comes out Friday."))
    if p["team"] not in pdata["injury_teams"]:
        w.append(("dim", f"The {nickname(p['team'])} haven't posted this week's injury report yet (Wednesday to Friday)."))
    if p["games_this_season"] < 2:
        w.append(("caution", f"{p['name']} has {p['games_this_season']} game(s) this season, so the projection leans on older games."))
    return w


def cmd_props(args):
    pdata = load_players(args.players)
    p = find_player(pdata["players"], args.player)
    r = Ratings(load_games(args.data))
    game = next((g for g in r.upcoming if p["team"] in (g["home_team"], g["away_team"])), None)
    opp = (game["away_team"] if game["home_team"] == p["team"] else game["home_team"]) if game else None

    print(f"{p['name']}, {p['pos']}, {TEAMS.get(p['team'], p['team'])}"
          + (f" vs {TEAMS.get(opp, opp)} ({game['weekday']} {game['gameday']})" if game else " (no game found this week)"))
    print(f"Stats through week {pdata['stats_through_week']} of {pdata['season']}; injury report for week "
          f"{pdata['injury_week']}; updated {pdata['updated']}.")
    for kind, text in player_warnings(p, pdata):
        print(Colour.wrap(text, kind))
    recent = ", ".join(f"wk {g['week']} {g['opp']}" for g in p["recent"])
    print(f"Last {len(p['recent'])} games: {recent}\n")

    if not args.prop:
        print(f"{'Prop':<18}{'Projection':>11}{'Fair line':>11}{'Over at fair':>14}{'Last 5':>24}")
        for s in props_for(p):
            info = pdata["stats"][s]
            mean, _ = project_prop(p, s, pdata, r, game)
            spread = pdata["spread"].get(s, {}).get(p["pos"])
            last5 = " ".join(f"{g[s]:g}" for g in p["recent"])
            if s == "anytime_td":
                yes = prop_probabilities("td", mean, None, 0.5)[0]
                print(f"{info['label']:<18}{mean:>11.2f}{'-':>11}{f'{yes:.1%} {fair_american(yes)}':>14}{last5:>24}")
                continue
            line = fair_prop_line(info["kind"], mean, spread)
            over, _, under = prop_probabilities(info["kind"], mean, spread, line)
            print(f"{info['label']:<18}{mean:>11.1f}{line:>11g}{over:>14.1%}{last5:>24}")
        print("\nFor one prop against a sportsbook's line, run for example:\n"
              f"  python3 football_odds.py props \"{p['name']}\" {props_for(p)[0]} LINE --odds -115 -105")
        return

    stat = PROP_ALIASES.get(args.prop.lower(), args.prop.lower())
    if stat not in p["avg"]:
        raise SystemExit(f"'{args.prop}' isn't a prop for {p['name']}. Options: {', '.join(props_for(p))}")
    info = pdata["stats"][stat]
    spread = pdata["spread"].get(stat, {}).get(p["pos"])
    mean, why = project_prop(p, stat, pdata, r, game)
    is_td = stat == "anytime_td"
    line = 0.5 if is_td else args.line
    if line is None:
        line = fair_prop_line(info["kind"], mean, spread)
        print(f"No line given, so using the model's fair line of {line:g}.")
    over, push, under = prop_probabilities(info["kind"], mean, spread, line)
    print(f"Projected {'touchdowns' if is_td else info['label'].lower()}: {mean:.2f}. Weighted average {p['avg'][stat]:.2f} over the last "
          f"{p['games']} games. {why}".rstrip())
    print(f"Last {len(p['recent'])} games: " + ", ".join(f"{g[stat]:g}" for g in p["recent"]) + "\n")

    names = ("Scores a TD", "No TD") if is_td else (f"Over {line:g}", f"Under {line:g}")
    header = f"{'Bet':<16}{'Win':>8}{'Push':>7}{'Fair':>7}"
    if args.odds:
        header += f"{'Book':>7}{'Edge':>9}{'Kelly':>8}  Rating"
    print(header)
    for i, (name, win, loss) in enumerate(((names[0], over, under), (names[1], under, over))):
        row = f"{name:<16}{win:>8.1%}{(f'{push:.1%}' if push else '-'):>7}{fair_american(win / (win + loss)):>7}"
        if args.odds:
            d = to_decimal(args.odds[i])
            edge = win * (d - 1) - loss
            rating = edge_rating(edge)
            row = Colour.wrap(row + f"{to_american(d):>7}{edge:>+9.1%}{kelly_text(win, loss, d, rating):>8}  {rating}", rating)
        print(row)
    print(Colour.wrap("\nPlayer props are hard to predict: in testing this model beat a plain season average by "
                      "only 1-3%, so treat big edges with suspicion and check the news.", "dim"))


# ----------------------------------------------------------------------- CLI

def half_point(x: float) -> float:
    return round(x * 2) / 2 + 0.0   # + 0.0 turns -0.0 into 0.0


def fmt_line(x: float) -> str:
    return f"{x:+g}" if x else "PK"


def signed(value: str) -> str:
    return f"+{value}" if value and not value.startswith("-") else value


def print_prediction(home: str, away: str, home_pts: float, away_pts: float, spread: float | None,
                     total_line: float | None, ml=None, spread_odds=None, total_odds=None,
                     margin_sd: float = MARGIN_SD, total_sd: float = TOTAL_SD, alt: bool = False):
    """Print moneyline, spread, total and team total probabilities, plus edges against
    any bookmaker odds given, and alternate lines when alt is set. ml and spread_odds
    are (home, away) pairs; total_odds is (over, under)."""
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
            line += f"{to_american(d):>7}{edge:>+9.1%}{kelly_text(win, loss, d, rating):>8}  {rating}"
            line = Colour.wrap(line, rating)
        print(line)

    print(f"\n{'Team total (fair line)':<22}{'Over':>8}{'Push':>7}{'Fair':>7}")
    for name, pts in ((away, away_pts), (home, home_pts)):
        line = half_point(pts)
        o, pu, u = line_probabilities(discrete_normal(pts, TEAM_SD, 0, 100), line)
        print(f"{name + ' ' + format(line, 'g'):<22}{o:>8.1%}{(f'{pu:.1%}' if pu else '-'):>7}{fair_american(o / (o + u)):>7}")

    if alt:
        print(f"\nAlternate spreads (fair odds)\n{home + ' line':<14}{'Covers':>8}{'Fair':>7}   {away + ' line':<14}{'Covers':>8}{'Fair':>7}")
        for off in range(-7, 8):
            L = spread + off
            c, _, n = line_probabilities(margin, -L)
            mark = " <" if off == 0 else ""
            print(f"{home + ' ' + fmt_line(L):<14}{c:>8.1%}{fair_american(c / (c + n)):>7}   "
                  f"{away + ' ' + fmt_line(-L):<14}{n:>8.1%}{fair_american(n / (c + n)):>7}{mark}")
        print(f"\nAlternate totals (fair odds)\n{'Line':<8}{'Over':>8}{'Fair':>7}{'Under':>8}{'Fair':>7}")
        for off in range(-7, 8):
            L = total_line + off
            if L <= 0:
                continue
            o, _, u = line_probabilities(total, L)
            print(f"{L:<8g}{o:>8.1%}{fair_american(o / (o + u)):>7}{u:>8.1%}{fair_american(u / (o + u)):>7}{' <' if off == 0 else ''}")

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
                     args.ml, args.spread_odds, args.total_odds, args.margin_sd, args.total_sd, args.alt)


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
                     ml, spread_odds, total_odds, args.margin_sd, args.total_sd, args.alt)


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
        p.add_argument("--alt", action="store_true", help="also show alternate spreads and totals")

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

    p = sub.add_parser("props", help="player props: all of a player's props, or one against a line")
    p.add_argument("player", help="player name or part of it, e.g. mahomes")
    p.add_argument("prop", nargs="?", help="prop, e.g. pass_yds, receptions, rush-yards, td")
    p.add_argument("line", nargs="?", type=float, help="the sportsbook's line, e.g. 249.5 (default: fair line)")
    p.add_argument("--odds", nargs=2, metavar=("OVER", "UNDER"), help="sportsbook odds for over and under (yes and no for td)")
    p.add_argument("--players", help="use a local players.json instead of downloading it")
    add_data(p)
    p.set_defaults(func=cmd_props)

    args = parser.parse_args()
    if args.no_color:
        Colour.enabled = False
    args.func(args)


if __name__ == "__main__":
    main()
