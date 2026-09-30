#!/usr/bin/env python3
"""Football odds toolkit: format conversion, bookmaker margin, and a Poisson match model.

Standard library only. Examples:

    python3 football_odds.py convert 2.50
    python3 football_odds.py convert 6/4
    python3 football_odds.py convert +150
    python3 football_odds.py margin 2.10 3.40 3.60
    python3 football_odds.py predict 1.6 1.1
    python3 football_odds.py predict 1.6 1.1 --odds 2.10 3.40 3.60
"""

import argparse
import math
from fractions import Fraction


# ---------------------------------------------------------------- conversion

def to_decimal(odds: str) -> float:
    """Parse decimal (2.5), fractional (6/4) or American (+150 / -200) odds."""
    s = odds.strip()
    if "/" in s:
        num, den = s.split("/")
        return 1 + float(num) / float(den)
    if s.startswith(("+", "-")):
        american = float(s)
        if american > 0:
            return 1 + american / 100
        return 1 + 100 / abs(american)
    value = float(s)
    if value <= 1:
        raise ValueError(f"decimal odds must be greater than 1, got {odds}")
    return value


def to_fractional(decimal: float) -> str:
    frac = Fraction(decimal - 1).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def to_american(decimal: float) -> str:
    if decimal >= 2:
        return f"+{round((decimal - 1) * 100)}"
    return f"-{round(100 / (decimal - 1))}"


def implied_probability(decimal: float) -> float:
    return 1 / decimal


# -------------------------------------------------------------------- margin

def remove_margin(decimals: list[float]) -> tuple[float, list[float]]:
    """Return the overround and the fair (margin-free) probabilities.

    Uses the proportional method: each implied probability is scaled so they sum to 1.
    """
    implied = [implied_probability(d) for d in decimals]
    book = sum(implied)
    return book - 1, [p / book for p in implied]


# ------------------------------------------------------------- Poisson model

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def score_matrix(home_xg: float, away_xg: float, max_goals: int = 10) -> list[list[float]]:
    """P(home scores i, away scores j), assuming independent Poisson goal counts."""
    home = [poisson_pmf(i, home_xg) for i in range(max_goals + 1)]
    away = [poisson_pmf(j, away_xg) for j in range(max_goals + 1)]
    return [[h * a for a in away] for h in home]


def market_probabilities(matrix: list[list[float]], line: float = 2.5) -> dict[str, float]:
    probs = {"home": 0.0, "draw": 0.0, "away": 0.0, "over": 0.0, "under": 0.0, "btts": 0.0}
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                probs["home"] += p
            elif i == j:
                probs["draw"] += p
            else:
                probs["away"] += p
            probs["over" if i + j > line else "under"] += p
            if i > 0 and j > 0:
                probs["btts"] += p
    return probs


def top_scores(matrix: list[list[float]], n: int = 5) -> list[tuple[str, float]]:
    scores = [(f"{i}-{j}", p) for i, row in enumerate(matrix) for j, p in enumerate(row)]
    return sorted(scores, key=lambda s: s[1], reverse=True)[:n]


def kelly_fraction(prob: float, decimal: float) -> float:
    """Kelly stake as a fraction of bankroll; 0 when there is no edge."""
    b = decimal - 1
    return max(0.0, (b * prob - (1 - prob)) / b)


# ----------------------------------------------------------------------- CLI

def fair_odds(prob: float) -> str:
    return f"{1 / prob:.2f}" if prob > 0 else "inf"


def cmd_convert(args):
    d = to_decimal(args.odds)
    print(f"Decimal:     {d:.3f}")
    print(f"Fractional:  {to_fractional(d)}")
    print(f"American:    {to_american(d)}")
    print(f"Implied %:   {implied_probability(d):.2%}")


def cmd_margin(args):
    decimals = [to_decimal(o) for o in args.odds]
    overround, fair = remove_margin(decimals)
    labels = ["Home", "Draw", "Away"] if len(decimals) == 3 else [f"#{i + 1}" for i in range(len(decimals))]
    print(f"Bookmaker margin: {overround:.2%}\n")
    print(f"{'Outcome':<8}{'Odds':>8}{'Implied':>10}{'Fair %':>10}{'Fair odds':>11}")
    for label, d, p in zip(labels, decimals, fair):
        print(f"{label:<8}{d:>8.2f}{implied_probability(d):>10.2%}{p:>10.2%}{fair_odds(p):>11}")


def cmd_predict(args):
    matrix = score_matrix(args.home_xg, args.away_xg)
    probs = market_probabilities(matrix, args.line)

    print(f"Expected goals: home {args.home_xg}, away {args.away_xg}\n")
    print(f"{'Market':<14}{'Prob':>8}{'Fair odds':>11}")
    rows = [
        ("Home win", probs["home"]),
        ("Draw", probs["draw"]),
        ("Away win", probs["away"]),
        (f"Over {args.line}", probs["over"]),
        (f"Under {args.line}", probs["under"]),
        ("BTTS yes", probs["btts"]),
        ("BTTS no", 1 - probs["btts"]),
    ]
    for name, p in rows:
        print(f"{name:<14}{p:>8.2%}{fair_odds(p):>11}")

    print("\nMost likely scores:")
    for score, p in top_scores(matrix):
        print(f"  {score:<6}{p:>7.2%}")

    if args.odds:
        print("\nValue vs bookmaker (1X2):")
        print(f"{'Outcome':<10}{'Odds':>7}{'Model':>9}{'Edge':>9}{'Kelly':>8}")
        for name, key, raw in zip(["Home", "Draw", "Away"], ["home", "draw", "away"], args.odds):
            d = to_decimal(raw)
            p = probs[key]
            edge = p * d - 1
            print(f"{name:<10}{d:>7.2f}{p:>9.2%}{edge:>+9.2%}{kelly_fraction(p, d):>8.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert odds between decimal, fractional and American")
    p.add_argument("odds", help="e.g. 2.50, 6/4, +150, -200")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("margin", help="bookmaker margin and fair odds for a market")
    p.add_argument("odds", nargs="+", help="odds for every outcome, e.g. 2.10 3.40 3.60")
    p.set_defaults(func=cmd_margin)

    p = sub.add_parser("predict", help="Poisson model from expected goals")
    p.add_argument("home_xg", type=float)
    p.add_argument("away_xg", type=float)
    p.add_argument("--line", type=float, default=2.5, help="over/under goals line (default 2.5)")
    p.add_argument("--odds", nargs=3, metavar=("HOME", "DRAW", "AWAY"), help="bookmaker 1X2 odds to compare")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
