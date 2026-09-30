#!/usr/bin/env python3
"""NFL odds toolkit: odds conversion, bookmaker vig, and a projected-points model.

Standard library only. Examples:

    python3 football_odds.py convert -110
    python3 football_odds.py convert +150
    python3 football_odds.py vig -110 -110
    python3 football_odds.py predict 24.5 21.5
    python3 football_odds.py predict 24.5 21.5 --spread -3.5 --total 44.5 \\
        --ml -165 +140 --spread-odds -110 -110 --total-odds -110 -110
"""

import argparse
import math
from fractions import Fraction

# NFL standard deviations, in points, of the final margin and of the total around
# closing betting lines, 2023-2025 seasons. Both can be overridden on the command line.
MARGIN_SD = 13.0
TOTAL_SD = 13.0


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


# --------------------------------------------------------------------- model

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


# ----------------------------------------------------------------------- CLI

def half_point(x: float) -> float:
    return round(x * 2) / 2


def fmt_line(x: float) -> str:
    return f"{x:+g}" if x else "PK"


def cmd_convert(args):
    d = to_decimal(args.odds)
    print(f"American:    {to_american(d)}")
    print(f"Decimal:     {d:.3f}")
    print(f"Fractional:  {to_fractional(d)}")
    print(f"Implied %:   {1 / d:.2%}")


def cmd_vig(args):
    decimals = [to_decimal(o) for o in args.odds]
    vig, fair = remove_vig(decimals)
    labels = ["Side 1", "Side 2"] if len(decimals) == 2 else [f"Side {i + 1}" for i in range(len(decimals))]
    print(f"Bookmaker vig: {vig:.2%}\n")
    print(f"{'Outcome':<9}{'Odds':>7}{'Implied':>10}{'Fair %':>9}{'Fair odds':>11}")
    for label, raw, d, p in zip(labels, args.odds, decimals, fair):
        print(f"{label:<9}{to_american(d):>7}{1 / d:>10.2%}{p:>9.2%}{fair_american(p):>11}")


def cmd_predict(args):
    home, away = args.home, args.away
    margin = margin_distribution(args.home_pts, args.away_pts, args.margin_sd)
    total = total_distribution(args.home_pts, args.away_pts, args.total_sd)

    proj_margin = args.home_pts - args.away_pts
    proj_total = args.home_pts + args.away_pts
    spread = args.spread if args.spread is not None else half_point(-proj_margin)
    total_line = args.total if args.total is not None else half_point(proj_total)

    print(f"{home} {args.home_pts:g} - {args.away_pts:g} {away}")
    print(f"Fair spread: {home} {fmt_line(-proj_margin)} · fair total: {proj_total:g}\n")

    # Home covers a spread of L when margin + L > 0, i.e. margin > -L.
    home_ml, _, away_ml = line_probabilities(margin, 0)
    cover, spread_push, no_cover = line_probabilities(margin, -spread)
    over, total_push, under = line_probabilities(total, total_line)

    rows = [
        (f"{home} moneyline", home_ml, 0.0, away_ml, args.ml, 0),
        (f"{away} moneyline", away_ml, 0.0, home_ml, args.ml, 1),
        (f"{home} {fmt_line(spread)}", cover, spread_push, no_cover, args.spread_odds, 0),
        (f"{away} {fmt_line(-spread)}", no_cover, spread_push, cover, args.spread_odds, 1),
        (f"Over {total_line:g}", over, total_push, under, args.total_odds, 0),
        (f"Under {total_line:g}", under, total_push, over, args.total_odds, 1),
    ]

    has_book = any(r[4] for r in rows)
    header = f"{'Market':<22}{'Win':>8}{'Push':>7}{'Fair':>7}"
    if has_book:
        header += f"{'Book':>7}{'Edge':>9}{'Kelly':>8}"
    print(header)
    for name, win, push, loss, book, idx in rows:
        # Fair odds ignore pushes, since a push returns the stake.
        line = f"{name:<22}{win:>8.1%}{(f'{push:.1%}' if push else '-'):>7}{fair_american(win / (win + loss)):>7}"
        if book:
            d = to_decimal(book[idx])
            edge = win * (d - 1) - loss
            line += f"{to_american(d):>7}{edge:>+9.1%}{kelly_fraction(win, loss, d):>8.1%}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert odds between American, decimal and fractional")
    p.add_argument("odds", help="e.g. -110, +150, 1.91, 10/11")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("vig", help="bookmaker vig and fair odds for a market")
    p.add_argument("odds", nargs="+", help="odds for every side, e.g. -110 -110")
    p.set_defaults(func=cmd_vig)

    p = sub.add_parser("predict", help="moneyline, spread and total from projected points")
    p.add_argument("home_pts", type=float, help="projected points for the home team")
    p.add_argument("away_pts", type=float, help="projected points for the away team")
    p.add_argument("--home", default="Home", help="home team name")
    p.add_argument("--away", default="Away", help="away team name")
    p.add_argument("--spread", type=float, help="home team's spread, e.g. -3.5 when favoured (default: fair line)")
    p.add_argument("--total", type=float, help="total points line (default: fair line)")
    p.add_argument("--ml", nargs=2, metavar=("HOME", "AWAY"), help="bookmaker moneyline odds")
    p.add_argument("--spread-odds", nargs=2, metavar=("HOME", "AWAY"), help="bookmaker spread odds, e.g. -110 -110")
    p.add_argument("--total-odds", nargs=2, metavar=("OVER", "UNDER"), help="bookmaker total odds, e.g. -110 -110")
    p.add_argument("--margin-sd", type=float, default=MARGIN_SD, help=f"margin standard deviation (default {MARGIN_SD:g})")
    p.add_argument("--total-sd", type=float, default=TOTAL_SD, help=f"total standard deviation (default {TOTAL_SD:g})")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
