"""Tests for football_odds.py. Standard library only:

    python3 -m unittest discover tests
"""

import io
import math
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import football_odds as fo  # noqa: E402

MINUS_110 = 1 + 100 / 110


# ---------------------------------------------------------------- conversion

class ToDecimalTest(unittest.TestCase):
    def test_american(self):
        self.assertAlmostEqual(fo.to_decimal("-110"), MINUS_110)
        self.assertAlmostEqual(fo.to_decimal("+150"), 2.5)
        self.assertAlmostEqual(fo.to_decimal("-200"), 1.5)
        self.assertAlmostEqual(fo.to_decimal("+100"), 2.0)

    def test_unsigned_american(self):
        self.assertAlmostEqual(fo.to_decimal("150"), 2.5)

    def test_decimal(self):
        self.assertAlmostEqual(fo.to_decimal("1.91"), 1.91)
        self.assertAlmostEqual(fo.to_decimal(" 2.5 "), 2.5)

    def test_fractional(self):
        self.assertAlmostEqual(fo.to_decimal("10/11"), MINUS_110)
        self.assertAlmostEqual(fo.to_decimal("6/4"), 2.5)

    def test_rejects_bad_odds(self):
        for odds in ("-50", "+99", "1", "0.5"):
            with self.subTest(odds=odds), self.assertRaises(ValueError):
                fo.to_decimal(odds)


class ToAmericanTest(unittest.TestCase):
    def test_round_trip(self):
        for odds in ("-110", "+150", "-200", "+100", "-1000", "+800"):
            with self.subTest(odds=odds):
                self.assertEqual(fo.to_american(fo.to_decimal(odds)), odds)

    def test_even_money_is_plus_100(self):
        self.assertEqual(fo.to_american(2.0), "+100")
        self.assertEqual(fo.to_american(1.999), "+100")

    def test_fractional(self):
        self.assertEqual(fo.to_fractional(2.5), "3/2")
        self.assertEqual(fo.to_fractional(MINUS_110), "10/11")

    def test_fair_american(self):
        self.assertEqual(fo.fair_american(0.5), "+100")
        self.assertEqual(fo.fair_american(0.6), "-150")
        self.assertEqual(fo.fair_american(0.4), "+150")
        self.assertEqual(fo.fair_american(0), "-")
        self.assertEqual(fo.fair_american(1), "-")


# ----------------------------------------------------------------------- vig

class RemoveVigTest(unittest.TestCase):
    def test_standard_market(self):
        vig, fair = fo.remove_vig([MINUS_110, MINUS_110])
        self.assertAlmostEqual(vig, 0.0476, places=4)   # the README's 4.76%
        self.assertAlmostEqual(fair[0], 0.5)
        self.assertAlmostEqual(fair[1], 0.5)

    def test_fair_probabilities_sum_to_one(self):
        _, fair = fo.remove_vig([fo.to_decimal("-165"), fo.to_decimal("+140")])
        self.assertAlmostEqual(sum(fair), 1.0)
        self.assertGreater(fair[0], fair[1])

    def test_no_vig_market(self):
        vig, fair = fo.remove_vig([2.0, 2.0])
        self.assertAlmostEqual(vig, 0.0)
        self.assertEqual(fair, [0.5, 0.5])


class EdgeRatingTest(unittest.TestCase):
    def test_bands(self):
        cases = {-0.01: "bad", 0.0: "caution", 0.02: "caution", 0.03: "good",
                 0.07: "good", 0.10: "good", 0.11: "caution"}
        for edge, rating in cases.items():
            with self.subTest(edge=edge):
                self.assertEqual(fo.edge_rating(edge), rating)


class KellyTest(unittest.TestCase):
    def test_positive_edge(self):
        self.assertAlmostEqual(fo.kelly_fraction(0.55, 0.45, 2.0), 0.10)

    def test_no_edge_is_zero(self):
        self.assertEqual(fo.kelly_fraction(0.5, 0.5, 2.0), 0.0)
        self.assertEqual(fo.kelly_fraction(0.4, 0.6, 2.0), 0.0)
        self.assertEqual(fo.kelly_fraction(0.0, 0.0, 2.0), 0.0)

    def test_pushes_are_left_out(self):
        # A 10% push chance shouldn't change the stake: 0.495 vs 0.405 is the same split as 0.55 vs 0.45.
        self.assertAlmostEqual(fo.kelly_fraction(0.495, 0.405, 2.0), 0.10)

    def test_text_only_for_good_bets(self):
        self.assertEqual(fo.kelly_text(0.55, 0.45, 2.0, "good"), "10.0%")
        self.assertEqual(fo.kelly_text(0.55, 0.45, 2.0, "caution"), "-")
        self.assertEqual(fo.kelly_text(0.55, 0.45, 2.0, "bad"), "-")


# --------------------------------------------------------------- game model

class GameModelTest(unittest.TestCase):
    def test_normal_cdf(self):
        self.assertAlmostEqual(fo.normal_cdf(0, 0, 1), 0.5)
        self.assertAlmostEqual(fo.normal_cdf(1.96, 0, 1), 0.975, places=3)

    def test_margin_has_no_ties(self):
        dist = fo.margin_distribution(24, 21)
        self.assertNotIn(0, dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_overtime_goes_to_three(self):
        plain = fo.discrete_normal(0, fo.MARGIN_SD, -100, 100)
        dist = fo.margin_distribution(20, 20)
        self.assertAlmostEqual(dist[3], plain[3] + plain[0] / 2)
        self.assertAlmostEqual(dist[-3], plain[-3] + plain[0] / 2)

    def test_even_game_is_a_coin_flip(self):
        home, push, away = fo.line_probabilities(fo.margin_distribution(20, 20), 0)
        self.assertAlmostEqual(home, 0.5)
        self.assertAlmostEqual(push, 0.0)
        self.assertAlmostEqual(away, 0.5)

    def test_favourite_wins_more(self):
        home, _, away = fo.line_probabilities(fo.margin_distribution(27, 20), 0)
        self.assertGreater(home, 0.65)
        self.assertAlmostEqual(home + away, 1.0)

    def test_whole_number_line_can_push(self):
        dist = fo.total_distribution(22, 22)
        over, push, under = fo.line_probabilities(dist, 44)
        self.assertGreater(push, 0.02)
        self.assertAlmostEqual(over, under)
        self.assertAlmostEqual(over + push + under, 1.0)

    def test_half_point_line_cannot_push(self):
        over, push, under = fo.line_probabilities(fo.total_distribution(22, 22), 44.5)
        self.assertEqual(push, 0.0)
        self.assertAlmostEqual(over + under, 1.0)

    def test_margin_sd_matches_constant(self):
        dist = fo.discrete_normal(0, fo.MARGIN_SD, -100, 100)
        variance = sum(k * k * p for k, p in dist.items())
        self.assertAlmostEqual(math.sqrt(variance), fo.MARGIN_SD, delta=0.1)


# ---------------------------------------------------------------- prop model

class PropModelTest(unittest.TestCase):
    def test_poisson_sums_to_one_with_right_mean(self):
        probs = [fo.negbin_pmf(k, 1.7, 1.0) for k in range(60)]
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertAlmostEqual(sum(k * p for k, p in enumerate(probs)), 1.7)

    def test_negative_binomial_sums_to_one_with_right_variance(self):
        mean, dispersion = 5.0, 1.8
        probs = [fo.negbin_pmf(k, mean, dispersion) for k in range(200)]
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertAlmostEqual(sum(k * p for k, p in enumerate(probs)), mean)
        variance = sum((k - mean) ** 2 * p for k, p in enumerate(probs))
        self.assertAlmostEqual(variance / mean, dispersion, places=4)

    def test_zero_mean(self):
        self.assertEqual(fo.negbin_pmf(0, 0, 1.5), 1.0)
        self.assertEqual(fo.negbin_pmf(2, 0, 1.5), 0.0)

    def test_anytime_td(self):
        yes, push, no = fo.prop_probabilities("td", 0.6, None, 0.5)
        self.assertAlmostEqual(no, math.exp(-0.6))
        self.assertAlmostEqual(yes, 1 - math.exp(-0.6))
        self.assertEqual(push, 0.0)

    def test_probabilities_sum_to_one(self):
        for kind, mean, spread in (("yards", 65.0, 0.6), ("count", 5.0, 1.4), ("td", 0.8, None)):
            for line in (0.5, 4.5, 5, 60, 64.5):
                with self.subTest(kind=kind, line=line):
                    over, push, under = fo.prop_probabilities(kind, mean, spread, line)
                    self.assertAlmostEqual(over + push + under, 1.0)
                    self.assertGreaterEqual(min(over, push, under), -1e-12)
                    if line != int(line):
                        self.assertEqual(push, 0.0)

    def test_yards_with_no_average(self):
        self.assertEqual(fo.prop_probabilities("yards", 0, 0.6, 9.5), (0.0, 0.0, 1.0))

    def test_yards_fair_line_sits_below_the_mean(self):
        # The lognormal is skewed right, so the 50/50 point is under the average.
        line = fo.fair_prop_line("yards", 70.0, 0.6)
        self.assertLess(line, 70.0)
        self.assertEqual(line % 1, 0.5)
        over, _, _ = fo.prop_probabilities("yards", 70.0, 0.6, line)
        self.assertAlmostEqual(over, 0.5, delta=0.02)

    def test_count_fair_line_is_closest_to_even(self):
        mean, spread = 5.3, 1.4
        line = fo.fair_prop_line("count", mean, spread)
        gap = abs(fo.prop_probabilities("count", mean, spread, line)[0] - 0.5)
        for other in (line - 1, line + 1):
            self.assertLessEqual(gap, abs(fo.prop_probabilities("count", mean, spread, other)[0] - 0.5))

    def test_higher_line_means_fewer_overs(self):
        low = fo.prop_probabilities("yards", 250.0, 0.3, 230.5)[0]
        high = fo.prop_probabilities("yards", 250.0, 0.3, 270.5)[0]
        self.assertGreater(low, high)


# ------------------------------------------------------------- team ratings

def game(season, week, home, away, home_score="NA", away_score="NA", location="Home"):
    return {"season": str(season), "week": str(week), "game_type": "REG", "location": location,
            "home_team": home, "away_team": away, "home_score": str(home_score), "away_score": str(away_score)}


def round_robin(season, scores):
    """Every team plays every other team home and away. `scores` maps team to points scored."""
    teams = list(scores)
    games, week = [], 1
    for home in teams:
        for away in teams:
            if home != away:
                games.append(game(season, week, home, away, scores[home], scores[away]))
                week += 1
    return games


class RatingsTest(unittest.TestCase):
    def test_equal_teams_are_rated_average(self):
        r = fo.Ratings(round_robin(2025, {"BUF": 20, "KC": 20, "MIA": 20, "NYJ": 20}))
        self.assertAlmostEqual(r.avg, 20.0)
        for team in ("BUF", "KC", "MIA", "NYJ"):
            self.assertAlmostEqual(r.off[team], 0.0, places=6)
            self.assertAlmostEqual(r.dfn[team], 0.0, places=6)

    def test_home_field(self):
        r = fo.Ratings(round_robin(2025, {"BUF": 20, "KC": 20, "MIA": 20, "NYJ": 20}))
        home, away = r.project("BUF", "KC")
        self.assertAlmostEqual(home - away, fo.HOME_FIELD)
        home, away = r.project("BUF", "KC", neutral=True)
        self.assertAlmostEqual(home, away)

    def test_high_scoring_team_has_better_offense(self):
        r = fo.Ratings(round_robin(2025, {"BUF": 30, "KC": 20, "MIA": 20, "NYJ": 20}))
        self.assertGreater(r.off["BUF"], 0)
        self.assertGreater(r.off["BUF"], r.off["KC"])
        home, away = r.project("BUF", "KC")
        self.assertGreater(home, away)

    def test_finds_season_and_next_week(self):
        games = round_robin(2024, {"BUF": 20, "KC": 20}) + round_robin(2025, {"BUF": 24, "KC": 17})
        games.append(game(2025, 3, "KC", "BUF"))
        r = fo.Ratings(games)
        self.assertEqual(r.season, 2025)
        self.assertEqual(r.games_this_season, 2)
        self.assertEqual(r.last_week, 2)
        self.assertEqual(r.next_week, 3)
        self.assertIsNotNone(r.find_upcoming("KC", "BUF"))
        self.assertIsNone(r.find_upcoming("BUF", "KC"))
        self.assertIn("2 games from 2025", r.describe())

    def test_shrinks_toward_average(self):
        # One blowout shouldn't be taken at face value.
        r = fo.Ratings([game(2025, 1, "BUF", "KC", 45, 10)])
        self.assertLess(r.off["BUF"] - r.off["KC"], 35 / 2)


# ------------------------------------------------------------ teams and players

class FindTeamTest(unittest.TestCase):
    def test_names(self):
        for text in ("KC", "kc", "Chiefs", "Kansas City", "kansas city chiefs", "  chiefs "):
            with self.subTest(text=text):
                self.assertEqual(fo.find_team(text), "KC")

    def test_unknown(self):
        with self.assertRaises(SystemExit):
            fo.find_team("Raptors")

    def test_every_team_finds_itself(self):
        for abbr, name in fo.TEAMS.items():
            with self.subTest(team=abbr):
                self.assertEqual(fo.find_team(abbr), abbr)
                self.assertEqual(fo.find_team(name), abbr)
                self.assertEqual(fo.find_team(fo.nickname(abbr)), abbr)


PLAYERS = [
    {"name": "Josh Allen", "pos": "QB", "team": "BUF"},
    {"name": "Josh Jacobs", "pos": "RB", "team": "GB"},
    {"name": "Patrick Mahomes", "pos": "QB", "team": "KC"},
]


class FindPlayerTest(unittest.TestCase):
    def test_exact_and_partial(self):
        self.assertEqual(fo.find_player(PLAYERS, "josh allen")["team"], "BUF")
        self.assertEqual(fo.find_player(PLAYERS, "mahomes")["team"], "KC")

    def test_ambiguous(self):
        with self.assertRaises(SystemExit) as ctx:
            fo.find_player(PLAYERS, "josh")
        self.assertIn("Josh Jacobs", str(ctx.exception))

    def test_missing(self):
        with self.assertRaises(SystemExit):
            fo.find_player(PLAYERS, "brady")

    def test_props_for_skips_tiny_averages(self):
        wr = {"pos": "WR", "avg": {"rec_yds": 60, "rec": 5, "rush_yds": 2, "anytime_td": 0.4}}
        self.assertEqual(fo.props_for(wr), ["rec_yds", "rec", "anytime_td"])


class FormattingTest(unittest.TestCase):
    def test_half_point(self):
        self.assertEqual(fo.half_point(3.2), 3.0)
        self.assertEqual(fo.half_point(3.3), 3.5)
        self.assertEqual(str(fo.half_point(-0.1)), "0.0")

    def test_fmt_line(self):
        self.assertEqual(fo.fmt_line(0), "PK")
        self.assertEqual(fo.fmt_line(3), "+3")
        self.assertEqual(fo.fmt_line(-6.5), "-6.5")

    def test_signed(self):
        self.assertEqual(fo.signed("150"), "+150")
        self.assertEqual(fo.signed("-110"), "-110")
        self.assertEqual(fo.signed(""), "")


# ----------------------------------------------------------------------- CLI

class CommandLineTest(unittest.TestCase):
    def run_cli(self, *args):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "football_odds.py"), "--no-color", *args],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_convert(self):
        out = self.run_cli("convert", "+150")
        self.assertIn("American:    +150", out)
        self.assertIn("Decimal:     2.500", out)
        self.assertIn("Fractional:  3/2", out)
        self.assertIn("Implied %:   40.00%", out)

    def test_vig(self):
        self.assertIn("Bookmaker vig: 4.76%", self.run_cli("vig", "-110", "-110"))

    def test_predict(self):
        out = self.run_cli("predict", "24.5", "21.5", "--spread", "-3", "--total", "44.5",
                           "--ml", "-165", "+140")
        self.assertIn("Fair spread: Home -3", out)
        self.assertRegex(out, r"Home moneyline\s+59\.1%.*-165.*bad")
        self.assertRegex(out, r"Home -3\s+48\.5%\s+4\.6%")

    def test_alternate_lines(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fo.print_prediction("Home", "Away", 24.5, 21.5, None, None, None, None, None,
                                fo.MARGIN_SD, fo.TOTAL_SD, True)
        self.assertIn("Fair spread: Home -3", buf.getvalue())
        self.assertIn("Team total", buf.getvalue())

    def test_bad_odds_fail(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "football_odds.py"), "convert", "-50"],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
