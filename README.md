# NFL odds calculator

[![Tests](https://github.com/evandergee/football-odds/actions/workflows/tests.yml/badge.svg)](https://github.com/evandergee/football-odds/actions/workflows/tests.yml)

A small toolkit for NFL betting maths. It turns team ratings and player stats into probabilities and fair odds for game lines (moneyline, spread, total, team totals, alternate lines, winning margin) and player props (passing, rushing and receiving yards, receptions, completions, touchdowns and more), and rates a sportsbook's price against them.

**Try it in your browser:**

- Game lines: https://evandergee.github.io/football-odds/
- Player props: https://evandergee.github.io/football-odds/props.html

What's here:

- **`index.html`** and **`props.html`**: the two web pages, sharing `assets/`.
- **`football_odds.py`**: the same calculator as a command-line script. It needs Python 3.10+ and nothing else.
- **`scripts/build_player_data.py`**: builds `data/players.json` from nflverse. A GitHub Actions job (`.github/workflows/update-data.yml`) runs it every six hours and commits the result when it changes.
- **`scripts/backtest_props.py`**: replays a past season week by week to measure the player-prop model.
- **`tests/`**: unit tests for the script. Run them with `python3 -m unittest discover tests`. GitHub Actions runs them on every push.

## Using the game lines page

1. **Pick a game** from this week's list, or choose any two teams. The projected points, spread and total fill in automatically, and the results appear at the bottom.
2. **Optional:** change the projected points if you know something the ratings don't, like an injured quarterback.
3. **Optional:** type in the odds from your sportsbook. For this week's games, typical odds are already filled in.

Fields marked with a red asterisk need a value, and empty ones are highlighted. Open **4. Team totals** to enter your sportsbook's team total lines. Below the main results are tables of fair odds for alternate spreads, alternate totals and winning-margin bands.

When you enter sportsbook odds, each bet is colour-coded by its edge:

- **Green, good price:** an edge of 3% to 10%.
- **Yellow, caution:** an edge under 3%, which is too thin to trust given the model's error, or over 10%, which usually means the model is missing news like an injury.
- **Red, bad price:** a negative edge. The model rates the bet below the sportsbook's price.

A Kelly stake is shown only for green bets. For thin or suspiciously big edges it can suggest reckless stakes, especially on heavy favourites.

## Using the player props page

1. **Pick a game** from this week's list.
2. **Pick a player and a prop.** Players are grouped by team, with injury statuses shown. Injured reserve is listed last.
3. **Type in the sportsbook's line.** It starts at the model's fair line. Add the over and under odds to get the same colour ratings as the game lines page.

The page shows the projection and why (the matchup adjustment), the player's last five games, warnings for injuries or thin data, and a table of every prop for that player with the model's fair line.

### How player props are projected

- **Average:** each player's per-game average for each stat over his last 20 games. Recent games count more (each older game counts 0.9 as much as the one after it), and last season's games count 30% as much.
- **Matchup:** passing props are adjusted for how much the opponent's pass defense allows compared with an average defense. Touchdown props are scaled by the team's projected points (from the team ratings) against its usual scoring. Rushing and receiving props aren't adjusted, because in the 2025 backtest that made them less accurate.
- **Spread of outcomes:** yards follow a skewed lognormal curve, touchdowns a Poisson distribution, and other counts a negative binomial. How widely results vary around the projection was measured from the 2024 and 2025 seasons. Because a few big games pull the average up, the fair line (50/50 point) sits a little below the projection for yards props.
- **Injuries:** this week's injury report from nflverse. Before game-day statuses come out (usually Friday), practice participation is shown instead.

### How accurate it is

`scripts/backtest_props.py` rebuilds the model before each week of a past season from earlier games only, and compares it with what happened. Tuned on 2025 and checked on 2024:

- Its projections beat a plain season-to-date average on 20 of 21 stat and position combinations in 2024, but only by 1–3%. Player props are hard to predict.
- Using each player's season average as a stand-in line, its over probabilities were calibrated to within a couple of points for predictions between 30% and 50%, where most bets fall. Its 50–60% predictions came true slightly more often than predicted, about 6 points more.

Sportsbooks set prop lines with more information than this (snap counts, depth charts, weather, late news), so a big edge usually means the model is missing something.

### Team ratings

Each team gets an offense and a defense rating, in points above or below an average team. They're fitted to every score from the current season, plus the previous season at 20% weight so early-season ratings aren't built on two or three games. Every team is also pulled toward average by the equivalent of six average games, and home field is worth 1.5 points. A team's projected points are the league average, plus its offense rating, minus the opponent's defense rating, plus or minus half the home-field edge.

Results, upcoming games and sportsbook lines come from [nflverse](https://github.com/nflverse/nfldata), loaded fresh each time the page opens.

In a backtest over the 2024 and 2025 seasons (448 games from week 4 on), the ratings missed actual margins by 10.2 points on average, against 9.6 for the sportsbooks' closing spreads. The sportsbooks are more accurate, so a big "edge" usually means the model is missing news.

## Command-line usage

See every team's offense and defense rating:

```bash
python3 football_odds.py teams
```

List this week's games, with the model's spread and total next to the sportsbook's:

```bash
python3 football_odds.py week
```

Get the full breakdown for one game, away team first. Teams can be abbreviations (`BUF`), nicknames (`Bills`) or cities (`Buffalo`). For this week's games, the sportsbook lines load automatically:

```bash
python3 football_odds.py game PIT CLE
python3 football_odds.py game Bills Chiefs --ml -165 +140
```

Each bet with sportsbook odds gets the same good / caution / bad rating as the web page, coloured green, yellow or red in the terminal. Add `--home-pts` or `--away-pts` to override the ratings, `--neutral` for a neutral-site game, or `--no-color` for plain output.

Add `--alt` to also print alternate spreads and totals. Team totals at the fair lines are always shown.

Check player props. With just a name, it lists every prop for that player's game this week with the model's projection and fair line. Add a prop, the sportsbook's line and the odds to rate one bet:

```bash
python3 football_odds.py props mahomes
python3 football_odds.py props "josh allen" pass_yds 249.5 --odds -115 -105
python3 football_odds.py props kelce receptions 5.5 --odds -130 +100
python3 football_odds.py props "james cook" td --odds +110 -140
```

Props can be written as `pass_yds`, `pass_td`, `pass_cmp`, `pass_att`, `pass_int`, `rush_yds`, `rush_att`, `rec_yds`, `rec`, `rush_rec_yds` or `td`, or in words like `receptions` or `rushing-yards`.

The `teams`, `week`, `game` and `props` commands download the latest results from nflverse each time they run. To work offline, download [games.csv](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) and add `--data games.csv`.

Convert odds between American, decimal and fractional formats:

```bash
python3 football_odds.py convert -110
python3 football_odds.py convert +150
```

Show the bookmaker's vig on a market and the fair odds with it removed:

```bash
python3 football_odds.py vig -110 -110
python3 football_odds.py vig -165 +140
```

Predict a game from projected points you supply yourself (home first, then away):

```bash
python3 football_odds.py predict 24.5 21.5
```

Name the teams, set the bookmaker's lines, and compare against their odds:

```bash
python3 football_odds.py predict 24.5 21.5 --home Chiefs --away Bills --spread -3 --total 44.5 --ml -165 +140 --spread-odds -110 -110 --total-odds -110 -110
```

The spread is always the home team's line, so `-3` means the home team is favoured by 3. Without `--spread` or `--total`, the script uses the fair lines from your projection.

The prediction shows:

- the fair spread and fair total
- moneyline win chances for both teams
- spread cover and push chances for both sides
- over, under and push chances for the total
- fair American odds for each of these, ignoring pushes, since a push returns your stake
- each team's total at its fair line
- when you give bookmaker odds, the edge on each bet, with a Kelly stake fraction for good prices

## How it works

- **Vig:** each side's implied probability is `1 / decimal odds`. Bookmakers price these to add up to more than 100%, and the excess is their vig. The vig is removed by scaling the probabilities so they add up to exactly 100%. At -110 on both sides the vig is 4.76%.
- **Margin and total:** the final margin is modelled as a bell curve (normal distribution) centred on the projected margin, with a standard deviation of 13 points. The total works the same way, centred on the projected total, also with a standard deviation of 13 points. Both figures are how far actual results landed from closing betting lines in the 2023–2025 seasons. Both curves are rounded to whole points so pushes on whole-number lines can be priced. You can change both standard deviations with `--margin-sd` and `--total-sd`.
- **Team totals:** each team's points follow a bell curve with a standard deviation of 9 points, again measured against 2023–2025 betting lines.
- **Overtime:** games tied after regulation are split evenly between a 3-point home win and a 3-point away win, because most overtime games end on a field goal. Ties are rare enough to ignore.
- **Edge:** `win chance × (decimal odds − 1) − loss chance`. A positive number means the model rates the bet as better than the price.
- **Kelly:** the fraction of a bankroll the Kelly criterion suggests when the edge is positive, adjusted for pushes.

## Limitations

- The model can only be as good as its inputs: the team ratings, player averages or projected points you give it.
- Real NFL margins cluster on key numbers like 3, 7 and 10. A smooth bell curve spreads that probability out, so it underrates pushes on those lines and overrates them elsewhere. Moving a line across 3 or 7 matters more in real life than this model shows.
- It ignores weather, rest and anything else that isn't already in the ratings or your projection. Injury reports are shown on the props page, but the model doesn't adjust for them, including a teammate being out.

## Disclaimer

This project is for learning and analysis only. It is not betting advice and makes no promise of profit. Bookmaker vig means most bettors lose money over time. If you do bet, do it only where it's legal for you, only if you're of legal age, and only with money you can afford to lose. If gambling stops being fun, help is available from [1-800-GAMBLER](https://www.1800gambler.net/).

## License

[MIT](LICENSE)
