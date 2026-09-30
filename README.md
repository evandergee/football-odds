# NFL odds calculator

A small toolkit for NFL betting maths. It converts between odds formats, measures a bookmaker's vig, and turns projected scores into moneyline, spread and total probabilities with fair odds.

**Try it in your browser:** https://evandergee.github.io/football-odds/

There are two versions:

- **`index.html`**: the calculator as a web page, which is what the link above shows. It builds team ratings from this season's results and lists this week's games with sportsbook lines.
- **`football_odds.py`**: a command-line script where you supply the projected points yourself. It needs Python 3.9+ and nothing else.

## Using the web page

1. **Pick a game** from this week's list, or choose any two teams. The projected points, spread and total fill in automatically, and the results appear at the bottom.
2. **Optional:** change the projected points if you know something the ratings don't, like an injured quarterback.
3. **Optional:** type in the odds from your sportsbook. For this week's games, typical odds are already filled in.

Fields marked with a red asterisk need a value, and empty ones are highlighted.

### Team ratings

Each team gets an offense and a defense rating, in points above or below an average team. They're fitted to every score from the current season, plus the previous season at 20% weight so early-season ratings aren't built on two or three games. Every team is also pulled toward average by the equivalent of six average games, and home field is worth 1.5 points. A team's projected points are the league average, plus its offense rating, minus the opponent's defense rating, plus or minus half the home-field edge.

Results, upcoming games and sportsbook lines come from [nflverse](https://github.com/nflverse/nfldata), loaded fresh each time the page opens.

In a backtest over the 2024 and 2025 seasons (448 games from week 4 on), the ratings missed actual margins by 10.2 points on average, against 9.6 for the sportsbooks' closing spreads. The sportsbooks are more accurate, so a big "edge" usually means the model is missing news.

## Command-line usage

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

Predict a game from each team's projected points (home first, then away):

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
- when you give bookmaker odds, the edge on each bet and a Kelly stake fraction

## How it works

- **Vig:** each side's implied probability is `1 / decimal odds`. Bookmakers price these to add up to more than 100%, and the excess is their vig. The vig is removed by scaling the probabilities so they add up to exactly 100%. At -110 on both sides the vig is 4.76%.
- **Margin and total:** the final margin is modelled as a bell curve (normal distribution) centred on the projected margin, with a standard deviation of 13 points. The total works the same way, centred on the projected total, also with a standard deviation of 13 points. Both figures are how far actual results landed from closing betting lines in the 2023–2025 seasons. Both curves are rounded to whole points so pushes on whole-number lines can be priced. You can change both standard deviations with `--margin-sd` and `--total-sd`.
- **Overtime:** games tied after regulation are split evenly between a 3-point home win and a 3-point away win, because most overtime games end on a field goal. Ties are rare enough to ignore.
- **Edge:** `win chance × (decimal odds − 1) − loss chance`. A positive number means the model rates the bet as better than the price.
- **Kelly:** the fraction of a bankroll the Kelly criterion suggests when the edge is positive, adjusted for pushes.

## Limitations

- The model can only be as good as the projected points you give it.
- Real NFL margins cluster on key numbers like 3, 7 and 10. A smooth bell curve spreads that probability out, so it underrates pushes on those lines and overrates them elsewhere. Moving a line across 3 or 7 matters more in real life than this model shows.
- It ignores injuries, weather, rest and anything else that isn't already in your projection.

## Disclaimer

This project is for learning and analysis only. It is not betting advice and makes no promise of profit. Bookmaker vig means most bettors lose money over time. If you do bet, do it only where it's legal for you, only if you're of legal age, and only with money you can afford to lose. If gambling stops being fun, help is available from [1-800-GAMBLER](https://www.1800gambler.net/).

## License

[MIT](LICENSE)
