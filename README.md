# NFL odds calculator

A small toolkit for NFL betting maths. It converts between odds formats, measures a bookmaker's vig, and turns projected scores into moneyline, spread and total probabilities with fair odds.

**Try it in your browser:** https://evandergee.github.io/football-odds/

There are two versions:

- **`index.html`**: the calculator as a web page, which is what the link above shows.
- **`football_odds.py`**: a command-line script. It needs Python 3.9+ and nothing else.

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
- **Margin and total:** the final margin is modelled as a bell curve (normal distribution) centred on the projected margin, with a standard deviation of 13.5 points. The total works the same way, centred on the projected total with a standard deviation of 10 points. Both curves are rounded to whole points so pushes on whole-number lines can be priced. You can change both standard deviations with `--margin-sd` and `--total-sd`.
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
