# Football odds

A small toolkit for football (soccer) odds maths. It converts between odds formats, measures a bookmaker's margin, and uses a Poisson model to turn expected goals into match probabilities and fair odds.

There are two versions:

- **`football_odds.py`**: a command-line script. It needs Python 3.9+ and nothing else.
- **`index.html`**: the same calculator in a web page. Open it in a browser, or host it with GitHub Pages.

## Command-line usage

Convert odds between decimal, fractional and American formats:

```bash
python3 football_odds.py convert 6/4
python3 football_odds.py convert -200
```

Show a bookmaker's margin and the fair odds with it removed:

```bash
python3 football_odds.py margin 2.10 3.40 3.60
```

Predict a match from each team's expected goals (home first, then away):

```bash
python3 football_odds.py predict 1.6 1.1
```

Compare the model with bookmaker home/draw/away odds, and change the over/under line:

```bash
python3 football_odds.py predict 1.6 1.1 --odds 2.10 3.40 3.60 --line 3.5
```

The prediction shows:

- home win, draw and away win chances, with fair odds for each
- over/under goals and both-teams-to-score chances
- the five most likely scores
- when you give bookmaker odds, the edge on each outcome and a Kelly stake fraction

## How it works

- **Margin:** each outcome's implied probability is `1 / decimal odds`. Bookmakers price these to add up to more than 100%, and the excess is their margin. The margin is removed by scaling the probabilities so they add up to exactly 100%.
- **Poisson model:** each team's goals are modelled as an independent Poisson distribution with the expected goals as its average. Combining the two gives the probability of every scoreline from 0-0 to 10-10, and every market is added up from those scorelines.
- **Edge:** `model probability × decimal odds − 1`. A positive number means the model rates the outcome as more likely than the odds imply.
- **Kelly:** the fraction of a bankroll the Kelly criterion suggests when the edge is positive.

## Limitations

- The model can only be as good as the expected goals you give it.
- Treating the two teams' goals as independent tends to underestimate draws a little. Models such as Dixon-Coles correct for this.
- It ignores team news, motivation, weather and anything else that isn't in the expected goals.

## Disclaimer

This project is for learning and analysis only. It is not betting advice and makes no promise of profit. Bookmaker margins mean most bettors lose money over time. If you do bet, do it only where it's legal for you, only if you're of legal age, and only with money you can afford to lose. If gambling stops being fun, help is available from services such as [BeGambleAware](https://www.begambleaware.org/) (UK) or [1-800-GAMBLER](https://www.1800gambler.net/) (US).

## License

[MIT](LICENSE)
