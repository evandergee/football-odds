// Shared code for the game-lines and player-props pages: odds maths, team
// ratings from nflverse results, and the coloured result cards.

const MARGIN_SD = 13, TOTAL_SD = 13, TEAM_SD = 9;          // points, from 2023-2025 results vs lines
const HOME_FIELD = 1.5, PRIOR_WEIGHT = 0.2, SHRINK = 6;   // team rating settings
// A positive edge below CAUTION_MIN is within the model's error; one above CAUTION_MAX
// usually means the model is missing news, since sportsbooks are more accurate than it is.
const CAUTION_MIN = 0.03, CAUTION_MAX = 0.10;
const GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv';

const TEAMS = {
  ARI: 'Arizona Cardinals', ATL: 'Atlanta Falcons', BAL: 'Baltimore Ravens', BUF: 'Buffalo Bills',
  CAR: 'Carolina Panthers', CHI: 'Chicago Bears', CIN: 'Cincinnati Bengals', CLE: 'Cleveland Browns',
  DAL: 'Dallas Cowboys', DEN: 'Denver Broncos', DET: 'Detroit Lions', GB: 'Green Bay Packers',
  HOU: 'Houston Texans', IND: 'Indianapolis Colts', JAX: 'Jacksonville Jaguars', KC: 'Kansas City Chiefs',
  LA: 'Los Angeles Rams', LAC: 'Los Angeles Chargers', LV: 'Las Vegas Raiders', MIA: 'Miami Dolphins',
  MIN: 'Minnesota Vikings', NE: 'New England Patriots', NO: 'New Orleans Saints', NYG: 'New York Giants',
  NYJ: 'New York Jets', PHI: 'Philadelphia Eagles', PIT: 'Pittsburgh Steelers', SEA: 'Seattle Seahawks',
  SF: 'San Francisco 49ers', TB: 'Tampa Bay Buccaneers', TEN: 'Tennessee Titans', WAS: 'Washington Commanders',
};

const $ = id => document.getElementById(id);
const teamName = abbr => TEAMS[abbr] || abbr;
const shortName = abbr => (TEAMS[abbr] || abbr).split(' ').pop();

// ------------------------------------------------------------- odds maths

// Parse American (-110, +150, 150), decimal (1.91) or fractional (10/11) odds.
function toDecimal(s) {
  s = String(s).trim();
  if (!s) return null;
  let d;
  if (s.includes('/')) { const [n, m] = s.split('/').map(Number); d = 1 + n / m; }
  else {
    const v = Number(s);
    if (/^[+-]/.test(s) || Math.abs(v) >= 100) {
      if (Math.abs(v) < 100) return NaN;
      d = v > 0 ? 1 + v / 100 : 1 + 100 / Math.abs(v);
    } else d = v;
  }
  return isFinite(d) && d > 1 ? d : NaN;
}

// Rounding first means a 50% price shows as +100 on both sides.
function toAmerican(d) {
  if (d >= 2 || Math.round(100 / (d - 1)) === 100) return '+' + Math.round((d - 1) * 100);
  return '-' + Math.round(100 / (d - 1));
}

const fairOdds = p => p > 0 && p < 1 ? toAmerican(1 / p) : '–';

function erf(x) {
  // Abramowitz and Stegun 7.1.26, accurate to about 1e-7.
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const cdf = (x, mean, sd) => 0.5 * (1 + erf((x - mean) / (sd * Math.SQRT2)));

// Probability of each whole-number result, from a normal curve rounded to integers.
function discreteNormal(mean, sd, lo, hi) {
  const dist = new Map();
  for (let k = lo; k <= hi; k++) dist.set(k, cdf(k + 0.5, mean, sd) - cdf(k - 0.5, mean, sd));
  return dist;
}

// [P(result > line), P(result == line), P(result < line)]
function lineProbs(dist, line) {
  let above = 0;
  for (const [k, p] of dist) if (k > line) above += p;
  const push = Number.isInteger(line) ? (dist.get(line) || 0) : 0;
  return [above, push, 1 - above - push];
}

// ------------------------------------------------------------ team ratings

function parseCsv(text) {
  const lines = text.trim().split('\n');
  const split = line => {
    const out = []; let cur = '', quoted = false;
    for (const ch of line) {
      if (ch === '"') quoted = !quoted;
      else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur.replace(/\r$/, ''));
    return out;
  };
  const head = split(lines[0]);
  return lines.slice(1).map(l => { const v = split(l); return Object.fromEntries(head.map((h, i) => [h, v[i]])); });
}

const played = g => g.home_score !== 'NA' && g.home_score !== '';

// Points for = league average + own offense - opponent defense + home field share.
// Solved by repeated averaging; SHRINK adds phantom average games so small samples regress.
function fitRatings(games, season) {
  const mine = {}, theirs = {};
  let total = 0, weight = 0;
  for (const g of games) {
    const w = +g.season === season ? 1 : PRIOR_WEIGHT;
    const h = g.location === 'Neutral' ? 0 : HOME_FIELD / 2;
    for (const [t, o, pts, hh] of [[g.home_team, g.away_team, +g.home_score, h], [g.away_team, g.home_team, +g.away_score, -h]]) {
      (mine[t] ||= []).push([o, pts, hh, w]);
      (theirs[o] ||= []).push([t, pts, hh, w]);
      total += pts * w; weight += w;
    }
  }
  const avg = total / weight;
  let off = {}, def = {};
  for (let i = 0; i < 40; i++) {
    const nOff = {}, nDef = {};
    for (const t in mine) {
      let s = 0, ws = 0;
      for (const [o, pts, hh, w] of mine[t]) { s += w * (pts - avg - hh + (def[o] || 0)); ws += w; }
      nOff[t] = s / (ws + SHRINK);
    }
    for (const t in theirs) {
      let s = 0, ws = 0;
      for (const [a, pts, hh, w] of theirs[t]) { s += w * (avg + (off[a] || 0) + hh - pts); ws += w; }
      nDef[t] = s / (ws + SHRINK);
    }
    off = nOff; def = nDef;
  }
  return {
    avg, off, def,
    // [home points, away points]
    project(home, away, neutral) {
      const h = neutral ? 0 : HOME_FIELD / 2;
      return [avg + (off[home] || 0) - (def[away] || 0) + h, avg + (off[away] || 0) - (def[home] || 0) - h];
    },
  };
}

// Load nflverse results: team ratings plus the next week's games with their lines.
async function loadNflData() {
  const res = await fetch(GAMES_URL);
  if (!res.ok) throw new Error(res.status);
  const games = parseCsv(await res.text());
  const done = games.filter(played);
  const season = Math.max(...done.map(g => +g.season));
  const current = done.filter(g => +g.season === season);
  const ratings = fitRatings(done.filter(g => +g.season >= season - 1), season);
  const weeks = current.filter(g => g.game_type === 'REG').map(g => +g.week);
  const lastWeek = weeks.length ? Math.max(...weeks) : 0;
  let note = `Ratings use ${current.length} games from ${season}` +
    (lastWeek ? ` (through week ${lastWeek})` : '') + ` plus ${season - 1} results at 20% weight.`;
  let upcoming = [], nextWeek = null;
  const future = games.filter(g => +g.season === season && !played(g));
  if (future.length) {
    nextWeek = Math.min(...future.map(g => +g.week));
    upcoming = future.filter(g => +g.week === nextWeek);
    note += ` Showing week ${nextWeek}'s games.`;
  }
  return { ratings, season, upcoming, nextWeek, note };
}

function gameLabel(g) {
  const day = new Date(g.gameday + 'T12:00:00').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  return `${day} · ${teamName(g.away_team)} @ ${teamName(g.home_team)}${g.location === 'Neutral' ? ' (neutral site)' : ''}`;
}

// ------------------------------------------------------------------ output

const pct = p => (p * 100).toFixed(1) + '%';
const fmtLine = x => x === 0 ? 'PK' : (x > 0 ? '+' : '') + x;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const signedPts = x => (x >= 0 ? '+' : '') + x.toFixed(1);

const RATING_LABELS = { good: 'Good price', caution: 'Caution', bad: 'Bad price' };
const edgeRating = edge => edge < 0 ? 'bad' : (edge < CAUTION_MIN || edge > CAUTION_MAX) ? 'caution' : 'good';

// A result card: the model's win chance and fair odds, plus the edge and a
// coloured rating when the sportsbook's decimal odds (book) are given.
function card(name, win, push, loss, book) {
  const fair = toAmerican((win + loss) / win);
  let extra = '', rating = '';
  if (book) {
    const edge = win * (book - 1) - loss;
    const b = book - 1, kelly = Math.max(0, (b * win - loss) / (b * (win + loss)));
    rating = edgeRating(edge);
    // Kelly only for good prices: for thin or suspiciously big edges it can
    // suggest reckless stakes, especially on heavy favourites.
    extra = `<div class="edge">Book ${toAmerican(book)} · edge ${edge > 0 ? '+' : ''}${(edge * 100).toFixed(1)}%` +
      (rating === 'good' ? ` · Kelly ${(kelly * 100).toFixed(1)}%` : '') + '</div>';
  }
  const badge = rating ? `<span class="badge">${RATING_LABELS[rating]}</span>` : '';
  return `<div class="card ${rating}">${badge}<div class="small">${esc(name)}</div><div class="big">${pct(win)}</div>` +
    `<div class="small">Fair odds ${fair}${push ? ` · push ${pct(push)}` : ''}</div>${extra}</div>`;
}

function vigText(label, a, b) {
  if (!a || !b) return '';
  return `${label} ${((1 / a + 1 / b - 1) * 100).toFixed(1)}%`;
}

// Mark each invalid field red and show the first problem's message. Each entry
// is [field id, is it bad, message]. Returns true when everything is valid.
function validate(problems, errId = 'err') {
  for (const [id, bad] of problems) $(id).classList.toggle('missing', bad);
  const first = problems.find(([, bad, msg]) => bad && msg);
  $(errId).textContent = first ? first[2] : '';
  return !first;
}
