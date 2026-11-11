# statpad-cbb

College basketball ranking model from CBBD play-by-play: possessions with on-court lineups, then a hierarchical RAPM for player and team ratings.

## Season convention

Use the season-ending year. `2027` means 2026–27.

## Model

### Likelihood

Each possession is modeled in points-per-possession (PPP):

```
points ≈ α + home · 1{home offense, non-neutral}
        + Σ_{i on offense} β_i  − Σ_{j on defense} β_j
        + τ_{off team} − τ_{def team}
        + κ_{off conf} − κ_{def conf}
        [+ optional design-mode box sums]
```

Estimation treats residuals as approximately iid Gaussian in PPP. That is a computational approximation: possessions within a game are dependent, so effective sample size is smaller than the possession count. Team/conference indicators are encoded **once per possession** (not once per on-court player) so group effects stay on the same PPP scale as the intercept.

### Hierarchy and identification

Player effects nest in teams; teams nest in conferences.

- `β` = player residual (offense / defense)
- `τ` = team effect
- `κ` = conference effect

The design is only weakly identified without constraints (shifting level between `β` and `τ` leaves predictions nearly unchanged when lineups stay within team). After the MAP fit we **center** minutes-weighted player residuals within each team and absorb `5 · mean(β)` into `τ` (because five players are on the floor while `τ` is once per possession), then center team effects within conference into `κ`. Level lives on the group terms; unit ratings stay invariant.

### Priors (ridge MAP)

Coefficients are fit as Gaussian MAP / ridge around prior means:

```
argmin_β  ‖y − Xβ‖² + Σ_j λ_j (β_j − μ_j)²
```

This is equivalent to `ε ~ N(0, 1)` and `β_j ~ N(μ_j, 1/λ_j)` with residual variance fixed at 1 — a regularization model, not a fully calibrated Bayesian posterior. λ values are chosen by game-blocked CV on held-out **game-margin RMSE** (possession SSE is only a diagnostic).

Prior means `μ` come from `--prior-season` when provided:

| Block | Prior mean |
| --- | --- |
| Intercept / home | prior-season α, home (with small λ_intercept / λ_home so the prior is not a no-op) |
| Team / conference | prior-season τ, κ |
| Players | 0, or box-informed means (below) |

There is no direct year-to-year player RAPM carryover beyond box means and the new team/conf membership.

### Box features

Default `box_mode=prior_mean`:

1. Standardize season-to-date box rates with **prior-season** mean/std when available (pathological prior moments fall back to sample).
2. Rates for players below 20 offensive possessions are treated as missing → league-mean z.
3. Residualize z-features **within team** so box does not restack team/conf level.
4. Freeze `γ` from the prior season and set player ridge targets `μ_i = x_i'γ`.
5. Box columns are **not** in `X` — early counting stats cannot outvote team priors through a free `γ`.

`box_mode=design` is legacy / seeding: on-court box sums enter `X` and `γ` is estimated. Use it when no prior `γ` exists.

### Team ratings and forecasts

A five-man unit’s expected PPP contribution is

```
unit_off = 5 · (minutes-weighted mean of β_off)
         + τ_off + κ_off
```

(and likewise on defense). Ratings are reported per 100 possessions (`× 100`). Game forecasts use these unit ratings, home advantage, and a single possessions-per-team pace factor. Observed offensive/defensive efficiency is shown for diagnostics; it is **not** opponent-adjusted. Optional RAPM/efficiency blends are post-hoc and should not be treated as fixing scale.

### Temporal backtests

Cut-date fits rebuild minutes and box rates from **train-window** possessions/plays only (no post-cut leakage). Default cut dates: Dec 1 / Jan 1 / Feb 1 / Mar 1.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put a College Basketball Data API bearer token in `.env` as `CBBD_API_KEY`. The live pull is not run until that key is present.

## Pipeline

```bash
# Optional prior season for group/box stabilization
python -m statpad_cbb ingest --season 2026
python -m statpad_cbb build-possessions --season 2026
python -m statpad_cbb build-features --season 2026
python -m statpad_cbb fit-rapm --season 2026 --prior hierarchical

python -m statpad_cbb ingest --season 2027
python -m statpad_cbb build-possessions --season 2027
python -m statpad_cbb build-features --season 2027
python -m statpad_cbb fit-rapm --season 2027 --prior hierarchical --prior-season 2026
# default box_mode=prior_mean; use --box-mode design for legacy ablation
python -m statpad_cbb evaluate --season 2027
python -m statpad_cbb backtest --season 2027 --prior hierarchical --prior-season 2026
python -m statpad_cbb eda --season 2027 --school Duke --player "Cameron Boozer"
```

Outputs: `team_ratings.parquet` / `.json`, model metrics under coverage, and EDA under `data/coverage/cbbd/{season}/eda/`. Ingest reuses existing raw JSON unless you pass `--force`.

## What is pulled

| Step | Endpoint | Cadence |
| --- | --- | --- |
| 1 | `GET /teams/directory?season={season}` | once per season |
| 2 | `GET /teams/roster?season={season}` | once per season |
| 3 | `GET /games?season={season}&startDateRange={start}&endDateRange={end}` | date-partitioned windows (14 days, split if a response hits 3,000 games) |
| 4 | `GET /plays/date?date={YYYY-MM-DD}` | once per unique UTC game date |

Expected request count: `2 + N_game_date_ranges + N_unique_UTC_game_dates`.

Box scores are not requested; minutes and counting stats are reconstructed from plays.

## Layout

```
src/statpad_cbb/ingest/        CBBD client, extractors, coverage
src/statpad_cbb/possessions/   lineup tracking + possession table
src/statpad_cbb/features/      player-season box/minutes features
src/statpad_cbb/model/         hierarchical RAPM fit, CV, roll-up
src/statpad_cbb/eda/           team and player diagnostic plots
data/raw/cbbd/{season}/        untouched API payloads
data/processed/cbbd/{season}/  teams/players/games/plays + parquet/JSON outputs
data/coverage/cbbd/{season}/   ingest + model coverage/metrics + eda/
tests/
```

Canonical IDs are the CBBD team, player, and game IDs.

## Tests

```bash
pytest
```
