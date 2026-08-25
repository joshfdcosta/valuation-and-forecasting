# Valuation and forecasting

A public portfolio piece in three parts: a discounted cash flow model you can move
the assumptions on, an LSTM that forecasts share prices, and the feedback loop that
grades the forecaster against reality and retrains it when it drifts.

Plus a statistical study of whether candlestick patterns predict anything at all.

The headline results are negative ones, and they are deliberately the headline:

```
3,360 out-of-sample predictions   2023-12-11 → 2026-08-21
direction accuracy                52.50%
model MAE                         $5.12
persistence baseline MAE          $5.05
skill vs baseline                 -1.39%
drift-triggered retrains          15

candlestick patterns tested       11
pattern/horizon tests             24
surviving Holm-Bonferroni          0
```

The model lost to "tomorrow's price equals today's." Not one candlestick pattern beat
chance once the multiple-testing correction was applied. On daily equity closes both
are the expected outcome, and reporting them plainly is the point of the project.

Exact figures shift slightly when the snapshot is regenerated, because training is
stochastic. The site reads them from the snapshot rather than hardcoding them.

## Architecture

The site is **static**. It has to be — a public page cannot depend on someone
starting a Python server with a 2GB torch install.

```
Offline (Python)                          In the browser (TypeScript)
────────────────                          ───────────────────────────
yfinance ─► features ─► LSTM              snapshot.json ─► React
              │                                              │
              └─► walk-forward backtest                      ├─► DCF, live
                  reconcile · score · drift                  └─► charts, hand-rolled SVG
                          │
                          └─► web/public/snapshot.json
```

The expensive, honest work happens once, offline. The DCF is ported to TypeScript so
the valuation is interactive with no latency and no backend. `tests/test_dcf.py` and
`web/src/lib/dcf.test.ts` pin both implementations to the same numbers, so the port
cannot silently drift from the reference.

The FastAPI app remains for running the pipeline live, but the published site does
not use it.

## Why the numbers can be trusted

Getting to a trustworthy negative result is harder than getting to a flattering
positive one. Three decisions do the work:

- **Targets are fractional offsets from the last close**, not raw prices. A model
  trained on 2019 dollar levels does not transfer to 2026 ones, and offsets make the
  naive baseline exactly zero — the thing to beat becomes explicit.
- **Splits are strictly chronological.** Shuffling a time series lets a model train
  on Thursday to predict Wednesday, which produces beautiful metrics and a worthless
  model.
- **The scaler is fitted on training data only**, so the test set's mean and variance
  never leak backwards.

The backtest is walk-forward: the model is refit on a rolling window and only ever
sees data from strictly before the anchor it is predicting from.

The candlestick study adds two more safeguards, because the naive version of that
study is one of the easiest ways in finance to fool yourself:

- **A stationary block bootstrap, not a t-test.** Overlapping forward returns are not
  independent — a 5-day return starting today shares four days with tomorrow's. A
  t-test reads that repetition as extra evidence and manufactures significance.
  Resampling contiguous blocks preserves the autocorrelation instead.
- **Holm-Bonferroni across the whole family of tests.** Eleven patterns at four
  horizons is 24 chances to clear p &lt; 0.05 by luck. Correcting for that is the
  difference between a finding and a coincidence you got attached to.

`tests/test_patterns.py` includes the guard that matters: run the whole study on a
synthetic random walk and zero patterns come back significant. A procedure that finds
edges in noise is worthless, so that case is tested explicitly.

Predictions are **append-then-reconcile**. A row is written with a null actual before
the candle exists; a later pass fills it in and scores it. Nothing can be edited after
the outcome is known, so the track record is a real one.

Retraining is a **periodic refit on a rolling window**, not online learning. Per-sample
weight updates on data this noisy drift into catastrophic forgetting and cannot be
cleanly rolled back. A refit is reproducible, auditable, and revertible to any prior
artefact.

## Layout

```
web/                        static site — this is what gets published
  src/lib/dcf.ts            DCF ported to TypeScript, runs in the browser
  src/lib/dcf.test.ts       parity tests against the Python reference
  src/components/Charts.tsx hand-rolled annotated SVG charts
  src/profile.ts            ← your name, links and bio live here
  public/snapshot.json      generated backtest, committed

scripts/build_snapshot.py   walk-forward backtest → snapshot.json
src/valuation/dcf.py        reference DCF implementation
src/data/                   yfinance retrieval, indicators, windowing
src/models/                 LSTM and the persistence/drift baselines
src/patterns/detect.py      vectorised candlestick pattern rules
src/patterns/study.py       block bootstrap + multiple-testing correction
src/pipeline/               train, predict, reconcile, drift-detect
src/storage/db.py           prediction ledger and training-run audit trail
app/main.py                 FastAPI, for running the pipeline live
```

## Running it

The site alone needs no Python:

```bash
npm install --prefix web
```

```bash
npm run dev --prefix web
```

To regenerate the backtest — this retrains the model repeatedly and takes a few
minutes:

```bash
python -m scripts.build_snapshot AAPL
```

Tests, both sides:

```bash
pytest tests -q
```

```bash
npm test --prefix web
```

## Deploying

`npm run build --prefix web` emits a static bundle to `web/dist`. Any static host
serves it — Vercel, Netlify, Cloudflare Pages, GitHub Pages. There is no server, no
database, and no runtime dependency on Python.

## Before you publish

Edit `web/src/profile.ts`. It holds the name, bio, skills and links, and it is the
only file with personal details in it.

## Known gaps

- **Exchange calendar is approximated with business days**, overcounting by roughly
  nine market holidays a year.
- **OHLC coherence is clamped after the fact.** The model predicts the four legs
  independently and can return a low above the open. A constrained head — predicting
  high and low as non-negative offsets from the body — is the better fix.
- **Single ticker.** No cross-sectional or sector features.
- **No transaction costs, slippage, or survivorship-bias handling.** There is no
  backtest here that would survive contact with execution reality, and none is
  claimed.
- **`yfinance` field names shift between releases.** `get_fundamentals` is defensive,
  but a missing field falls back to a default assumption rather than failing loudly.

Not investment advice.
