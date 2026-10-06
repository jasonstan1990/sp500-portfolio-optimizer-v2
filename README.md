# S&P 500 Portfolio Optimizer V4 — Quant Reliability Edition

Streamlit research application for screening, factor ranking, deterministic portfolio optimization, Black–Litterman views, point-in-time aware walk-forward validation, risk analytics, stress testing, and investment allocation.

> **Research tool, not investment advice.** The current optimizer is fitted to historical observations. Use the Walk-Forward tab for out-of-sample validation and interpret all model outputs as estimates, not forecasts.

## What changed in V4

V4 completes the reliability roadmap from V3:

1. **Walk-forward backtest** — monthly or quarterly rebalance; each decision uses data available through the prior trading session.
2. **Transaction costs + turnover** — weight drift is simulated between rebalances; trading costs are deducted when target weights change.
3. **Current S&P 500 universe refresh** — the offline updater refreshes current constituents from Wikipedia, with a saved-universe fallback.
4. **Point-in-time safeguards** — historical membership is reconstructed from constituent changes. Fundamental snapshots are accumulated from V4 onward. Before the first saved historical fundamental snapshot, walk-forward ranking automatically disables growth/value rather than leaking today's fundamentals into the past.
5. **Advanced metrics** — CAGR, Sharpe, Sortino, max drawdown, Calmar, alpha, beta, tracking error, information ratio, correlation, VaR/CVaR, concentration and effective-N.
6. **Modular architecture** — numerical, data, universe, ranking, backtest, and risk logic are separated from the Streamlit UI.
7. **UI upgrade** — workflow tabs, portfolio scorecard, sector allocation, drawdown, rolling validation metrics, turnover and audit log.
8. **Expanded stress testing** — transparent proxy scenarios for +200 bp rates, recession, technology crash, inflation shock, broad equity shock, plus historical episodes.
9. **Expanded tests** — optimizer bounds, Black–Litterman convention, ADTV windows, sector aliases, corrupted database detection, point-in-time fundamental selection, universe reconstruction, transaction-cost drag, no-future-fundamentals check, publication safety and Windows replace retry.
10. **Optional monthly automation** — a GitHub Actions workflow can test, update, validate and commit `cache.db` automatically. The local Windows updater remains the fallback.

## Existing V3 reliability fixes retained

- Daily / weekly / monthly momentum windows are calendar-consistent.
- 60-day ADTV always uses daily price and volume observations.
- Max-Sharpe and Min-Vol use constrained SLSQP optimization; Monte Carlo is visual only and deterministic.
- Black–Litterman posterior returns are treated as excess returns, avoiding double subtraction of the risk-free rate.
- Stress-sector aliases map Yahoo sector names such as `Healthcare`, `Consumer Cyclical`, and `Basic Materials`.
- SQLite connections close before atomic publication; staged DB integrity is checked; Windows file-lock replacement retries are included.

## Files

- `app.py` — Streamlit UI/orchestration only.
- `portfolio_math.py` — covariance, bounded weights, optimizers, Black–Litterman, risk contributions.
- `ranking.py` — screening and factor ranking.
- `backtest.py` — point-in-time aware walk-forward engine.
- `risk.py` — performance metrics, rolling analytics, stress tests.
- `universe.py` — current S&P 500 parsing and historical membership reconstruction.
- `data_store.py` — SQLite schema, snapshots, PIT fundamentals.
- `update_data.py` — offline / CI data publisher.
- `.github/workflows/monthly_update.yml` — optional scheduled updater.

## Run the public app

Python 3.11+:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The public app does **not** download market data. It reads `.portfolio_cache/cache.db` only.

## Update locally on Windows

Double-click:

```text
UPDATE_MONTHLY.bat
```

Wait for `SUCCESS`. Then upload only:

```text
.portfolio_cache/cache.db
```

to the same path in GitHub. Do not upload `cache.before_update.db` or `cache.new.db`.

## What the updater does

1. Reads the last valid snapshot as fallback state.
2. Attempts to refresh current S&P 500 constituents and historical changes from Wikipedia.
3. Builds a union of current and historical symbols required for point-in-time backtests.
4. Downloads adjusted daily price / volume history from Yahoo Finance.
5. Requires valid current prices for at least 95% of current constituents.
6. Retains historical/delisted symbols when sufficient historical data exist even if they do not trade today.
7. Refreshes current fundamentals; aborts if more than 10% fail.
8. Saves the current fundamental table as a true as-of snapshot for future PIT backtests.
9. Builds daily, weekly and monthly snapshots.
10. Creates a staged SQLite DB, runs `PRAGMA integrity_check`, closes connections and atomically replaces the live DB.

## Point-in-time limitations

V4 does **not fabricate historical fundamentals**. The first V4 updater run starts a real history of fundamental snapshots, and subsequent monthly runs add new as-of snapshots. For dates before the first snapshot, walk-forward validation uses price-based factors only (momentum, quality and low volatility) and disables the growth screen/value factor.

Historical S&P 500 membership is reconstructed from the public constituent-change table. This is materially better than using today's 500 names throughout history, but it is still not a licensed CRSP/S&P point-in-time constituent database and may be incomplete around symbol changes, mergers or older events.

## Automated monthly update

See `AUTOMATION.md`. The bundled GitHub Actions workflow:

- runs on the 5th of each month,
- runs all tests,
- runs the updater,
- commits `cache.db` only when validation succeeds.

If external services rate-limit the CI runner, use the local `UPDATE_MONTHLY.bat`; failure never intentionally replaces the live cache.

## Methodology notes

- Price data are adjusted prices from Yahoo Finance.
- Earnings growth is Yahoo `earningsGrowth` (quarterly YoY), not analyst consensus forecast growth.
- Ledoit–Wolf shrinkage covariance is used for optimization.
- Synthetic stress tests are transparent scenario assumptions, not forecasts or a structural macro model.
- Transaction costs are modeled as basis points per traded notional at each rebalance.
- No tax, bid/ask spread model beyond the chosen cost assumption, market impact, borrow cost or execution slippage model is included.
