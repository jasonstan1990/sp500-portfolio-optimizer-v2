# S&P 500 Portfolio Optimizer v2

Streamlit application with optional sector screening, saved market-data snapshots,
Monte Carlo portfolio selection, Black–Litterman views, and risk charts.

## Run

Install Python 3.11 or newer, then:

```sh
pip install -r requirements.txt
streamlit run app.py
```

## Sector selection

In Screening (Sprint 2), enable **Filter by sector**, then choose one or several
**Include sectors**. Disabled means all sectors. Enabled with no selection stops
calculation and asks for a selection. Sector screening combines with the other
filters before ranking and portfolio selection. Impossible weight constraints
produce a clear error. Sampling preserves weight bounds, including tight cases.

## Monthly update on Windows

Extract the full project and double-click `UPDATE_MONTHLY.bat`. Wait for SUCCESS.
Commit the updated `.portfolio_cache/cache.db`. The public app performs no market
data downloads. `START_HERE_GR.txt` provides Greek update instructions.

## New repository

Suggested name: `sp500-portfolio-optimizer-v2`.
Create an empty repository on GitHub (without an initial README). Upload this
folder's contents, including `.portfolio_cache/cache.db`. Do not upload the ZIP
as the app itself. GitHub Desktop can add and publish the complete folder while
respecting `.gitignore`. For Streamlit deployment select the new repository,
branch `main`, and entry point `app.py`.

## Data and validation limitations

The bundled database is the original March 2026 snapshot. Run the updater for
new data. It retains the original constituent list, which is not current or
point-in-time S&P 500 membership. Updated growth data uses Yahoo earningsGrowth,
not analyst forecast growth. The updater keeps completed weekly/monthly periods.

Historical performance uses weights fitted to historical data and excludes costs;
it is not independent out-of-sample evidence. Monte Carlo selects the best sampled
portfolio, not a guaranteed global optimum. Weekly/monthly ranking lookback lengths
still use the original daily observation counts and need a separate methodology
revision. Black–Litterman excess-return/risk-free conventions need further review.

Checks: updater failure protection and partial-download tests; Python syntax;
120 sampled portfolios tested against bounds and sum-to-one constraints.
No complete live Yahoo download, Windows run, or Streamlit UI run has been verified.
