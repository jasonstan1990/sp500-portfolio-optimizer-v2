"""S&P 500 Portfolio Optimizer V4 — public snapshot-only Streamlit application."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import BacktestConfig, run_walk_forward
from data_store import list_fundamental_snapshots, load_snapshot
from portfolio_math import (
    FREQ_ANNUAL,
    RISK_HORIZON_LABEL,
    black_litterman_posterior_excess,
    ledoit_wolf_cov,
    optimize_max_excess_sharpe,
    optimize_max_sharpe,
    optimize_min_volatility,
    sample_weights,
    volatility_risk_contributions,
)
from ranking import pct_change_safe, screen_and_rank
from risk import (
    concentration_metrics,
    drawdown_series,
    historical_stress_table,
    performance_metrics,
    rolling_beta,
    rolling_sharpe,
    synthetic_stress_table,
)

st.set_page_config(page_title="S&P 500 Portfolio Optimizer V4", layout="wide")

CACHE_DIR = ".portfolio_cache"
DB_PATH = os.path.join(CACHE_DIR, "cache.db")
BENCH = "^GSPC"


@st.cache_data(show_spinner=False)
def cached_snapshot(key: str, db_version: int | None):
    return load_snapshot(DB_PATH, key)


@st.cache_data(show_spinner=False)
def cached_fundamental_history(db_version: int | None):
    return list_fundamental_snapshots(DB_PATH)


@st.cache_data(show_spinner="Running point-in-time walk-forward backtest…")
def cached_walk_forward(db_version: int | None, config_json: str):
    daily = load_snapshot(DB_PATH, "daily")
    if daily is None:
        raise RuntimeError("Daily snapshot is unavailable.")
    _, _, payload = daily
    data = payload["data"]
    cfg = BacktestConfig(**json.loads(config_json))
    current = data.get("current_constituents", data.get("tickers", []))
    changes = data.get("constituent_changes", pd.DataFrame(columns=["Date", "Added", "Removed"]))
    fundamentals = list_fundamental_snapshots(DB_PATH)
    return run_walk_forward(
        data["close_df"],
        data["vol_df"],
        list(current),
        changes,
        fundamentals,
        cfg,
        bench=BENCH,
    )


def fmt_pct(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}%}"


def fmt_num(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def wealth_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.dropna()).cumprod()


def frontier_cloud(mu_ann, cov_ann, rf, n, min_w, max_w, count):
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(int(count)):
        w = sample_weights(rng, n, min_w, max_w)
        ret = float(mu_ann @ w)
        vol = float(np.sqrt(max(w.T @ cov_ann @ w, 0.0)))
        sharpe = (ret - rf) / vol if vol > 1e-12 else np.nan
        rows.append((ret, vol, sharpe))
    return np.asarray(rows)


def sector_weight_table(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Weight": weights, "Sector": sectors.reindex(weights.index).fillna("Unknown")})
    return frame.groupby("Sector", as_index=False)["Weight"].sum().sort_values("Weight", ascending=False)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
# 📈 S&P 500 Portfolio Optimizer V4
### Screen → Rank → Optimize → Validate → Stress → Invest

Build a transparent quantitative portfolio from saved, validated market snapshots.
"""
)
st.caption(
    "Research tool, not investment advice. Current optimization is fitted to historical data; use the Walk-Forward tab for out-of-sample validation."
)
st.markdown("---")

if not os.path.exists(DB_PATH):
    st.error("Missing .portfolio_cache/cache.db. Upload or restore the published snapshot database.")
    st.stop()

db_version = os.stat(DB_PATH).st_mtime_ns

# -----------------------------------------------------------------------------
# Sidebar — current portfolio configuration
# -----------------------------------------------------------------------------
st.sidebar.header("1 · Portfolio Configuration")
frequency = st.sidebar.selectbox("Data frequency", ["daily", "weekly", "monthly"], index=0)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 1.0, 0.02, 0.005)

latest = cached_snapshot(frequency, db_version)
daily_latest = cached_snapshot("daily", db_version)
if latest is None or daily_latest is None:
    st.error("Required published snapshots are missing. Run the updater and publish cache.db.")
    st.stop()

published_at, meta, payload = latest
_, daily_meta, daily_payload = daily_latest
close_full = payload["data"]["close_df"].copy()
vol_full = payload["data"]["vol_df"].copy()
info_df = payload["data"]["info_df"].copy()
current_tickers = list(payload["data"].get("current_constituents", payload["data"].get("tickers", [])))
daily_close_full = daily_payload["data"]["close_df"].copy()
daily_vol_full = daily_payload["data"]["vol_df"].copy()
constituent_changes = daily_payload["data"].get(
    "constituent_changes", pd.DataFrame(columns=["Date", "Added", "Removed"])
)

last_data_date = pd.Timestamp(daily_close_full.index.max()).normalize()
first_data_date = pd.Timestamp(close_full.index.min()).normalize()

start_default = max(first_data_date, pd.Timestamp("2020-01-01"))
start_date = st.sidebar.date_input("Analysis start", value=start_default.date(), min_value=first_data_date.date(), max_value=last_data_date.date())
end_date = st.sidebar.date_input("Analysis end", value=last_data_date.date(), min_value=first_data_date.date(), max_value=last_data_date.date())
if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
    st.error("Analysis start must be before analysis end.")
    st.stop()

st.sidebar.header("2 · Screening")
adtv_usd_min = st.sidebar.number_input("Min ADTV ($, true 60 trading days)", min_value=0, value=1_000_000, step=100_000)
vol_mult_spx = st.sidebar.slider("Max volatility vs S&P 500 (×)", 0.5, 5.0, 3.0, 0.1)
beta_min, beta_max = st.sidebar.slider("Beta range", 0.3, 5.0, (0.8, 1.3), 0.05)
min_growth = st.sidebar.slider("Min earnings growth (YoY)", -0.50, 1.00, 0.00, 0.05)

sector_series = info_df.get("Sector", pd.Series("Unknown", index=info_df.index)).fillna("Unknown").astype(str)
sector_options = sorted(sector_series.reindex(current_tickers).fillna("Unknown").unique())
filter_sectors = st.sidebar.checkbox("Filter by sector", value=False)
selected_sectors = st.sidebar.multiselect("Include sectors", sector_options, disabled=not filter_sectors)
if filter_sectors and not selected_sectors:
    st.sidebar.warning("Choose at least one sector or turn sector filtering off.")

st.sidebar.header("3 · Ranking")
top_n = st.sidebar.slider("Top-N stocks", 5, 80, 30, 1)
w_mom = st.sidebar.slider("Momentum weight", 0.0, 1.0, 0.40, 0.05)
w_qual = st.sidebar.slider("Quality weight", 0.0, 1.0, 0.30, 0.05)
w_val = st.sidebar.slider("Value weight", 0.0, 1.0, 0.20, 0.05)
w_lvol = st.sidebar.slider("Low-vol weight", 0.0, 1.0, 0.10, 0.05)
factor_weights = {"momentum": w_mom, "quality": w_qual, "value": w_val, "lowvol": w_lvol}
if abs(sum(factor_weights.values()) - 1.0) > 1e-9:
    st.sidebar.info("Ranking weights are automatically normalized to 100%.")

st.sidebar.header("4 · Portfolio Constraints")
max_weight = st.sidebar.number_input("Max weight per stock", 0.0, 1.0, 0.10, 0.01)
min_weight = st.sidebar.number_input("Min weight per stock", 0.0, 1.0, 0.00, 0.01)
num_portfolios = st.sidebar.slider("Monte Carlo dots (visual only)", 250, 10_000, 2_000, 250)

st.sidebar.header("5 · Black–Litterman")
enable_bl = st.sidebar.checkbox("Enable sector views", value=False)
bl_delta = st.sidebar.number_input("Risk aversion δ", 0.1, 10.0, 2.5, 0.1)
bl_tau = st.sidebar.number_input("Tau (τ)", 0.001, 1.0, 0.05, 0.01)

st.sidebar.header("6 · Investment Helper")
investment_amount = st.sidebar.number_input("Investment amount ($)", min_value=0.0, value=40_000.0, step=100.0)
allow_fractional = st.sidebar.checkbox("Allow fractional shares", value=True)

# -----------------------------------------------------------------------------
# Data status
# -----------------------------------------------------------------------------
status_cols = st.columns(4)
status_cols[0].metric("Data through", f"{last_data_date:%Y-%m-%d}")
status_cols[1].metric("Current constituents", f"{len(current_tickers)}")
status_cols[2].metric("Snapshot", f"{published_at[:10]}")
status_cols[3].metric("Frequency", frequency.title())

universe_source = meta.get("universe_source", "Saved snapshot universe")
st.caption(f"Universe source: {universe_source}")
if meta.get("universe_refresh_fallback"):
    st.warning("The last updater could not refresh the public constituent source and used the saved universe fallback.")
if (pd.Timestamp.today().normalize() - last_data_date).days > 45:
    st.warning("Market data are more than 45 days old. Run the monthly updater.")
if meta.get("stale_fundamentals"):
    st.warning(f"{len(meta['stale_fundamentals'])} current stocks retained prior fundamental values at the last update.")
if meta.get("missing_tickers"):
    st.info(f"{len(meta['missing_tickers'])} current constituents were excluded because valid current prices were unavailable.")

# -----------------------------------------------------------------------------
# Current screening / ranking
# -----------------------------------------------------------------------------
analysis_close = close_full.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
analysis_vol = vol_full.reindex(analysis_close.index)
daily_for_screen = daily_close_full.loc[: pd.Timestamp(end_date)].copy()
daily_vol_for_screen = daily_vol_full.reindex(daily_for_screen.index)

if analysis_close.empty or BENCH not in analysis_close.columns:
    st.error("No valid data exist in the selected analysis period.")
    st.stop()

ranking = screen_and_rank(
    analysis_close,
    daily_for_screen,
    daily_vol_for_screen,
    info_df,
    current_tickers,
    bench=BENCH,
    frequency=frequency,
    risk_free_rate=risk_free_rate,
    adtv_usd_min=adtv_usd_min,
    vol_mult_spx=vol_mult_spx,
    beta_min=beta_min,
    beta_max=beta_max,
    min_growth=min_growth,
    selected_sectors=selected_sectors if filter_sectors else None,
    top_n=top_n,
    factor_weights=factor_weights,
    use_growth_filter=True,
    use_value_factor=True,
)

if ranking.rank_table.empty:
    st.error("No stocks passed the current screening/ranking rules. Relax the filters.")
    st.stop()
selected = ranking.rank_table.index.tolist()

if min_weight > max_weight or len(selected) * max_weight < 1 - 1e-10 or len(selected) * min_weight > 1 + 1e-10:
    st.error(
        f"Weight constraints cannot allocate 100% across {len(selected)} selected stocks. "
        f"The maximum weight must be at least {1/len(selected):.1%}."
    )
    st.stop()

returns_all = pct_change_safe(analysis_close)
stock_returns = returns_all[selected].dropna(how="any")
if len(stock_returns) < 30:
    st.error("Not enough aligned return history for the selected portfolio.")
    st.stop()

# Optimization range is intentionally separate from the broad screening range.
date_range = st.slider(
    "Optimization / fit window",
    min_value=stock_returns.index.min().to_pydatetime(),
    max_value=stock_returns.index.max().to_pydatetime(),
    value=(stock_returns.index.min().to_pydatetime(), stock_returns.index.max().to_pydatetime()),
    format="YYYY-MM-DD",
)
fit_returns = stock_returns.loc[date_range[0] : date_range[1]].dropna(how="any")
if len(fit_returns) < 30:
    st.error("Optimization window is too short.")
    st.stop()

f_ann = FREQ_ANNUAL[frequency]
mu_ann = fit_returns.mean().values * f_ann
cov_fit = ledoit_wolf_cov(fit_returns)
cov_ann = cov_fit.values * f_ann

try:
    w_opt = optimize_max_sharpe(mu_ann, cov_ann, risk_free_rate, min_weight, max_weight)
    w_min = optimize_min_volatility(cov_ann, min_weight, max_weight)
except Exception as exc:
    st.error(f"Portfolio optimizer failed: {exc}")
    st.stop()

opt_weights = pd.Series(w_opt, index=selected, name="Optimal Weight")
min_weights = pd.Series(w_min, index=selected, name="Min Risk Weight")

# Black-Litterman views
sector_map = info_df.get("Sector", pd.Series("Unknown", index=selected)).reindex(selected).fillna("Unknown")
market_caps = pd.to_numeric(info_df.get("MarketCap", pd.Series(index=selected, dtype=float)).reindex(selected), errors="coerce")
if market_caps.notna().sum() == 0 or market_caps.fillna(0).sum() <= 0:
    market_weights = pd.Series(1.0 / len(selected), index=selected)
else:
    mc = market_caps.fillna(market_caps.median())
    market_weights = mc / mc.sum()

bl_weights = None
bl_mu_excess = None
bl_view_rows = []
if enable_bl:
    st.sidebar.markdown("**BL sector views**")
    sectors_in_selection = sorted(sector_map.unique())
    chosen_sectors = st.sidebar.multiselect("Pick up to 2 sectors", sectors_in_selection, max_selections=2)
    views = []
    for sec in chosen_sectors:
        expected = st.sidebar.slider(f"{sec}: expected excess return", -20.0, 20.0, 3.0, 0.5) / 100.0
        confidence = st.sidebar.slider(f"{sec}: confidence", 1, 99, 60, 1) / 100.0
        mask = np.array([1.0 if sector_map[t] == sec else 0.0 for t in selected], dtype=float)
        if mask.sum() > 0:
            p = mask / mask.sum()
            views.append((p, expected, confidence))
            bl_view_rows.append((sec, expected, confidence))
    if views:
        try:
            bl_mu_excess = black_litterman_posterior_excess(
                pd.DataFrame(cov_ann, index=selected, columns=selected),
                market_weights,
                views,
                delta=bl_delta,
                tau=bl_tau,
            )
            bl_weights = pd.Series(
                optimize_max_excess_sharpe(bl_mu_excess.values, cov_ann, min_weight, max_weight),
                index=selected,
                name="BL Weight",
            )
        except Exception as exc:
            st.warning(f"Black–Litterman calculation failed: {exc}")

# Current fit-window analytics
port_fit = pd.Series(fit_returns.values @ w_opt, index=fit_returns.index, name="Optimal")
min_fit = pd.Series(fit_returns.values @ w_min, index=fit_returns.index, name="Min Risk")
bench_fit = returns_all[BENCH].reindex(fit_returns.index).fillna(0.0)
fit_metrics = performance_metrics(port_fit, bench_fit, risk_free_rate=risk_free_rate, periods_per_year=f_ann)
concentration = concentration_metrics(opt_weights)
sector_weights = sector_weight_table(opt_weights, sector_map)

expected_ret = float(mu_ann @ w_opt)
expected_vol = float(np.sqrt(max(w_opt.T @ cov_ann @ w_opt, 0.0)))
expected_sharpe = (expected_ret - risk_free_rate) / expected_vol if expected_vol > 1e-12 else np.nan

st.subheader("Current Portfolio Scorecard")
score = st.columns(6)
score[0].metric("Historical ann. return", fmt_pct(expected_ret))
score[1].metric("Expected volatility", fmt_pct(expected_vol))
score[2].metric("Sharpe", fmt_num(expected_sharpe))
score[3].metric("Beta", fmt_num(fit_metrics.get("Beta")))
score[4].metric("Largest position", fmt_pct(concentration.get("Largest Weight")))
score[5].metric("Effective # stocks", fmt_num(concentration.get("Effective N"), 1))

# Frontier cloud is display-only; optimization points are deterministic SLSQP.
cloud = frontier_cloud(mu_ann, cov_ann, risk_free_rate, len(selected), min_weight, max_weight, num_portfolios)
frontier_fig = go.Figure()
frontier_fig.add_trace(
    go.Scatter(
        x=cloud[:, 1],
        y=cloud[:, 0],
        mode="markers",
        marker=dict(color=cloud[:, 2], colorscale="Viridis", showscale=True, size=5),
        name="Feasible samples",
        text=[f"Return {r:.2%} | Vol {v:.2%} | Sharpe {s:.2f}" for r, v, s in cloud],
    )
)
frontier_fig.add_trace(
    go.Scatter(
        x=[expected_vol], y=[expected_ret], mode="markers", marker=dict(size=13, symbol="star"), name="True Max Sharpe"
    )
)
min_ret = float(mu_ann @ w_min)
min_vol = float(np.sqrt(max(w_min.T @ cov_ann @ w_min, 0.0)))
frontier_fig.add_trace(
    go.Scatter(x=[min_vol], y=[min_ret], mode="markers", marker=dict(size=13, symbol="diamond"), name="True Min Vol")
)
if bl_weights is not None:
    bl_vol = float(np.sqrt(max(bl_weights.values.T @ cov_ann @ bl_weights.values, 0.0)))
    bl_excess = float(bl_mu_excess.values @ bl_weights.values)
    frontier_fig.add_trace(
        go.Scatter(x=[bl_vol], y=[bl_excess + risk_free_rate], mode="markers", marker=dict(size=12, symbol="x"), name="BL Max Sharpe")
    )
frontier_fig.update_layout(title="Efficient Frontier — Monte Carlo cloud + deterministic optimizers", xaxis_title="Annual volatility", yaxis_title="Annual return")

# -----------------------------------------------------------------------------
# Main workflow tabs
# -----------------------------------------------------------------------------
tab_overview, tab_screen, tab_opt, tab_backtest, tab_risk, tab_reports = st.tabs(
    ["Overview", "Screen & Rank", "Optimize", "Walk-Forward", "Risk & Stress", "Reports"]
)

with tab_overview:
    st.info(
        "The chart below is an in-sample fit-window simulation. It is useful for describing the fitted portfolio, "
        "but it is not evidence of future performance. Use Walk-Forward for out-of-sample validation."
    )
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(x=port_fit.index, y=wealth_curve(port_fit), name="Optimal (fit window)"))
    cum_fig.add_trace(go.Scatter(x=min_fit.index, y=wealth_curve(min_fit), name="Min Risk (fit window)"))
    cum_fig.add_trace(go.Scatter(x=bench_fit.index, y=wealth_curve(bench_fit), name="S&P 500"))
    if bl_weights is not None:
        bl_fit = pd.Series(fit_returns.values @ bl_weights.values, index=fit_returns.index)
        cum_fig.add_trace(go.Scatter(x=bl_fit.index, y=wealth_curve(bl_fit), name="BL (fit window)"))
    cum_fig.update_layout(title="Growth of $1 — fitted historical window", yaxis_title="Growth of 1", xaxis_title="Date")
    st.plotly_chart(cum_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sector_fig = go.Figure(go.Bar(x=sector_weights["Sector"], y=sector_weights["Weight"]))
        sector_fig.update_layout(title="Sector allocation", yaxis_tickformat=".0%", xaxis_title="Sector", yaxis_title="Weight")
        st.plotly_chart(sector_fig, use_container_width=True)
    with c2:
        dd = drawdown_series(port_fit)
        dd_fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown"))
        dd_fig.update_layout(title="Portfolio drawdown — fit window", yaxis_tickformat=".0%", xaxis_title="Date")
        st.plotly_chart(dd_fig, use_container_width=True)

with tab_screen:
    st.subheader("Screening")
    st.write(f"**{len(ranking.screened)} / {ranking.diagnostics.get('available', len(current_tickers))}** available stocks pass all current filters.")
    st.dataframe(
        ranking.screen_table.style.format(
            {
                "ADTV $ (60d)": "{:,.0f}",
                "Vol (ann)": "{:.2%}",
                "Beta": "{:.2f}",
                "Earnings Growth (YoY)": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    st.subheader("Top-N ranked universe")
    st.caption(f"Effective factor weights: {ranking.diagnostics.get('effective_factor_weights', {})}")
    st.dataframe(
        ranking.rank_table.style.format(
            {
                "Momentum(6/12m)": "{:.2%}",
                "Sharpe(1y ann.)": "{:.2f}",
                "Earnings Yield (1/PE)": "{:.2%}",
                "LowVol (1/σ)": "{:.2f}",
                "Composite Z": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

with tab_opt:
    st.subheader("Deterministic portfolio optimization")
    st.caption("Max-Sharpe and Min-Vol are solved with constrained SLSQP. Monte Carlo is used only to visualize the feasible region.")
    c1, c2 = st.columns(2)
    with c1:
        opt_df = pd.DataFrame({"Ticker": selected, "Optimal Weight": w_opt, "Sector": sector_map.values}).sort_values("Optimal Weight", ascending=False)
        st.markdown("**Max-Sharpe weights**")
        st.dataframe(opt_df.style.format({"Optimal Weight": "{:.2%}"}), use_container_width=True)
    with c2:
        min_df = pd.DataFrame({"Ticker": selected, "Min Risk Weight": w_min, "Sector": sector_map.values}).sort_values("Min Risk Weight", ascending=False)
        st.markdown("**Minimum-volatility weights**")
        st.dataframe(min_df.style.format({"Min Risk Weight": "{:.2%}"}), use_container_width=True)
    st.plotly_chart(frontier_fig, use_container_width=True)

    st.subheader("Black–Litterman")
    if not enable_bl:
        st.info("Enable Black–Litterman views in the sidebar to build a posterior allocation.")
    elif bl_weights is None:
        st.warning("Add at least one valid sector view.")
    else:
        st.caption("BL posterior returns are treated as excess returns; the risk-free rate is not subtracted twice.")
        bl_df = pd.DataFrame({"Ticker": selected, "BL Weight": bl_weights.values, "Sector": sector_map.values}).sort_values("BL Weight", ascending=False)
        st.dataframe(bl_df.style.format({"BL Weight": "{:.2%}"}), use_container_width=True)
        if bl_view_rows:
            st.dataframe(pd.DataFrame(bl_view_rows, columns=["Sector", "Expected Excess Return", "Confidence"]).style.format({"Expected Excess Return": "{:.2%}", "Confidence": "{:.0%}"}), use_container_width=True)

with tab_backtest:
    st.subheader("Point-in-time walk-forward validation")
    st.caption(
        "At each rebalance the model sees only information available through the prior trading session. "
        "Historical constituent changes are reconstructed when stored; current fundamentals are never leaked backwards."
    )
    hist = cached_fundamental_history(db_version)
    bt_status = st.columns(3)
    bt_status[0].metric("Stored PIT fundamental snapshots", len(hist))
    bt_status[1].metric("Constituent change events", len(constituent_changes) if constituent_changes is not None else 0)
    archive_count = len(daily_payload["data"].get("archive_tickers", current_tickers))
    bt_status[2].metric("Daily historical symbols", archive_count)

    b1, b2, b3 = st.columns(3)
    with b1:
        bt_start = st.date_input(
            "Backtest start",
            value=max(pd.Timestamp("2021-01-01"), pd.Timestamp(daily_close_full.index.min())).date(),
            min_value=pd.Timestamp(daily_close_full.index.min()).date(),
            max_value=last_data_date.date(),
            key="bt_start",
        )
        bt_end = st.date_input(
            "Backtest end",
            value=last_data_date.date(),
            min_value=pd.Timestamp(daily_close_full.index.min()).date(),
            max_value=last_data_date.date(),
            key="bt_end",
        )
    with b2:
        bt_rebalance = st.selectbox("Rebalance", ["monthly", "quarterly"], index=0)
        bt_training = st.selectbox("Training window", [126, 252, 504, 756], index=1, format_func=lambda x: f"{x} trading days")
    with b3:
        bt_cost = st.number_input("Transaction cost (bps per traded notional)", 0.0, 200.0, 10.0, 1.0)
        bt_optimizer = st.selectbox("Walk-forward optimizer", ["max_sharpe", "min_vol"], format_func=lambda x: "Max Sharpe" if x == "max_sharpe" else "Minimum Volatility")

    if st.button("▶ Run Walk-Forward Backtest", type="primary", use_container_width=True):
        cfg_dict = {
            "start_date": str(bt_start),
            "end_date": str(bt_end),
            "rebalance": bt_rebalance,
            "training_days": int(bt_training),
            "top_n": int(top_n),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
            "optimizer": bt_optimizer,
            "risk_free_rate": float(risk_free_rate),
            "transaction_cost_bps": float(bt_cost),
            "adtv_usd_min": float(adtv_usd_min),
            "vol_mult_spx": float(vol_mult_spx),
            "beta_min": float(beta_min),
            "beta_max": float(beta_max),
            "min_growth": float(min_growth),
            "factor_weights": factor_weights,
        }
        try:
            result = cached_walk_forward(db_version, json.dumps(cfg_dict, sort_keys=True))
            st.session_state["walk_forward_result"] = result
            st.session_state["walk_forward_config"] = cfg_dict
        except Exception as exc:
            st.error(f"Walk-forward backtest failed: {exc}")

    bt = st.session_state.get("walk_forward_result")
    if bt is None:
        st.info("Run the backtest once to generate out-of-sample analytics.")
    else:
        for warning in bt.warnings:
            st.warning(warning)
        if bt.net_returns.empty:
            st.error("No investable walk-forward period was produced with the selected constraints.")
        else:
            m = bt.metrics
            cards = st.columns(7)
            cards[0].metric("CAGR", fmt_pct(m.get("CAGR")))
            cards[1].metric("Sharpe", fmt_num(m.get("Sharpe")))
            cards[2].metric("Max Drawdown", fmt_pct(m.get("Max Drawdown")))
            cards[3].metric("Alpha", fmt_pct(m.get("Alpha")))
            cards[4].metric("Beta", fmt_num(m.get("Beta")))
            cards[5].metric("Info Ratio", fmt_num(m.get("Information Ratio")))
            cards[6].metric("Avg turnover", fmt_pct(m.get("Average One-way Turnover")))

            bt_cum = go.Figure()
            bt_cum.add_trace(go.Scatter(x=bt.net_returns.index, y=wealth_curve(bt.net_returns), name="Portfolio net"))
            bt_cum.add_trace(go.Scatter(x=bt.gross_returns.index, y=wealth_curve(bt.gross_returns), name="Portfolio gross", line=dict(dash="dot")))
            bt_cum.add_trace(go.Scatter(x=bt.benchmark_returns.index, y=wealth_curve(bt.benchmark_returns), name="S&P 500"))
            bt_cum.update_layout(title="Walk-forward growth of $1", yaxis_title="Growth of 1", xaxis_title="Date")
            st.plotly_chart(bt_cum, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                bt_dd = drawdown_series(bt.net_returns)
                fig = go.Figure(go.Scatter(x=bt_dd.index, y=bt_dd, fill="tozeroy", name="Drawdown"))
                fig.update_layout(title="Walk-forward drawdown", yaxis_tickformat=".0%", xaxis_title="Date")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                rs = rolling_sharpe(bt.net_returns, 63, risk_free_rate, 252)
                rb = rolling_beta(bt.net_returns, bt.benchmark_returns, 63)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rs.index, y=rs, name="63d rolling Sharpe"))
                fig.add_trace(go.Scatter(x=rb.index, y=rb, name="63d rolling Beta"))
                fig.update_layout(title="Rolling validation metrics", xaxis_title="Date")
                st.plotly_chart(fig, use_container_width=True)

            if not bt.rebalance_log.empty:
                successful = bt.rebalance_log[bt.rebalance_log["Status"] == "rebalanced"].copy()
                if not successful.empty:
                    turn_fig = go.Figure(go.Bar(x=successful["Trade Date"], y=successful["Turnover"]))
                    turn_fig.update_layout(title="One-way turnover by rebalance", yaxis_tickformat=".0%", xaxis_title="Trade date")
                    st.plotly_chart(turn_fig, use_container_width=True)
                st.markdown("**Rebalance audit log**")
                st.dataframe(bt.rebalance_log, use_container_width=True)
            st.caption(
                f"Actual invested period: {bt.metadata.get('actual_start')} → {bt.metadata.get('end')} | "
                f"PIT universe: {bt.metadata.get('pit_universe')} | Fundamental snapshots: {bt.metadata.get('fundamental_snapshots')}"
            )

with tab_risk:
    st.subheader("Advanced risk analytics")
    metric_order = [
        "CAGR", "Annual Return", "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar",
        "Alpha", "Beta", "Tracking Error", "Information Ratio", "Correlation"
    ]
    metric_rows = []
    for name in metric_order:
        if name in fit_metrics:
            value = fit_metrics[name]
            display = fmt_pct(value) if name in {"CAGR", "Annual Return", "Annual Volatility", "Max Drawdown", "Alpha", "Tracking Error"} else fmt_num(value)
            metric_rows.append((name, display))
    st.table(pd.DataFrame(metric_rows, columns=["Metric", "Value"]))

    horizon = RISK_HORIZON_LABEL[frequency]
    st.markdown(f"**Historical VaR / CVaR ({horizon} observations)**")
    var_rows = [
        (f"VaR 95% ({horizon})", fmt_pct(fit_metrics.get("VaR 95%"))),
        (f"CVaR 95% ({horizon})", fmt_pct(fit_metrics.get("CVaR 95%"))),
        (f"VaR 99% ({horizon})", fmt_pct(fit_metrics.get("VaR 99%"))),
        (f"CVaR 99% ({horizon})", fmt_pct(fit_metrics.get("CVaR 99%"))),
    ]
    st.table(pd.DataFrame(var_rows, columns=["Metric", "Value"]))

    rc = volatility_risk_contributions(w_opt, cov_ann, labels=selected)
    rc_pct = rc / rc.sum() if abs(rc.sum()) > 1e-12 else rc
    rc_df = pd.DataFrame({"Ticker": selected, "Risk Contribution": rc.values, "Risk Contribution %": rc_pct.values, "Sector": sector_map.values}).sort_values("Risk Contribution %", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        rc_fig = go.Figure(go.Bar(x=rc_df.head(20)["Ticker"], y=rc_df.head(20)["Risk Contribution %"]))
        rc_fig.update_layout(title="Top volatility risk contributors", yaxis_tickformat=".0%", xaxis_title="Ticker")
        st.plotly_chart(rc_fig, use_container_width=True)
    with c2:
        rc_sector = rc_df.groupby("Sector", as_index=False)["Risk Contribution"].sum().sort_values("Risk Contribution", ascending=False)
        rc_sector["Risk Contribution %"] = rc_sector["Risk Contribution"] / rc_sector["Risk Contribution"].sum()
        sec_fig = go.Figure(go.Bar(x=rc_sector["Sector"], y=rc_sector["Risk Contribution %"]))
        sec_fig.update_layout(title="Risk contribution by sector", yaxis_tickformat=".0%", xaxis_title="Sector")
        st.plotly_chart(sec_fig, use_container_width=True)

    st.subheader("Stress tests")
    synth = synthetic_stress_table(opt_weights, sector_map)
    hist_stress = historical_stress_table(port_fit)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Transparent synthetic macro proxies**")
        st.dataframe(synth.style.format({"Portfolio Return": "{:.2%}"}), use_container_width=True)
    with s2:
        st.markdown("**Historical episodes in the available series**")
        if hist_stress.empty:
            st.info("Selected fit window does not contain the predefined historical stress periods.")
        else:
            st.dataframe(hist_stress.style.format({"Portfolio Return": "{:.2%}"}), use_container_width=True)
    st.caption("Synthetic stress shocks are explicit scenario assumptions, not forecasts or a structural macro model.")

with tab_reports:
    st.subheader("Downloads")
    st.download_button(
        "⬇️ Top-N ranking CSV",
        data=ranking.rank_table.reset_index().to_csv(index=False).encode(),
        file_name="topN_ranked.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Optimal weights CSV",
        data=pd.DataFrame({"Ticker": selected, "Optimal Weight": w_opt, "Sector": sector_map.values}).to_csv(index=False).encode(),
        file_name="optimal_weights.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Minimum-risk weights CSV",
        data=pd.DataFrame({"Ticker": selected, "Min Risk Weight": w_min, "Sector": sector_map.values}).to_csv(index=False).encode(),
        file_name="minrisk_weights.csv",
        mime="text/csv",
    )
    if bl_weights is not None:
        st.download_button(
            "⬇️ Black–Litterman weights CSV",
            data=pd.DataFrame({"Ticker": selected, "BL Weight": bl_weights.values}).to_csv(index=False).encode(),
            file_name="black_litterman_weights.csv",
            mime="text/csv",
        )
    bt = st.session_state.get("walk_forward_result")
    if bt is not None and not bt.net_returns.empty:
        bt_returns = pd.concat([bt.net_returns, bt.gross_returns, bt.benchmark_returns], axis=1)
        st.download_button("⬇️ Walk-forward returns CSV", bt_returns.to_csv().encode(), "walk_forward_returns.csv", "text/csv")
        st.download_button("⬇️ Rebalance log CSV", bt.rebalance_log.to_csv(index=False).encode(), "rebalance_log.csv", "text/csv")
        if not bt.weights_history.empty:
            st.download_button("⬇️ Walk-forward weights CSV", bt.weights_history.to_csv().encode(), "walk_forward_weights.csv", "text/csv")

    st.subheader("Investment allocation")
    last_prices = daily_close_full[selected].loc[: pd.Timestamp(end_date)].ffill().iloc[-1].dropna()
    alloc = pd.DataFrame({"Ticker": selected, "Weight": w_opt}).merge(last_prices.rename("Price"), left_on="Ticker", right_index=True, how="inner")
    alloc["Target $"] = investment_amount * alloc["Weight"]
    if allow_fractional:
        alloc["Shares"] = alloc["Target $"] / alloc["Price"]
    else:
        alloc["Shares"] = np.floor(alloc["Target $"] / alloc["Price"])
    alloc["Cost"] = alloc["Shares"] * alloc["Price"]
    total_cost = float(alloc["Cost"].sum())
    remaining = investment_amount - total_cost
    st.dataframe(
        alloc.sort_values("Weight", ascending=False).style.format(
            {"Weight": "{:.2%}", "Price": "${:,.2f}", "Target $": "${:,.2f}", "Shares": "{:.4f}", "Cost": "${:,.2f}"}
        ),
        use_container_width=True,
    )
    st.write(f"**Total cost:** ${total_cost:,.2f} | **Remaining cash:** ${remaining:,.2f}")
