"""Point-in-time aware walk-forward portfolio backtester."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from data_store import latest_fundamentals_asof
from portfolio_math import ledoit_wolf_cov, optimize_max_sharpe, optimize_min_volatility
from ranking import pct_change_safe, screen_and_rank
from risk import performance_metrics
from universe import changes_coverage_start, membership_on_date


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str | pd.Timestamp = "2021-01-01"
    end_date: str | pd.Timestamp | None = None
    rebalance: str = "monthly"  # monthly | quarterly
    training_days: int = 252
    top_n: int = 30
    min_weight: float = 0.0
    max_weight: float = 0.10
    optimizer: str = "max_sharpe"  # max_sharpe | min_vol
    risk_free_rate: float = 0.02
    transaction_cost_bps: float = 10.0
    adtv_usd_min: float = 1_000_000
    vol_mult_spx: float = 3.0
    beta_min: float = 0.8
    beta_max: float = 1.3
    min_growth: float = 0.0
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {"momentum": 0.40, "quality": 0.30, "value": 0.20, "lowvol": 0.10}
    )


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    benchmark_returns: pd.Series
    weights_history: pd.DataFrame
    rebalance_log: pd.DataFrame
    metrics: dict[str, float]
    warnings: list[str]
    metadata: dict


def _signal_dates(index: pd.DatetimeIndex, start, end, rebalance: str) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if idx.empty:
        return []
    frame = pd.DataFrame(index=idx)
    if rebalance == "monthly":
        keys = [idx.to_period("M")]
    elif rebalance == "quarterly":
        keys = [idx.to_period("Q")]
    else:
        raise ValueError("rebalance must be 'monthly' or 'quarterly'")
    # Last trading day in each period is the signal date; orders trade next session.
    return list(frame.groupby(keys).tail(1).index)


def _next_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(pd.Timestamp(date), side="right")
    return None if pos >= len(index) else pd.Timestamp(index[pos])


def _target_for_asof(
    asof: pd.Timestamp,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    current_members: list[str],
    constituent_changes: pd.DataFrame | None,
    fundamental_history: dict[pd.Timestamp, pd.DataFrame],
    cfg: BacktestConfig,
    bench: str,
):
    hist_close = close.loc[:asof]
    hist_volume = volume.reindex(hist_close.index)
    members = sorted(membership_on_date(asof, current_members, constituent_changes))

    info_asof = latest_fundamentals_asof(fundamental_history, asof)
    fundamentals_available = info_asof is not None
    # Never leak today's fundamentals backwards.  Before the first saved PIT
    # snapshot, the backtest intentionally becomes price-only.
    ranking = screen_and_rank(
        hist_close,
        hist_close,
        hist_volume,
        info_asof,
        members,
        bench=bench,
        frequency="daily",
        risk_free_rate=cfg.risk_free_rate,
        adtv_usd_min=cfg.adtv_usd_min,
        vol_mult_spx=cfg.vol_mult_spx,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
        min_growth=cfg.min_growth,
        selected_sectors=None,
        top_n=cfg.top_n,
        factor_weights=cfg.factor_weights,
        use_growth_filter=fundamentals_available,
        use_value_factor=fundamentals_available,
    )
    selected = ranking.rank_table.index.tolist()
    min_assets = max(2, int(math.ceil(1.0 / cfg.max_weight - 1e-12))) if cfg.max_weight > 0 else 10**9
    if len(selected) < min_assets:
        return None, {
            "reason": f"Only {len(selected)} ranked stocks; at least {min_assets} required by max-weight constraint.",
            "fundamentals_available": fundamentals_available,
            "members": len(members),
            "screened": len(ranking.screened),
        }

    train_close = hist_close[selected].tail(cfg.training_days + 1)
    train_ret = pct_change_safe(train_close)
    # Keep only assets with enough usable observations, then recompute ranking order.
    min_obs = min(126, max(60, cfg.training_days // 2))
    usable = [t for t in selected if t in train_ret and train_ret[t].notna().sum() >= min_obs]
    if len(usable) < min_assets:
        return None, {
            "reason": f"Only {len(usable)} assets have sufficient training history.",
            "fundamentals_available": fundamentals_available,
            "members": len(members),
            "screened": len(ranking.screened),
        }
    usable = usable[: cfg.top_n]
    aligned = train_ret[usable].dropna(how="any")
    if len(aligned) < min_obs:
        return None, {
            "reason": f"Only {len(aligned)} aligned observations after cleaning.",
            "fundamentals_available": fundamentals_available,
            "members": len(members),
            "screened": len(ranking.screened),
        }

    cov_ann = ledoit_wolf_cov(aligned).values * 252.0
    mu_ann = aligned.mean().values * 252.0
    if cfg.optimizer == "min_vol":
        weights = optimize_min_volatility(cov_ann, cfg.min_weight, cfg.max_weight)
    else:
        weights = optimize_max_sharpe(mu_ann, cov_ann, cfg.risk_free_rate, cfg.min_weight, cfg.max_weight)
    target = pd.Series(weights, index=usable, dtype=float)
    target = target[target > 1e-10]
    target = target / target.sum()
    return target, {
        "reason": "ok",
        "fundamentals_available": fundamentals_available,
        "members": len(members),
        "screened": len(ranking.screened),
        "selected": len(target),
        "effective_factor_weights": ranking.diagnostics.get("effective_factor_weights", {}),
    }


def run_walk_forward(
    daily_close: pd.DataFrame,
    daily_volume: pd.DataFrame,
    current_members: list[str],
    constituent_changes: pd.DataFrame | None,
    fundamental_history: dict[pd.Timestamp, pd.DataFrame] | None,
    cfg: BacktestConfig,
    *,
    bench: str = "^GSPC",
) -> BacktestResult:
    close = daily_close.copy().sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    volume = daily_volume.reindex(close.index).copy()
    fundamental_history = fundamental_history or {}
    if bench not in close.columns:
        raise ValueError("Benchmark is missing from daily data.")

    all_returns = pct_change_safe(close)
    end = pd.Timestamp(cfg.end_date) if cfg.end_date is not None else close.index.max()
    requested_start = pd.Timestamp(cfg.start_date)
    signal_start = max(close.index.min(), requested_start - pd.Timedelta(days=int(cfg.training_days * 1.7)))
    signals = _signal_dates(close.index, signal_start, end, cfg.rebalance)
    trade_to_signal = {}
    for signal in signals:
        trade = _next_trading_day(close.index, signal)
        if trade is not None and trade <= end and trade >= requested_start:
            trade_to_signal[trade] = signal

    simulation_days = close.index[(close.index >= requested_start) & (close.index <= end)]
    if simulation_days.empty:
        raise ValueError("Backtest date range has no observations.")

    current_weights = pd.Series(dtype=float)
    net_values = []
    gross_values = []
    bench_values = []
    out_dates = []
    weight_rows = []
    logs = []
    warnings: list[str] = []
    cost_rate = float(cfg.transaction_cost_bps) / 10_000.0

    coverage_start = changes_coverage_start(constituent_changes)
    pit_universe = constituent_changes is not None and not constituent_changes.empty
    if not pit_universe:
        warnings.append("No constituent-change history is stored; backtest uses the saved current universe.")
    elif coverage_start is not None and requested_start < coverage_start:
        warnings.append(
            f"Constituent change history begins {coverage_start:%Y-%m-%d}; earlier membership reconstruction may be incomplete."
        )
    if not fundamental_history:
        warnings.append(
            "No historical fundamental snapshots exist yet. Backtest is PIT-safe by disabling growth/value until snapshots accumulate."
        )

    for day in simulation_days:
        rebalance_cost = 0.0
        if day in trade_to_signal:
            signal = trade_to_signal[day]
            target, diag = _target_for_asof(
                signal,
                close,
                volume,
                current_members,
                constituent_changes,
                fundamental_history,
                cfg,
                bench,
            )
            if target is None:
                logs.append(
                    {
                        "Trade Date": day,
                        "Signal Date": signal,
                        "Status": "skipped",
                        "Reason": diag.get("reason"),
                        "Turnover": np.nan,
                        "Cost": np.nan,
                        "Stocks": 0,
                        "PIT Fundamentals": diag.get("fundamentals_available", False),
                    }
                )
            else:
                union = current_weights.index.union(target.index)
                old = current_weights.reindex(union, fill_value=0.0)
                new = target.reindex(union, fill_value=0.0)
                two_sided_turnover = float((new - old).abs().sum())
                one_way_turnover = two_sided_turnover / 2.0
                rebalance_cost = cost_rate * two_sided_turnover
                current_weights = target.copy()
                logs.append(
                    {
                        "Trade Date": day,
                        "Signal Date": signal,
                        "Status": "rebalanced",
                        "Reason": "ok",
                        "Turnover": one_way_turnover,
                        "Cost": rebalance_cost,
                        "Stocks": len(target),
                        "PIT Fundamentals": diag.get("fundamentals_available", False),
                    }
                )
                row = {"Date": day, **{k: float(v) for k, v in target.items()}}
                weight_rows.append(row)

        day_ret = all_returns.loc[day] if day in all_returns.index else pd.Series(dtype=float)
        if current_weights.empty:
            gross = 0.0
        else:
            asset_ret = day_ret.reindex(current_weights.index).fillna(0.0).astype(float)
            gross = float((current_weights * asset_ret).sum())
        net = (1.0 - rebalance_cost) * (1.0 + gross) - 1.0
        bench_ret = float(day_ret.get(bench, 0.0))

        out_dates.append(day)
        gross_values.append(gross)
        net_values.append(net)
        bench_values.append(bench_ret)

        # Let weights drift naturally with relative asset returns between rebalances.
        if not current_weights.empty:
            asset_ret = day_ret.reindex(current_weights.index).fillna(0.0).astype(float)
            denom = 1.0 + gross
            if denom > 1e-12:
                current_weights = current_weights * (1.0 + asset_ret) / denom
                current_weights = current_weights[current_weights.abs() > 1e-12]
                current_weights = current_weights / current_weights.sum()

    net_s = pd.Series(net_values, index=pd.DatetimeIndex(out_dates), name="Portfolio Net")
    gross_s = pd.Series(gross_values, index=net_s.index, name="Portfolio Gross")
    bench_s = pd.Series(bench_values, index=net_s.index, name="S&P 500")
    # Drop the pre-investment flat prefix before the first successful rebalance.
    log_df = pd.DataFrame(logs)
    successful = log_df[log_df["Status"] == "rebalanced"] if not log_df.empty else pd.DataFrame()
    if not successful.empty:
        first_trade = pd.Timestamp(successful.iloc[0]["Trade Date"])
        net_s = net_s.loc[first_trade:]
        gross_s = gross_s.loc[first_trade:]
        bench_s = bench_s.loc[first_trade:]
    else:
        warnings.append("No successful rebalance occurred with the selected constraints.")

    weights_df = pd.DataFrame(weight_rows)
    if not weights_df.empty:
        weights_df = weights_df.set_index("Date").fillna(0.0).sort_index()
    metrics = performance_metrics(
        net_s,
        bench_s,
        risk_free_rate=cfg.risk_free_rate,
        periods_per_year=252,
    )
    total_cost = float(log_df["Cost"].fillna(0).sum()) if not log_df.empty else 0.0
    avg_turnover = float(log_df.loc[log_df["Status"] == "rebalanced", "Turnover"].mean()) if not successful.empty else np.nan
    metrics["Transaction Costs (sum of return drag)"] = total_cost
    metrics["Average One-way Turnover"] = avg_turnover

    metadata = {
        "requested_start": str(requested_start.date()),
        "actual_start": None if net_s.empty else str(net_s.index.min().date()),
        "end": str(end.date()),
        "rebalance": cfg.rebalance,
        "training_days": cfg.training_days,
        "pit_universe": pit_universe,
        "constituent_history_start": None if coverage_start is None else str(coverage_start.date()),
        "fundamental_snapshots": len(fundamental_history),
        "transaction_cost_bps": cfg.transaction_cost_bps,
    }
    return BacktestResult(net_s, gross_s, bench_s, weights_df, log_df, metrics, warnings, metadata)
