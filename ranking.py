"""Screening and factor-ranking logic shared by the UI and walk-forward engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_math import (
    FREQ_ANNUAL,
    MOMENTUM_WINDOWS,
    annualize_mean_std,
    compute_adtv_usd,
    compute_beta,
    zscore,
)


@dataclass
class RankingResult:
    screened: list[str]
    screen_table: pd.DataFrame
    rank_table: pd.DataFrame
    diagnostics: dict


def pct_change_safe(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.ffill().pct_change(fill_method=None).dropna(how="all")


def _cumulative_return(ret: pd.DataFrame, window: int) -> pd.Series:
    if ret.empty:
        return pd.Series(dtype=float)
    n = min(window, len(ret))
    return (1.0 + ret.iloc[-n:]).prod() - 1.0


def screen_and_rank(
    close: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_volume: pd.DataFrame,
    info: pd.DataFrame | None,
    universe: list[str],
    *,
    bench: str = "^GSPC",
    frequency: str = "daily",
    risk_free_rate: float = 0.02,
    adtv_usd_min: float = 1_000_000,
    vol_mult_spx: float = 3.0,
    beta_min: float = 0.8,
    beta_max: float = 1.3,
    min_growth: float = 0.0,
    selected_sectors: list[str] | None = None,
    top_n: int = 30,
    factor_weights: dict[str, float] | None = None,
    use_growth_filter: bool = True,
    use_value_factor: bool = True,
) -> RankingResult:
    """Screen and rank a universe using only data supplied by the caller.

    Passing a historical ``info`` snapshot makes the function point-in-time safe.
    If no historical fundamentals exist, callers should set ``use_growth_filter``
    and ``use_value_factor`` to False rather than leaking today's fundamentals.
    """
    if factor_weights is None:
        factor_weights = {"momentum": 0.40, "quality": 0.30, "value": 0.20, "lowvol": 0.10}
    if frequency not in FREQ_ANNUAL:
        raise ValueError(f"Unsupported frequency: {frequency}")

    close = close.sort_index()
    daily_close = daily_close.sort_index()
    daily_volume = daily_volume.reindex(daily_close.index)
    returns_all = pct_change_safe(close)
    if bench not in returns_all.columns:
        raise ValueError("Benchmark is missing from the selected data window.")

    available = [t for t in universe if t in returns_all.columns and returns_all[t].notna().sum() >= 30]
    if not available:
        return RankingResult([], pd.DataFrame(), pd.DataFrame(), {"reason": "no_available_tickers"})

    returns = returns_all[available]
    spx_ret = returns_all[bench].dropna()
    _, spx_sigma_ann, _ = annualize_mean_std(spx_ret, frequency)
    vols = returns.std() * np.sqrt(FREQ_ANNUAL[frequency])
    betas = compute_beta(returns_all[available + [bench]], bench)

    daily_names = [t for t in available if t in daily_close.columns and t in daily_volume.columns]
    adtv = compute_adtv_usd(daily_close[daily_names], daily_volume[daily_names], 60).reindex(available)

    if info is None:
        info = pd.DataFrame(index=available)
    else:
        info = info.reindex(available).copy()

    sectors = info.get("Sector", pd.Series("Unknown", index=available)).fillna("Unknown").astype(str)
    growth = pd.to_numeric(
        info.get("AnalystGrowth", pd.Series(np.nan, index=available)), errors="coerce"
    )

    masks = pd.DataFrame(index=available)
    masks["liquidity"] = adtv >= float(adtv_usd_min)
    masks["volatility"] = vols <= float(vol_mult_spx) * float(spx_sigma_ann)
    masks["beta"] = betas.between(float(beta_min), float(beta_max), inclusive="both")
    masks["growth"] = growth >= float(min_growth) if use_growth_filter else True
    if selected_sectors:
        masks["sector"] = sectors.isin(selected_sectors)
    else:
        masks["sector"] = True

    screened = masks.index[masks.fillna(False).all(axis=1)].tolist()
    screen_table = pd.DataFrame(
        {
            "Sector": sectors,
            "Sector included": masks["sector"],
            "ADTV $ (60d)": adtv,
            "Vol (ann)": vols,
            "Beta": betas,
            "Earnings Growth (YoY)": growth,
            "Pass": masks.fillna(False).all(axis=1),
        }
    ).sort_values("ADTV $ (60d)", ascending=False)

    if not screened:
        return RankingResult([], screen_table, pd.DataFrame(), {"reason": "screen_empty"})

    w6, w12 = MOMENTUM_WINDOWS[frequency]
    hist_n = min(len(returns), w12)
    ret_win = returns.iloc[-hist_n:]
    mom6 = _cumulative_return(ret_win, w6)
    mom12 = _cumulative_return(ret_win, w12)
    momentum = (0.5 * mom6 + 0.5 * mom12).reindex(screened)

    f = FREQ_ANNUAL[frequency]
    rf_period = risk_free_rate / f
    quality = (((ret_win.mean() - rf_period) / ret_win.std()) * np.sqrt(f)).reindex(screened)
    lowvol = (1.0 / (ret_win.std() * np.sqrt(f)).replace(0, np.nan)).reindex(screened)

    pe = pd.to_numeric(info.get("PE", pd.Series(np.nan, index=available)), errors="coerce")
    earn_yield = (1.0 / pe.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).reindex(screened)

    raw_factors = {
        "momentum": momentum,
        "quality": quality,
        "value": earn_yield,
        "lowvol": lowvol,
    }
    enabled = ["momentum", "quality", "lowvol"]
    if use_value_factor and earn_yield.notna().sum() >= max(3, int(0.25 * len(screened))):
        enabled.append("value")

    requested = {k: max(0.0, float(factor_weights.get(k, 0.0))) for k in enabled}
    total = sum(requested.values())
    if total <= 0:
        requested = {k: 1.0 for k in enabled}
        total = float(len(enabled))
    effective_weights = {k: v / total for k, v in requested.items()}

    z_factors = {}
    for name in enabled:
        s = raw_factors[name].replace([np.inf, -np.inf], np.nan)
        if s.notna().any():
            s = s.fillna(s.median())
        else:
            s = pd.Series(0.0, index=screened)
        z_factors[name] = zscore(s)

    composite = sum(effective_weights[k] * z_factors[k] for k in enabled).sort_values(ascending=False)
    ranked = composite.index.tolist()[: min(int(top_n), len(composite))]
    rank_table = pd.DataFrame(
        {
            "Sector": sectors.reindex(ranked),
            "Momentum(6/12m)": momentum.reindex(ranked),
            "Sharpe(1y ann.)": quality.reindex(ranked),
            "Earnings Yield (1/PE)": earn_yield.reindex(ranked),
            "LowVol (1/σ)": lowvol.reindex(ranked),
            "Composite Z": composite.reindex(ranked),
        },
        index=ranked,
    )
    rank_table.index.name = "Ticker"

    diagnostics = {
        "available": len(available),
        "screened": len(screened),
        "ranked": len(ranked),
        "effective_factor_weights": effective_weights,
        "fundamentals_used": bool(use_growth_filter or "value" in enabled),
        "value_factor_used": "value" in enabled,
        "growth_filter_used": bool(use_growth_filter),
    }
    return RankingResult(screened, screen_table, rank_table, diagnostics)
