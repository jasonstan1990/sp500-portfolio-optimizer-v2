"""Risk, performance and stress-test analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_math import normalize_stress_sector


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = returns.dropna().astype(float)
    wealth = (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if pd.notna(b) and abs(float(b)) > 1e-12 else np.nan


def performance_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {}
    n = len(r)
    years = n / periods_per_year
    wealth = (1.0 + r).cumprod()
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    ann_mean = float(r.mean() * periods_per_year)
    sharpe = _safe_div(ann_mean - risk_free_rate, vol)
    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else np.nan
    sortino = _safe_div(ann_mean - risk_free_rate, downside_vol)
    dd = drawdown_series(r)
    max_dd = float(dd.min()) if not dd.empty else np.nan
    calmar = _safe_div(cagr, abs(max_dd)) if pd.notna(max_dd) else np.nan

    q95 = float(r.quantile(0.05))
    q99 = float(r.quantile(0.01))
    var95 = -q95
    var99 = -q99
    cvar95 = -float(r[r <= q95].mean()) if (r <= q95).any() else np.nan
    cvar99 = -float(r[r <= q99].mean()) if (r <= q99).any() else np.nan

    result = {
        "CAGR": cagr,
        "Annual Return": ann_mean,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "VaR 95%": var95,
        "CVaR 95%": cvar95,
        "VaR 99%": var99,
        "CVaR 99%": cvar99,
    }

    if benchmark is not None:
        joined = pd.concat([r.rename("p"), benchmark.rename("b")], axis=1).dropna()
        if len(joined) >= 30:
            p = joined["p"]
            b = joined["b"]
            var_b = float(b.var(ddof=1))
            beta = float(p.cov(b) / var_b) if var_b > 0 else np.nan
            rf_period = risk_free_rate / periods_per_year
            alpha_period = float((p.mean() - rf_period) - beta * (b.mean() - rf_period)) if pd.notna(beta) else np.nan
            alpha = alpha_period * periods_per_year if pd.notna(alpha_period) else np.nan
            active = p - b
            tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
            active_return = float(active.mean() * periods_per_year)
            info_ratio = _safe_div(active_return, tracking_error)
            corr = float(p.corr(b))
            result.update(
                {
                    "Alpha": alpha,
                    "Beta": beta,
                    "Tracking Error": tracking_error,
                    "Information Ratio": info_ratio,
                    "Correlation": corr,
                }
            )
    return result


def concentration_metrics(weights: pd.Series) -> dict[str, float]:
    w = weights.dropna().astype(float)
    w = w[w.abs() > 1e-12]
    if w.empty:
        return {"HHI": np.nan, "Effective N": np.nan, "Largest Weight": np.nan}
    hhi = float((w**2).sum())
    return {
        "HHI": hhi,
        "Effective N": _safe_div(1.0, hhi),
        "Largest Weight": float(w.max()),
    }


def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> pd.Series:
    r = returns.astype(float)
    mean = r.rolling(window).mean() * periods_per_year
    vol = r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return (mean - risk_free_rate) / vol.replace(0, np.nan)


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    joined = pd.concat([returns.rename("p"), benchmark.rename("b")], axis=1)
    cov = joined["p"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var(ddof=1)
    return cov / var.replace(0, np.nan)


# Scenario shocks are intentionally transparent, user-facing proxies rather than a
# claim of a structural macro model.  Values are one-period equity shocks.
STRESS_SCENARIOS = {
    "Rates +200 bps": {
        "Technology": -0.12,
        "Communication Services": -0.09,
        "Consumer Discretionary": -0.10,
        "Financial Services": -0.03,
        "Industrials": -0.06,
        "Health Care": -0.04,
        "Energy": -0.02,
        "Materials": -0.05,
        "Consumer Defensive": -0.03,
        "Real Estate": -0.14,
        "Utilities": -0.10,
        "Unknown": -0.07,
    },
    "Recession": {
        "Technology": -0.18,
        "Communication Services": -0.16,
        "Consumer Discretionary": -0.22,
        "Financial Services": -0.20,
        "Industrials": -0.21,
        "Health Care": -0.09,
        "Energy": -0.24,
        "Materials": -0.20,
        "Consumer Defensive": -0.08,
        "Real Estate": -0.18,
        "Utilities": -0.09,
        "Unknown": -0.16,
    },
    "Technology crash": {
        "Technology": -0.30,
        "Communication Services": -0.22,
        "Consumer Discretionary": -0.15,
        "Financial Services": -0.08,
        "Industrials": -0.08,
        "Health Care": -0.05,
        "Energy": -0.03,
        "Materials": -0.05,
        "Consumer Defensive": -0.03,
        "Real Estate": -0.07,
        "Utilities": -0.03,
        "Unknown": -0.10,
    },
    "Inflation shock": {
        "Technology": -0.14,
        "Communication Services": -0.10,
        "Consumer Discretionary": -0.13,
        "Financial Services": -0.04,
        "Industrials": -0.08,
        "Health Care": -0.05,
        "Energy": 0.08,
        "Materials": 0.03,
        "Consumer Defensive": -0.04,
        "Real Estate": -0.10,
        "Utilities": -0.06,
        "Unknown": -0.07,
    },
    "Broad equity -20%": {k: -0.20 for k in [
        "Technology", "Communication Services", "Consumer Discretionary", "Financial Services",
        "Industrials", "Health Care", "Energy", "Materials", "Consumer Defensive",
        "Real Estate", "Utilities", "Unknown"
    ]},
}


def synthetic_stress_table(weights: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["Scenario", "Portfolio Return"])
    sectors = sectors.reindex(weights.index).map(normalize_stress_sector).fillna("Unknown")
    rows = []
    for name, shock_map in STRESS_SCENARIOS.items():
        shocks = sectors.map(lambda s: shock_map.get(s, shock_map.get("Unknown", 0.0)))
        result = float((weights * shocks).sum())
        rows.append((name, result))
    return pd.DataFrame(rows, columns=["Scenario", "Portfolio Return"])


def historical_stress_table(returns: pd.Series) -> pd.DataFrame:
    r = returns.dropna()
    periods = {
        "COVID crash (2020-02-19 to 2020-03-23)": ("2020-02-19", "2020-03-23"),
        "2022 inflation/rates selloff": ("2022-01-03", "2022-10-12"),
    }
    rows = []
    for name, (start, end) in periods.items():
        piece = r.loc[start:end]
        if len(piece) >= 2:
            rows.append((name, float((1.0 + piece).prod() - 1.0)))
    if len(r) >= 21:
        monthly = (1.0 + r).resample("ME").prod() - 1.0
        if not monthly.empty:
            worst = monthly.idxmin()
            rows.append((f"Worst calendar month ({worst:%Y-%m})", float(monthly.loc[worst])))
    return pd.DataFrame(rows, columns=["Scenario", "Portfolio Return"])
