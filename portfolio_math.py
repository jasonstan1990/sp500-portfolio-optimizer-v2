"""Quant helpers for the S&P 500 Portfolio Optimizer.

This module is deliberately UI-free so the numerical methods can be tested
without starting Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

FREQ_ANNUAL = {"daily": 252, "weekly": 52, "monthly": 12}
MOMENTUM_WINDOWS = {
    "daily": (126, 252),
    "weekly": (26, 52),
    "monthly": (6, 12),
}
RISK_HORIZON_LABEL = {
    "daily": "1-day",
    "weekly": "1-week",
    "monthly": "1-month",
}


def annualize_mean_std(ret: pd.Series, freq: str = "daily"):
    f = FREQ_ANNUAL[freq]
    mu = ret.mean() * f
    sigma = ret.std() * np.sqrt(f)
    return mu, sigma, f


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.dropna()
    if clean.empty or clean.shape[0] < 2:
        raise ValueError("Not enough aligned return observations for covariance estimation.")
    lw = LedoitWolf().fit(clean.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


def compute_adtv_usd(close: pd.DataFrame, vol: pd.DataFrame, lookback_days: int = 60) -> pd.Series:
    """Average daily dollar volume using DAILY close and DAILY volume data."""
    if close.empty or vol.empty:
        return pd.Series(dtype=float)
    n = min(int(lookback_days), len(close))
    recent_close = close.iloc[-n:]
    recent_vol = vol.reindex(recent_close.index).iloc[-n:]
    return recent_close.mul(recent_vol, axis=0).mean(axis=0)


def compute_beta(returns: pd.DataFrame, bench_col: str) -> pd.Series:
    if bench_col not in returns.columns:
        return pd.Series(index=returns.columns, dtype=float)
    rm = returns[bench_col].dropna()
    betas = {}
    for c in returns.columns:
        if c == bench_col:
            continue
        ri = returns[c].reindex(rm.index).dropna()
        common = ri.index.intersection(rm.index)
        if len(common) < 30:
            betas[c] = np.nan
            continue
        market = rm.loc[common].to_numpy(dtype=float)
        asset = ri.loc[common].to_numpy(dtype=float)
        varm = np.var(market, ddof=1)
        cov = np.cov(asset, market, ddof=1)[0, 1]
        betas[c] = cov / varm if varm > 0 else np.nan
    return pd.Series(betas)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if pd.notna(std) and std != 0 else s * 0


def validate_weight_bounds(n: int, min_w: float, max_w: float) -> None:
    if n <= 0:
        raise ValueError("Portfolio must contain at least one asset.")
    if min_w < 0 or max_w > 1 or min_w > max_w:
        raise ValueError("Invalid portfolio weight bounds.")
    if n * max_w < 1 - 1e-10 or n * min_w > 1 + 1e-10:
        raise ValueError("Weight bounds cannot allocate 100% across the selected assets.")


def project_to_bounded_simplex(x, min_w: float, max_w: float) -> np.ndarray:
    """Project a vector onto sum(w)=1 with identical lower/upper bounds."""
    x = np.asarray(x, dtype=float)
    validate_weight_bounds(len(x), min_w, max_w)
    lo = float(np.min(x - max_w))
    hi = float(np.max(x - min_w))
    for _ in range(100):
        mid = (lo + hi) / 2.0
        s = np.clip(x - mid, min_w, max_w).sum()
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(x - (lo + hi) / 2.0, min_w, max_w)
    # Remove tiny floating error while respecting the bounds.
    residual = 1.0 - w.sum()
    if abs(residual) > 1e-12:
        if residual > 0:
            room = max_w - w
        else:
            room = w - min_w
        idx = np.where(room > 1e-12)[0]
        if len(idx):
            j = idx[np.argmax(room[idx])]
            w[j] += residual
    return w


def sample_weights(rng: np.random.Generator, n: int, min_w: float, max_w: float) -> np.ndarray:
    return project_to_bounded_simplex(rng.random(n), min_w, max_w)


def _solve(objective, n: int, min_w: float, max_w: float) -> np.ndarray:
    validate_weight_bounds(n, min_w, max_w)
    x0 = np.full(n, 1.0 / n)
    bounds = [(min_w, max_w)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"Numerical optimizer failed: {res.message}")
    return project_to_bounded_simplex(res.x, min_w, max_w)


def optimize_min_volatility(cov_ann, min_w: float, max_w: float) -> np.ndarray:
    cov_ann = np.asarray(cov_ann, dtype=float)
    n = cov_ann.shape[0]

    def objective(w):
        variance = float(w.T @ cov_ann @ w)
        return np.sqrt(max(variance, 0.0))

    return _solve(objective, n, min_w, max_w)


def optimize_max_sharpe(mu_ann, cov_ann, risk_free_rate: float, min_w: float, max_w: float) -> np.ndarray:
    mu_ann = np.asarray(mu_ann, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    n = len(mu_ann)

    def objective(w):
        variance = float(w.T @ cov_ann @ w)
        sigma = np.sqrt(max(variance, 0.0))
        if sigma <= 1e-12:
            return 1e6
        return -float((mu_ann @ w - risk_free_rate) / sigma)

    return _solve(objective, n, min_w, max_w)


def optimize_max_excess_sharpe(mu_excess_ann, cov_ann, min_w: float, max_w: float) -> np.ndarray:
    """Maximize Sharpe when expected returns are already EXCESS returns."""
    mu_excess_ann = np.asarray(mu_excess_ann, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    n = len(mu_excess_ann)

    def objective(w):
        variance = float(w.T @ cov_ann @ w)
        sigma = np.sqrt(max(variance, 0.0))
        if sigma <= 1e-12:
            return 1e6
        return -float((mu_excess_ann @ w) / sigma)

    return _solve(objective, n, min_w, max_w)


_STRESS_SECTOR_ALIASES = {
    "Technology": "Technology",
    "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Cyclical": "Consumer Discretionary",
    "Financial Services": "Financial Services",
    "Financials": "Financial Services",
    "Industrials": "Industrials",
    "Health Care": "Health Care",
    "Healthcare": "Health Care",
    "Energy": "Energy",
    "Materials": "Materials",
    "Basic Materials": "Materials",
    "Consumer Defensive": "Consumer Defensive",
    "Consumer Staples": "Consumer Defensive",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Unknown": "Unknown",
}


def normalize_stress_sector(sector: str) -> str:
    if sector is None or pd.isna(sector):
        return "Unknown"
    return _STRESS_SECTOR_ALIASES.get(str(sector), "Unknown")


def black_litterman_posterior_excess(
    cov_ann: pd.DataFrame,
    market_weights: pd.Series,
    views: list[tuple[np.ndarray, float, float]],
    *,
    delta: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected EXCESS returns.

    ``views`` items are ``(P_row, expected_excess_return, confidence)``.
    Confidence must be in (0, 1).  The returned vector is excess return, so callers
    must not subtract the risk-free rate again when maximizing its Sharpe ratio.
    """
    if tau <= 0:
        raise ValueError("Black-Litterman tau must be positive.")
    sigma = cov_ann.astype(float)
    w = market_weights.reindex(sigma.index).fillna(0.0).astype(float)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    pi = delta * (sigma.values @ w.values)
    if not views:
        return pd.Series(pi, index=sigma.index)

    p_rows, q_values, omega_values = [], [], []
    tau_sigma = sigma.values * tau
    for p, q, confidence in views:
        p = np.asarray(p, dtype=float)
        if len(p) != len(sigma):
            raise ValueError("Black-Litterman view vector has incorrect length.")
        conf = float(np.clip(confidence, 1e-6, 1 - 1e-6))
        view_var = float(p @ tau_sigma @ p.T)
        omega = max(view_var * (1.0 - conf) / conf, 1e-12)
        p_rows.append(p)
        q_values.append(float(q))
        omega_values.append(omega)

    P = np.vstack(p_rows)
    Q = np.asarray(q_values)
    Omega = np.diag(omega_values)
    inv_tau = np.linalg.pinv(tau_sigma)
    middle = inv_tau + P.T @ np.linalg.pinv(Omega) @ P
    rhs = inv_tau @ pi + P.T @ np.linalg.pinv(Omega) @ Q
    posterior = np.linalg.pinv(middle) @ rhs
    return pd.Series(posterior, index=sigma.index)


def volatility_risk_contributions(weights, cov_ann, labels=None) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_ann, dtype=float)
    variance = float(w.T @ cov @ w)
    sigma = np.sqrt(max(variance, 0.0))
    if sigma <= 1e-12:
        values = np.zeros_like(w)
    else:
        marginal = cov @ w / sigma
        values = w * marginal
    return pd.Series(values, index=labels)
