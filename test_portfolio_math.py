import unittest
import numpy as np
import pandas as pd

from portfolio_math import (
    MOMENTUM_WINDOWS,
    RISK_HORIZON_LABEL,
    compute_adtv_usd,
    normalize_stress_sector,
    optimize_max_excess_sharpe,
    optimize_max_sharpe,
    optimize_min_volatility,
    project_to_bounded_simplex,
    sample_weights,
)


class PortfolioMathTests(unittest.TestCase):
    def test_frequency_windows_are_calendar_consistent(self):
        self.assertEqual(MOMENTUM_WINDOWS["daily"], (126, 252))
        self.assertEqual(MOMENTUM_WINDOWS["weekly"], (26, 52))
        self.assertEqual(MOMENTUM_WINDOWS["monthly"], (6, 12))
        self.assertEqual(RISK_HORIZON_LABEL["weekly"], "1-week")
        self.assertEqual(RISK_HORIZON_LABEL["monthly"], "1-month")

    def test_bounded_projection_and_sampling(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            w = sample_weights(rng, 20, 0.01, 0.10)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
            self.assertGreaterEqual(float(w.min()), 0.01 - 1e-10)
            self.assertLessEqual(float(w.max()), 0.10 + 1e-10)
        x = np.array([100.0, 0.0, 0.0, 0.0])
        w = project_to_bounded_simplex(x, 0.10, 0.40)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        self.assertTrue(np.all(w >= 0.10 - 1e-10))
        self.assertTrue(np.all(w <= 0.40 + 1e-10))

    def test_exact_optimizers_are_feasible_and_deterministic(self):
        mu = np.array([0.12, 0.08, 0.06])
        cov = np.array([
            [0.040, 0.006, 0.004],
            [0.006, 0.025, 0.003],
            [0.004, 0.003, 0.018],
        ])
        w1 = optimize_max_sharpe(mu, cov, 0.02, 0.0, 0.80)
        w2 = optimize_max_sharpe(mu, cov, 0.02, 0.0, 0.80)
        wmin = optimize_min_volatility(cov, 0.0, 0.80)
        self.assertTrue(np.allclose(w1, w2, atol=1e-9))
        self.assertAlmostEqual(float(w1.sum()), 1.0, places=10)
        self.assertAlmostEqual(float(wmin.sum()), 1.0, places=10)
        self.assertLessEqual(float(w1.max()), 0.80 + 1e-10)
        equal = np.ones(3) / 3
        vol_min = np.sqrt(wmin @ cov @ wmin)
        vol_equal = np.sqrt(equal @ cov @ equal)
        self.assertLessEqual(vol_min, vol_equal + 1e-10)

    def test_black_litterman_excess_optimizer_uses_excess_returns_directly(self):
        mu_excess = np.array([0.06, 0.02])
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        w = optimize_max_excess_sharpe(mu_excess, cov, 0.0, 1.0)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=10)
        self.assertGreater(w[0], w[1])

    def test_adtv_is_60_daily_observations(self):
        idx = pd.bdate_range("2026-01-01", periods=100)
        close = pd.DataFrame({"A": 10.0}, index=idx)
        volume = pd.DataFrame({"A": np.r_[np.full(40, 1.0), np.full(60, 3.0)]}, index=idx)
        adtv = compute_adtv_usd(close, volume, 60)
        self.assertAlmostEqual(float(adtv["A"]), 30.0)

    def test_stress_sector_aliases_cover_yahoo_names(self):
        self.assertEqual(normalize_stress_sector("Healthcare"), "Health Care")
        self.assertEqual(normalize_stress_sector("Consumer Cyclical"), "Consumer Discretionary")
        self.assertEqual(normalize_stress_sector("Basic Materials"), "Materials")
        self.assertEqual(normalize_stress_sector("Something New"), "Unknown")


if __name__ == "__main__":
    unittest.main()
