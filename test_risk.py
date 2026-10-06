import unittest
import numpy as np
import pandas as pd

from risk import concentration_metrics, performance_metrics, synthetic_stress_table


class RiskTests(unittest.TestCase):
    def test_advanced_metrics_are_finite_on_clean_series(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        x = np.arange(300)
        bench = pd.Series(0.0003 + 0.008 * np.sin(x / 13), index=idx)
        port = 0.0002 + 0.8 * bench + pd.Series(0.003 * np.cos(x / 7), index=idx)
        m = performance_metrics(port, bench, risk_free_rate=0.02, periods_per_year=252)
        for key in ["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Alpha", "Beta", "Tracking Error", "Information Ratio"]:
            self.assertIn(key, m)
            self.assertTrue(np.isfinite(m[key]))

    def test_concentration_effective_n(self):
        w = pd.Series([0.25, 0.25, 0.25, 0.25])
        m = concentration_metrics(w)
        self.assertAlmostEqual(m["HHI"], 0.25)
        self.assertAlmostEqual(m["Effective N"], 4.0)
        self.assertAlmostEqual(m["Largest Weight"], 0.25)

    def test_stress_sector_alias_mapping_prevents_silent_omission(self):
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        sectors = pd.Series(["Healthcare", "Consumer Cyclical"], index=["A", "B"])
        table = synthetic_stress_table(weights, sectors)
        self.assertEqual(len(table), 5)
        self.assertTrue(np.isfinite(table["Portfolio Return"]).all())


if __name__ == "__main__":
    unittest.main()
