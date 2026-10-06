import unittest
import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_walk_forward


def synthetic_market(n_assets=12, periods=900):
    idx = pd.bdate_range("2021-01-01", periods=periods)
    t = np.arange(periods)
    bench_r = 0.00025 + 0.006 * np.sin(t / 17.0)
    bench_price = 100 * np.cumprod(1 + bench_r)
    close = {"^GSPC": bench_price}
    volume = {"^GSPC": np.full(periods, 5_000_000.0)}
    for i in range(n_assets):
        r = 0.00025 + (0.85 + 0.02 * i) * 0.006 * np.sin(t / 17.0) + 0.002 * np.cos(t / (5.0 + i))
        close[f"T{i}"] = (50 + i) * np.cumprod(1 + r)
        volume[f"T{i}"] = np.full(periods, 2_000_000.0 + i * 100_000)
    return pd.DataFrame(close, index=idx), pd.DataFrame(volume, index=idx)


class BacktestTests(unittest.TestCase):
    def test_walk_forward_runs_and_never_uses_future_fundamentals(self):
        close, volume = synthetic_market()
        members = [c for c in close.columns if c != "^GSPC"]
        future_info = pd.DataFrame(
            {"PE": 15.0, "AnalystGrowth": 0.2, "Sector": "Technology", "MarketCap": 1e10},
            index=members,
        )
        history = {pd.Timestamp("2030-01-01"): future_info}
        cfg = BacktestConfig(
            start_date="2023-01-03",
            end_date=str(close.index[-1].date()),
            rebalance="quarterly",
            training_days=252,
            top_n=8,
            min_weight=0.0,
            max_weight=0.25,
            transaction_cost_bps=10,
            adtv_usd_min=0,
            vol_mult_spx=10.0,
            beta_min=0.0,
            beta_max=5.0,
        )
        result = run_walk_forward(close, volume, members, pd.DataFrame(), history, cfg)
        self.assertGreater(len(result.net_returns), 100)
        successful = result.rebalance_log[result.rebalance_log["Status"] == "rebalanced"]
        self.assertGreater(len(successful), 1)
        self.assertFalse(successful["PIT Fundamentals"].any())
        self.assertAlmostEqual(float(result.weights_history.sum(axis=1).iloc[0]), 1.0, places=8)

    def test_transaction_costs_reduce_net_but_not_gross(self):
        close, volume = synthetic_market()
        members = [c for c in close.columns if c != "^GSPC"]
        common = dict(
            start_date="2023-01-03",
            end_date=str(close.index[-1].date()),
            rebalance="quarterly",
            training_days=252,
            top_n=8,
            min_weight=0.0,
            max_weight=0.25,
            adtv_usd_min=0,
            vol_mult_spx=10.0,
            beta_min=0.0,
            beta_max=5.0,
        )
        zero = run_walk_forward(close, volume, members, pd.DataFrame(), {}, BacktestConfig(**common, transaction_cost_bps=0))
        costly = run_walk_forward(close, volume, members, pd.DataFrame(), {}, BacktestConfig(**common, transaction_cost_bps=50))
        self.assertTrue(np.allclose(zero.gross_returns.values, costly.gross_returns.values, atol=1e-12))
        self.assertLess(float((1 + costly.net_returns).prod()), float((1 + zero.net_returns).prod()))


if __name__ == "__main__":
    unittest.main()
