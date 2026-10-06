import unittest
import numpy as np
import pandas as pd

from ranking import screen_and_rank


class RankingTests(unittest.TestCase):
    def test_missing_sectors_and_fundamentals_can_run_price_only(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        t = np.arange(300)
        close = {"^GSPC": 100 * np.cumprod(1 + 0.0002 + 0.004 * np.sin(t/11))}
        vol = {"^GSPC": np.full(300, 2e6)}
        members = []
        for i in range(8):
            ticker = f"T{i}"; members.append(ticker)
            r = 0.0003 + (0.9 + i*.02) * 0.004 * np.sin(t/11) + 0.001*np.cos(t/(5+i))
            close[ticker] = (50+i) * np.cumprod(1+r)
            vol[ticker] = np.full(300, 2e6)
        close = pd.DataFrame(close, index=idx)
        vol = pd.DataFrame(vol, index=idx)
        result = screen_and_rank(
            close, close, vol, None, members,
            adtv_usd_min=0, vol_mult_spx=10, beta_min=0, beta_max=5,
            min_growth=0, top_n=5, use_growth_filter=False, use_value_factor=False,
        )
        self.assertEqual(len(result.rank_table), 5)
        self.assertFalse(result.diagnostics["fundamentals_used"])
        self.assertTrue((result.screen_table["Sector"] == "Unknown").all())


if __name__ == "__main__":
    unittest.main()
