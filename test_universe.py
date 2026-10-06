import unittest
import pandas as pd

from universe import membership_on_date, normalize_symbol, parse_wikipedia_tables


class UniverseTests(unittest.TestCase):
    def test_symbol_normalization(self):
        self.assertEqual(normalize_symbol("BRK.B"), "BRK-B")
        self.assertEqual(normalize_symbol(" bf.b "), "BF-B")
        self.assertIsNone(normalize_symbol(None))

    def test_membership_reconstruction_undoes_future_changes(self):
        current = ["A", "B", "NEW"]
        changes = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-10", "2025-01-10"]),
                "Added": ["B", "NEW"],
                "Removed": ["OLD1", "OLD2"],
            }
        )
        at_2023 = membership_on_date("2023-12-31", current, changes)
        self.assertEqual(at_2023, {"A", "OLD1", "OLD2"})
        at_2024 = membership_on_date("2024-06-01", current, changes)
        self.assertEqual(at_2024, {"A", "B", "OLD2"})
        at_2026 = membership_on_date("2026-01-01", current, changes)
        self.assertEqual(at_2026, {"A", "B", "NEW"})

    def test_parse_wikipedia_like_tables(self):
        current = pd.DataFrame(
            {
                "Symbol": [f"T{i}" for i in range(400)],
                "Security": [f"Company {i}" for i in range(400)],
                "GICS Sector": ["Technology"] * 400,
            }
        )
        changes = pd.DataFrame(
            {
                ("Date", "Date"): ["January 1, 2025"] * 10,
                ("Added", "Ticker"): ["NEW"] * 10,
                ("Removed", "Ticker"): ["OLD"] * 10,
            }
        )
        c, h = parse_wikipedia_tables([current, changes])
        self.assertEqual(len(c), 400)
        self.assertEqual(h.iloc[0]["Added"], "NEW")
        self.assertEqual(h.iloc[0]["Removed"], "OLD")


if __name__ == "__main__":
    unittest.main()
