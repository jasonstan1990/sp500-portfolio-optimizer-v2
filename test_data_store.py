import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_store import connect, integrity_check, latest_fundamentals_asof, list_fundamental_snapshots, pack


class DataStoreTests(unittest.TestCase):
    def test_schema_migration_and_fundamental_asof_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cache.db"
            conn = connect(db)
            f1 = pd.DataFrame({"PE": [10.0]}, index=["A"])
            f2 = pd.DataFrame({"PE": [12.0]}, index=["A"])
            conn.execute("INSERT INTO fundamental_snapshots VALUES(?,?,?)", ("2026-01-31", "p1", pack(f1)))
            conn.execute("INSERT INTO fundamental_snapshots VALUES(?,?,?)", ("2026-02-28", "p2", pack(f2)))
            conn.commit(); conn.close()
            history = list_fundamental_snapshots(db)
            self.assertIsNone(latest_fundamentals_asof(history, pd.Timestamp("2025-12-31")))
            chosen = latest_fundamentals_asof(history, pd.Timestamp("2026-02-15"))
            self.assertEqual(float(chosen.loc["A", "PE"]), 10.0)
            self.assertEqual(integrity_check(db), "ok")

    def test_corrupted_database_is_not_reported_as_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "bad.db"
            db.write_bytes(b"not-a-sqlite-database")
            with self.assertRaises(sqlite3.DatabaseError):
                integrity_check(db)


if __name__ == "__main__":
    unittest.main()
