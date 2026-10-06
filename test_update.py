import tempfile
import unittest
from pathlib import Path
from contextlib import closing
import sqlite3
from unittest import mock

import pandas as pd

import update_data as u
from data_store import ensure_schema


class UpdateTests(unittest.TestCase):
    def test_missing_benchmark_and_stale_data_rejected(self):
        with self.assertRaises(RuntimeError):
            u.validate_current_prices(pd.DataFrame(), pd.DataFrame(), ["A"])
        old = pd.DataFrame({"^GSPC": [10]}, index=pd.to_datetime(["2020-01-01"]))
        with self.assertRaises(RuntimeError):
            u.validate_current_prices(old, old, ["A"])

    def test_partial_current_download_reported(self):
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=40)
        tickers = [f"T{i}" for i in range(20)]
        frame = pd.DataFrame(100.0, index=idx, columns=tickers + ["^GSPC"])
        frame["T0"] = float("nan")
        valid, missing, _ = u.validate_current_prices(frame, frame, tickers)
        self.assertEqual(missing, ["T0"])
        self.assertEqual(len(valid), 19)

    def test_archive_accepts_delisted_history(self):
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=100)
        close = pd.DataFrame({"OLD": 10.0, "NOW": 20.0}, index=idx)
        volume = pd.DataFrame({"OLD": 100.0, "NOW": 100.0}, index=idx)
        close.loc[idx[-30]:, "OLD"] = float("nan")
        archive = u.identify_archive_prices(close, volume, ["OLD", "NOW"])
        self.assertEqual(archive, ["NOW", "OLD"])

    def test_replace_retries_windows_file_lock(self):
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(32, "file in use")
            return None

        with mock.patch.object(u.os, "replace", side_effect=flaky_replace), mock.patch.object(u.time, "sleep", return_value=None):
            u._replace_with_retry(Path("a"), Path("b"), attempts=5, delay=0.01)
        self.assertEqual(calls["n"], 3)

    def test_failed_publication_preserves_database_and_success_adds_pit_fundamentals(self):
        original = u.DB
        try:
            with tempfile.TemporaryDirectory() as tmp:
                u.DB = Path(tmp) / "cache.db"
                with closing(sqlite3.connect(u.DB)) as con:
                    ensure_schema(con)
                    con.execute(
                        "INSERT INTO published_snapshots(snapshot_key,published_at,meta_json,payload_blob) VALUES('daily','old','{}',X'00')"
                    )
                    con.commit()
                before = u.DB.read_bytes()
                good = {"meta": {"published_at": "new"}, "data": {}}
                fundamentals = pd.DataFrame({"PE": [10.0]}, index=["A"])
                with self.assertRaises(KeyError):
                    u.publish({"daily": good, "weekly": {"meta": {}}}, "2026-10-01", fundamentals)
                self.assertEqual(before, u.DB.read_bytes())

                u.publish({k: good for k in ["daily", "weekly", "monthly"]}, "2026-10-01", fundamentals)
                with closing(sqlite3.connect(u.DB)) as con:
                    self.assertEqual(con.execute("SELECT count(*) FROM published_snapshots").fetchone()[0], 3)
                    self.assertEqual(con.execute("SELECT count(*) FROM fundamental_snapshots").fetchone()[0], 1)
                    self.assertEqual(con.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            u.DB = original


if __name__ == "__main__":
    unittest.main()
