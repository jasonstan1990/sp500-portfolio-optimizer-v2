import tempfile
import unittest
from pathlib import Path
import sqlite3
import pandas as pd
import update_data as u

class UpdateTests(unittest.TestCase):
    def test_missing_benchmark_and_stale_data_rejected(self):
        with self.assertRaises(RuntimeError):
            u.validate_prices(pd.DataFrame(), pd.DataFrame(), ['A'])
        old = pd.DataFrame({'^GSPC':[10]}, index=pd.to_datetime(['2020-01-01']))
        with self.assertRaises(RuntimeError):
            u.validate_prices(old, old, ['A'])

    def test_partial_download_reported(self):
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=40)
        tickers = [f'T{i}' for i in range(20)]
        frame = pd.DataFrame(100., index=idx, columns=tickers+['^GSPC'])
        frame['T0'] = float('nan')
        valid, missing = u.validate_prices(frame, frame, tickers)
        self.assertEqual(missing, ['T0'])
        self.assertEqual(len(valid), 19)

    def test_failed_publication_preserves_database(self):
        original = u.DB
        try:
            with tempfile.TemporaryDirectory() as tmp:
                u.DB = Path(tmp)/'cache.db'
                with sqlite3.connect(u.DB) as con:
                    con.execute('CREATE TABLE published_snapshots(snapshot_key TEXT,published_at TEXT,meta_json TEXT,payload_blob BLOB)')
                    con.execute("INSERT INTO published_snapshots VALUES('daily','old','{}',X'00')")
                before = u.DB.read_bytes()
                good = {'meta':{'published_at':'new'}, 'data':{}}
                with self.assertRaises(KeyError):
                    u.publish({'daily':good,'weekly':{'meta':{}}})
                self.assertEqual(before, u.DB.read_bytes())
                u.publish({k:good for k in ['daily','weekly','monthly']})
                with sqlite3.connect(u.DB) as con:
                    self.assertEqual(con.execute('SELECT count(*) FROM published_snapshots').fetchone()[0], 3)
        finally:
            u.DB = original

if __name__ == '__main__':
    unittest.main()
