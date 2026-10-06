"""Monthly, offline-public-app snapshot publisher. Run only on your own computer."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import closing
import json
import pickle
import sqlite3
import zlib
import shutil
import os
import time
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / '.portfolio_cache' / 'cache.db'
BENCH = '^GSPC'


def load_seed():
    if not DB.exists():
        raise RuntimeError('Missing .portfolio_cache/cache.db. Extract the complete package first.')
    with closing(sqlite3.connect(DB)) as con:
        row = con.execute("SELECT payload_blob FROM published_snapshots WHERE snapshot_key='daily' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError('No daily snapshot to supply the stock list.')
    return pickle.loads(zlib.decompress(row[0]))


def validate_prices(close, volume, tickers):
    if BENCH not in close or close[BENCH].dropna().empty:
        raise RuntimeError('Benchmark download failed. Old data retained.')
    latest = close[BENCH].last_valid_index()
    if (pd.Timestamp.now().normalize() - latest).days > 7:
        raise RuntimeError('Downloaded benchmark is stale. Old data retained.')
    valid = [t for t in tickers if t in close and t in volume
             and close[t].notna().sum() >= 30 and close[t].last_valid_index() == latest
             and (close[t].dropna() > 0).all() and volume[t].fillna(0).gt(0).any()]
    missing = sorted(set(tickers) - set(valid))
    if len(valid) < .95 * len(tickers):
        raise RuntimeError(f'Too many missing stocks ({len(missing)}). Retry later. Old data retained.')
    return valid, missing


def publish(payloads):
    """Build a complete replacement database; failure never damages live snapshots."""
    backup = DB.with_name('cache.before_update.db')
    staged = DB.with_name('cache.new.db')
    with closing(sqlite3.connect(DB)) as src, closing(sqlite3.connect(backup)) as dst:
        src.backup(dst)
        dst.commit()
    shutil.copy2(backup, staged)
    try:
        with closing(sqlite3.connect(staged)) as con:
            for freq, p in payloads.items():
                con.execute('DELETE FROM published_snapshots WHERE snapshot_key=?', (freq,))
                con.execute('INSERT INTO published_snapshots(snapshot_key,published_at,meta_json,payload_blob) VALUES(?,?,?,?)',
                            (freq, p['meta']['published_at'], json.dumps(p['meta']), zlib.compress(pickle.dumps(p, protocol=4))))
            con.commit()
            con.execute('VACUUM')
        os.replace(staged, DB)
    finally:
        if staged.exists():
            staged.unlink()


def main():
    import yfinance as yf
    seed = load_seed()
    tickers = seed['data']['tickers']
    start = seed['meta'].get('start_date', '2020-01-01')
    # Exclude today's potentially incomplete US trading session.
    end = datetime.now(timezone.utc).date().isoformat()
    print(f'Downloading daily history for {len(tickers)} stocks through the last completed day...', flush=True)
    frames = []
    symbols = list(dict.fromkeys(list(tickers) + [BENCH]))
    for i in range(0, len(symbols), 40):
        batch = symbols[i:i+40]
        data = yf.download(batch, start=start, end=end, interval='1d', auto_adjust=True,
                           group_by='column', threads=4, progress=False, timeout=30)
        if not data.empty:
            frames.append(data)
        print(f'Prices: {min(i+40, len(symbols))}/{len(symbols)}', flush=True)
        time.sleep(.5)
    if not frames:
        raise RuntimeError('No prices received. Check internet access and retry.')
    data = pd.concat(frames, axis=1)
    close, volume = data['Close'].copy(), data['Volume'].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    close = close.sort_index().loc[lambda x: ~x.index.duplicated()]
    volume = volume.reindex(close.index)
    valid, missing = validate_prices(close, volume, tickers)
    close, volume = close[valid+[BENCH]], volume[valid+[BENCH]]
    info = seed['data']['info_df'].reindex(valid).copy()
    stale = []
    field_map = {'PE':'trailingPE', 'DividendYield':'dividendYield', 'Sector':'sector',
                 'MarketCap':'marketCap', 'AnalystGrowth':'earningsGrowth'}
    for i, ticker in enumerate(valid, 1):
        try:
            fresh = yf.Ticker(ticker).info
            if not fresh or not fresh.get('marketCap'):
                raise ValueError('No fundamental data')
            for column, field in field_map.items():
                info.loc[ticker, column] = fresh.get(field, float('nan') if column != 'Sector' else 'Unknown')
        except Exception:
            stale.append(ticker)
        if i % 25 == 0:
            print(f'Fundamentals: {i}/{len(valid)}', flush=True)
        time.sleep(.15)
    if len(stale) > .1 * len(valid):
        raise RuntimeError('More than 10% of fundamentals failed. Old data retained; retry later.')
    published = datetime.now(timezone.utc).isoformat()
    payloads = {}
    for freq, rule in [('daily', None), ('weekly', 'W-FRI'), ('monthly', 'ME')]:
        c = close if rule is None else close.resample(rule).last()
        v = volume if rule is None else volume.resample(rule).sum(min_count=1)
        # Keep only completed calendar weeks/months.
        if rule:
            cutoff = pd.Timestamp(end) - pd.Timedelta(days=1)
            c = c.loc[c.index <= cutoff]
            v = v.reindex(c.index)
        meta = dict(seed['meta'], frequency=freq, interval={'daily':'1d','weekly':'1wk','monthly':'1mo'}[freq],
                    published_at=published, end_date=str(c.index.max().date()),
                    tickers_count=len(valid), tickers_preview=valid[:20],
                    missing_tickers=missing, stale_fundamentals=stale,
                    universe_note='Original saved stock universe; not automatically reconstituted.',
                    growth_definition='Yahoo earningsGrowth (quarterly year-on-year), not analyst forecast',
                    prices_adjusted=True)
        payloads[freq] = {'meta':meta, 'data':{'close_df':c, 'vol_df':v, 'info_df':info, 'tickers':valid}}
    publish(payloads)
    report = {'published_at':published, 'last_daily_price':str(close.index.max().date()),
              'excluded_stocks':missing, 'fundamentals_retained_from_previous_snapshot':stale}
    (ROOT/'UPDATE_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSUCCESS. Upload .portfolio_cache/cache.db to the same folder on GitHub.')
    print('Last daily price:', report['last_daily_price'])
    print('Excluded stocks:', missing, '\nOld fundamentals retained:', stale)
    print('A local backup is in .portfolio_cache/cache.before_update.db (do not upload).')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nUPDATE FAILED: {exc}')
        raise SystemExit(1)
