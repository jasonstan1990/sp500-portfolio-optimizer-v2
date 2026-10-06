"""Offline/CI snapshot publisher for the S&P 500 Portfolio Optimizer V4.

The updater is transactional: all downloads and validation happen before a staged
SQLite database replaces the live cache.  If anything fails, the old cache remains
untouched.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
import json
import os
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from data_store import ensure_schema, load_snapshot, pack
from universe import fetch_sp500_universe, historical_symbol_union, normalize_symbol

ROOT = Path(__file__).resolve().parent
DB = ROOT / ".portfolio_cache" / "cache.db"
BENCH = "^GSPC"


def load_seed():
    if not DB.exists():
        raise RuntimeError("Missing .portfolio_cache/cache.db. Extract the complete package first.")
    latest = load_snapshot(DB, "daily")
    if latest is None:
        raise RuntimeError("No daily snapshot to supply fallback data.")
    return latest[2]


def _latest_market_date(close: pd.DataFrame) -> pd.Timestamp:
    if BENCH not in close or close[BENCH].dropna().empty:
        raise RuntimeError("Benchmark download failed. Old data retained.")
    latest = pd.Timestamp(close[BENCH].last_valid_index()).normalize()
    if (pd.Timestamp.now(tz=None).normalize() - latest).days > 7:
        raise RuntimeError("Downloaded benchmark is stale. Old data retained.")
    return latest


def validate_current_prices(close, volume, current_tickers):
    latest = _latest_market_date(close)
    valid = [
        t
        for t in current_tickers
        if t in close
        and t in volume
        and close[t].notna().sum() >= 30
        and pd.Timestamp(close[t].last_valid_index()).normalize() == latest
        and (close[t].dropna() > 0).all()
        and volume[t].fillna(0).gt(0).any()
    ]
    missing = sorted(set(current_tickers) - set(valid))
    if len(valid) < 0.95 * len(current_tickers):
        raise RuntimeError(f"Too many current constituents missing prices ({len(missing)}). Retry later. Old data retained.")
    return valid, missing, latest


def identify_archive_prices(close, volume, candidates):
    """Keep historical constituents even if they no longer trade today."""
    valid = []
    for t in candidates:
        if t not in close or t not in volume:
            continue
        prices = close[t].dropna()
        if len(prices) >= 30 and (prices > 0).all() and volume[t].fillna(0).gt(0).any():
            valid.append(t)
    return sorted(set(valid))


def _replace_with_retry(src: Path, dst: Path, attempts: int = 6, delay: float = 0.75):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(delay * attempt)
    raise last_exc


def publish(payloads, fundamental_asof: str, fundamental_df: pd.DataFrame):
    """Build and validate a complete replacement DB, then atomically publish it."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    backup = DB.with_name("cache.before_update.db")
    staged = DB.with_name("cache.new.db")

    # Explicit closing is essential on Windows before os.replace().
    with closing(sqlite3.connect(DB)) as src, closing(sqlite3.connect(backup)) as dst:
        ensure_schema(src)
        ensure_schema(dst)
        src.backup(dst)
        dst.commit()
    shutil.copy2(backup, staged)

    try:
        with closing(sqlite3.connect(staged)) as con:
            ensure_schema(con)
            for freq, payload in payloads.items():
                con.execute("DELETE FROM published_snapshots WHERE snapshot_key=?", (freq,))
                con.execute(
                    """
                    INSERT INTO published_snapshots(snapshot_key,published_at,meta_json,payload_blob)
                    VALUES(?,?,?,?)
                    """,
                    (
                        freq,
                        payload["meta"]["published_at"],
                        json.dumps(payload["meta"], default=str),
                        pack(payload),
                    ),
                )
            # Accumulate true as-of fundamental snapshots going forward.  Existing
            # dates are replaced because a rerun on the same market date is a repair.
            con.execute(
                """
                INSERT INTO fundamental_snapshots(asof_date,published_at,payload_blob)
                VALUES(?,?,?)
                ON CONFLICT(asof_date) DO UPDATE SET
                    published_at=excluded.published_at,
                    payload_blob=excluded.payload_blob
                """,
                (fundamental_asof, payloads["daily"]["meta"]["published_at"], pack(fundamental_df)),
            )
            con.commit()
            con.execute("VACUUM")
            integrity = con.execute("PRAGMA integrity_check").fetchone()
            if not integrity or str(integrity[0]).lower() != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        _replace_with_retry(staged, DB)
    finally:
        if staged.exists():
            staged.unlink()


def _extract_ohlcv(data: pd.DataFrame, batch: list[str]):
    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data["Close"].copy()
            volume = data["Volume"].copy()
        except KeyError:
            return pd.DataFrame(), pd.DataFrame()
    else:
        # yfinance can flatten a one-symbol response.
        if "Close" not in data or "Volume" not in data or len(batch) != 1:
            return pd.DataFrame(), pd.DataFrame()
        close = data[["Close"]].rename(columns={"Close": batch[0]})
        volume = data[["Volume"]].rename(columns={"Volume": batch[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(batch[0])
    if isinstance(volume, pd.Series):
        volume = volume.to_frame(batch[0])
    return close, volume


def _download_history(symbols: list[str], start: str, end: str):
    import yfinance as yf

    close_frames, volume_frames = [], []
    for i in range(0, len(symbols), 40):
        batch = symbols[i : i + 40]
        data = yf.download(
            batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            threads=4,
            progress=False,
            timeout=30,
        )
        c, v = _extract_ohlcv(data, batch)
        if not c.empty:
            close_frames.append(c)
            volume_frames.append(v)
        print(f"Prices: {min(i + 40, len(symbols))}/{len(symbols)}", flush=True)
        time.sleep(0.4)
    if not close_frames:
        raise RuntimeError("No prices received. Check internet access and retry.")
    close = pd.concat(close_frames, axis=1)
    volume = pd.concat(volume_frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    close = close.sort_index().loc[lambda x: ~x.index.duplicated()]
    volume = volume.reindex(close.index)
    return close, volume


def _get_universe(seed):
    seed_current = [normalize_symbol(t) for t in seed["data"].get("tickers", [])]
    seed_current = [t for t in seed_current if t]
    try:
        current_table, changes = fetch_sp500_universe()
        current = current_table.index.tolist()
        source = "Wikipedia: List of S&P 500 companies"
        print(f"Universe: fetched {len(current)} current constituents and {len(changes)} change events.")
        return current, current_table, changes, source, False
    except Exception as exc:
        print(f"WARNING: current-universe refresh failed ({exc}). Using saved universe.")
        old_changes = seed["data"].get("constituent_changes", pd.DataFrame(columns=["Date", "Added", "Removed"]))
        old_table = pd.DataFrame(index=seed_current)
        if "info_df" in seed["data"] and "Sector" in seed["data"]["info_df"]:
            old_table["Sector"] = seed["data"]["info_df"]["Sector"].reindex(seed_current)
        return seed_current, old_table, old_changes, "Saved fallback universe", True


def _refresh_fundamentals(current_valid, seed_info, current_table):
    import yfinance as yf

    columns = ["PE", "DividendYield", "Sector", "MarketCap", "AnalystGrowth"]
    info = seed_info.reindex(current_valid).copy() if seed_info is not None else pd.DataFrame(index=current_valid)
    for col in columns:
        if col not in info:
            info[col] = np.nan if col != "Sector" else "Unknown"
    stale = []
    field_map = {
        "PE": "trailingPE",
        "DividendYield": "dividendYield",
        "Sector": "sector",
        "MarketCap": "marketCap",
        "AnalystGrowth": "earningsGrowth",
    }
    for i, ticker in enumerate(current_valid, 1):
        try:
            fresh = yf.Ticker(ticker).info
            if not fresh or not fresh.get("marketCap"):
                raise ValueError("No fundamental data")
            for column, field in field_map.items():
                value = fresh.get(field, np.nan if column != "Sector" else "Unknown")
                info.loc[ticker, column] = value
        except Exception:
            stale.append(ticker)
            # At minimum preserve the current GICS sector from the index source.
            if ticker in current_table.index and "Sector" in current_table.columns:
                info.loc[ticker, "Sector"] = current_table.loc[ticker, "Sector"]
        if i % 25 == 0:
            print(f"Fundamentals: {i}/{len(current_valid)}", flush=True)
        time.sleep(0.12)
    if len(stale) > 0.10 * len(current_valid):
        raise RuntimeError("More than 10% of current fundamentals failed. Old data retained; retry later.")
    return info, stale


def main():
    seed = load_seed()
    start = str(seed["meta"].get("start_date", "2020-01-01"))
    end = datetime.now(timezone.utc).date().isoformat()  # yf end is exclusive; excludes incomplete current session.

    current, current_table, changes, universe_source, universe_fallback = _get_universe(seed)
    old_archive = seed["data"].get("archive_tickers", seed["data"].get("tickers", []))
    candidates = sorted(
        set(historical_symbol_union(current, changes, start_date=start))
        | {normalize_symbol(t) for t in old_archive if normalize_symbol(t)}
    )
    symbols = list(dict.fromkeys(candidates + [BENCH]))
    print(
        f"Downloading daily history for {len(candidates)} current/historical symbols through the last completed day...",
        flush=True,
    )
    close, volume = _download_history(symbols, start, end)
    current_valid, missing_current, latest = validate_current_prices(close, volume, current)
    archive_valid = identify_archive_prices(close, volume, candidates)

    # Keep the benchmark plus all historical symbols in DAILY for PIT backtesting.
    daily_cols = [t for t in archive_valid if t in close.columns]
    daily_close = close[daily_cols + [BENCH]].copy()
    daily_volume = volume.reindex(columns=daily_cols + [BENCH]).copy()

    seed_info = seed["data"].get("info_df", pd.DataFrame())
    info, stale = _refresh_fundamentals(current_valid, seed_info, current_table)

    published = datetime.now(timezone.utc).isoformat()
    payloads = {}
    for freq, rule in [("daily", None), ("weekly", "W-FRI"), ("monthly", "ME")]:
        if rule is None:
            c, v = daily_close, daily_volume
        else:
            base_cols = current_valid + [BENCH]
            c = daily_close.reindex(columns=base_cols).resample(rule).last()
            v = daily_volume.reindex(columns=base_cols).resample(rule).sum(min_count=1)
            cutoff = pd.Timestamp(end) - pd.Timedelta(days=1)
            c = c.loc[c.index <= cutoff]
            v = v.reindex(c.index)
        meta = dict(
            seed["meta"],
            frequency=freq,
            interval={"daily": "1d", "weekly": "1wk", "monthly": "1mo"}[freq],
            published_at=published,
            end_date=str(c.index.max().date()),
            tickers_count=len(current_valid),
            tickers_preview=current_valid[:20],
            missing_tickers=missing_current,
            stale_fundamentals=stale,
            universe_source=universe_source,
            universe_refresh_fallback=universe_fallback,
            universe_note=(
                "Current S&P 500 constituents refreshed by the updater; daily snapshot also retains historical symbols "
                "from the constituent-change log for point-in-time walk-forward tests."
            ),
            growth_definition="Yahoo earningsGrowth (quarterly year-on-year), not analyst forecast",
            prices_adjusted=True,
            pit_fundamentals_note=(
                "Fundamental history is accumulated from V4 updater runs only; earlier dates use price-only factors in PIT-safe backtests."
            ),
        )
        payloads[freq] = {
            "meta": meta,
            "data": {
                "close_df": c,
                "vol_df": v,
                "info_df": info,
                "tickers": current_valid,
                "current_constituents": current_valid,
                "archive_tickers": archive_valid if freq == "daily" else current_valid,
                "constituent_changes": changes,
            },
        }

    fundamental_asof = str(latest.date())
    publish(payloads, fundamental_asof, info)
    report = {
        "published_at": published,
        "last_daily_price": fundamental_asof,
        "current_constituents": len(current_valid),
        "historical_symbols_retained": len(archive_valid),
        "excluded_current_stocks": missing_current,
        "fundamentals_retained_from_previous_snapshot": stale,
        "universe_source": universe_source,
        "universe_refresh_fallback": universe_fallback,
        "constituent_change_events": int(len(changes)),
    }
    (ROOT / "UPDATE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSUCCESS. Upload .portfolio_cache/cache.db to the same folder on GitHub.")
    print("Last daily price:", report["last_daily_price"])
    print("Current constituents:", report["current_constituents"])
    print("Historical symbols retained:", report["historical_symbols_retained"])
    print("Excluded current stocks:", missing_current)
    print("Old fundamentals retained:", stale)
    print("A local backup is in .portfolio_cache/cache.before_update.db (do not upload).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nUPDATE FAILED: {exc}")
        raise SystemExit(1)
