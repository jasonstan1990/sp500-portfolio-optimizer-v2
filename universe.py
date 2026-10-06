"""S&P 500 constituent utilities.

Network functions are used only by the offline updater.  The Streamlit app uses the
constituent-change table stored inside the published snapshot.
"""
from __future__ import annotations

from io import StringIO
from typing import Iterable

import pandas as pd

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(symbol) -> str | None:
    """Normalize index symbols to Yahoo Finance notation (BRK.B -> BRK-B)."""
    if symbol is None or pd.isna(symbol):
        return None
    s = str(symbol).strip().upper()
    if not s or s in {"NAN", "NONE", "—", "-"}:
        return None
    return s.replace(".", "-")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(x) for x in tup if str(x) != "nan").strip()
            for tup in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_col(columns: Iterable[str], *needles: str) -> str | None:
    lowered = {str(c).lower(): str(c) for c in columns}
    for low, original in lowered.items():
        if all(n.lower() in low for n in needles):
            return original
    return None


def parse_wikipedia_tables(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse Wikipedia's current constituent and historical changes tables.

    This is deliberately tolerant to small heading changes and MultiIndex columns.
    """
    current = None
    changes = None
    for raw in tables:
        df = _flatten_columns(raw)
        sym_col = _find_col(df.columns, "symbol")
        sector_col = _find_col(df.columns, "gics", "sector")
        if current is None and sym_col and sector_col and len(df) >= 400:
            name_col = _find_col(df.columns, "security")
            current = pd.DataFrame(
                {
                    "Ticker": df[sym_col].map(normalize_symbol),
                    "Security": df[name_col].astype(str) if name_col else "",
                    "Sector": df[sector_col].astype(str),
                }
            ).dropna(subset=["Ticker"])
            current = current.drop_duplicates("Ticker").set_index("Ticker")
            continue

        date_col = _find_col(df.columns, "date")
        add_col = _find_col(df.columns, "added", "ticker")
        rem_col = _find_col(df.columns, "removed", "ticker")
        if changes is None and date_col and (add_col or rem_col) and len(df) >= 10:
            changes = pd.DataFrame(
                {
                    "Date": pd.to_datetime(df[date_col], errors="coerce"),
                    "Added": df[add_col].map(normalize_symbol) if add_col else None,
                    "Removed": df[rem_col].map(normalize_symbol) if rem_col else None,
                }
            )
            changes = changes.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if current is None:
        raise RuntimeError("Could not identify the current S&P 500 constituents table.")
    if changes is None:
        changes = pd.DataFrame(columns=["Date", "Added", "Removed"])
    return current, changes


def fetch_sp500_universe(timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch current constituents and index changes from Wikipedia.

    Wikipedia is a convenient public source, not a licensed index feed.  The updater
    records the source and falls back to the previous saved universe on failure.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SP500PortfolioOptimizer/4.0; educational research)"
    }
    response = requests.get(WIKIPEDIA_URL, headers=headers, timeout=timeout)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return parse_wikipedia_tables(tables)


def membership_on_date(
    asof,
    current_members: Iterable[str],
    changes: pd.DataFrame | None,
) -> set[str]:
    """Reconstruct membership as-of a date by undoing later index changes.

    Starting from today's/current membership, every change after *asof* is reversed:
    an added constituent is removed and a removed constituent is restored.
    """
    target = pd.Timestamp(asof).normalize()
    members = {normalize_symbol(x) for x in current_members}
    members.discard(None)
    if changes is None or changes.empty:
        return members

    events = changes.copy()
    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    events = events.dropna(subset=["Date"]).sort_values("Date", ascending=False)
    for _, row in events.iterrows():
        if pd.Timestamp(row["Date"]).normalize() <= target:
            continue
        added = normalize_symbol(row.get("Added"))
        removed = normalize_symbol(row.get("Removed"))
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return members


def changes_coverage_start(changes: pd.DataFrame | None) -> pd.Timestamp | None:
    if changes is None or changes.empty or "Date" not in changes:
        return None
    dates = pd.to_datetime(changes["Date"], errors="coerce").dropna()
    return None if dates.empty else dates.min().normalize()


def historical_symbol_union(
    current_members: Iterable[str],
    changes: pd.DataFrame | None,
    *,
    start_date=None,
) -> list[str]:
    """Symbols worth downloading to support point-in-time backtests."""
    symbols = {normalize_symbol(x) for x in current_members}
    symbols.discard(None)
    if changes is not None and not changes.empty:
        events = changes.copy()
        events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
        if start_date is not None:
            events = events[events["Date"] >= pd.Timestamp(start_date)]
        for col in ("Added", "Removed"):
            if col in events:
                symbols.update(x for x in events[col].map(normalize_symbol).tolist() if x)
    return sorted(symbols)
