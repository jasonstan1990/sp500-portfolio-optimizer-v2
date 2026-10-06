"""SQLite persistence helpers for the S&P 500 Portfolio Optimizer.

The public Streamlit app is intentionally snapshot-only: it never downloads market
or fundamental data.  The offline updater owns all network activity and publishes a
validated SQLite file atomically.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import json
import pickle
import sqlite3
import zlib

import pandas as pd


def pack(value) -> bytes:
    return zlib.compress(pickle.dumps(value, protocol=4))


def unpack(blob: bytes):
    return pickle.loads(zlib.decompress(blob))


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Migrate older cache databases forward without deleting data."""
    if not table_exists(conn, "published_snapshots"):
        conn.execute(
            """
            CREATE TABLE published_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_key TEXT NOT NULL,
                published_at TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                payload_blob BLOB NOT NULL
            )
            """
        )
    else:
        cols = get_columns(conn, "published_snapshots")
        if "snapshot_key" not in cols:
            conn.execute("ALTER TABLE published_snapshots ADD COLUMN snapshot_key TEXT")
            conn.execute(
                """
                UPDATE published_snapshots
                SET snapshot_key = COALESCE(snapshot_key, 'daily')
                WHERE snapshot_key IS NULL
                """
            )

    cols = get_columns(conn, "published_snapshots")
    if "id" in cols:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_published_snapshots_key_id
            ON published_snapshots(snapshot_key, id)
            """
        )
    else:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_published_snapshots_key
            ON published_snapshots(snapshot_key)
            """
        )

    # Point-in-time fundamentals are accumulated over monthly updater runs.  The
    # first V4 update starts the history; we never fabricate historical values.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamental_snapshots (
            asof_date TEXT PRIMARY KEY,
            published_at TEXT NOT NULL,
            payload_blob BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fundamental_snapshots_asof
        ON fundamental_snapshots(asof_date)
        """
    )

    conn.commit()


def connect(
    db_path: str | Path,
    *,
    check_same_thread: bool = True,
    migrate: bool = True,
) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    if migrate:
        ensure_schema(conn)
    return conn


def load_snapshot(db_path: str | Path, snapshot_key: str):
    """Return (published_at, meta, payload) for the latest snapshot key."""
    with closing(connect(db_path, migrate=False)) as conn:
        if not table_exists(conn, "published_snapshots"):
            return None
        cols = get_columns(conn, "published_snapshots")
        if "snapshot_key" in cols:
            order = "id DESC" if "id" in cols else "rowid DESC"
            row = conn.execute(
                f"""
                SELECT published_at, meta_json, payload_blob
                FROM published_snapshots
                WHERE snapshot_key = ?
                ORDER BY {order}
                LIMIT 1
                """,
                (snapshot_key,),
            ).fetchone()
        else:
            # Very old caches pre-date keyed snapshots and are treated as daily.
            if snapshot_key != "daily":
                return None
            order = "id DESC" if "id" in cols else "rowid DESC"
            row = conn.execute(
                f"SELECT published_at, meta_json, payload_blob FROM published_snapshots ORDER BY {order} LIMIT 1"
            ).fetchone()
    if not row:
        return None
    published_at, meta_json, payload_blob = row
    return published_at, json.loads(meta_json), unpack(payload_blob)


def list_fundamental_snapshots(db_path: str | Path) -> dict[pd.Timestamp, pd.DataFrame]:
    """Load all stored point-in-time fundamental snapshots keyed by as-of date."""
    with closing(connect(db_path, migrate=False)) as conn:
        if not table_exists(conn, "fundamental_snapshots"):
            return {}
        rows = conn.execute(
            """
            SELECT asof_date, payload_blob
            FROM fundamental_snapshots
            ORDER BY asof_date
            """
        ).fetchall()
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for asof, blob in rows:
        value = unpack(blob)
        if isinstance(value, pd.DataFrame):
            result[pd.Timestamp(asof)] = value
    return result


def latest_fundamentals_asof(
    history: dict[pd.Timestamp, pd.DataFrame],
    asof: pd.Timestamp,
) -> pd.DataFrame | None:
    eligible = [d for d in history if d <= pd.Timestamp(asof).normalize()]
    if not eligible:
        return None
    return history[max(eligible)]


def integrity_check(db_path: str | Path) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute("PRAGMA integrity_check").fetchone()
    return "" if not row else str(row[0])
