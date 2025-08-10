#!/usr/bin/env python3
"""
ingest_messages.py
Ingest messages from CSV or JSONL into SQLite with a minimal schema:

CREATE TABLE IF NOT EXISTS messages (
  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id INTEGER,
  timestamp TEXT,
  role TEXT,
  content TEXT
);

- Ignores any source message_id; SQLite autoincrements it.
- Accepts inputs with columns: thread_id, role, content, and any of:
    - timestamp  (ISO string or UNIX epoch: s/ms/us/ns)
    - ts_iso     (ISO string)
    - ts_unix    (UNIX epoch: s/ms/us/ns)
- Cleans and coerces types, drops empty content rows, deduplicates.
- Works from CLI *and* safely inside notebooks (defaults when no CLI flags).
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


MIN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id INTEGER,
  timestamp TEXT,
  role TEXT,
  content TEXT
);
"""

# Optional: a uniqueness guard to prevent duplicates on re-runs.
# Comment out if you don't want it.
UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_minimal
ON messages (thread_id, timestamp, role, content);
"""


def infer_unix_unit(series: pd.Series) -> str:
    """Infer likely UNIX unit (s/ms/us/ns) from magnitude."""
    s = pd.to_numeric(series, errors="coerce").dropna().abs()
    if s.empty:
        return "s"
    m = s.median()
    if m > 1e17:
        return "ns"
    if m > 1e14:
        return "us"
    if m > 1e11:
        return "ms"
    return "s"


def parse_timestamp_columns(df: pd.DataFrame) -> pd.Series:
    """
    Return a single pandas datetime Series using priority:
    ts_iso → ts_unix (auto unit) → timestamp (ISO) → timestamp (numeric auto unit).
    """
    final = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # 1) ts_iso
    if "ts_iso" in df.columns:
        ts_iso = pd.to_datetime(df["ts_iso"], errors="coerce")
        final = final.fillna(ts_iso)

    # 2) ts_unix
    if "ts_unix" in df.columns:
        ts_unix_num = pd.to_numeric(df["ts_unix"], errors="coerce")
        unit = infer_unix_unit(ts_unix_num)
        ts_unix_dt = pd.to_datetime(ts_unix_num, unit=unit, errors="coerce")
        final = final.fillna(ts_unix_dt)

    # 3) timestamp as ISO
    if "timestamp" in df.columns:
        ts_try = pd.to_datetime(df["timestamp"], errors="coerce")
        final = final.fillna(ts_try)

        # 4) timestamp as numeric (only where still NaT)
        mask_nat = final.isna()
        if mask_nat.any():
            as_num = pd.to_numeric(df.loc[mask_nat, "timestamp"], errors="coerce")
            unit2 = infer_unix_unit(as_num)
            ts_num_dt = pd.to_datetime(as_num, unit=unit2, errors="coerce")
            final.loc[mask_nat] = ts_num_dt

    return final


def load_input(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    if ext == ".csv":
        return pd.read_csv(path)
    # try jsonl then csv
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.read_csv(path)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a frame with only the minimal columns:
    thread_id (int), timestamp (YYYY-mm-dd HH:MM:SS), role (str), content (str)
    """
    df = df.copy()

    # ensure columns exist
    for col in ["thread_id", "role", "content"]:
        if col not in df.columns:
            df[col] = np.nan

    # thread_id → int (when possible)
    df["thread_id"] = pd.to_numeric(df["thread_id"], errors="coerce").astype("Int64")

    # timestamp → normalized string
    ts = parse_timestamp_columns(df)
    df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    # role/content → strings
    df["role"] = df["role"].astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()

    # drop rows with empty content or missing timestamp
    df = df[~df["content"].isna() & (df["content"] != "")]
    df = df[~df["timestamp"].isna()]

    # keep only minimal columns
    out = df[["thread_id", "timestamp", "role", "content"]].copy()

    # dedupe (exact duplicates)
    out = out.drop_duplicates()

    # fill thread_id nulls with -1 (optional) or drop; here we drop nulls
    out = out[~out["thread_id"].isna()]

    # convert Int64 to int for sqlite executemany tuples
    out["thread_id"] = out["thread_id"].astype(int)

    return out


def insert_rows(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    sql = "INSERT OR IGNORE INTO messages (thread_id, timestamp, role, content) VALUES (?, ?, ?, ?)"
    data = list(df.itertuples(index=False, name=None))
    conn.executemany(sql, data)
    conn.commit()


def run(input_path: str, db_path: str, create: bool = True, unique_index: bool = True):
    df_raw = load_input(input_path)
    df_min = normalize_frame(df_raw)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if create:
        cur.executescript(MIN_TABLE_SQL)
        if unique_index:
            cur.executescript(UNIQUE_INDEX_SQL)
        conn.commit()

    insert_rows(conn, df_min)

    total = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"✅ Ingested {len(df_min):,} rows. Total rows in messages: {total:,}")
    conn.close()


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa
        return get_ipython() is not None
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to messages.csv or messages_raw.jsonl")
    parser.add_argument("--db", default="claims.db", help="Path to SQLite DB (default: claims.db)")
    parser.add_argument("--no-create", action="store_true", help="Do not create table/index")
    parser.add_argument("--no-unique", action="store_true", help="Do not create unique index (allows dupes)")
    args = parser.parse_args()

    if not args.input:
        if _in_notebook():
            # Notebook-friendly defaults
            print("⚠️  --input not provided; defaulting to 'messages_raw.jsonl' for notebook use.")
            input_path = "messages_raw.jsonl"
        else:
            raise SystemExit("Error: --input is required (path to CSV or JSONL).")
    else:
        input_path = args.input

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run(
        input_path=input_path,
        db_path=args.db,
        create=(not args.no_create),
        unique_index=(not args.no_unique),
    )


if __name__ == "__main__":
    main()