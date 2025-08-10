# fix_timestamps.py
import sqlite3
import pandas as pd

DB_PATH = "claims.db"

def fix_timestamps():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT rowid, * FROM messages", conn)

    # Only fill if timestamp is NULL
    mask = df["timestamp"].isna()
    df.loc[mask, "timestamp"] = df.loc[mask, "ts_iso"]

    # If duplicates exist, offset them slightly to keep uniqueness
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["thread_id", "timestamp"])

    for tid, group in df.groupby("thread_id"):
        seen_times = {}
        for idx, row in group.iterrows():
            ts = row["timestamp"]
            while ts in seen_times:
                ts = ts + pd.Timedelta(seconds=1)
            seen_times[ts] = True
            df.at[idx, "timestamp"] = ts

    # Save back (overwrite table)
    df.to_sql("messages", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Timestamps repaired in {DB_PATH}")

if __name__ == "__main__":
    fix_timestamps()