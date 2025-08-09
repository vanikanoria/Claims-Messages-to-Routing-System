import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

CREATE_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id INTEGER,
    timestamp TEXT,
    role TEXT,
    content TEXT
);
"""

# Uniqueness via a companion table to keep it simple with your schema
CREATE_DEDUP = """
CREATE TABLE IF NOT EXISTS message_dedup (
    content_hash TEXT PRIMARY KEY,
    msg_id INTEGER,
    FOREIGN KEY (msg_id) REFERENCES messages(id) ON DELETE CASCADE
);
"""

CREATE_REJECTS = """
CREATE TABLE IF NOT EXISTS rejected_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_line TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
"""

def _parse_timestamp(ts: Optional[str]) -> Optional[str]:
    """Return ISO 'YYYY-MM-DD HH:MM:SS' or None if unparseable."""
    if not ts:
        return None
    s = str(ts).strip()
    # Try ISO with or without Z
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        # standardize to naive local ISO (readable, sortable as text)
        return dt.astimezone().replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    # Try unix seconds
    try:
        t = float(s)
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _to_int_or_none(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _norm_role(x) -> str:
    if not x:
        return "unknown"
    return str(x).strip().lower()

def _to_text(x) -> str:
    return "" if x is None else str(x)

def _content_hash(thread_id: Optional[int], ts: Optional[str], role: str, content: str) -> str:
    base = f"{thread_id or ''}|{ts or ''}|{role}|{content}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def ingest_messages(jsonl_path="messages.jsonl", db_path="claims.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(CREATE_MESSAGES)
    cur.executescript(CREATE_DEDUP)
    cur.executescript(CREATE_REJECTS)
    cur.executescript(CREATE_INDEXES)

    inserted = 0
    duplicates = 0
    rejected = 0
    skipped_empty = 0

    p = Path(jsonl_path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                cur.execute("INSERT INTO rejected_messages(raw_line, reason) VALUES (?,?)", (raw, "invalid_json"))
                rejected += 1
                continue

            # Normalize fields
            ts = _parse_timestamp(obj.get("timestamp") or obj.get("created_at") or obj.get("time"))
            role = _norm_role(obj.get("role"))
            content = _to_text(obj.get("content"))
            thread_id = _to_int_or_none(obj.get("thread_id"))

            # Basic validation
            if not ts:
                cur.execute("INSERT INTO rejected_messages(raw_line, reason) VALUES (?,?)", (raw, "missing_or_bad_timestamp"))
                rejected += 1
                continue

            if content.strip() == "":
                # You can choose to keep empties; here we skip & log
                skipped_empty += 1
                continue

            # Dedup check (idempotency)
            chash = _content_hash(thread_id, ts, role, content)
            # If hash exists, it's a duplicate
            cur.execute("SELECT 1 FROM message_dedup WHERE content_hash = ?", (chash,))
            if cur.fetchone():
                duplicates += 1
                continue

            # Insert message
            cur.execute(
                "INSERT INTO messages(thread_id, timestamp, role, content) VALUES (?,?,?,?)",
                (thread_id, ts, role, content)
            )
            msg_id = cur.lastrowid
            # Record hash
            cur.execute(
                "INSERT INTO message_dedup(content_hash, msg_id) VALUES (?,?)",
                (chash, msg_id)
            )

            inserted += 1

    conn.commit()
    conn.close()
    print(f"✅ Done. Inserted: {inserted}, duplicates skipped: {duplicates}, empty skipped: {skipped_empty}, rejected: {rejected}")