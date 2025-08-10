# 📄 Claims Messages Analysis & Dashboard

This repository contains a complete pipeline for **ingesting, processing, classifying, and analyzing claims-related messages**, along with a **Streamlit dashboard** for interactive exploration.

---

## 📂 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| **`ingest_messages.py`** | Loads raw message data (`messages_raw.jsonl`) into a SQLite database (`claims.db`) for further processing. |
| **`analysis.ipynb`** | Exploratory Data Analysis (EDA) on message content, volume, intents, and conversation durations. Produces visualizations and statistics. |
| **`topic_modeling.ipynb`** | Applies NLP topic modeling to group messages into thematic categories. |
| **`summarization.ipynb`** | Generates summaries of long conversation threads using LLM-based summarization. |
| **`streamlit_claims_dashboard.py`** | Interactive dashboard for exploring message intents, conversation patterns, and trends over time. Connects to `claims.db`. |
| **`claims.db`** (+ `.db-shm` & `.db-wal`) | SQLite database containing ingested and enriched message data. |
| **
| **`conversation_durations_with_quartiles.png`** | Visualization of conversation duration distribution. |

---

## ⚙️ Installation

## 1) Prerequisites
- Python 3.9+  
- Git (optional)  
- (Optional) OpenAI API key for LLM-based summaries/intents

---

## 2) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# or on Windows:
# venv\Scripts\activate

---

## 3) Install dependencies
```bash
pip install -r requirements.txt

---

## 4) Project data inputs
	•	messages.csv → the original raw messages CSV you were given
Required columns: thread_id, timestamp, role, content
	•	timestamp can be ISO or UNIX.
	•	messages_raw.jsonl → same data with anomalies (if provided).

Place the file(s) at the repo root (or pass a path to the ingest script).

## 5) Ingest → SQLite

This creates/updates claims.db with a normalized messages table.

Quick start (CSV)

```bash
python ingest_messages.py --input messages.csv --db claims.db --create

Alternate (JSONL with anomalies)

python ingest_messages.py --input messages_raw.jsonl --db claims.db --create

Flags (common):
	•	--input : path to messages.csv or messages_raw.jsonl
	•	--db    : path to SQLite DB (default: claims.db)
	•	--create: create tables if missing (idempotent)
	•	--wal   : enable WAL mode for faster writes (optional)

Resulting table (minimal schema):

```sql

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id INTEGER,
  timestamp TEXT,
  role TEXT,
  content TEXT
);

## 6) Run the Streamlit dashboard

streamlit run streamlit_claims_dashboard.py

