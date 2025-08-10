# 📄 Claims Messages Analysis & Dashboard

This repository contains a complete pipeline for **ingesting, processing, classifying, and analyzing claims-related messages**, along with a **Streamlit dashboard** for interactive exploration.


## 📂 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| **`ingest_messages.py`** | Loads raw message data (`messages_raw.jsonl`) into a SQLite database (`claims.db`) for further processing. |
| **`analysis.ipynb`** | Exploratory Data Analysis (EDA) on message content, volume, intents, and conversation durations. Produces visualizations and statistics. |
| **`get_intents.ipynb`** | In this notebook I built a hybrid regex-LLM intent classifier and before building that model I built the following: (1) LDA-based (Latent Discrimination Analysis) topic classification model (2)Regex-based (Regular Expressions) rules based model (3) LLM-based model with probability-based primary model selection |
| **`get_summary_and_subject.ipynb`** | Generates (1) One-line Summaries  (2) Subjects of long conversation threads using LLM-based summarization. |
| **`streamlit_claims_dashboard.py`** | Interactive dashboard for exploring message intents, conversation patterns, and trends over time. Connects to `claims.db`. |
| **`claims.db`** (+ `.db-shm` & `.db-wal`) | SQLite database containing ingested and enriched message data. |
| **`messages.csv`** | CSV file contained original message data. |
| **`messages_raw.jsonl`** | Message data as newline-delimited JSON with realistic anomalies (missing fields, bad types, duplicates, etc.). |


## ⚙️ Installation

## 1) Prerequisites
- Python 3.9+  
- Git (optional)  
- (Optional) OpenAI API key for LLM-based summaries/intents


## 2) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

## 3) Install dependencies
```bash
pip install -r requirements-local.txt #the requirements.txt file is for the streamlit app
python -m spacy download en_core_web_sm
# or for scispaCy:
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

## 4) Project data inputs
	•	messages.csv → the original raw messages CSV you were given
Required columns: thread_id, timestamp, role, content
	•	timestamp can be ISO or UNIX.
	•	messages_raw.jsonl → same data with anomalies (if provided).

Place the file(s) at the repo root (or pass a path to the ingest script).

## 5) Ingest → SQLite

This creates/updates claims.db with a normalized messages table from the messages_raw.jsonl file.


```
python ingest_messages.py --input messages_raw.jsonl --db claims.db
```
Flags (common):
	•	--input : path to messages.csv or messages_raw.jsonl
	•	--db    : path to SQLite DB (default: claims.db)

Resulting table (minimal schema):

```sql

CREATE TABLE IF NOT EXISTS messages (
  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id INTEGER,
  timestamp TEXT,
  role TEXT,
  content TEXT
);
```

## 6) Run the Streamlit dashboard

```bash 
streamlit run streamlit_claims_dashboard.py

```

What it does
	•	Connects to claims.db
	•	Robustly parses timestamps from timestamp (or falls back to ts_iso/ts_unix if present)
	•	Provides a single date range filter (with a unique key to avoid duplicates)
	•	Visuals:
	•	% of messages by intent (multi-label if all_intents exists; otherwise primary intent)
	•	Volume by role
	•	Messages over time (daily/weekly/monthly)
	•	Intents over time (daily/weekly/monthly)
	•	KPIs:
	•	Total messages in window
	•	(Optional) median response times if a response_times table exists

Upload DB option
In the sidebar you can upload a .db file if you don’t want to use the local claims.db.


## 7) Troubleshooting Tips

* No data after filters. Adjust the sidebar: Check that your CSV/DB file actually contains rows for the selected date range.
  
* Ensure your timestamp column is populated—run the timestamp fix script if needed.
```bash 
python fix_timestamps.py
```
* Deployment errors on Streamlit Cloud (Could not install packages due to an OSError): This is usually due to OS-specific packages in requirements.txt (like appnope); Remove platform-specific packages or use conditional installs.

* Dashboard loads but graphs are blank: Check that your DB has primary_intent and all_intents populated. If empty, re-run ingestion + AI integration scripts.

## 8) What’s next
	•	Deploy Streamlit to Streamlit Community Cloud
	•	Enhance classifier with fine-tuned model
	•	Add SLA/first-response metrics and alerts
