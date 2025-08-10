# streamlit_claims_dashboard.py
import json
import difflib
import sqlite3
import tempfile
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "/Users/vanikanoria/Desktop/practice/GainLife/Claims-Messages-to-Routing-System/claims.db"

st.set_page_config(page_title="Claims Messaging Dashboard", layout="wide")

# =========================
# Utilities
# =========================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
        .str.lower()
    )
    return df

def choose_time_column(df: pd.DataFrame) -> str | None:
    # Priority: ISO-ish first, then numeric unix
    priority = ["timestamp", "ts_iso", "datetime", "created_at", "date", "sent_at", "ts", "ts_unix"]
    for c in priority:
        if c in df.columns:
            return c
    return None

def _infer_unix_unit(s: pd.Series) -> str:
    """Infer seconds/ms/us/ns from median magnitude."""
    s = pd.to_numeric(s, errors="coerce").dropna().abs()
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

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = choose_time_column(df)
    if not col:
        raise ValueError("No timestamp-like column found.")

    # Try ISO/datetime strings first
    if col != "ts_unix":
        ts = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        # If we totally failed and column looks numeric, fall back to unit inference
        if ts.notna().sum() == 0 and pd.api.types.is_numeric_dtype(df[col]):
            unit = _infer_unix_unit(df[col])
            ts = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit=unit, errors="coerce")
    else:
        # Numeric unix path
        unit = _infer_unix_unit(df[col])
        ts = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit=unit, errors="coerce")

    # Drop impossible years to avoid overflows/garbage (tweak if needed)
    ts = ts.mask(~ts.dt.year.between(2000, 2035))

    # Finalize
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Optional debug to verify parsing
    # st.write({"source_col": col, "inferred_unit": unit if col=="ts_unix" else "ISO",
    #           "min": df["timestamp"].min(), "max": df["timestamp"].max()})

    return df

def parse_list_safe(x):
    if isinstance(x, list): return x
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except Exception:
            pass
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]  # single label fallback

def percent_df(series: pd.Series, label_name: str = "intent") -> pd.DataFrame:
    s = series.fillna("Unclassified").replace({"": "Unclassified", "nan": "Unclassified", "None": "Unclassified"})
    vc = s.value_counts(dropna=False, normalize=True).sort_values(ascending=False) * 100.0
    return vc.rename_axis(label_name).reset_index(name="pct")

def messages_over_time(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.set_index("timestamp").resample(rule).size().rename("messages").reset_index()

def intent_over_time(df: pd.DataFrame, rule: str, multi_col: Optional[str], primary_col: str) -> pd.DataFrame:
    if multi_col:
        long = df[["timestamp", multi_col]].explode(multi_col).dropna(subset=[multi_col]).copy()
        long = long.rename(columns={multi_col: "intent"})
    else:
        tmp = df[["timestamp", primary_col]].copy()
        tmp["intent"] = tmp[primary_col].replace({"nan": "Unclassified", "None": "Unclassified", "": "Unclassified"})
        long = tmp[["timestamp", "intent"]]
    ts = (long.groupby([pd.Grouper(key="timestamp", freq=rule), "intent"])
                .size().reset_index(name="count"))
    return ts

def coerce_role(df: pd.DataFrame) -> pd.DataFrame:
    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.strip().str.lower()
    return df

def response_times_pair(df_in: pd.DataFrame) -> pd.DataFrame:
    need = {"thread_id", "timestamp", "role"}
    if not need.issubset(df_in.columns):
        return pd.DataFrame(columns=["thread_id", "prev_role", "curr_role", "delta_minutes", "ts"])
    d = df_in[["thread_id", "timestamp", "role"]].dropna().sort_values(["thread_id", "timestamp"]).copy()
    out = []
    last = {}
    for r in d.itertuples(index=False):
        t, ts, role = r.thread_id if "thread_id" in d.columns else None, r.timestamp, r.role
        if t in last:
            p_role, p_ts = last[t]
            if {p_role, role} <= {"claimant", "adjuster"} and p_role != role:
                delta = (ts - p_ts).total_seconds() / 60.0
                if 0 < delta < 60 * 24 * 30:
                    out.append((t, p_role, role, float(delta), ts))
        last[t] = (role, ts)
    return pd.DataFrame(out, columns=["thread_id", "prev_role", "curr_role", "delta_minutes", "ts"])

def kpi_median(rt_df, prev_role, curr_role):
    if rt_df.empty:
        return "—"
    sub = rt_df.query("prev_role == @prev_role and curr_role == @curr_role")
    return f"{sub['delta_minutes'].median():.1f} min" if not sub.empty else "—"

# =========================
# Data loading (cached)
# =========================
@st.cache_data(show_spinner=False)

def load_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    return df

@st.cache_data(show_spinner=False)
def load_sqlite_table(db_bytes: bytes, table: str) -> pd.DataFrame:
    # Write UploadedFile to a temp .db so sqlite3 can open it
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        tmp.write(db_bytes)
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
    return df

@st.cache_data(show_spinner=False)
def list_sqlite_tables(db_bytes: bytes) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        tmp.write(db_bytes)
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        rows = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
        conn.close()
    return rows["name"].tolist()

@st.cache_data(show_spinner=False)
def run_sql(db_bytes: bytes, sql: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        tmp.write(db_bytes)
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        df = pd.read_sql_query(sql, conn)
        conn.close()
    return df

# =========================
# UI: Source selector
# =========================
st.title("Claims Messaging Dashboard")

source = st.sidebar.radio("Data Source", ["CSV", "SQLite DB"], horizontal=True)

if source == "CSV":
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if not upload:
        st.info("Upload a CSV. We’ll auto-detect the timestamp column (e.g., ts_iso).")
        st.stop()
    raw = load_csv(upload)

else:
    db_file = st.file_uploader("Upload SQLite .db", type=["db", "sqlite"])
    if not db_file:
        st.info("Upload a SQLite database file (e.g., claims.db).")
        st.stop()

    # List tables and let user pick
    try:
        tables = list_sqlite_tables(db_file.getvalue())
    except Exception as e:
        st.error(f"Could not read tables: {e}")
        st.stop()

    if not tables:
        st.error("No tables found in this database.")
        st.stop()

    table_choice = st.selectbox("Select table", tables, index=max(0, tables.index("messages") if "messages" in tables else 0))

    st.checkbox("Use custom SQL (advanced)", key="use_sql")
    if st.session_state.use_sql:
        sql = st.text_area("SQL query", value=f"SELECT * FROM {table_choice} LIMIT 1000")
        if st.button("Run SQL"):
            try:
                raw = run_sql(db_file.getvalue(), sql)
                st.success(f"Query returned {len(raw)} rows.")
            except Exception as e:
                st.error(f"SQL error: {e}")
                st.stop()
        else:
            st.stop()
    else:
        raw = load_sqlite_table(db_file.getvalue(), table_choice)

# =========================
# Normalize + prepare
# =========================
df = normalize_headers(raw)
df = parse_timestamp(df)
df = coerce_role(df)

# Multi-intent field (your column name)
multi_col = None
if "all_intents" in df.columns:
    df["all_intents"] = df["all_intents"].apply(parse_list_safe)
    if df["all_intents"].apply(lambda x: isinstance(x, list) and len(x) > 0).any():
        multi_col = "all_intents"

# Primary intent column (prefer hybrid)
primary_col = None
if "primary_intent" in df.columns:
    primary_col = "primary_intent"
elif "intent_primary_hybrid" in df.columns:  # optional fallback
    primary_col = "intent_primary_hybrid"
elif "intent" in df.columns:                 # optional fallback
    primary_col = "intent"

if primary_col is None:
    df["intent_fallback"] = "Unclassified"
    primary_col = "intent_fallback"
else:
    df[primary_col] = (
        df[primary_col].astype(str).str.strip()
          .replace({"": "Unclassified", "nan": "Unclassified", "None": "Unclassified"})
    )
# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

roles = sorted(df["role"].dropna().unique().tolist()) if "role" in df.columns else []
role_sel = st.sidebar.multiselect("Roles", roles, default=roles) if roles else None
# Ensure we have valid timestamps
if "timestamp" not in df.columns:
    st.error("No 'timestamp' column after parsing.")
    st.stop()

ts_valid = df["timestamp"].dropna()
if ts_valid.empty:
    # Fallback to today's date window if nothing valid
    today = pd.Timestamp.today().normalize()
    min_d = today.date()
    max_d = today.date()
else:
    min_d = ts_valid.min().date()
    max_d = ts_valid.max().date()

# Now the widget can't receive NaT
date_input = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
    key="date_range_main"   # unique key
)

# Normalize selection
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_d, end_d = date_input
else:
    start_d = end_d = date_input

# Apply filter safely
mask = df["timestamp"].between(pd.Timestamp(start_d), pd.Timestamp(end_d))
df_f = df.loc[mask].copy()
date_input = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key='other')
if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
    start_d, end_d = date_input
else:
    start_d = end_d = date_input

med_opt = "All"
if "medical_flag" in df.columns:
    med_opt = st.sidebar.selectbox("Medical flag", ["All", "Medical only", "Non-medical"])

mask = df["timestamp"].between(pd.Timestamp(start_d), pd.Timestamp(end_d))
if role_sel is not None:
    mask &= df["role"].isin(role_sel)
if "medical_flag" in df.columns:
    if med_opt == "Medical only":
        mask &= df["medical_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    elif med_opt == "Non-medical":
        mask &= ~df["medical_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

df_f = df.loc[mask].copy()

with st.expander("Debug snapshot", expanded=False):
    st.write("Rows (filtered):", len(df_f))
    st.write("Columns:", list(df_f.columns))
    st.write("Date range:", df_f["timestamp"].min() if not df_f.empty else None,
             "→", df_f["timestamp"].max() if not df_f.empty else None)
    st.dataframe(df_f.head(10))

if df_f.empty:
    st.warning("No data after filters. Adjust the sidebar.")
    st.stop()

# =========================
# KPIs
# =========================
rt = response_times_pair(df_f)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total messages", f"{len(df_f):,}")
c2.metric("Median RT: claimant → adjuster", kpi_median(rt, "claimant", "adjuster"))
c3.metric("Median RT: adjuster → claimant", kpi_median(rt, "adjuster", "claimant"))
med_pct = (df_f["medical_flag"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).mean() * 100.0) if "medical_flag" in df_f.columns else 0.0
c4.metric("Medical messages", f"{int(med_pct/100*len(df_f)):,} ({med_pct:.1f}%)")

st.markdown("---")

# =========================
# % by intent
# =========================

st.subheader("% of messages by intent")

def percent_df(series: pd.Series, label_name="intent"):
    s = series.fillna("Unclassified").replace({"": "Unclassified", "nan": "Unclassified", "None": "Unclassified"})
    vc = s.value_counts(normalize=True).sort_values(ascending=False) * 100.0
    return vc.rename_axis(label_name).reset_index(name="pct")

try:
    if multi_col:
        long = df.explode(multi_col).dropna(subset=[multi_col]).rename(columns={multi_col: "intent"})
        intent_pct = percent_df(long["intent"], "intent")
    else:
        intent_pct = percent_df(df["primary_intent"], "intent")

    if intent_pct.empty:
        st.info("No intents to display.")
    else:
        fig1 = px.bar(intent_pct, x="intent", y="pct", text="pct")
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig1.update_layout(yaxis_title="% of messages", xaxis_title="Intent",
                           uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig1, use_container_width=True)
except Exception as e:
    st.error(f"Intent % plot error: {e}")
    st.write("Columns:", list(df.columns))
    st.write("Head:", df.head())

# =========================
# Volume by role
# =========================
st.subheader("Volume of messages by role")
try:
    if "role" not in df_f.columns:
        st.info("No 'role' column to aggregate.")
    else:
        role_cnt = df_f["role"].value_counts(dropna=False).rename_axis("role").reset_index(name="count")
        fig2 = px.bar(role_cnt, x="role", y="count", text="count")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(yaxis_title="Messages", xaxis_title="Role")
        st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Role volume plot error: {e}")

st.markdown("---")

# =========================
# Messages over time
# =========================
st.subheader("Messages over time")
try:
    agg = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="msg_agg")
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[agg]
    vol_ts = messages_over_time(df_f, rule)
    fig3 = px.line(vol_ts, x="timestamp", y="messages", markers=True)
    fig3.update_layout(xaxis_title=f"{agg} timeline", yaxis_title="Count")
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.error(f"Messages-over-time plot error: {e}")

# =========================
# Intent over time
# =========================
st.subheader("Intent over time")
try:
    agg2 = st.radio("Aggregation for intent", ["Daily", "Weekly", "Monthly"], horizontal=True, key="intent_agg")
    rule2 = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[agg2]
    ts_int = intent_over_time(df_f, rule2, multi_col=multi_col, primary_col=primary_col)
    fig4 = px.line(ts_int, x="timestamp", y="count", color="intent", markers=True)
    fig4.update_layout(xaxis_title=f"{agg2} timeline", yaxis_title="Count")
    st.plotly_chart(fig4, use_container_width=True)
except Exception as e:
    st.error(f"Intent-over-time plot error: {e}")

st.markdown("---")

# =========================
# Table + topics
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("Message table (filtered)")
    show_cols = [c for c in ["timestamp","thread_id","role","topic","intent_primary_hybrid","intent","content","medical_flag"] if c in df_f.columns]
    st.dataframe(
        df_f[show_cols].sort_values("timestamp", ascending=False).reset_index(drop=True),
        height=450,
        use_container_width=True
    )

with right:
    st.subheader("Top topics")
    if "topic" in df_f.columns:
        top_topics = (df_f["topic"].replace({"nan":"Unlabeled","None":"Unlabeled","": "Unlabeled"})
                      .value_counts().head(15).rename_axis("topic").reset_index(name="count"))
        fig_topics = px.bar(top_topics, x="count", y="topic", orientation="h", text="count")
        fig_topics.update_traces(textposition="outside")
        fig_topics.update_layout(xaxis_title="Count", yaxis_title="Topic")
        st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.info("No 'topic' column found.")