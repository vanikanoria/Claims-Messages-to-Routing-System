
# streamlit_claims_dashboard.py
import os
import sqlite3
from datetime import date
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Claims Messaging Dashboard", layout="wide")

# ------------------------------
# Data Loading
# ------------------------------
@st.cache_data(show_spinner=False)
def load_from_sqlite(db_path: str, table: str = "messages") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    return df

@st.cache_data(show_spinner=False)
def load_from_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# ------------------------------
# Utilities
# ------------------------------
def coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        # try common fallback names
        for alt in ("ts_iso", "created_at", "time", "date"):
            if alt in df.columns:
                df[col] = pd.to_datetime(df[alt], errors="coerce")
                break
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns we use are present
    for c in ("thread_id", "timestamp", "role", "content"):
        if c not in df.columns:
            df[c] = None
    # Standardize dtypes
    df["role"] = df["role"].astype(str).str.strip().str.lower()
    df["content"] = df["content"].astype(str)
    # Primary intent: prefer 'intent_primary' else derive later
    if "intent_primary" in df.columns:
        df["intent_primary"] = df["intent_primary"].astype(str).str.strip()
        df.loc[df["intent_primary"].isin(["", "nan", "None"]), "intent_primary"] = "Unclassified"
    else:
        df["intent_primary"] = "Unclassified"
    # If multi-label exists, try to parse lists stored as strings
    if "intents" in df.columns:
        df["intents"] = df["intents"].apply(_safe_parse_list)
    return df

def _safe_parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        # try JSON
        import json
        try:
            v = json.loads(s)
            return v if isinstance(v, list) else []
        except Exception:
            pass
    # comma-separated fallback
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [] if s in ("", "nan", "None") else [s]

def percent_series(series: pd.Series) -> pd.DataFrame:
    vc = series.value_counts(dropna=False, normalize=True).sort_values(ascending=False) * 100.0
    out = vc.rename_axis(series.name or "label").reset_index(name="pct")
    out.rename(columns={out.columns[0]: "label"}, inplace=True)
    return out

def compute_rt_from_messages(msgs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute response deltas (minutes) when role switches claimant <-> adjuster within a thread.
    Excludes gaps <= 0 or > 30 days.
    """
    if not {"thread_id", "timestamp", "role"}.issubset(msgs.columns):
        return pd.DataFrame(columns=["thread_id","prev_role","curr_role","delta_minutes","ts_iso"])
    df = msgs[["thread_id","timestamp","role"]].dropna().copy()
    df = df.sort_values(["thread_id","timestamp"])
    rows = []
    prev = {}
    for r in df.itertuples(index=False):
        t, ts, role = r.thread_id, r.timestamp, r.role
        if t in prev:
            p_role, p_ts = prev[t]
            if {p_role, role} <= {"claimant","adjuster"} and p_role != role:
                delta = (ts - p_ts).total_seconds()/60.0
                if 0 < delta < 60*24*30:
                    rows.append((t, p_role, role, float(delta), ts))
        prev[t] = (role, ts)
    return pd.DataFrame(rows, columns=["thread_id","prev_role","curr_role","delta_minutes","ts_iso"])

def median_rt(rt: pd.DataFrame, prev_role: str, curr_role: str) -> Optional[float]:
    if rt.empty:
        return None
    sub = rt.query("prev_role == @prev_role and curr_role == @curr_role")
    return None if sub.empty else float(sub["delta_minutes"].median())

# ------------------------------
# Sidebar: Data Input
# ------------------------------
st.sidebar.header("Data")
source = st.sidebar.radio("Load data from", ["SQLite (.db)", "CSV upload"], horizontal=True)

df = pd.DataFrame()
if source == "SQLite (.db)":
    db_path = st.sidebar.text_input("SQLite DB path", value="claims.db")
    table = st.sidebar.text_input("Table name", value="messages")
    if st.sidebar.button("Load"):
        if not os.path.exists(db_path):
            st.error(f"DB not found: {db_path}")
        else:
            df = load_from_sqlite(db_path, table)
else:
    up = st.sidebar.file_uploader("Upload CSV (export of your messages table)", type=["csv"])
    if up is not None:
        df = load_from_csv(up)

if df.empty:
    st.info("Load your data (SQLite table or CSV) to see the dashboard.")
    st.stop()

# Normalize/prepare
df = coerce_datetime(df, "timestamp")
df = normalize_columns(df)
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")
df_time = df.set_index("timestamp")

# Sidebar filters
st.sidebar.header("Filters")
roles = sorted(df["role"].dropna().unique().tolist())
role_filter = st.sidebar.multiselect("Role(s)", roles, default=roles)
min_d, max_d = df_time.index.min().date(), df_time.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)

df_f = df_time.loc[str(date_range[0]):str(date_range[-1])]
if role_filter:
    df_f = df_f[df_f["role"].isin(role_filter)]

# ------------------------------
# KPIs (using MEDIAN)
# ------------------------------
rt = compute_rt_from_messages(df_f.reset_index())  # build on the filtered range
med_c2a = median_rt(rt, "claimant", "adjuster")
med_a2c = median_rt(rt, "adjuster", "claimant")

c1, c2, c3 = st.columns(3)
c1.metric("Total messages", f"{len(df_f):,}")
c2.metric("Median RT: claimant → adjuster", f"{med_c2a:.1f} min" if med_c2a is not None else "—")
c3.metric("Median RT: adjuster → claimant", f"{med_a2c:.1f} min" if med_a2c is not None else "—")

st.markdown("---")

# ------------------------------
# % of messages by intent
# ------------------------------
st.subheader("% of messages by intent")
if "intents" in df_f.columns and df_f["intents"].apply(lambda x: isinstance(x, list) and len(x)>0).any():
    long = df_f.reset_index().explode("intents").dropna(subset=["intents"])
    intent_pct = percent_series(long["intents"].rename("intent"))
else:
    intent_pct = percent_series(df_f["intent_primary"].rename("intent"))

fig1 = px.bar(intent_pct, x="intent", y="pct", text="pct")
fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig1.update_layout(yaxis_title="% of messages", xaxis_title="Intent", uniformtext_minsize=8, uniformtext_mode='hide')
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------
# Volume of messages by role
# ------------------------------
st.subheader("Volume of messages by role")
role_cnt = df_f["role"].value_counts(dropna=False).rename_axis("role").reset_index(name="count")
fig2 = px.bar(role_cnt, x="role", y="count", text="count")
fig2.update_traces(textposition="outside")
fig2.update_layout(yaxis_title="Messages", xaxis_title="Role")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ------------------------------
# Messages over time (line)
# ------------------------------
st.subheader("Messages over time")
freq = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, key="msg_freq")
rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]
vol_ts = df_f["content"].resample(rule).size().rename("messages").reset_index()
fig3 = px.line(vol_ts, x="timestamp", y="messages", markers=True)
fig3.update_layout(xaxis_title=f"{freq} timeline", yaxis_title="Count")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Intent over time (line)
# ------------------------------
st.subheader("Intent over time")
if "intents" in df_f.columns and df_f["intents"].apply(lambda x: isinstance(x, list) and len(x)>0).any():
    long = df_f.reset_index().explode("intents").dropna(subset=["intents"])
    intent_ts = (
        long.groupby([pd.Grouper(key="timestamp", freq=rule), "intents"])
            .size()
            .reset_index(name="count")
            .rename(columns={"intents": "intent"})
    )
else:
    tmp = df_f.reset_index()
    tmp["intent"] = tmp["intent_primary"].fillna("Unclassified")
    intent_ts = (
        tmp.groupby([pd.Grouper(key="timestamp", freq=rule), "intent"])
            .size()
            .reset_index(name="count")
    )

fig4 = px.line(intent_ts, x="timestamp", y="count", color="intent", markers=True)
fig4.update_layout(xaxis_title=f"{freq} timeline", yaxis_title="Count")
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Data preview
# ------------------------------
with st.expander("Preview data (filtered)"):
    st.dataframe(df_f.reset_index().head(50))