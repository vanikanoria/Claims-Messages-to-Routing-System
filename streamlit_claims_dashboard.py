#!/usr/bin/env python3
# streamlit_claims_dashboard.py
# Claims Messages Dashboard (SQLite -> Streamlit)

import ast
import sqlite3
from contextlib import closing
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =======================
# Data loading & parsing
# =======================
def load_messages(db_path: str) -> pd.DataFrame:
    """Load the 'messages' table and normalize types/columns we need."""
    with closing(sqlite3.connect(db_path)) as con:
        df = pd.read_sql_query("SELECT * FROM messages", con)

    # Timestamp: prefer 'timestamp', else 'ts_iso'
    ts = pd.to_datetime(df.get("timestamp"), errors="coerce", utc=False)
    if ts.isna().all() and "ts_iso" in df.columns:
        ts = pd.to_datetime(df["ts_iso"], errors="coerce", utc=False)
    df["timestamp"] = ts

    # Normalize roles (keep ALL roles for Overview volume chart)
    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.strip().str.lower()

    # Primary intent (your column name)
    if "primary_intent" in df.columns:
        s = df["primary_intent"].astype(str)
        s = s.where(~s.isin(["", "None", "nan"]), np.nan)
        df["primary_intent_use"] = s
    else:
        df["primary_intent_use"] = np.nan  # not present

    # Multi-intents: all_intents (list or string representation)
    def _parse_listish(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                try:
                    v = ast.literal_eval(s)
                    return list(v) if isinstance(v, (list, tuple)) else []
                except Exception:
                    return []
            return [s] if s else []
        return []

    if "all_intents" in df.columns:
        df["multi_intents_list"] = df["all_intents"].apply(_parse_listish)
    else:
        df["multi_intents_list"] = [[] for _ in range(len(df))]

    # Subject/Summary clean-up (if present)
    for col in ["subject", "summary"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"^```json\s*$", "", regex=True)
                .replace({"None": np.nan, "nan": np.nan})
                .str.strip()
            )

    # Keep valid timestamps
    df = df[df["timestamp"].notna()].copy()
    df.sort_values(["thread_id", "timestamp"], inplace=True)
    return df


# =======================
# KPI helpers
# =======================
def _dir_response_deltas(df: pd.DataFrame, from_role: str, to_role: str) -> List[pd.Timedelta]:
    """For each message by `from_role`, if the next in thread is `to_role`, record delta."""
    out: List[pd.Timedelta] = []
    for _, g in df.groupby("thread_id", sort=False):
        g = g.sort_values("timestamp")
        roles = g["role"].values
        times = g["timestamp"].values
        for i in range(len(g) - 1):
            if roles[i] == from_role and roles[i + 1] == to_role:
                dt = pd.Timestamp(times[i + 1]) - pd.Timestamp(times[i])
                if pd.notna(dt) and pd.Timedelta(0) <= dt <= pd.Timedelta(days=30):
                    out.append(dt)
    return out


def median_hours(deltas: List[pd.Timedelta]) -> Optional[float]:
    if not deltas:
        return None
    return float(np.median([d.total_seconds() / 3600 for d in deltas]))


# =======================
# Intent math (exact, per cohort)
# =======================
def primary_pct(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    % by primary intent within the cohort:
      drop NA -> value_counts(normalize=True) * 100
    """
    s = df_group["primary_intent_use"].dropna()
    if s.empty:
        return pd.DataFrame(columns=["intent", "pct"])
    pct = (s.value_counts(normalize=True) * 100).round(1)
    out = pct.reset_index()
    out.columns = ["intent", "pct"]
    return out


def multi_pct(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-intent % within the cohort:
      explode -> value_counts() * 100 / (# messages in cohort)
    Totals may exceed 100% because one message can have multiple intents.
    """
    n = len(df_group)
    if n == 0:
        return pd.DataFrame(columns=["intent", "pct"])
    ex = df_group.explode("multi_intents_list")
    ex = ex[ex["multi_intents_list"].notna() & (ex["multi_intents_list"] != "")]
    if ex.empty:
        return pd.DataFrame(columns=["intent", "pct"])
    pct = (ex["multi_intents_list"].value_counts() * 100.0 / n).round(1)
    out = pct.reset_index()
    out.columns = ["intent", "pct"]
    return out


def intents_trend_lines(df_group: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Weekly line series per intent (prefer primary; else multi)."""
    if df_group["primary_intent_use"].notna().any():
        base = df_group.dropna(subset=["primary_intent_use"]).copy()
        base["intent_for_trend"] = base["primary_intent_use"]
    else:
        base = df_group.explode("multi_intents_list")
        base = base[base["multi_intents_list"].notna() & (base["multi_intents_list"] != "")]
        base["intent_for_trend"] = base["multi_intents_list"]

    if base.empty:
        return pd.DataFrame(columns=["timestamp", "intent_for_trend", "count"])

    base = base.set_index("timestamp").sort_index()
    ts = base.groupby([pd.Grouper(freq=freq), "intent_for_trend"]).size().reset_index(name="count")
    return ts


# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Claims Messages Dashboard", layout="wide")
st.title("📊 Claims Messages Dashboard")
st.caption("Source: SQLite database → table `messages`")

# Sidebar — DB & filters
st.sidebar.header("Data")
db_path = st.sidebar.text_input("SQLite DB path", value="claims.db")

# Load data
try:
    df = load_messages(db_path)
except Exception as e:
    st.error(f"Failed to load DB: {e}")
    st.stop()

if df.empty:
    st.warning("No rows found after timestamp parsing. Check your DB.")
    st.stop()

# Date filter (global)
min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
min_d, max_d = min_ts.date(), max_ts.date()

st.sidebar.subheader("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
    key="date_range_unique",  # unique key avoids duplicate ID errors
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = (min_d, max_d)

# Optional role filter for Overview & AI tabs (NOT used for Intents to avoid leakage)
roles_present = sorted(df["role"].dropna().unique().tolist())
sel_roles = st.sidebar.multiselect("Roles (Overview & AI tabs)", roles_present, default=roles_present)

# Apply filters for Overview/AI tabs
mask_date = (df["timestamp"].dt.date >= start_d) & (df["timestamp"].dt.date <= end_d)
df_overview = df[mask_date].copy()
if sel_roles:
    df_overview = df_overview[df_overview["role"].isin(sel_roles)]

if df_overview.empty:
    st.warning("No data after filters.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total messages", f"{len(df_overview):,}")
with c2:
    d1 = _dir_response_deltas(df_overview, "claimant", "adjuster")
    st.metric("Median response (Claimant → Adjuster)", f"{median_hours(d1):.1f} h" if d1 else "—")
with c3:
    d2 = _dir_response_deltas(df_overview, "adjuster", "claimant")
    st.metric("Median response (Adjuster → Claimant)", f"{median_hours(d2):.1f} h" if d2 else "—")
with c4:
    st.metric("Active threads", f"{df_overview['thread_id'].nunique():,}")

st.divider()

tab_overview, tab_ai, tab_intents = st.tabs(["Overview", "AI Insights", "Intent Analysis"])

# -------------------- Overview --------------------
with tab_overview:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Volume of Messages over Time")
        mv = (
            df_overview.set_index("timestamp")
            .sort_index()["content"]
            .resample("M")
            .count()
            .rename("count")
            .reset_index()
        )
        if not mv.empty:
            fig = px.line(mv, x="timestamp", y="count", markers=True)
            fig.update_layout(yaxis_title="Messages", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time series in current filter.")

    with right:
        st.subheader("Volume of Messages by Role")
        # Show ALL roles present in the date range (ignore sidebar role filter here)
        role_counts = df[mask_date]["role"].value_counts()
        if not role_counts.empty:
            role_df = role_counts.reset_index()
            role_df.columns = ["role", "count"]
            fig2 = px.bar(role_df, x="role", y="count", text="count")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No roles in this date range.")

# -------------------- AI Integration --------------------
with tab_ai:
    st.subheader("AI Integration — Subjects, Summaries, and Intents")
    keep = ["message_id", "thread_id", "timestamp", "role", "content"]
    for col in ["subject", "summary", "primary_intent", "all_intents"]:
        if col in df_overview.columns:
            keep.append(col)

    table = df_overview[keep].copy()

    # shorten long text for the grid
    table["content"] = table["content"].astype(str).str.slice(0, 240)
    if "summary" in table.columns:
        table["summary"] = table["summary"].astype(str).str.slice(0, 300)

    # Pretty column names
    pretty_names = {
        "message_id": "Message ID",
        "thread_id": "Thread ID",
        "timestamp": "Timestamp",
        "role": "Role",
        "content": "Content",
        "subject": "Subject",
        "summary": "Summary",
        "primary_intent": "Primary Intent",
        "all_intents": "All Intents"
    }
    table.rename(columns=pretty_names, inplace=True)

    st.dataframe(table.reset_index(drop=True), use_container_width=True)

    # -------------------- Intents --------------------
# -------------------- Intents (Monthly) --------------------
with tab_intents:
    st.subheader("Claimant Intents — Monthly")

    # Build cohort from *date-only* filtered frame (avoid role-filter leakage)
    df_date = df[mask_date].copy()
    df_claim = df_date[df_date["role"] == "claimant"].copy()

    if df_claim.empty:
        st.info("No claimant messages in the selected date range.")
        st.stop()

    # --- % by primary intent (within claimants only)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Primary Intent of Claimants")
        p = primary_pct(df_claim)  # same math; % of claimant messages overall
        if p.empty:
            st.info("No primary intents available.")
        else:
            fig = px.bar(
                p, x="intent", y="pct", text="pct",
                title="Share of Claimant Messages by Primary Intent"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                yaxis_title="% of claimant messages",
                xaxis_title="Primary Intent",
                uniformtext_minsize=10, uniformtext_mode="hide",
                margin=dict(t=60)
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Distributino of all Intents of Claimants")
        m = multi_pct(df_claim)  # explode list; denom = # claimant msgs
        if m.empty:
            st.info("No multi-intents available.")
        else:
            fig = px.bar(
                m, x="intent", y="pct", text="pct",
                title="Share of Claimant Messages Containing Each Intent"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                yaxis_title="% of claimant messages",
                xaxis_title="Intents",
                uniformtext_minsize=10, uniformtext_mode="hide",
                margin=dict(t=60)
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Intent trend over time — separate chart per intent (Monthly)
    st.markdown("### Monthly Intent Trend — Claimants")

    # Reuse your helper, but pass monthly frequency
    tr = intents_trend_lines(df_claim, freq="MS")  # MS = month start

    if tr.empty:
        st.info("No intents to trend for claimants in this date range.")
    else:
        # Order intents by total volume so the most important appear first
        intent_order = (
            tr.groupby("intent_for_trend")["count"].sum().sort_values(ascending=False).index.tolist()
        )

        selected_intents = st.multiselect(
            "Select intents to display",
            options=intent_order,
            default=intent_order[:min(6, len(intent_order))]
        )

        if not selected_intents:
            st.info("Select at least one intent to view the trend.")
        else:
            # One separate chart per intent (monthly)
            for intent_name in selected_intents:
                sub = tr[tr["intent_for_trend"] == intent_name].copy()
                if sub.empty:
                    continue
                # Optional: show Month label nicely
                sub["Month"] = sub["timestamp"].dt.to_period("M").dt.to_timestamp()

                fig = px.line(
                    sub, x="Month", y="count", markers=True,
                    title=f"Monthly Volume — {intent_name} (Claimants)"
                )
                fig.update_layout(
                    yaxis_title="Messages",
                    xaxis_title="Month",
                    margin=dict(t=60)
                )
                st.plotly_chart(fig, use_container_width=True)