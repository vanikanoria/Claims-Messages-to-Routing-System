
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Claims Messaging KPIs", page_icon="📊", layout="wide")
st.title("📊 Claims Messaging KPIs & Visuals")

st.markdown("""
**Upload a CSV / JSON / JSONL file** with at least these columns:
- `timestamp` (ISO datetime)
- `role` (e.g., claimant / adjuster)
- `content` (text of the message)

Optional but recommended:
- `thread_id` (conversation id)
- `intents` (JSON array of labels) and/or `intent_primary`
""")

# ---------------------------
# Helpers
# ---------------------------
def _normalize_df(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".jsonl") or name.endswith(".ndjson"):
        rows = [json.loads(l) for l in uploaded_file.getvalue().decode("utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type. Upload CSV / JSON / JSONL.")
        return pd.DataFrame()

    # Standardize columns
    # timestamp
    tcol = None
    for c in ["timestamp","created_at","time","date"]:
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        st.error("No timestamp column found. Please include a 'timestamp' column."); return pd.DataFrame()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(None)
    df = df[~df[tcol].isna()].copy()
    df = df.sort_values(tcol)

    # role
    rcol = None
    for c in ["role","sender_role","from_role"]:
        if c in df.columns:
            rcol = c; break
    if rcol is None:
        df["role"] = "unknown"; rcol = "role"
    else:
        df[rcol] = df[rcol].astype(str).str.lower().str.strip()

    # content
    ccol = None
    for c in ["content","text","message"]:
        if c in df.columns:
            ccol = c; break
    if ccol is None:
        df["content"] = ""; ccol = "content"

    # thread_id
    thcol = None
    for c in ["thread_id","conversation_id","claim_id"]:
        if c in df.columns:
            thcol = c; break
    if thcol is None:
        # fabricate thread ids if missing (not ideal, but enables response-time calc)
        df["thread_id"] = df.groupby(rcol).cumcount()
        thcol = "thread_id"

    # intents
    if "intents" in df.columns:
        def to_list(x):
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, str) and x.strip().startswith("["):
                try: return json.loads(x)
                except Exception: return []
            return []
        df["intents"] = df["intents"].apply(to_list)
    else:
        df["intents"] = [[] for _ in range(len(df))]

    # primary intent
    if "intent_primary" not in df.columns:
        df["intent_primary"] = df["intents"].apply(lambda xs: xs[0] if xs else "Unclassified")

    # rename unified
    df.rename(columns={tcol:"timestamp", rcol:"role", thcol:"thread_id", ccol:"content"}, inplace=True)
    return df

def compute_response_times(df: pd.DataFrame) -> pd.DataFrame:
    """Compute response times when the role switches inside a thread.
       Returns columns: thread_id, direction, response_minutes, timestamp"""
    if df.empty: return pd.DataFrame(columns=["thread_id","direction","response_minutes","timestamp"])
    out = []
    for tid, g in df.sort_values("timestamp").groupby("thread_id"):
        prev_time, prev_role = None, None
        for _, row in g.iterrows():
            t, role = row["timestamp"], row["role"]
            if prev_time is not None and role != prev_role and {"claimant","adjuster"} <= {role, prev_role} | {"claimant","adjuster"}:
                direction = f"{prev_role}→{role}"
                delta_min = (t - prev_time).total_seconds() / 60.0
                if 0 < delta_min < 60*24*30:  # under 30 days
                    out.append({"thread_id": tid, "direction": direction, "response_minutes": delta_min, "timestamp": t})
            prev_time, prev_role = t, role
    return pd.DataFrame(out)

def pct_messages_by_intent(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        xs = r["intents"] if isinstance(r["intents"], (list,tuple)) and len(r["intents"])>0 else [r.get("intent_primary","Unclassified")]
        for it in xs:
            rows.append({"intent": it})
    d = pd.DataFrame(rows)
    if d.empty: return d
    counts = d.value_counts("intent").reset_index(name="count")
    counts["percent"] = 100 * counts["count"] / counts["count"].sum()
    return counts.sort_values("percent", ascending=False)

def weekly_escalation_pct_by_role(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    tmp = df.copy()
    tmp["week"] = tmp["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    # treat any intent containing "escalation" (case-insensitive) as escalation/complaint
    def is_escalation(row):
        intents = row["intents"] if isinstance(row["intents"], (list,tuple)) else []
        intents = [*intents, row.get("intent_primary","")]
        return any("escalation" in str(x).lower() for x in intents)
    tmp["is_escalation"] = tmp.apply(is_escalation, axis=1)
    grp = tmp.groupby(["week","role"]).agg(
        total=("content","count"),
        escalations=("is_escalation","sum")
    ).reset_index()
    grp["pct_escalation"] = np.where(grp["total"]>0, 100*grp["escalations"]/grp["total"], 0.0)
    return grp

# ---------------------------
# UI
# ---------------------------
uploaded = st.file_uploader("Upload dataset", type=["csv","json","jsonl","ndjson"])
df = _normalize_df(uploaded)

if df.empty:
    st.stop()

# Filters
min_dt, max_dt = df["timestamp"].min().date(), df["timestamp"].max().date()
start, end = st.sidebar.date_input("Date range", (min_dt, max_dt))
if isinstance(start, tuple):
    start, end = start
df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end) + pd.Timedelta(days=1))]

roles = sorted(df["role"].unique())
role_sel = st.sidebar.multiselect("Roles", roles, default=roles)
df = df[df["role"].isin(role_sel)]

st.divider()

# ---------------------------
# KPIs
# ---------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total messages processed", f"{len(df):,}")

# response times
rt = compute_response_times(df)
with col2:
    c_to_a = rt[rt["direction"]=="claimant→adjuster"]["response_minutes"]
    st.metric("Avg response time (claimant→adjuster)", f"{c_to_a.mean():.1f} min" if len(c_to_a) else "—")

with col3:
    a_to_c = rt[rt["direction"]=="adjuster→claimant"]["response_minutes"]
    st.metric("Avg response time (adjuster→claimant)", f"{a_to_c.mean():.1f} min" if len(a_to_c) else "—")

with col4:
    wk = weekly_escalation_pct_by_role(df)
    recent = wk[wk["week"]==wk["week"].max()] if not wk.empty else pd.DataFrame()
    if not recent.empty:
        avg_pct = recent["pct_escalation"].mean()
        st.metric("% Escalation (last week avg)", f"{avg_pct:.1f}%")
    else:
        st.metric("% Escalation (last week avg)", "—")

st.divider()

# ---------------------------
# Visual 1: % of messages by intents
# ---------------------------
st.subheader("% of messages by intents")
pct = pct_messages_by_intent(df)
if pct.empty:
    st.info("No intent data found. Include `intents` (list) or `intent_primary`.")
else:
    fig = px.bar(pct, x="intent", y="percent", title="% of Messages by Intent")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Visual 2: Volume of messages by role
# ---------------------------
st.subheader("Volume of messages by role")
by_role = df.value_counts("role").reset_index(name="count")
fig2 = px.bar(by_role, x="role", y="count", title="Message Volume by Role")
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Visual 3: % Escalation by week and role
# ---------------------------
st.subheader("% of messages classified as Escalation/Complaint by week and role")
wk = weekly_escalation_pct_by_role(df)
if wk.empty:
    st.info("Not enough data to compute weekly escalation rates.")
else:
    fig3 = px.line(wk, x="week", y="pct_escalation", color="role",
                   markers=True, title="% Escalation by Week and Role")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# Raw table (optional)
# ---------------------------
with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)
