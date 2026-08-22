
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from app.db import DB_PATH, get_conn, get_stats

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("💸 LLM Cost Autopilot — Dashboard")

window = st.sidebar.selectbox("Time window", ["All time", "Last 24h", "Last 7d", "Last 30d"])
window_map = {"All time": None, "Last 24h": 1, "Last 7d": 7, "Last 30d": 30}
days = window_map[window]
since_seconds = days * 86400 if days else None

stats = get_stats(since_seconds=since_seconds)

if stats["total_requests"] == 0:
    st.info("No requests logged yet. Send some traffic to the API "
            "(see `scripts/load_test.py`) and refresh this page.")
    st.stop()

# --- Headline "money shot" metric --------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", f"{stats['total_requests']:,}")
col2.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
col3.metric(
    f"Cost saved vs. all-{stats['baseline_model']}",
    f"{stats['cost_savings_pct']:.1f}%",
    help=f"Hypothetical cost if every request went to {stats['baseline_model']}: "
         f"${stats['hypothetical_cost_if_all_on_most_expensive_model_usd']:.4f}",
)
col4.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")

st.markdown(
    f"### 🎯 Routing to cheaper models saved **{stats['cost_savings_pct']:.1f}%** "
    f"compared to sending everything to `{stats['baseline_model']}`."
)

col5, col6 = st.columns(2)
col5.metric("Avg quality score (LLM-as-judge)", f"{stats['avg_quality_score']:.2f} / 1.00")
col6.metric("Escalation rate", f"{stats['escalation_rate_pct']:.1f}%",
            help="Share of requests where the verifier caught a quality gap and flagged escalation.")

st.divider()

# --- Routing distribution -----------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Routing distribution")
    dist_df = pd.DataFrame(stats["routing_distribution"])
    if not dist_df.empty:
        st.bar_chart(dist_df.set_index("model")["count"])
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Cost by model")
    if not dist_df.empty:
        st.bar_chart(dist_df.set_index("model")["cost_usd"])

st.divider()

# --- Cost over time -------------------------------------------------------
st.subheader("Cost per day")
with get_conn() as conn:
    df = pd.read_sql_query(
        "SELECT timestamp, cost, complexity_tier, routed_model, quality_score, escalated "
        "FROM requests ORDER BY timestamp",
        conn,
    )

if not df.empty:
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    daily_cost = df.groupby("date")["cost"].sum()
    st.line_chart(daily_cost)

    st.subheader("Quality score distribution")
    scored = df.dropna(subset=["quality_score"])
    if not scored.empty:
        st.bar_chart(scored["quality_score"].value_counts(bins=10).sort_index())

    st.subheader("Recent requests")
    st.dataframe(
        df.tail(50).sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.caption(f"Reading from `{DB_PATH}`")
