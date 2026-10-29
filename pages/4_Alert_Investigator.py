"""
4_Alert_Investigator.py
-----------------------
Alert Investigator page: groups flagged intervals into incidents, explains each one
with rules, and (optionally) asks an LLM to rewrite the findings in plain English.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_detection import select_threshold
from src.investigator import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DISCLAIMER,
    build_digest,
    incident_facts,
    run_investigation,
    summarise_with_fallback,
)

st.set_page_config(page_title="Alert Investigator", page_icon=None, layout="wide")
st.markdown("## Alert Investigator")
st.caption(
    "An automated investigation step: group flagged intervals into incidents, compare each "
    "with typical usage, rank them, and write an alert digest."
)

# ── Guard ───────────────────────────────────────────────────────────────────
trained    = st.session_state.get("trained", False)
scores     = st.session_state.get("scores", None)
val_scores = st.session_state.get("val_scores", None)
threshold  = st.session_state.get("threshold", None)
test_df    = st.session_state.get("test_df", None)
train_df   = st.session_state.get("train_df", None)

if not trained or scores is None or test_df is None or train_df is None or val_scores is None:
    st.warning("**Model not trained yet.** Go to Model Lab and click **Train RBM** first.")
    st.page_link("pages/3_Model_Lab.py", label="Go to Model Lab")
    st.stop()

# ── Sidebar settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Investigator Settings")
    pct = st.slider("Alert threshold (percentile of val scores)", 80, 99, 95, 1)
    max_gap = st.slider(
        "Merge flags closer than (intervals)", 1, 8, 2,
        help="Flags this many 15-minute intervals apart or less are merged into one incident.",
    )
    top_n = st.slider("Incidents in the digest", 3, 10, 5)

    st.divider()
    st.markdown("### Optional AI summary")
    use_llm = st.checkbox("Enhance summary with AI (Groq)", value=False)
    api_key = ""
    base_url = DEFAULT_BASE_URL
    model = DEFAULT_MODEL
    if use_llm:
        api_key = st.text_input(
            "API key", type="password",
            help="Used only for this session. Or set the GROQ_API_KEY environment variable.",
        )
        model = st.text_input("Model", value=DEFAULT_MODEL)
        base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
        st.caption("Only the findings (times and readings) are sent, never your raw data file.")

# ── Run the investigation ───────────────────────────────────────────────────
thr = select_threshold(val_scores, float(pct))
n_flagged = int(np.sum(scores > thr))
incidents = run_investigation(test_df, scores, thr, train_df, val_scores, max_gap=max_gap)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flagged intervals", f"{n_flagged:,}")
c2.metric("Incidents", f"{len(incidents):,}")
c3.metric("High severity", sum(1 for i in incidents if i["level"] == "High"))
c4.metric("Longest incident", f"{max((i['span_intervals'] for i in incidents), default=0) * 15} min")

st.info(DISCLAIMER)
st.divider()

if not incidents:
    st.success("No incidents at this threshold.")
    st.stop()

tab_digest, tab_table, tab_detail = st.tabs(["Alert digest", "Incident table", "Investigate an incident"])

# ── Digest ──────────────────────────────────────────────────────────────────
with tab_digest:
    if use_llm:
        if st.button("Generate AI summary", type="primary"):
            with st.spinner("Asking the language model..."):
                text, used, err = summarise_with_fallback(
                    incidents, n_flagged, True, api_key or None, base_url, model, top_n
                )
            st.session_state["digest_text"] = text
            st.session_state["digest_used_llm"] = used
            st.session_state["digest_err"] = err
        text = st.session_state.get("digest_text")
        if text:
            if st.session_state.get("digest_err"):
                st.warning(st.session_state["digest_err"])
            st.caption("AI-written summary" if st.session_state.get("digest_used_llm") else "Rule-based digest")
            st.text(text)
            st.download_button("Download digest", text, file_name="alert_digest.txt")
        else:
            st.caption("Click **Generate AI summary**. The rule-based digest is shown below meanwhile.")
            st.text(build_digest(incidents, n_flagged, top_n))
    else:
        digest = build_digest(incidents, n_flagged, top_n)
        st.caption("Rule-based digest (works offline). Turn on the AI option in the sidebar for a friendlier rewrite.")
        st.text(digest)
        st.download_button("Download digest", digest, file_name="alert_digest.txt")

# ── Table ───────────────────────────────────────────────────────────────────
with tab_table:
    rows = [{
        "ID": i["id"],
        "Severity": i["level"],
        "Start": i["start"],
        "End": i["end"],
        "Flagged intervals": i["n_flagged"],
        "Peak score": round(i["peak_score"], 3),
        "Label": i["label"],
    } for i in incidents]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ── Detail ──────────────────────────────────────────────────────────────────
with tab_detail:
    options = {f"#{i['id']}  {i['level']}  {i['start']:%Y-%m-%d %H:%M}  -  {i['label']}": i for i in incidents[:200]}
    choice = st.selectbox("Choose an incident (most severe first)", list(options))
    inc = options[choice]
    st.markdown(f"**{inc['label']}**")
    st.text(incident_facts(inc))

    comp = pd.DataFrame([{
        "Reading": f,
        "Actual (mean)": round(s["actual"], 3),
        "Typical (training, same hours)": round(s["typical"], 3),
        "Ratio": round(s["ratio"], 2) if not np.isnan(s["ratio"]) else None,
    } for f, s in inc["findings"].items()])
    st.dataframe(comp, hide_index=True)
    st.caption("Typical = median of the training period for the same hours of the day.")