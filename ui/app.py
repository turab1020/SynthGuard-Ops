"""SynthGuard-Ops UI: live income predictions with SHAP explanations, from the deployed API.

Run:  streamlit run ui/app.py        (SYNTHGUARD_API_URL, default: the Render API)
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
from client import Api, api_url, contribution_rows, field_specs

FAVICON = Path(__file__).with_name("favicon.svg")
st.set_page_config(page_title="SynthGuard-Ops", page_icon=str(FAVICON), layout="wide")
COLORS = alt.Scale(domain=["towards >50K", "towards <=50K"], range=["#2e7d32", "#c62828"])


@st.cache_resource
def get_api(base: str) -> Api:
    return Api(base)


@st.cache_data(ttl=300, show_spinner=False)
def load_meta(base: str) -> tuple[dict, dict, dict]:
    api = get_api(base)
    health = api.wake()
    return health, api.info(), api.openapi()


# ------------------------------------------------------------------ sidebar: which API
base = st.sidebar.text_input("API URL", value=api_url()).rstrip("/")
st.sidebar.caption("A sleeping free instance can take about a minute to wake up.")
if st.sidebar.button("Reconnect"):
    load_meta.clear()

try:
    with st.spinner(f"Connecting to {base} …"):
        health, info, openapi = load_meta(base)
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {base}: {exc}")
    st.stop()

st.sidebar.success("API is up")
status = [f"**Model version:** {health.get('model_version', '?')}",
          f"**Trained on:** {health.get('trained_on') or '?'}"]  # fmt: skip
if health.get("commit"):
    status.append(f"**Commit:** `{health['commit'][:7]}`")
st.sidebar.markdown("  \n".join(status))
st.sidebar.markdown(f"[API docs]({base}/docs) · [metrics]({base}/metrics)")

st.title("SynthGuard-Ops")
st.write(
    "Income classifier trained **only on synthetic data** from the champion generator and "
    "checked on a fixed real hold-out. Enter a person's details to get a live prediction, "
    "its confidence, and the SHAP contribution of each feature."
)

tabs = ["Predict"] + (["Generate"] if info.get("generate_enabled") else [])
views = st.tabs(tabs)

# ------------------------------------------------------------------ Predict
with views[0]:
    specs = field_specs(openapi)
    with st.form("predict"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(specs):
            with cols[i % 3]:
                if f.kind == "category":
                    index = f.options.index(f.default)
                    values[f.name] = st.selectbox(f.label, f.options, index=index)
                else:
                    values[f.name] = int(st.number_input(f.label, min_value=f.minimum,
                                                         max_value=f.maximum, value=f.default,
                                                         step=1))  # fmt: skip
        top_k = st.slider("Features to explain", 3, len(specs), 5)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        try:
            st.session_state["result"] = get_api(base).predict(values, top_k=top_k)
        except ValueError as exc:
            st.error(f"The API rejected the input: {exc}")
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("result")
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result["label"])
        c2.metric("P(income > 50K)", f"{result['probability_gt_50k']:.1%}")
        c3.metric("Confidence", f"{result['confidence']:.1%}")

        exp = result["explanation"]
        rows = pd.DataFrame(contribution_rows(exp))
        st.subheader("Why: SHAP contributions (log-odds)")
        chart = (
            alt.Chart(rows)
            .mark_bar()
            .encode(
                x=alt.X("contribution:Q", title="contribution to log-odds of >50K"),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("direction:N", scale=COLORS, title=None),
                tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
            )
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"Base value {exp['base_value']:+.3f} + sum of contributions = logit "
            f"{exp['logit']:+.3f} → probability {result['probability_gt_50k']:.3f}. "
            f"Model version {result['model_version']}."
        )

# ------------------------------------------------------------------ Generate (local only)
if "Generate" in tabs:
    with views[1]:
        st.write("Sample rows from the champion synthetic-data generator.")
        n = st.number_input("Rows", min_value=1, max_value=1000, value=10, step=1)
        if st.button("Generate"):
            try:
                with st.spinner("Sampling …"):
                    out = get_api(base).generate(int(n))
                st.caption(f"Generator: {out.get('generator', {})}")
                st.dataframe(pd.DataFrame(out["rows"]), width="stretch")
            except requests.RequestException as exc:
                st.error(f"Generation failed: {exc}")
else:
    st.sidebar.caption("The Generate tab appears when the API has /generate enabled (local run).")
