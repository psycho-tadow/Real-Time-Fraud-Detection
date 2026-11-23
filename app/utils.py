"""Shared helpers: page setup, styling, model loading, explanations."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 2.4rem 2.2rem; border-radius: 18px;
  background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
  border: 1px solid #1F2A44; margin-bottom: 1.2rem;}
.hero h1 {color: #FFFFFF; margin: 0 0 .4rem 0; font-size: 2.3rem;}
.hero p {color: #CBD5E1; font-size: 1.08rem; margin: 0;}
.card {background: #111A2E; border: 1px solid #1F2A44; border-radius: 14px;
  padding: 1.1rem 1.2rem; height: 100%;}
.card h4 {margin: 0 0 .45rem 0; color: #93C5FD; font-size: 1.02rem;}
.card p {margin: 0; color: #CBD5E1; font-size: .95rem; line-height: 1.5;}
.pill {display: inline-block; padding: .18rem .7rem; border-radius: 999px;
  background: #1E3A8A; color: #DBEAFE; font-size: .82rem; margin: 0 .35rem .4rem 0;}
.verdict {border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
  font-size: 1.6rem; font-weight: 700;}
.verdict.fraud {background: #7F1D1D; color: #FEE2E2; border: 1px solid #EF4444;}
.verdict.legit {background: #064E3B; color: #D1FAE5; border: 1px solid #10B981;}
.note {color: #94A3B8; font-size: .88rem;}
</style>
"""


def setup(title, icon):
    """Call first on every page."""
    st.set_page_config(page_title=f"{title} | Fraud Detection", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def pills(items):
    st.markdown("".join(f'<span class="pill">{i}</span>' for i in items),
                unsafe_allow_html=True)


def show_figure(name, caption=None):
    """Show a chart saved from your notebook; explain how to create it if missing."""
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Chart `{name}` not found yet. Save it from your notebook into "
                f"`reports/figures/` (see the figure-saving cell) and push to GitHub.")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import shap
    model = joblib.load(ROOT / "models" / "model.pkl")
    cfg = json.loads((ROOT / "models" / "threshold.json").read_text())
    return model, cfg, shap.TreeExplainer(model)


@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "data" / "sample.csv")


def score(df):
    model, cfg, _ = load_model()
    return model.predict_proba(df[cfg["features"]])[:, 1]


def explain_row(row, top=8):
    """Top SHAP contributions for one transaction (positive = pushes toward fraud)."""
    _, cfg, explainer = load_model()
    vals = explainer.shap_values(row[cfg["features"]])[0]
    s = pd.Series(vals, index=cfg["features"])
    return s.reindex(s.abs().sort_values(ascending=False).index)[:top]
