import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import setup, hero, pills
from site_config import GITHUB_URL, AUTHOR, LINKEDIN_URL

setup("About", "ℹ️")
hero("About this project", f"Built by {AUTHOR} as an end-to-end machine learning portfolio project.")

st.markdown("### Tech stack")
pills(["Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "imbalanced-learn",
       "SHAP", "Streamlit", "pytest", "Git & GitHub"])

st.markdown("### Links")
st.markdown(f"- Source code: [{GITHUB_URL}]({GITHUB_URL})")
if LINKEDIN_URL:
    st.markdown(f"- LinkedIn: [{LINKEDIN_URL}]({LINKEDIN_URL})")

st.markdown("### Project structure")
st.code("""data/        sample data for the demo (full dataset not stored)
notebooks/   EDA and modelling
src/         data loading and time split, metrics, cost threshold
models/      trained model and chosen threshold
app/         this website
tests/       unit tests""", language="text")

st.markdown("### Run it yourself")
st.code("""python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app/app.py""", language="bash")

st.markdown("### What I would do next")
st.markdown("""
- Monitor score drift and retrain on a schedule with experiment tracking.
- Add customer-history features such as spending velocity.
- Serve the model behind an API with latency monitoring.
- Evaluate on a second dataset to check the method generalises.
""")
