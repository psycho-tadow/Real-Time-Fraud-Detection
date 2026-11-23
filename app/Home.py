import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils import setup, hero, card, pills
from site_config import DATA, active_results

setup("Home", "🛡️")

hero("Real-Time Fraud Detection System",
     "An end-to-end machine learning project: from highly imbalanced data to an "
     "explainable, cost-aware fraud detector you can try live.")

res, label = active_results()
saved = 1 - res["cost"] / res["cost_nothing"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraud rate in the data", DATA["fraud_rate"])
c2.metric("PR-AUC", f"{res['pr_auc']:.2f}")
c3.metric("Frauds caught", f"{res['tp']} of {res['tp'] + res['fn']}")
c4.metric("Cost vs. no model", f"-{saved:.1%}")
st.markdown(f'<p class="note">Headline numbers are from the {label}.</p>',
            unsafe_allow_html=True)

st.write("")
st.subheader("What this project shows")
a, b, c = st.columns(3)
with a:
    card("The problem",
         "Fraud is about 1 in 600 transactions. A model that always says 'legit' is "
         "99.8% accurate and useless, so the project is built around better metrics.")
with b:
    card("The method",
         "Time-based validation, careful leakage control, class-imbalance strategies "
         "compared fairly, and a decision threshold chosen from business cost.")
with c:
    card("The product",
         "A deployed app that scores transactions, shows the decision and explains "
         "the top reasons behind every score with SHAP.")

st.write("")
st.subheader("Take the tour")
st.markdown("""
- **Learn**: how real-time fraud detection works, in plain language.
- **Data & EDA**: what the data looks like and why it is hard.
- **Method**: how the model was built and evaluated, and every decision along the way.
- **Live Demo**: score real transactions, change features, upload your own CSV.
- **Results**: final numbers, explanations and honest limitations.
- **About**: tech stack, code link and how to run it yourself.
""")

st.write("")
pills(["Python", "pandas", "scikit-learn", "XGBoost", "SHAP", "Streamlit", "100% free to host"])
