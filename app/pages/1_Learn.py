import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from utils import setup, hero, card
from site_config import VALID

setup("Learn", "📘")
hero("Learn: how fraud detection works",
     "Everything you need to understand this project, no prior knowledge required.")

t1, t2, t3, t4, t5 = st.tabs(
    ["The problem", "Why it is hard", "A real system", "Metrics", "Glossary"])

with t1:
    st.markdown("""
### What is real-time fraud detection?
Every time a card is used, the bank must decide whether to **approve** or **block**
the payment, and it has to decide in a fraction of a second, while the customer is
standing at the till or waiting on a checkout page.

A machine learning model helps by giving every transaction a **fraud score** from
0 to 1. A **threshold** then turns the score into an action: score above the
threshold means flag it, otherwise let it through.

### Two ways to be wrong
""")
    a, b = st.columns(2)
    with a:
        card("Missed fraud (false negative)",
             "The bank or customer loses the money. In this project a missed fraud "
             "costs the transaction amount.")
    with b:
        card("False alarm (false positive)",
             "A genuine customer is blocked or an analyst wastes time. In this project "
             "each false alarm costs a fixed amount.")
    st.markdown("""
These two mistakes do **not** cost the same, which is why the final threshold in
this project is chosen from cost, not from a default cutoff of 0.5 by habit.
""")

with t2:
    st.markdown("### Six reasons this problem is hard")
    items = [
        ("1. Extreme imbalance",
         "Fraud is a tiny fraction of transactions (about 0.17% here). Models "
         "trained naively learn to ignore it."),
        ("2. Unequal costs",
         "A missed fraud and a false alarm cost different amounts, so plain "
         "accuracy is the wrong yardstick."),
        ("3. Speed",
         "The decision must come back in milliseconds, which limits how complex "
         "the model and its inputs can be."),
        ("4. Fraudsters adapt",
         "Attackers change tactics once they are caught, so patterns drift over "
         "time (concept drift) and models go stale."),
        ("5. Delayed labels",
         "The truth often arrives weeks later, for example when a customer "
         "disputes a charge, so fresh data is never fully labelled."),
        ("6. Explanations matter",
         "Analysts and regulators need to know why something was flagged, so a "
         "black-box score is not enough."),
    ]
    for i in range(0, 6, 3):
        cols = st.columns(3)
        for col, (t, b) in zip(cols, items[i:i + 3]):
            with col:
                card(t, b)
        st.write("")

with t3:
    st.markdown("### How a production system is organised")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1E3A8A", fontcolor="white",
        color="#3B82F6", fontname="Helvetica"];
  edge [color="#64748B", fontcolor="#94A3B8", fontname="Helvetica"];
  T [label="Card\\ntransaction"];
  F [label="Build\\nfeatures"];
  R [label="Rules\\n(hard blocks)"];
  M [label="ML model\\nscore"];
  D [label="Threshold\\ndecision"];
  A [label="Approve", fillcolor="#065F46", color="#10B981"];
  V [label="Review /\\nBlock", fillcolor="#7F1D1D", color="#EF4444"];
  L [label="Labels:\\nanalysts, disputes", fillcolor="#334155", color="#64748B"];
  T -> F -> R -> M -> D;
  D -> A; D -> V;
  V -> L [style=dashed]; L -> M [style=dashed, label="retrain"];
}
""")
    st.markdown("""
Real systems usually combine **rules** (for obvious cases) with a **model** (for
subtle patterns), send uncertain cases to human analysts, and **retrain** as new
confirmed labels arrive.

**What this project covers:** the model, its evaluation, the cost-based decision
threshold, explanations and a deployed app.

**What it does not cover (future work):** streaming infrastructure, a feature store
with customer history such as spending velocity, and drift monitoring in production.
""")

with t4:
    st.markdown("### Why accuracy is useless here")
    st.markdown("""
Fraud is about 0.17% of transactions, so a model that predicts **'legit' for
everything** is about **99.8% accurate** and catches **zero** fraud.
So we use metrics that focus on the rare class:
""")
    a, b, c = st.columns(3)
    with a:
        card("Precision", "Of the transactions we flagged, what share were really fraud?")
    with b:
        card("Recall", "Of all real frauds, what share did we catch?")
    with c:
        card("PR-AUC", "Summarises the precision-recall trade-off across every threshold. "
             "Random guessing scores about the fraud rate, roughly 0.001.")
    st.markdown("### A worked example (validation set, threshold 0.5)")
    cm = pd.DataFrame(
        [[VALID["tn"], VALID["fp"]], [VALID["fn"], VALID["tp"]]],
        index=["Actually legit", "Actually fraud"],
        columns=["Predicted legit", "Predicted fraud"])
    st.table(cm)
    st.markdown(f"""
- **Precision** = {VALID['tp']} / ({VALID['tp']} + {VALID['fp']}) = **{VALID['tp'] / (VALID['tp'] + VALID['fp']):.2f}**
- **Recall** = {VALID['tp']} / ({VALID['tp']} + {VALID['fn']}) = **{VALID['tp'] / (VALID['tp'] + VALID['fn']):.3f}**
""")

with t5:
    gloss = {
        "Class imbalance": "When one class is far rarer than the other. Here fraud is about 1 in 600.",
        "Time-based split": "Training on the past and testing on the future, the way a model is really used. A random split can leak future patterns into training.",
        "Data leakage": "Information from outside the training data sneaking into the model, which makes results look better than they will be in reality.",
        "Threshold": "The score cutoff above which a transaction is flagged as fraud.",
        "SMOTE": "A method that creates synthetic examples of the rare class. It must only be applied to training data.",
        "scale_pos_weight": "An XGBoost setting that makes each fraud count for more during training, so the rare class is not ignored.",
        "Gradient boosting (XGBoost)": "Builds many small decision trees one after another, each correcting the previous ones' mistakes.",
        "SHAP": "A method that assigns each feature a contribution to a single prediction, so the model can be explained transaction by transaction.",
        "Concept drift": "When the patterns in real data change over time so a trained model slowly becomes less accurate.",
        "PCA features (V1 to V28)": "In this dataset the original features were transformed and anonymised for privacy, so they have no business names.",
    }
    for term, text in gloss.items():
        with st.expander(term):
            st.write(text)
