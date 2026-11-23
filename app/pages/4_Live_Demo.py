import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from utils import setup, hero, load_model, load_sample, score, explain_row

setup("Live Demo", "🔍")
hero("Live demo",
     "Pick a transaction, change its features, and watch the model react.")

model, cfg, _ = load_model()
sample = load_sample()
features, threshold = cfg["features"], cfg["threshold"]

tab1, tab2 = st.tabs(["Try a transaction", "Score a CSV file"])

with tab1:
    kind = st.radio("Show", ["All", "Actual fraud", "Actual legit"], horizontal=True)
    pool = sample if kind == "All" else sample[sample["Class"] == (1 if kind == "Actual fraud" else 0)]
    idx = st.selectbox(
        "Transaction", pool.index.tolist(),
        format_func=lambda i: f"#{i}  |  amount {sample.loc[i, 'Amount']:.2f}")
    row = sample.loc[[idx], features].copy()
    actual = "Fraud" if sample.loc[idx, "Class"] == 1 else "Legit"

    with st.expander("What-if: change the most influential features"):
        for f in ["V14", "V4", "V12", "V10", "Amount"]:
            if f in features:
                lo, hi = float(sample[f].min()), float(sample[f].max())
                row[f] = st.slider(f, lo, hi, float(row[f].iloc[0]), key=f"{f}_{idx}")

    s = float(score(row)[0])
    flagged = s >= threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud score", f"{s:.3f}")
    with c2:
        cls = "fraud" if flagged else "legit"
        st.markdown(f'<div class="verdict {cls}">{"FRAUD" if flagged else "LEGIT"}</div>',
                    unsafe_allow_html=True)
    c3.metric("Actual label (dataset)", actual)
    st.progress(min(max(s, 0.0), 1.0))
    st.markdown(f'<p class="note">Flagged when the score is at least {threshold} '
                f'(threshold chosen from cost).</p>', unsafe_allow_html=True)

    st.subheader("Why this score? Top reasons")
    reasons = explain_row(row)
    st.bar_chart(reasons)
    up = [f for f, v in reasons.items() if v > 0][:3]
    down = [f for f, v in reasons.items() if v < 0][:3]
    if up:
        st.write("Pushing toward **fraud**: " + ", ".join(up))
    if down:
        st.write("Pushing toward **legit**: " + ", ".join(down))
    st.caption("Features V1 to V28 are anonymised, so they have no business meaning. "
               "Bars are SHAP values: they explain what the model relies on, not what causes fraud.")

with tab2:
    st.write("Upload a CSV with the same columns as the training features "
             "(V1 to V28 and Amount). Download the example to see the format.")
    example = sample.drop(columns=["Class"]).head(50)
    st.download_button("Download example CSV", example.to_csv(index=False), "example.csv")
    up_file = st.file_uploader("Upload CSV", type="csv")
    if up_file is not None:
        import pandas as pd
        df = pd.read_csv(up_file)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing[:8])}")
        else:
            out = df.copy()
            out["fraud_score"] = score(df)
            out["flagged"] = out["fraud_score"] >= threshold
            a, b = st.columns(2)
            a.metric("Transactions scored", f"{len(out):,}")
            b.metric("Flagged as fraud", f"{int(out['flagged'].sum()):,}")
            flagged_df = out[out["flagged"]].sort_values("fraud_score", ascending=False)
            st.dataframe(out.head(100))
            st.download_button("Download flagged transactions",
                               flagged_df.to_csv(index=False), "flagged.csv")
