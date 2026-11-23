import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from utils import setup, hero, show_figure
from site_config import TEST, VALID

setup("Results", "🏆")
hero("Results", "Final numbers, what drives the model, and where it falls short.")

st.markdown("### Final test-set result")
if TEST["pr_auc"] is None:
    st.info("The held-out test set has not been scored yet. Run the final test cell once "
            "and fill the TEST numbers in `app/site_config.py`. Validation results are "
            "shown on the Method page.")
else:
    a, b, c, d = st.columns(4)
    a.metric("PR-AUC", f"{TEST['pr_auc']:.3f}")
    b.metric("Recall", f"{TEST['recall']:.3f}")
    c.metric("Precision", f"{TEST['precision']:.3f}")
    d.metric("Cost saved", f"{1 - TEST['cost'] / TEST['cost_nothing']:.1%}")
    st.table(pd.DataFrame(
        [[TEST["tn"], TEST["fp"]], [TEST["fn"], TEST["tp"]]],
        index=["Actually legit", "Actually fraud"],
        columns=["Predicted legit", "Predicted fraud"]))
    st.caption(f"Total cost {TEST['cost']:.2f} versus {TEST['cost_nothing']:.2f} "
               "if no transaction were flagged.")

st.markdown("### What drives the model")
show_figure("shap_summary.png",
            "Each dot is a transaction. Right of zero pushes toward fraud; red means a high feature value.")
st.markdown("""
V4 and V11 push toward fraud when their values are **high**; V14, V12 and V10 push toward
fraud when their values are **low**. A few features carry most of the signal. Since the
features are anonymised, the model's reasoning cannot be translated into business terms.
""")

st.markdown("### Limitations, honestly")
st.markdown(f"""
- **Small evaluation sets:** validation and test contain only {55} and {52} frauds, so every
  metric has wide uncertainty and small differences between models are not meaningful.
- **Two days of data:** seasonality and long-term drift cannot be studied.
- **Anonymised features:** limited feature engineering and no business-level explanations.
- **Assumed costs:** the threshold depends on assumed costs and should be re-tuned with real ones.
- **Not production infrastructure:** no streaming pipeline, customer-history features or monitoring.
""")


st.markdown("""
**Validation vs test:** scores are lower on the test set (PR-AUC 0.85 on validation, 0.76 on test).
Validation was used to choose the model and the threshold, so it is slightly optimistic, and the
test set is the cleaner estimate. With only 52 frauds, recall carries roughly +/-0.12 of uncertainty.
The cost saving is also sensitive to a few large-amount missed frauds.
""")