import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from utils import setup, hero, show_figure
from site_config import MODELS, THRESHOLDS, CHOSEN_THRESHOLD, FALSE_ALARM_COST

setup("Method", "🧪")
hero("Method: how the model was built",
     "Every decision, and the reason for it.")

st.markdown("### 1. Avoiding leakage")
st.markdown("""
- **Duplicates removed before splitting**, so identical rows cannot appear in both train and test.
- **Time-based split**, so the future never leaks into training.
- **Scaling and SMOTE live inside a pipeline**, so they are fitted on training data only.
- **`Time` is not used as a feature**, because validation and test values lie outside the training range.
- **The test set was used once**, at the very end.
""")

st.markdown("### 2. Models compared (validation set)")
df = pd.DataFrame(MODELS, columns=["Model", "PR-AUC", "Recall @ 90% precision", "Note"])
st.table(df.style.format({"PR-AUC": "{:.4f}", "Recall @ 90% precision": "{:.4f}"}))
st.markdown("""
**What this shows:** handling the imbalance clearly helps (no weights is much worse), but
the gaps between the better models are within noise, because validation contains only
55 frauds, and one fraud moves recall by about 1.8 points. **XGBoost** was chosen as the
final model: it is slightly ahead, handles imbalance natively and pairs well with SHAP.
It is not claimed to be dramatically better than the simple baseline.
""")

st.markdown("### 3. Choosing the decision threshold from cost")
st.markdown(f"""
Assumptions: a **missed fraud costs its transaction amount**, a **false alarm costs
{FALSE_ALARM_COST}**. Total cost was computed for every threshold on the validation set.
""")
th = pd.DataFrame(THRESHOLDS, columns=["Threshold", "Total cost", "Flagged", "Recall"])
st.table(th.style.format({"Threshold": "{:.2f}", "Total cost": "{:.2f}", "Recall": "{:.3f}"}))
show_figure("cost_curve.png", "Cost versus threshold on the validation set.")
st.markdown(f"""
**Chosen threshold: {CHOSEN_THRESHOLD}.** The exact minimum (0.54) sits right next to a cost
cliff: just above it, the model starts missing several larger-amount frauds and cost jumps.
{CHOSEN_THRESHOLD} catches the same frauds as lower thresholds, costs about 6% more than the
minimum and is far safer if new data shifts slightly. The costs are assumptions, so the
chosen threshold should be revisited with real business figures.
""")
