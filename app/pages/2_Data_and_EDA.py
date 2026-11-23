import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from utils import setup, hero, show_figure
from site_config import DATA, SPLITS

setup("Data & EDA", "📊")
hero("Data & exploratory analysis",
     "What the data contains, how imbalanced it is, and what to watch out for.")

st.markdown(f"""
### The dataset
The **Credit Card Fraud Detection** dataset (ULB Machine Learning Group) holds anonymised
European card transactions covering about two days. After removing duplicate rows it has
**{DATA['rows']:,} transactions**, of which **{DATA['frauds']} are fraud ({DATA['fraud_rate']})**.
""")

st.table(pd.DataFrame({
    "Column": ["Time", "V1 to V28", "Amount", "Class"],
    "Meaning": ["Seconds since the first transaction",
                "Anonymised features (PCA components), so no business names",
                "Transaction amount",
                "Label: 0 = legitimate, 1 = fraud"],
}))

st.markdown("### Class imbalance")
show_figure("01_class_balance.png", "Fraud is a sliver of all transactions.")

st.markdown("### Transaction amount")
show_figure("02_amount.png", "Amounts are heavily right-skewed, so a log scale is easier to read.")

st.markdown("### Time of day")
show_figure("03_hour.png", "Hours with few transactions give noisy fraud rates, so they are not over-interpreted.")

st.markdown("### Why the split is by time")
st.table(pd.DataFrame(SPLITS, columns=["Split", "Rows", "Frauds", "Fraud rate"]))
st.markdown("""
Data is sorted by time: the model learns from the earliest 70%, is tuned on the next 15%
and is judged once on the final 15%. This mirrors real use, where tomorrow's
transactions are scored by a model trained on yesterday's.
""")
