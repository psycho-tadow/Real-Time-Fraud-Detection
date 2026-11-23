# Real-Time Fraud Detection

Detects fraudulent credit card transactions from highly imbalanced data. A Streamlit app will serve the final model.

**Status:** Day 2 of 7 done (EDA, time-based split, baseline).

## Data
Kaggle Credit Card Fraud Detection (ULB): 283,726 transactions after removing duplicates, 473 frauds (about 0.17%).

## What's done
- **EDA:** checked missing values and duplicates, class imbalance, `Amount` and `Time`.
- **Time-based split (70/15/15):** train on the past, validate and test on the future.

| Split | Rows | Frauds |
| Train | 198,608 | 366 |
| Validation | 42,559 | 55 |
| Test | 42,559 | 52 |

- **Baseline:** logistic regression (class-weighted, scaler inside a pipeline).

| Model | PR-AUC (valid) | Recall @ 90% precision |
| Random guess | 0.0013 | n/a |
| Logistic regression | 0.8372 | 0.7636 |

## Limitation
Validation and test have only about 50 frauds each, so scores are noisy.

## Setup
```bash
python -m venv .venv --prompt fraud-detection
source .venv/Scripts/activate
pip install -r requirements.txt
```
Put `creditcard.csv` in `data/`.

## Next
XGBoost/LightGBM, cost-based threshold, SHAP, Streamlit app.




all models land at about 0.84-0.85 PR-AUC, and the differences are within noise. Class weighting clearly helped, and no model clearly wins. XGBoost is the final model, because it's slightly ahead, handles imbalance natively, and works well with SHAP.

