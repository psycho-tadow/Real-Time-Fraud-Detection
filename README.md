# Real-Time Fraud Detection System

An end-to-end machine learning project that detects fraudulent credit card transactions in highly imbalanced data. It uses time-aware validation, a cost-based decision threshold and SHAP explanations, and is served through a free, public Streamlit website.

**Live demo:** [https://real-time-fraud-detector.streamlit.app/]  ·  **Author:** [Ansh Tomar]

## Highlights

- **Honest evaluation:** time-based train/validation/test split, with the test set used once at the end.
- **Right metrics for rare events:** PR-AUC and recall at fixed precision, not accuracy.
- **Business-aware decisions:** the fraud threshold is chosen by minimizing cost (a missed fraud costs its amount, a false alarm costs a fixed 5), not by defaulting to 0.5.
- **Explainable:** every score comes with its top SHAP reasons.
- **Deployed:** a multi-page website with a learning guide, method write-up, live scoring demo and CSV upload. Hosting is free.

## Results

Final model: **XGBoost**, threshold **0.5**. Scored once on the held-out test set (the last 15% of transactions in time order).

| Metric (test set) | Value |
|---|---|
| PR-AUC | 0.757 |
| Frauds caught | 39 of 52 (recall 0.75) |
| Precision | 0.59 (66 transactions flagged) |
| False alarms | 27 |
| Total cost | 2,507 vs 6,169 if nothing is flagged (59.4% lower) |

For comparison, random guessing scores a PR-AUC of about 0.001. Validation scores were higher (PR-AUC 0.852, 48 of 55 frauds caught). The model and threshold were chosen on validation, so validation is slightly optimistic and the test result is the fair estimate.

### Models compared (validation set)

| Model | PR-AUC | Recall @ 90% precision |
|---|---|---|
| Logistic regression, no class weights | 0.779 | 0.491 |
| Logistic regression, balanced | 0.837 | 0.764 |
| Logistic regression + SMOTE | 0.841 | 0.764 |
| **XGBoost** | **0.852** | **0.782** |
| LightGBM (regularized) | 0.836 | 0.764 |

Handling the class imbalance clearly helps. The differences between the better models are within noise, because validation has only 55 frauds. XGBoost was chosen because it is slightly ahead, handles imbalance natively and works well with SHAP.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from the ULB Machine Learning Group: anonymised European card transactions covering two days. After removing duplicate rows it contains 283,726 transactions, of which 473 are fraud (about 0.17%).

- `Time`: seconds since the first transaction
- `V1`-`V28`: anonymised features (PCA components, so they have no business names)
- `Amount`: transaction amount
- `Class`: 0 = legitimate, 1 = fraud

The full CSV is not stored in this repository. A 355-row sample (`data/sample.csv`) powers the demo.

## Method

1. **EDA** (`notebooks/01_EDA.ipynb`): missing values, duplicates, class imbalance, `Amount` and time patterns.
2. **Leakage control:** duplicates removed before splitting; scaling and SMOTE live inside a pipeline so they are fitted on training data only; `Time` is not used as a feature.
3. **Time-based split (70/15/15):** train on the past, validate and test on the future.

   | Split | Rows | Frauds |
   |---|---|---|
   | Train | 198,608 | 366 |
   | Validation | 42,559 | 55 |
   | Test | 42,559 | 52 |

4. **Modelling** (`notebooks/02_Modelling.ipynb`): a logistic regression baseline, then imbalance strategies (class weights, SMOTE) and gradient-boosted trees.
5. **Cost-based threshold:** total cost was computed for every threshold on validation. The exact minimum (0.54) sat next to a cost cliff, so **0.5** was chosen instead: it catches the same frauds, costs about 6% more than the minimum, and is more robust if new data shifts.
6. **Explainability:** SHAP summary plots and per-transaction reasons.
7. **Deployment:** the model and threshold are saved and served by the Streamlit app.

## Limitations

- Validation and test contain only 55 and 52 frauds, so every metric is uncertain (recall on the test set is roughly 0.62 to 0.85). Small differences between models should not be over-interpreted.
- The cost saving is sensitive to a few large-amount frauds, and it fell from 96% on validation to 59% on test.
- The data covers two days, so seasonality and long-term drift cannot be studied.
- Features are anonymised, so there is little feature engineering and no business-level explanation.
- The costs are assumptions and should be replaced with real business figures.
- This is a model and an app, not production infrastructure: there is no streaming pipeline, customer-history features or drift monitoring.

## Run it yourself

```bash
git clone https://github.com/psycho-tadow/Real-Time-Fraud-Detection.git
cd Real-Time-Fraud-Detection
python -m venv .venv
source .venv/Scripts/activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

To rerun the notebooks, install `requirements-dev.txt` instead and place `creditcard.csv` in `data/`. Run the tests with `python -m pytest`.

## Project structure

```
app/          Streamlit website (Home.py + pages/)
data/         sample.csv for the demo (full dataset not included)
models/       trained XGBoost model and chosen threshold
notebooks/    01_EDA.ipynb, 02_Modelling.ipynb
reports/      figures used by the website
src/          data loading and time split, metrics, cost threshold
tests/        unit tests
.streamlit/   site theme
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, Streamlit, pytest, Git.

## Future work

- Monitor score drift and retrain on a schedule with experiment tracking.
- Add customer-history features such as spending velocity.
- Serve the model behind an API with latency monitoring.
- Validate the method on a second dataset.

## Contact

[Ansh Tomar] · [https://www.linkedin.com/in/ansh-tomar/]