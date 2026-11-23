"""Edit the numbers and links on this page, and the whole site updates.

All METRICS below were measured in your notebooks. The TEST block stays empty
until you run the final test-set cell once (Day 7) and copy the numbers here.
"""

GITHUB_URL = "https://github.com/psycho-tadow/Real-Time-Fraud-Detection"   
AUTHOR = "Ansh Tomar"                                        
LINKEDIN_URL = "https://www.linkedin.com/in/ansh-tomar/"                                        

# Dataset after dropping duplicate rows (check with len(df) and df["Class"].sum())
DATA = {"rows": 283726, "frauds": 473, "fraud_rate": "0.17%"}

SPLITS = [  # name, rows, frauds, fraud rate
    ("Train", 198608, 366, "0.18%"),
    ("Validation", 42559, 55, "0.13%"),
    ("Test", 42559, 52, "0.12%"),
]

MODELS = [  # name, PR-AUC (validation), recall @ 90% precision, note
    ("Logistic regression (no weights)", 0.7789, 0.4909, "Ignores the rare class"),
    ("Logistic regression (balanced)", 0.8372, 0.7636, "Baseline"),
    ("Logistic regression + SMOTE", 0.8408, 0.7636, "Same as baseline within noise"),
    ("XGBoost", 0.8521, 0.7818, "Chosen model"),
    ("LightGBM (regularized)", 0.8360, 0.7636, "Default settings overfit badly (PR-AUC 0.07)"),
]

THRESHOLDS = [  # threshold, total cost, transactions flagged, recall
    (0.30, 511.55, 118, 0.873),
    (0.40, 371.55, 90, 0.873),
    (0.45, 331.55, 82, 0.873),
    (0.50, 296.55, 75, 0.873),
    (0.54, 278.77, 70, 0.855),
    (0.60, 502.80, 64, 0.800),
]
CHOSEN_THRESHOLD = 0.5
FALSE_ALARM_COST = 5

# Measured on the validation set at threshold 0.5 (XGBoost)
VALID = dict(pr_auc=0.8521, recall_at_90p=0.7818, flagged=75, precision=0.64,
             recall=0.873, cost=296.55, cost_nothing=8396.11,
             tp=48, fp=27, fn=7, tn=42477)

# Fill these in AFTER running the final test cell once. Leave None until then.
TEST = dict(pr_auc=None, recall_at_90p=None, flagged=None, precision=None,
            recall=None, cost=None, cost_nothing=None,
            tp=None, fp=None, fn=None, tn=None)


def active_results():
    """Return (numbers, label). Uses TEST once filled, otherwise VALID."""
    if TEST.get("pr_auc") is not None:
        return TEST, "test set"
    return VALID, "validation set"

TEST = dict(pr_auc=0.7571, recall_at_90p=0.75, flagged=66, precision=0.591,
            recall=0.75, cost=2507.4, cost_nothing=6168.88,
            tp=39, fp=27, fn=13, tn=42480)