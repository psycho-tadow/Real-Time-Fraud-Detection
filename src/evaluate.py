from sklearn.metrics import average_precision_score,precision_recall_curve

def pr_auc(y_true,scores):
    return average_precision_score(y_true,scores)

def recall_at_precision(y_true,scores,min_precision=0.90):
    precision,recall,_=precision_recall_curve(y_true,scores)
    ok=precision>=min_precision
    return recall[ok].max() if ok.any() else 0.0