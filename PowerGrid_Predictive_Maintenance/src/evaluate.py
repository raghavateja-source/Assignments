import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_predictions(y_true, probability, threshold=.5):
    pred = (probability >= threshold).astype(int)
    return {
        'Precision': precision_score(y_true, pred),
        'Recall': recall_score(y_true, pred),
        'F1': f1_score(y_true, pred),
        'ROC-AUC': roc_auc_score(y_true, probability),
        'PR-AUC': average_precision_score(y_true, probability),
    }
