"""
Evaluator Module for Financial Distress Prediction.
Computes performance metrics and builds summary comparison tables.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a single model on the test data and returns a dictionary of metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
    - confusion_matrix
    - classification_report (as string)
    """
    y_pred = model.predict(X_test)
    
    # Calculate probability scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
        
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }
    
    return metrics

def compare_models(results_dict):
    """
    Compares all evaluated models and returns a DataFrame sorted by F1 and ROC-AUC.
    - results_dict: Dictionary with model name as key and its evaluation metrics dict as value.
    """
    comparison_data = []
    for name, metrics in results_dict.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })
    
    df_compare = pd.DataFrame(comparison_data)
    df_compare = df_compare.sort_values(by=['F1', 'ROC-AUC'], ascending=False).reset_index(drop=True)
    return df_compare
