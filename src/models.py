"""
Models Module for Financial Distress Prediction.
Defines the dictionary of classical ML models, trains them, and runs cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate

# Define models dict with fixed random states
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

def train_all_models(X_train, y_train):
    """
    Trains all models in the MODELS dictionary on training data.
    Returns a dictionary of fitted model objects.
    """
    fitted_models = {}
    for name, model in MODELS.items():
        # Create a new clone/instance to avoid modifications to global dict
        from sklearn.base import clone
        clf = clone(model)
        clf.fit(X_train, y_train)
        fitted_models[name] = clf
    return fitted_models

def cross_validate_all(models, X, y, cv=5):
    """
    Runs StratifiedKFold cross-validation on each model in the models dictionary.
    Returns a DataFrame containing the mean and std of:
    accuracy, precision, recall, f1, and roc_auc.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = []
    
    for name, model in models.items():
        # Perform cross-validation
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        
        row = {'Model': name}
        for metric in scoring:
            mean_score = np.mean(scores[f'test_{metric}'])
            std_score = np.std(scores[f'test_{metric}'])
            row[f'{metric}_mean'] = mean_score
            row[f'{metric}_std'] = std_score
            
        cv_results.append(row)
        
    return pd.DataFrame(cv_results)
