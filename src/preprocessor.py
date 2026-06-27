"""
Preprocessing Module for Financial Distress Prediction.
Chains median imputation and standard scaling, and handles class imbalance using SMOTE.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def get_pipeline():
    """
    Constructs and returns an sklearn Pipeline chaining:
    1. SimpleImputer with median strategy.
    2. StandardScaler.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline

def fit_transform_train(pipeline, X_train):
    """
    Fits the preprocessing pipeline on training data and transforms it.
    Returns a pandas DataFrame with original column names and index.
    """
    X_transformed = pipeline.fit_transform(X_train)
    return pd.DataFrame(X_transformed, columns=X_train.columns, index=X_train.index)

def transform_test(pipeline, X_test):
    """
    Transforms the test data using a fitted preprocessing pipeline.
    Returns a pandas DataFrame with original column names and index.
    """
    X_transformed = pipeline.transform(X_test)
    return pd.DataFrame(X_transformed, columns=X_test.columns, index=X_test.index)

def apply_smote(X_train, y_train):
    """
    Applies SMOTE to training features and labels to handle class imbalance.
    Returns resampled X and y as a pandas DataFrame and Series respectively.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    if isinstance(X_train, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if isinstance(y_train, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)
        
    return X_resampled, y_resampled
