"""
Data Loading Module for Financial Distress Prediction.

Download data from: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
Place year1.arff through year5.arff in data/raw/
"""

import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split

def load_arff(filepath):
    """
    Reads a .arff file using scipy.io.arff and returns a pandas DataFrame.
    Decodes byte strings to unicode strings and converts the class column to integer.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Decode byte string columns to utf-8 strings
    for col in df.columns:
        if df[col].dtype == object:
            # Check if values are bytes
            first_val = df[col].iloc[0] if len(df[col]) > 0 else None
            if isinstance(first_val, bytes):
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                
    # If target is 'class', ensure it is converted to integer
    if 'class' in df.columns:
        df['class'] = df['class'].astype(int)
        
    return df

def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generates a synthetic Polish Companies Bankruptcy dataset matching the real schema.
    - Features Attr1 to Attr64 (with ~5% missing values)
    - Target column 'class' containing integers 0 and 1
    - Class imbalance ratio of ~97:3 (97% healthy, 3% distressed)
    - 'year' column (1-5)
    - Correlates Attr2 (ROA), Attr5 (Liquidity), and Attr6 (Debt/Equity) with target for realistic modeling
    """
    rng = np.random.RandomState(random_state)
    
    # Generate class labels with 97:3 ratio
    y = rng.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    
    data = {}
    # Generate 64 features
    for i in range(1, 65):
        col_name = f"Attr{i}"
        if i == 2:  # Attr2: ROA
            vals = np.where(y == 0, rng.normal(0.06, 0.04, size=n_samples), rng.normal(-0.18, 0.12, size=n_samples))
        elif i == 5:  # Attr5: Liquidity ratio
            vals = np.where(y == 0, rng.normal(0.6, 0.3, size=n_samples), rng.normal(-0.3, 0.5, size=n_samples))
        elif i == 6:  # Attr6: Debt/Equity
            vals = np.where(y == 0, rng.normal(0.35, 0.25, size=n_samples), rng.normal(1.8, 1.2, size=n_samples))
        else:
            mean = rng.uniform(-0.5, 0.5)
            scale = rng.uniform(0.1, 2.0)
            vals = rng.normal(mean, scale, size=n_samples)
            
        # Introduce 5% missing values (NaN) to simulate real missing data
        mask = rng.rand(n_samples) < 0.05
        vals[mask] = np.nan
        data[col_name] = vals
        
    data['class'] = y
    data['year'] = rng.choice([1, 2, 3, 4, 5], size=n_samples)
    
    df = pd.DataFrame(data)
    return df

def load_all_years(data_dir):
    """
    Loads all 5 Polish bankruptcy ARFF files (year1.arff to year5.arff) from data_dir,
    adds a 'year' column (1-5), and concatenates them.
    
    If files do not exist, falls back to generating a synthetic dataset.
    
    Download data from: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
    Place year1.arff through year5.arff in data/raw/
    """
    dfs = []
    missing_files = False
    
    # Try loading all 5 years
    for y in range(1, 6):
        filename = f"year{y}.arff"
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files = True
            break
        try:
            df = load_arff(filepath)
            df['year'] = y
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            missing_files = True
            break
            
    if missing_files:
        print("Raw ARFF files not found in data/raw/. "
              "Generating synthetic Polish bankruptcy dataset for demonstration.")
        return generate_synthetic_data()
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def get_train_test_split(df, test_size=0.2, random_state=42):
    """
    Splits the combined dataframe into train and test sets.
    Renames target column "class" to "distress".
    Returns X_train, X_test, y_train, y_test.
    """
    df_clean = df.copy()
    if 'class' in df_clean.columns:
        df_clean = df_clean.rename(columns={'class': 'distress'})
        
    target_col = 'distress'
    # 'year' should not be used as an input feature for ML models
    feature_cols = [col for col in df_clean.columns if col not in [target_col, 'year']]
    
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
