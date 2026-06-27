"""
Main execution script for Financial Distress Prediction.
Runs the full pipeline end-to-end including loading, preprocessing, modeling, evaluation, and clustering.
"""

import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans

from src.data_loader import load_all_years, get_train_test_split
from src.preprocessor import get_pipeline, fit_transform_train, transform_test, apply_smote
from src.models import train_all_models
from src.evaluator import evaluate_model, compare_models

def run_pipeline(data_dir):
    print("=" * 60)
    print("STARTING FINANCIAL DISTRESS PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print(f"\n[Step 1] Loading data from directory: {data_dir}")
    df = load_all_years(data_dir)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Train-Test Split
    print("\n[Step 2] Splitting data into train and test sets (Stratified, 80/20)...")
    X_train, X_test, y_train, y_test = get_train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train class ratio (healthy/distressed): {len(y_train[y_train==0])}/{len(y_train[y_train==1])}")
    
    # 3. Preprocessing
    print("\n[Step 3] Building and fitting preprocessing pipeline...")
    pipeline = get_pipeline()
    X_train_scaled = fit_transform_train(pipeline, X_train)
    X_test_scaled = transform_test(pipeline, X_test)
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE on training data to balance classes...")
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    print(f"Resampled training set shape: {X_train_res.shape}")
    print(f"Resampled class ratio: {len(y_train_res[y_train_res==0])}/{len(y_train_res[y_train_res==1])}")
    
    # 4. Modeling
    print("\n[Step 4] Training all classical ML models on balanced training set...")
    fitted_models = train_all_models(X_train_res, y_train_res)
    
    # 5. Evaluation & Comparison
    print("\n[Step 5] Evaluating models on scaled test set...")
    results = {}
    for name, model in fitted_models.items():
        results[name] = evaluate_model(model, X_test_scaled, y_test)
        
    df_compare = compare_models(results)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (Sorted by F1 and ROC-AUC)")
    print("=" * 60)
    print(df_compare.to_string(index=False))
    print("=" * 60)
    
    # Print best model
    best_model_name = df_compare.iloc[0]['Model']
    best_model_f1 = df_compare.iloc[0]['F1']
    print(f"\nBest model by F1: {best_model_name} with F1={best_model_f1:.4f}")
    
    # 6. Clustering (Risk Archetypes)
    print("\n[Step 6] Running KMeans clustering (k=4) on full preprocessed dataset...")
    # Preprocess the entire feature space (excluding year and target distress)
    df_clean = df.copy()
    if 'class' in df_clean.columns:
        df_clean = df_clean.rename(columns={'class': 'distress'})
    
    y_full = df_clean['distress']
    X_full = df_clean.drop(columns=['distress', 'year'], errors='ignore')
    
    # Preprocess full feature set
    pipeline_full = get_pipeline()
    X_full_scaled = fit_transform_train(pipeline_full, X_full)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_full_scaled)
    
    # Compute characteristics of each cluster on raw/unscaled features for interpretability
    cluster_metrics = {}
    for cid in range(4):
        mask = (cluster_labels == cid)
        cluster_metrics[cid] = {
            'distress_rate': y_full.iloc[mask].mean(),
            'debt_equity': X_full['Attr6'].iloc[mask].mean(),
            'roa': X_full['Attr2'].iloc[mask].mean(),
            'liquidity': X_full['Attr5'].iloc[mask].mean(),
        }
        
    # Programmatic cluster naming:
    # 1. Highest distress_rate -> "Distressed"
    remaining_clusters = list(range(4))
    distressed_cid = max(remaining_clusters, key=lambda c: cluster_metrics[c]['distress_rate'])
    cluster_names = {distressed_cid: "Distressed"}
    remaining_clusters.remove(distressed_cid)

    # 2. Remaining with highest debt-to-equity (Attr6) -> "Highly Leveraged"
    highly_leveraged_cid = max(remaining_clusters, key=lambda c: cluster_metrics[c]['debt_equity'])
    cluster_names[highly_leveraged_cid] = "Highly Leveraged"
    remaining_clusters.remove(highly_leveraged_cid)

    # 3. Remaining with lowest liquidity (Attr5) -> "Liquidity Stressed"
    liquidity_stressed_cid = min(remaining_clusters, key=lambda c: cluster_metrics[c]['liquidity'])
    cluster_names[liquidity_stressed_cid] = "Liquidity Stressed"
    remaining_clusters.remove(liquidity_stressed_cid)

    # 4. Remaining -> "Stable Growth"
    stable_growth_cid = remaining_clusters[0]
    cluster_names[stable_growth_cid] = "Stable Growth"
    
    # Build cluster summary table
    cluster_summary_data = []
    for cid in range(4):
        metrics = cluster_metrics[cid]
        name = cluster_names[cid]
        cluster_summary_data.append({
            'Cluster ID': cid,
            'Archetype Name': name,
            'Distress Rate': f"{metrics['distress_rate']*100:.2f}%",
            'Mean Debt/Equity (Attr6)': f"{metrics['debt_equity']:.4f}",
            'Mean ROA (Attr2)': f"{metrics['roa']:.4f}",
            'Mean Liquidity (Attr5)': f"{metrics['liquidity']:.4f}"
        })
        
    df_cluster_summary = pd.DataFrame(cluster_summary_data)
    df_cluster_summary = df_cluster_summary.sort_values(by='Cluster ID').reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print("K-MEANS RISK ARCHETYPE CLUSTERING SUMMARY")
    print("=" * 60)
    print(df_cluster_summary.to_string(index=False))
    print("=" * 60)
    
    print("\nPipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(base_dir, "data", "raw")
    
    parser = argparse.ArgumentParser(description="Financial Distress Prediction Pipeline")
    parser.add_argument("--data_dir", type=str, default=default_data_dir,
                        help="Path to the directory containing ARFF files")
    args = parser.parse_args()
    
    run_pipeline(args.data_dir)
