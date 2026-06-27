"""
Visualizer Module for Financial Distress Prediction.
Contains reusable matplotlib and seaborn plotting functions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

# Apply professional seaborn settings
sns.set_theme(style="whitegrid")

def plot_roc_curves(models, X_test, y_test, save_path=None):
    """
    Plots ROC curves for all models in the dictionary.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(cm, labels, title, save_path=None):
    """
    Plots a confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_feature_importances(importances, feature_names, top_n=20, save_path=None):
    """
    Plots feature importances as a horizontal bar chart.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = np.array(feature_names)[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_names, palette='viridis', hue=top_names, legend=False)
    plt.xlabel('Relative Importance Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_cluster_pca(X, cluster_labels, cluster_names, save_path=None):
    """
    Applies PCA (n_components=2) and plots a 2D scatter plot colored by cluster.
    Annotates the cluster centroids with their names.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = cluster_labels
    df_pca['ClusterName'] = df_pca['Cluster'].map(cluster_names)
    
    plt.figure(figsize=(10, 8))
    
    unique_names = list(cluster_names.values())
    colors = sns.color_palette("Set1", n_colors=len(unique_names))
    
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='ClusterName', data=df_pca, 
        palette=dict(zip(unique_names, colors)), alpha=0.7, s=50
    )
    
    # Compute and annotate centroids
    for cid, name in cluster_names.items():
        cluster_points = df_pca[df_pca['Cluster'] == cid]
        if len(cluster_points) > 0:
            centroid_x = cluster_points['PCA1'].mean()
            centroid_y = cluster_points['PCA2'].mean()
            
            plt.scatter(
                centroid_x, centroid_y, color='black', marker='X', 
                s=200, edgecolor='white', linewidth=1.5, zorder=10
            )
            plt.annotate(
                name, (centroid_x, centroid_y),
                textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8, ec="black"),
                zorder=11
            )
            
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)', fontsize=12)
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)', fontsize=12)
    plt.title('K-Means Risk Archetypes (PCA Projection)', fontsize=14, fontweight='bold', pad=15)
    plt.legend(title='Risk Archetypes', loc='best', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_sector_robustness(sector_results_df, save_path=None):
    """
    Plots a grouped bar chart of F1, ROC-AUC, Precision, and Recall per sector.
    """
    melted_df = sector_results_df.melt(
        id_vars=['Sector'], 
        value_vars=['Precision', 'Recall', 'F1', 'ROC-AUC'],
        var_name='Metric', value_name='Score'
    )
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Sector', y='Score', hue='Metric', data=melted_df, palette='muted')
    
    plt.ylim([0, 1.05])
    plt.xlabel('Sectors / Industries', fontsize=12, fontweight='bold')
    plt.ylabel('Score Value', fontsize=12, fontweight='bold')
    plt.title('Model Robustness Across Sectors', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
