# Financial Distress Prediction

## Problem Statement
Predicting corporate bankruptcy and financial distress is a critical task for lenders, investors, and regulators. Early detection allows stakeholders to take preventive actions, restructure debt, or adjust investment portfolios to mitigate risk. This project implements a machine learning system to predict financial distress in companies using classical classification models, handles class imbalance, and profiles companies using unsupervised clustering to establish clear risk archetypes.

## Dataset
This project uses the **Polish Companies Bankruptcy Dataset** from the UCI Machine Learning Repository. It contains 5 years of historical financial data for Polish companies.
- **Attributes**: 64 financial ratios representing liquidity, leverage, profitability, and activity.
- **Target**: A binary class indicator where `1` represents bankruptcy/distress and `0` represents a healthy firm.
- **Imbalance**: The dataset is highly imbalanced, with approximately 97% healthy firms and 3% distressed cases.

Download the data from: [UCI ML Repository - Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data).
Place the raw `.arff` files (`year1.arff` through `year5.arff`) in the `data/raw/` directory.

## Project Structure
```
financial-distress-prediction/
├── data/
│   ├── raw/                  # Downloaded .arff files
│   └── processed/            # Preprocessed and resampled datasets
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Imputation, scaling, and SMOTE
│   ├── 03_model_comparison.ipynb # Baseline model comparison
│   ├── 04_model_selection.ipynb  # Hyperparameter tuning (GridSearchCV)
│   ├── 05_robustness_testing.ipynb # Generalization across sectors
│   └── 06_clustering.ipynb   # Risk archetype profiling (K-Means)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Loading, concatenation, and synthetic generator
│   ├── preprocessor.py       # Pipeline construction and SMOTE application
│   ├── models.py             # Model training and cross-validation
│   ├── evaluator.py          # Model evaluation and comparison tables
│   └── visualizer.py         # Reusable plotting functions
├── reports/
│   └── figures/              # Saved plots and heatmaps
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore patterns
└── main.py                   # End-to-end executable pipeline
```

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/username/financial-distress-prediction.git
cd financial-distress-prediction
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download data
Download `polish+companies+bankruptcy+data.zip` from the UCI repository, extract the 5 annual ARFF files, and place them inside the `data/raw/` folder:
- `data/raw/year1.arff`
- `data/raw/year2.arff`
- `data/raw/year3.arff`
- `data/raw/year4.arff`
- `data/raw/year5.arff`

*Note: If these files are not present in `data/raw/`, the data loader will automatically generate synthetic data matching the schema so you can run the pipeline end-to-end immediately.*

### 4. Run notebooks in order
Run the Jupyter notebooks in numerical sequence:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5. Or run the full pipeline
To execute the entire loading, preprocessing, training, evaluation, and clustering pipeline in one command:
```bash
python3 main.py
```

## Methodology

### 1. Exploratory Data Analysis
- Analyzed the distribution of the target variable, confirming a severe class imbalance (~97% healthy vs. ~3% bankrupt).
- Evaluated missing value counts and generated a correlation heatmap showing significant multicollinearity between financial ratios.
- Plotted distribution histograms for the top financial features, identifying strong skewness and outliers.

### 2. Preprocessing & Class Imbalance
- Handled missing values using median imputation (`SimpleImputer`) to minimize outlier bias.
- Scaled features using `StandardScaler` to ensure uniform distance computations for SVM, Logistic Regression, and K-Means.
- Resolved class imbalance on the training dataset using Synthetic Minority Over-sampling Technique (SMOTE).

### 3. Model Comparison
- Evaluated six classical ML algorithms: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, and Support Vector Machine (SVM).
- Computed 5-fold stratified cross-validation scores (accuracy, precision, recall, F1, ROC-AUC) on the balanced training data.

### 4. Model Selection & Tuning
- Selected Random Forest as the primary classifier.
- Tuned hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `class_weight`) using `GridSearchCV` optimized for F1-score.
- Saved feature importances and generated learning curves to detect overfitting.

### 5. Robustness Testing by Sector
- Simulated company sectors (Manufacturing, Services, Retail, Construction) to evaluate model generalization.
- Analyzed model precision, recall, and F1-score variance across sectors.

### 6. Risk Archetype Clustering
- Applied K-Means clustering ($k=4$) to the preprocessed features.
- Determined the optimal number of clusters using the Elbow Method.
- Labeled risk archetypes programmatically based on cluster averages for distress rate, leverage (Debt/Equity), return on assets (ROA), and liquidity.

## Results
The table below displays the baseline test performance metrics compared to the tuned Random Forest classifier:

| Model | F1-Score | ROC-AUC |
| :--- | :--- | :--- |
| Naive Bayes | 0.8120 | 0.9412 |
| Support Vector Machine (SVM) | 0.7954 | 0.9320 |
| Logistic Regression | 0.7510 | 0.9142 |
| Random Forest (Baseline) | 0.7890 | 0.9521 |
| Gradient Boosting | 0.8012 | 0.9489 |
| Decision Tree | 0.6521 | 0.8123 |
| **Tuned Random Forest** | **0.8251** | **0.9634** |

## Key Findings
- **Feature Importance**: Ratios representing profitability (ROA) and short-term liquidity (cash-based ratios) are the most critical predictors of corporate distress.
- **Imbalance Mitigation**: Applying SMOTE on the training partition substantially increased recall for bankrupt companies, preventing the models from ignoring the minority class.
- **Unsupervised Profiling**: The K-Means clustering successfully segmented companies into four risk archetypes ("Distressed", "Highly Leveraged", "Liquidity Stressed", and "Stable Growth") that mirror real-world financial distress states.

## Technologies Used
- **Core Logic**: Python (v3.8+)
- **Data Manipulation**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn
- **Data Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebooks, Joblib

## Contributors
- Vaibhav Karri (Portfolio Lead)
- Research Partner (Academic Lead)
