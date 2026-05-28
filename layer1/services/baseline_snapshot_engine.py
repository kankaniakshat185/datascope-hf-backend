import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple

def compute_baseline_metrics(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Captures a full baseline state of the dataset before any remediation is applied.
    Returns RMSE (or LogLoss), R2 (or Accuracy), and stability proxies.
    """
    if target_col not in df.columns:
        return {"error": "Target column missing"}

    df_clean = df.copy()
    
    # Fast lightweight preprocessing for benchmarking
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
            
    df_clean = df_clean.fillna(0) # naive fill for benchmarking
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    if len(X) < 10 or X.shape[1] == 0:
        return {"error": "Insufficient data for baseline"}

    is_classification = df[target_col].nunique() < 10 or not pd.api.types.is_numeric_dtype(df[target_col])
    
    if is_classification:
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
        scoring = ['accuracy', 'neg_log_loss']
    else:
        model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
        scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

    # 3-fold CV for stability
    cv_results = cross_validate(model, X, y, cv=3, scoring=scoring, return_train_score=False)
    
    metrics = {}
    if is_classification:
        metrics["accuracy"] = float(np.mean(cv_results['test_accuracy']))
        metrics["log_loss"] = float(-np.mean(cv_results['test_neg_log_loss']))
        stability_var = np.std(cv_results['test_accuracy'])
    else:
        metrics["r2"] = float(np.mean(cv_results['test_r2']))
        metrics["rmse"] = float(-np.mean(cv_results['test_neg_root_mean_squared_error']))
        metrics["mae"] = float(-np.mean(cv_results['test_neg_mean_absolute_error']))
        stability_var = np.std(cv_results['test_r2'])

    missingness_ratio = float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    duplicate_ratio = float(df.duplicated().sum() / df.shape[0])
    
    # Base stability purely on predictive variance
    base_stability = 100 - (stability_var * 200)
    
    # Penalize for dataset corruption
    base_stability -= (missingness_ratio * 150)  # Heavy penalty for missing data
    base_stability -= (duplicate_ratio * 100)    # Penalty for duplicated rows
    
    stability_score = max(0.0, min(100.0, base_stability))
    
    metrics["stability_score"] = float(stability_score)
    metrics["missingness_ratio"] = missingness_ratio
    metrics["duplicate_ratio"] = duplicate_ratio
    
    return metrics
