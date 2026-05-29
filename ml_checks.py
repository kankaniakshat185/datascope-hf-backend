import pandas as pd
import numpy as np
import pandas.api.types as ptypes
from sklearn.ensemble import IsolationForest

def run_ml_checks(df: pd.DataFrame, target_col: str) -> list:
    issues = []
    
    # Define features
    features = [c for c in df.columns if c != target_col]
    
    # 1. Class Imbalance (only if target is likely categorical)
    if target_col in df.columns and (df[target_col].nunique() < 20 or ptypes.is_object_dtype(df[target_col])):
        counts = df[target_col].value_counts(normalize=True)
        min_class_ratio = counts.min()
        if min_class_ratio < 0.1: # Less than 10% representation
            ratio_str = f"1:{int(counts.max()/min_class_ratio)}" if min_class_ratio > 0 else "1:inf"
            issues.append({
                "type": "class_imbalance",
                "column": target_col,
                "ratio": ratio_str,
                "min_class": min_class_ratio * 100
            })
            
    # 2. Highly Correlated Features (Numerical only)
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        for col in upper.columns:
            correlated_with = upper[col][upper[col] > 0.85].index.tolist()
            if correlated_with:
                issues.append({
                    "type": "high_correlation",
                    "column": col,
                    "correlated_with": correlated_with,
                })
    
    # 3. Outliers (using IQR on numerical features per column)
    if len(num_df.columns) > 0:
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) < 10:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = len(outliers)
            
            if len(series) > 0:
                outlier_perc = (outlier_count / len(series)) * 100
                if outlier_perc > 1:
                    issues.append({
                        "type": "outliers",
                        "column": col,
                        "percentage": outlier_perc,
                        "count": int(outlier_count)
                    })

    return issues
