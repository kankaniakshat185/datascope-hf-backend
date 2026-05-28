import pandas as pd
import numpy as np
from typing import Dict, Any, List
import copy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def compute_vif_proxy(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, float]:
    """Computes a simplified VIF/redundancy proxy using R^2 of each feature against others."""
    # We use a fast approximation: highest R^2 with any single other feature, or max correlation
    if len(num_cols) < 2: return {col: 1.0 for col in num_cols}
    corr_matrix = df[num_cols].corr().abs()
    vif_proxy = {}
    for col in num_cols:
        max_r2 = (corr_matrix[col].drop(col).max()) ** 2
        # VIF = 1 / (1 - R^2). Cap at 10 to avoid inf
        vif = 1.0 / (1.0 - max_r2 + 1e-5)
        vif_proxy[col] = min(vif, 10.0)
    return vif_proxy

def arbitrate_governance_signals(
    df: pd.DataFrame,
    target_column: str,
    ml_issues: List[Dict[str, Any]], 
    outlier_pct: float,
    shap_insights: List[str],
    impact_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    FEATURE INTERPRETATION & SEMANTIC REASONING ENGINE.
    Replaces random metric reporting with context-aware statistical reasoning.
    """
    
    arbitrated_issues = []
    suppressed_signals = []
    
    # -------------------------------------------------------------
    # 0. Feature Relationship Profiling
    # -------------------------------------------------------------
    num_df = df.select_dtypes(include=[np.number])
    num_cols = num_df.columns.tolist()
    if target_column in num_cols: num_cols.remove(target_column)
    
    corr_matrix = num_df.corr().abs() if len(num_cols) > 0 else pd.DataFrame()
    vif_scores = compute_vif_proxy(num_df, num_cols)
    
    # Encode for MI
    df_clean = df.copy()
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    df_clean = df_clean.fillna(0)
    
    X = df_clean.drop(columns=[target_column]) if target_column in df_clean.columns else df_clean
    y = df_clean[target_column] if target_column in df_clean.columns else None
    
    mutual_info = {}
    if y is not None and len(X) > 0 and len(X.columns) > 0:
        is_classification = df[target_column].nunique() < 20 or not pd.api.types.is_numeric_dtype(df[target_column])
        try:
            if is_classification:
                mi = mutual_info_classif(X, y, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)
            mutual_info = {col: val for col, val in zip(X.columns, mi)}
        except:
            pass

    # -------------------------------------------------------------
    # 1. Leakage Arbitration & Semantic Interpretation
    # -------------------------------------------------------------
    has_leakage = False
    leakage_explanation = ""
    leakage_confidence = 0
    worst_leakage_category = None
    
    raw_leakage_issues = [i for i in ml_issues if i.get("type") == "data_leakage"]
    other_issues = [i for i in ml_issues if i.get("type") != "data_leakage"]
    
    for leaky_issue in raw_leakage_issues:
        col = leaky_issue.get("column", "")
        # Since leakage is now handled by the dedicated statistical engine (detect_feature_leakage),
        # we bypass the old Category A/B/C/D rules entirely.
        leaky_issue["severity"] = "LOW"
        leaky_issue["description"] = f"**Informational:** '{col}' flagged by legacy checks. Refer to the new Governance Leakage Engine for definitive arbitration."
        arbitrated_issues.append(leaky_issue)
            
    # -------------------------------------------------------------
    # 2. Other Issues Arbitration
    # -------------------------------------------------------------
    for issue in other_issues:
        if issue.get("type") == "high_correlation":
            suppressed_signals.append({
                "type": "correlation_suppressed",
                "column": issue.get("column"),
                "reason": "Tree ensembles naturally tolerate correlation. Suppressed from governance UI."
            })
            continue
            
        elif issue.get("type") == "outliers":
            if outlier_pct < 5.0:
                suppressed_signals.append({
                    "type": "weak_anomaly_suppressed",
                    "column": issue.get("column"),
                    "reason": "Weak isolated anomaly observation. Cross-engine consensus is low."
                })
                continue
                
        arbitrated_issues.append(issue)

    return {
        "arbitrated_issues": arbitrated_issues,
        "suppressed_signals": suppressed_signals,
        "has_leakage": has_leakage,
        "leakage_explanation": leakage_explanation,
        "leakage_confidence": leakage_confidence,
        "worst_leakage_category": worst_leakage_category
    }

