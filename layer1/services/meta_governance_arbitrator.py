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
        if impact_data and col in impact_data:
            feat_metrics = impact_data[col]
            importance = feat_metrics.get("importance_score", 0)
            ablation = feat_metrics.get("performance_impact", 0)
            
            # Additional semantic signals
            vif = vif_scores.get(col, 1.0)
            mi_score = mutual_info.get(col, 0.0)
            max_corr = 0.0
            if col in corr_matrix.columns:
                max_corr = corr_matrix[col].drop(col, errors='ignore').max()
                if pd.isna(max_corr): max_corr = 0.0
                
            redundancy_score = max(max_corr, min(vif / 10.0, 1.0))
            
            base_confidence = min(100, importance * 150)
            
            # Semantic Interpretation Matrix
            if importance > 0.05 and ablation < 0.05 and redundancy_score > 0.7:
                # CASE 2: REDUNDANT INFORMATIVE FEATURE
                leaky_issue["severity"] = "LOW" # UI Color: Yellow/Greenish
                leaky_issue["type"] = "redundant_informative"
                leaky_issue["description"] = (
                    "**CASE 2 — REDUNDANT INFORMATIVE FEATURE**\n\n"
                    f"**What happened:** The trained model currently relies heavily on '{col}' (Permutation Impact: {importance:.2f}), "
                    f"but its total removal barely harms performance (Ablation Drop: {ablation*100:.1f}%).\n\n"
                    "**Why it happened:** This usually indicates feature redundancy. The dataset contains other features with overlapping "
                    f"information (Max Correlation: {max_corr:.2f}, VIF: {vif:.1f}). Removing it allows the model to substitute it with alternatives.\n\n"
                    "**Is this dangerous?:** No. This is normal statistical behavior. It is NOT leakage.\n\n"
                    "**What to do next:** No immediate action required. You can optionally drop it to simplify the model."
                )
                leaky_issue["suggestion"] = "Learn More: [Feature Redundancy & Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)"
                leaky_issue["confidence"] = base_confidence * 0.9
                arbitrated_issues.append(leaky_issue)
                if worst_leakage_category not in ["B", "C", "D"]: worst_leakage_category = "A"

            elif importance > 0.05 and ablation >= 0.05 and redundancy_score < 0.6:
                # CASE 1: CORE STABLE FEATURE
                leaky_issue["severity"] = "LOW" # UI Color: Green
                leaky_issue["type"] = "core_stable_feature"
                leaky_issue["description"] = (
                    "**CASE 1 — CORE STABLE FEATURE**\n\n"
                    f"**What happened:** This feature independently drives model performance (Permutation Impact: {importance:.2f}, Ablation Drop: {ablation*100:.1f}%).\n\n"
                    "**Why it happened:** Removing this feature significantly harms the model because no alternative feature can fully replace "
                    f"its predictive information (Redundancy Score: {redundancy_score:.2f}).\n\n"
                    "**Is this dangerous?:** No. This is a foundational predictive feature. The model genuinely depends on it.\n\n"
                    "**What to do next:** Keep this feature. It is highly valuable."
                )
                leaky_issue["suggestion"] = "Learn More: [Permutation Importance vs Feature Ablation](https://scikit-learn.org/stable/modules/permutation_importance.html)"
                leaky_issue["confidence"] = base_confidence
                arbitrated_issues.append(leaky_issue)
                if worst_leakage_category not in ["C", "D"]: worst_leakage_category = "B"

            elif importance > 0.20 and ablation >= 0.10 and mi_score > 0.5:
                # CASE 3: POSSIBLE FEATURE LEAKAGE (Requires Consensus)
                has_leakage = True
                leakage_explanation = f"Cross-engine consensus confirms '{col}' unnaturally dictates model performance (Ablation Drop: {ablation*100:.1f}%, High Mutual Information)."
                leaky_issue["severity"] = "CRITICAL" # UI Color: Red
                leaky_issue["type"] = "feature_leakage"
                leaky_issue["description"] = (
                    "**CASE 3 — POSSIBLE FEATURE LEAKAGE**\n\n"
                    f"**What happened:** This feature behaves too perfectly across multiple engines. It causes severe model collapse when removed (Ablation Drop: {ablation*100:.1f}%) "
                    f"and holds unnatural statistical overlap with the target (Mutual Information: {mi_score:.2f}).\n\n"
                    "**Why it happened:** The feature may directly encode or proxy the target variable, meaning it contains future information "
                    "the model should not legitimately know at prediction time.\n\n"
                    "**Is this dangerous?:** YES. The model will appear to perform perfectly in training, but fail completely in the real world.\n\n"
                    "**What to do next:** Deployment Blocked. Remove this feature and retrain."
                )
                leaky_issue["suggestion"] = "Learn More: [Feature Leakage in Machine Learning](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742)"
                leaky_issue["confidence"] = min(100, base_confidence * 1.5)
                arbitrated_issues.append(leaky_issue)
                worst_leakage_category = "D"
            else:
                # General Observation (Not strictly any case, but safe)
                leaky_issue["severity"] = "LOW"
                leaky_issue["description"] = f"**Informational:** '{col}' is highly correlated with the target, but lacks the extreme ablation sensitivity required to trigger a leakage consensus. Considered safe."
                leaky_issue["suggestion"] = "Monitor. Strong predictors are good, but require cross-validation tracking."
                arbitrated_issues.append(leaky_issue)
        else:
            leaky_issue["severity"] = "LOW"
            leaky_issue["description"] = f"**Observation:** '{col}' is highly correlated with the target, but causal ablation data is unavailable. Weak leakage confidence without causal proof."
            leaky_issue["suggestion"] = "Monitor. Strong predictors require ablation testing before governance escalation."
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

