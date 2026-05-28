import pandas as pd
import numpy as np
from typing import Dict, Any, List
import math
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def compute_target_candidates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    HUMAN-IN-THE-LOOP TARGET PROFILING ENGINE
    Analyzes all columns to suggest the most likely target variables without auto-selecting.
    """
    valid_cols = [str(c) for c in df.columns if not str(c).startswith("Unnamed:")]
    total_rows = len(df)
    
    candidates = []
    
    priority_names = ["target", "label", "class", "outcome", "status", "price", "churn", "survived"]
    
    for col in valid_cols:
        col_data = df[col]
        nunique = col_data.nunique()
        missing = col_data.isnull().sum() / total_rows
        dtype = str(col_data.dtype)
        
        # Skip purely empty or id-like columns
        if missing > 0.8:
            continue
        if nunique == total_rows and total_rows > 10:
            continue
            
        score = 0.0
        reasons = []
        is_binary = False
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        # 1. Semantic Naming
        if col.lower() in priority_names or col.lower() == "y":
            score += 0.4
            reasons.append("Exact match with common ML label naming conventions.")
        elif any(name in col.lower() for name in priority_names):
            score += 0.2
            reasons.append("Partial match with common ML label naming conventions.")
            
        # 2. Binary Detection (Classification)
        if nunique == 2:
            is_binary = True
            score += 0.5
            reasons.append("Binary distribution detected (standard classification target).")
            # Check class balance
            val_counts = col_data.value_counts(normalize=True)
            if min(val_counts) > 0.1:
                score += 0.1
                reasons.append("Classes are relatively balanced.")
            else:
                reasons.append("Warning: Highly imbalanced binary classes.")
                
        # 3. Numeric Variance (Regression)
        if is_numeric and nunique > 10 and not is_binary:
            score += 0.3
            reasons.append("Continuous numeric variable with sufficient variance (regression-compatible).")
            
        # 4. Low Cardinality Categorical
        if not is_numeric and 2 < nunique < 20:
            score += 0.2
            reasons.append("Low cardinality categorical variable (multi-class classification).")
            
        # 5. Penalties
        if missing > 0.2:
            score -= 0.2
            reasons.append("Warning: Contains >20% missing values. Requires imputation if selected.")
            
        if not reasons:
            reasons.append("Valid column but no specific ML target heuristics triggered.")
            
        # Cap score at 0.99 for realism
        final_score = min(0.99, max(0.01, score))
        
        candidates.append({
            "column": col,
            "score": round(final_score, 2),
            "reason": reasons,
            "dtype": dtype,
            "nunique": int(nunique),
            "missing_pct": round(float(missing) * 100, 1),
            "proxies": identify_potential_proxies(df, col)
        })
        
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def identify_potential_proxies(df: pd.DataFrame, target: str) -> List[Dict[str, Any]]:
    """
    FUTURE SAFETY GUARDS: Scans for columns that might be near-duplicates or proxies of the target.
    """
    valid_cols = [str(c) for c in df.columns if not str(c).startswith("Unnamed:") and c != target]
    target_data = df[target]
    
    proxies = []
    
    # Simple semantic overlap
    target_clean = target.lower().replace("_", "").replace("-", "")
    
    for col in valid_cols:
        col_clean = col.lower().replace("_", "").replace("-", "")
        if target_clean in col_clean or col_clean in target_clean:
            proxies.append({
                "column": col,
                "reason": f"Semantic overlap with target name ('{col}' ~ '{target}')."
            })
            continue
            
        # Correlation if both numeric
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target_data):
            corr = abs(df[col].corr(target_data))
            if corr > 0.95:
                proxies.append({
                    "column": col,
                    "reason": f"Extremely high mathematical correlation ({corr:.2f}) with target. Likely a duplicate or data leakage."
                })
                
    return proxies
