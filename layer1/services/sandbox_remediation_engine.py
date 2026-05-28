import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def apply_sandbox_remediation(df: pd.DataFrame, issue: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies the proposed remediation to a sandboxed candidate dataframe.
    NEVER modifies the original dataframe directly.
    """
    candidate_df = df.copy()
    issue_type = issue.get("type", "")
    col = issue.get("column", None)

    try:
        if issue_type == "MISSING_VALUES" and col:
            if pd.api.types.is_numeric_dtype(candidate_df[col]):
                candidate_df[col] = candidate_df[col].fillna(candidate_df[col].median())
            else:
                candidate_df[col] = candidate_df[col].fillna(candidate_df[col].mode()[0] if not candidate_df[col].mode().empty else "Unknown")
                
        elif issue_type == "OUTLIERS" and col:
            if pd.api.types.is_numeric_dtype(candidate_df[col]):
                q1 = candidate_df[col].quantile(0.25)
                q3 = candidate_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                candidate_df[col] = np.clip(candidate_df[col], lower, upper)
                
        elif issue_type == "HIGH_CARDINALITY" and col:
            # Keep top 10 categories, bucket rest into 'Other'
            top_cats = candidate_df[col].value_counts().nlargest(10).index
            candidate_df[col] = candidate_df[col].apply(lambda x: x if x in top_cats else 'Other')
            
        elif issue_type == "HIGH_CORRELATION" and col:
            # Drop one of the highly correlated columns
            candidate_df = candidate_df.drop(columns=[col])
            
        elif issue_type == "CLASS_IMBALANCE" and col:
            # Basic random oversampling of minority classes
            max_size = candidate_df[col].value_counts().max()
            lst = [candidate_df]
            for class_index, group in candidate_df.groupby(col):
                if len(group) < max_size:
                    lst.append(group.sample(max_size - len(group), replace=True))
            candidate_df = pd.concat(lst)
            
        elif issue_type == "OUTLIERS_CONSENSUS":
            # This is a full row drop if we assume the engine identified anomalous rows
            # We don't have row indexes in this simple issue format, but we'll mock a generic capping across all numerics
            for c in candidate_df.select_dtypes(include=[np.number]).columns:
                q1 = candidate_df[c].quantile(0.01)
                q3 = candidate_df[c].quantile(0.99)
                candidate_df[c] = np.clip(candidate_df[c], q1, q3)

    except Exception as e:
        # If remediation fails, return original and we'll catch it in the delta engine
        pass

    return candidate_df
