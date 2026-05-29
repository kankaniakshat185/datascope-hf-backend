import pandas as pd
import numpy as np
import re

def detect_pii(df: pd.DataFrame) -> list:
    pii_issues = []
    
    pii_patterns = {
        "Email Addresses": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "Phone Numbers": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "Social Security Numbers": r'\b\d{3}-\d{2}-\d{4}\b',
        "Credit Card Numbers": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
    }
    
    for col in df.select_dtypes(include=['object', 'string']).columns:
        col_data = df[col].dropna().astype(str)
        if len(col_data) == 0:
            continue
            
        sample_size = min(1000, len(col_data))
        sample_data = col_data.sample(sample_size, random_state=42)
        
        for pii_type, pattern in pii_patterns.items():
            matches = sample_data.str.contains(pattern, regex=True).sum()
            match_percentage = (matches / sample_size) * 100
            
            if match_percentage > 5:
                pii_issues.append({
                    "type": "pii_detected",
                    "column": col,
                    "pii_type": pii_type,
                    "percentage": match_percentage
                })
                break
                
    return pii_issues

def run_validators(df: pd.DataFrame) -> list:
    issues = []
    
    # 1. Missing Values Validation
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            issues.append({
                "type": "missing_values",
                "column": col,
                "percentage": percentage,
                "count": count
            })
            
    # 2. Uniqueness Validation
    import re
    for col in df.columns:
        is_id_pattern = bool(re.search(r'(^id$|_id$|^id_|_id_|^uuid$|^guid$|^record_id$|^customer_id$|^transaction_id$)', str(col).lower()))
        
        if is_id_pattern:
            if not df[col].is_unique:
                duplicates = df.duplicated(subset=[col]).sum()
                issues.append({
                    "type": "uniqueness_violation",
                    "column": col,
                    "count": int(duplicates),
                    "percentage": (duplicates / len(df)) * 100
                })
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                duplicates = df.duplicated(subset=[col]).sum()
                if duplicates > 0:
                    # Limit the flood of info messages by checking ratio
                    ratio = duplicates / len(df)
                    if ratio < 0.9: # Normal measurements will naturally repeat
                        issues.append({
                            "type": "repeated_values",
                            "column": col,
                            "count": int(duplicates),
                            "percentage": ratio * 100
                        })
    # 3. Categorical Validation (Cardinality check)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].nunique()
        if unique_vals > len(df) * 0.9 and unique_vals > 50:
            issues.append({
                "type": "high_cardinality",
                "column": col,
                "count": unique_vals,
            })
            
    # 4. PII Detection
    pii_issues = detect_pii(df)
    issues.extend(pii_issues)
            
    return issues

def run_custom_rules(df: pd.DataFrame, rules: list) -> list:
    issues = []
    
    for rule in rules:
        col = rule.get("column")
        if not col or col not in df.columns:
            continue
            
        rule_type = rule.get("type")
        val = rule.get("value")
        
        col_data = df[col]
        invalid_count = 0
        
        if rule_type == "min" and val is not None:
            invalid_count = (pd.to_numeric(col_data, errors='coerce') < val).sum()
        elif rule_type == "max" and val is not None:
            invalid_count = (pd.to_numeric(col_data, errors='coerce') > val).sum()
        elif rule_type == "in" and isinstance(val, list):
            invalid_count = (~col_data.isin(val)).sum()
        elif rule_type == "not_null":
            invalid_count = col_data.isnull().sum()
            
        if invalid_count > 0:
            percentage = (invalid_count / len(df)) * 100
            issues.append({
                "type": "custom_rule_violation",
                "column": col,
                "rule_type": rule_type,
                "expected": val,
                "percentage": percentage,
                "count": int(invalid_count)
            })
            
    return issues
