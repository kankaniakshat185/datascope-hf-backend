def format_suggestions(issue: dict, impact: float) -> dict:
    """
    Converts raw issue into structured UI-friendly output.
    Keeps impact numeric for sorting + string for display.
    """

    issue_type = issue.get("type")
    impact_val = float(impact) if isinstance(impact, (int, float)) else 0.0

    # ----------------------------
    # 🔥 Priority based on IMPACT
    # ----------------------------
    if impact_val > 5 or issue_type in ["data_leakage"]:
        severity = "HIGH"
    elif impact_val > 2:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # ----------------------------
    # Base structure
    # ----------------------------
    formatted = {
        "type": issue_type,
        "severity": severity,
        "impact": impact_val,  # ✅ numeric for sorting
        "impact_display": f"+{round(impact_val, 2)}%" if impact_val >= 0 else f"{round(impact_val, 2)}%",
        "description": "",
        "suggestion": "",
    }

    # ----------------------------
    # Issue-specific formatting
    # ----------------------------

    if issue_type == "missing_values":
        col = issue.get("column", "unknown")
        perc = issue.get("percentage", 0)

        formatted["description"] = f"Missing values in '{col}' ({perc:.1f}%) affecting model reliability"

        if isinstance(col, str) and any(k in col.lower() for k in ["age", "price", "amount"]):
            formatted["suggestion"] = "Use median imputation for robustness"
        else:
            formatted["suggestion"] = "Impute missing values (mean/median/mode)"

    elif issue_type == "class_imbalance":
        ratio = issue.get("ratio", "unknown")

        formatted["description"] = f"Severe class imbalance detected ({ratio})"
        formatted["suggestion"] = "Apply SMOTE, oversampling, or class weights"

    elif issue_type == "high_correlation":
        col = issue.get("column", "unknown")
        corrs = issue.get("correlated_with", [])

        formatted["description"] = f"Feature '{col}' is highly correlated with {', '.join(corrs)}"
        formatted["suggestion"] = f"Drop '{col}' to reduce multicollinearity"

    elif issue_type == "outliers":
        col = issue.get("column", "unknown")
        perc = issue.get("percentage", 0)

        formatted["description"] = f"Outliers detected in '{col}' ({perc:.1f}% of data) impacting model stability"
        formatted["suggestion"] = "Remove or cap extreme values using IQR or Z-score"

    elif issue_type == "data_leakage":
        col = issue.get("column", "unknown")

        formatted["description"] = f"Data leakage: '{col}' is strongly correlated with target"
        formatted["suggestion"] = f"Remove '{col}' to prevent model overfitting"

    elif issue_type == "uniqueness_violation":
        col = issue.get("column", "unknown")

        formatted["description"] = f"Duplicate identifiers found in ID column '{col}'"
        formatted["suggestion"] = "Ensure unique identifiers or drop duplicates"


    elif issue_type == "high_cardinality":
        col = issue.get("column", "unknown")

        formatted["description"] = f"Column '{col}' has very high cardinality"
        formatted["suggestion"] = "Apply encoding (target/embedding) or drop feature"

    elif issue_type == "pii_detected":
        col = issue.get("column", "unknown")
        pii_type = issue.get("pii_type", "Sensitive Data")
        perc = issue.get("percentage", 0)
        
        formatted["description"] = f"Security Risk: '{col}' contains {pii_type} ({perc:.1f}%)"
        formatted["suggestion"] = "Anonymize, hash, or drop this column before training!"
        formatted["severity"] = "HIGH" # Override to HIGH severity because it's a security risk
        formatted["impact"] = max(formatted["impact"], 10.0) # Ensure it sorts near the top

    elif issue_type == "custom_rule_violation":
        col = issue.get("column", "unknown")
        rule_type = issue.get("rule_type", "")
        expected = issue.get("expected", "")
        
        formatted["description"] = f"Rule Violation: '{col}' failed '{rule_type}' ({expected})"
        formatted["suggestion"] = "Review the data points that violate your custom business logic"
        formatted["severity"] = "HIGH"
        formatted["impact"] = max(formatted["impact"], 8.0)

    else:
        formatted["description"] = f"Detected issue: {issue_type}"
        formatted["suggestion"] = "Further investigation required"

    return formatted