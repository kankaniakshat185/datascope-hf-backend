import math
from typing import Dict, Any, List

def evaluate_governance_rules(
    ml_issues: List[Dict[str, Any]], 
    layer1_data: Dict[str, Any], 
    shap_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Deterministic Rules Engine for ML Governance.
    Evaluates ML validation artifacts to enforce state transitions and approval workflows.
    Returns the recommended Governance state and a list of human-readable audit reasons.
    """
    audit_logs = []
    final_status = "APPROVED"  # Default optimistic state
    
    # 1. Total Impact Evaluation
    total_impact = 0.0
    for issue in ml_issues:
        impact = issue.get("impact", 0)
        try:
            total_impact += float(impact)
        except (ValueError, TypeError):
            pass
            
    if total_impact > 30.0:
        final_status = "REJECTED"
        audit_logs.append({
            "rule": "IMPACT_THRESHOLD_EXCEEDED",
            "severity": "CRITICAL",
            "message": f"Cumulative dataset degradation impact is {total_impact:.1f}%, exceeding the 30% tolerance."
        })
    elif total_impact > 15.0:
        if final_status == "APPROVED": final_status = "AWAITING_REVIEW"
        audit_logs.append({
            "rule": "IMPACT_WARNING",
            "severity": "HIGH",
            "message": f"Cumulative dataset degradation impact is {total_impact:.1f}%. Human review required."
        })

    # 2. Consensus Outlier Engine Evaluation
    outlier_pct = 0.0
    if layer1_data and "outlier_analysis" in layer1_data:
        summary = layer1_data["outlier_analysis"].get("summary", {})
        outlier_pct = summary.get("percentage_flagged", 0)
        
    if outlier_pct > 15.0:
        final_status = "REJECTED"
        audit_logs.append({
            "rule": "SEVERE_OUTLIER_CONTAMINATION",
            "severity": "CRITICAL",
            "message": f"Layer 1 Consensus Engine flagged {outlier_pct:.1f}% of data as severe anomalies. Hard rejection threshold is 15%."
        })
    elif outlier_pct > 5.0:
        if final_status == "APPROVED": final_status = "RETRAINING_RECOMMENDED"
        audit_logs.append({
            "rule": "MODERATE_OUTLIER_CONTAMINATION",
            "severity": "MEDIUM",
            "message": f"Consensus Engine flagged {outlier_pct:.1f}% anomalies. Retraining with robust capping is recommended."
        })

    # 3. Critical ML Vulnerability Check
    critical_issues = [i for i in ml_issues if i.get("severity") == "CRITICAL"]
    if critical_issues:
        final_status = "REJECTED"
        for issue in critical_issues:
            audit_logs.append({
                "rule": "CRITICAL_ML_VULNERABILITY",
                "severity": "CRITICAL",
                "message": f"Critical vulnerability detected: {issue.get('description', 'Unknown')}"
            })

    # 4. SHAP Feature Leakage / Bias Evaluation
    if shap_data and not shap_data.get("error"):
        insights = shap_data.get("insights", [])
        for insight in insights:
            if "dominant" in str(insight).lower() or "overwhelming" in str(insight).lower():
                if final_status in ["APPROVED", "AWAITING_REVIEW"]:
                    final_status = "RETRAINING_RECOMMENDED"
                audit_logs.append({
                    "rule": "SHAP_FEATURE_LEAKAGE",
                    "severity": "HIGH",
                    "message": f"SHAP engine flagged potential target leakage or severe bias: {insight}"
                })

    # If everything is perfect, add an approval log
    if final_status == "APPROVED":
        audit_logs.append({
            "rule": "ALL_CHECKS_PASSED",
            "severity": "INFO",
            "message": "Dataset passed all deterministic governance thresholds. Auto-approved."
        })
        
    return {
        "recommended_status": final_status,
        "audit_logs": audit_logs
    }
