from typing import Dict, Any, List
import copy

def arbitrate_governance_signals(
    ml_issues: List[Dict[str, Any]], 
    outlier_pct: float,
    shap_insights: List[str],
    impact_data: Dict[str, Any],
    corr_matrix: Any = None
) -> Dict[str, Any]:
    """
    CENTRAL META-GOVERNANCE ARBITRATION LAYER.
    Coordinates across engines, resolves contradictions, calibrates severity, 
    and suppresses weak signals before emission.
    """
    
    arbitrated_issues = []
    suppressed_signals = []
    
    # -------------------------------------------------------------
    # 1. Leakage Arbitration (Permutation vs Ablation vs Correlation)
    # -------------------------------------------------------------
    has_leakage = False
    leakage_explanation = ""
    leakage_confidence = 0
    
    # Collect all potential leakage signals
    raw_leakage_issues = [i for i in ml_issues if i.get("type") == "data_leakage"]
    other_issues = [i for i in ml_issues if i.get("type") != "data_leakage"]
    
    for leaky_issue in raw_leakage_issues:
        col = leaky_issue.get("column", "")
        # Check against ablation if impact_data exists
        if impact_data and col in impact_data:
            feat_metrics = impact_data[col]
            importance = feat_metrics.get("importance_score", 0)
            ablation = feat_metrics.get("performance_impact", 0)
            
            # CONTRADICTION RULE: High permutation, low ablation = Redundancy
            if importance > 0.4 and ablation < 0.05:
                suppressed_signals.append({
                    "type": "feature_redundancy",
                    "column": col,
                    "reason": f"Permutation importance is high but ablation drop is near zero ({ablation:.4f}). Interpreting as distributed feature redundancy, NOT leakage."
                })
                # Downgrade issue
                leaky_issue["severity"] = "LOW"
                leaky_issue["description"] = f"Observation: '{col}' is a strong but non-critical predictor. Its ablation impact is low ({ablation:.4f}) due to ensemble redundancy."
                leaky_issue["suggestion"] = "No immediate action required. Feature substitution prevents leakage collapse."
                leaky_issue["type"] = "feature_redundancy"
                arbitrated_issues.append(leaky_issue)
                continue
            
            # CONFIRMATION RULE: High importance AND high ablation
            if importance > 0.6 and ablation > 0.2:
                has_leakage = True
                leakage_confidence = 95
                leakage_explanation = f"Confirmed Proxy Target: '{col}' causes catastrophic collapse when ablated ({ablation*100:.1f}% drop)."
                leaky_issue["severity"] = "CRITICAL"
                leaky_issue["description"] = leakage_explanation
                leaky_issue["confidence"] = leakage_confidence
                arbitrated_issues.append(leaky_issue)
                continue
                
        # If no impact data to verify, downgrade pure correlation to observation
        leaky_issue["severity"] = "LOW"
        leaky_issue["description"] = f"Observation: '{col}' is highly correlated with the target. Weak leakage confidence without causal proof."
        leaky_issue["suggestion"] = "Monitor. Strong predictors require ablation testing before governance escalation."
        arbitrated_issues.append(leaky_issue)
        
    # -------------------------------------------------------------
    # 2. Correlation Arbitration
    # -------------------------------------------------------------
    for issue in other_issues:
        if issue.get("type") == "high_correlation":
            # Suppress. Tree ensembles tolerate this.
            suppressed_signals.append({
                "type": "correlation_suppressed",
                "column": issue.get("column"),
                "reason": "Tree ensembles naturally tolerate correlation. Suppressed from governance UI."
            })
            continue
            
        elif issue.get("type") == "outliers":
            # -------------------------------------------------------------
            # 3. Anomaly Arbitration
            # -------------------------------------------------------------
            if outlier_pct < 5.0:
                # Contradiction: Single engine flagged outliers, but consensus is low
                suppressed_signals.append({
                    "type": "weak_anomaly_suppressed",
                    "column": issue.get("column"),
                    "reason": "Weak isolated anomaly observation. Cross-engine consensus is low."
                })
                continue
                
        # Emit validated issues only
        arbitrated_issues.append(issue)

    return {
        "arbitrated_issues": arbitrated_issues,
        "suppressed_signals": suppressed_signals,
        "has_leakage": has_leakage,
        "leakage_explanation": leakage_explanation,
        "leakage_confidence": leakage_confidence
    }
