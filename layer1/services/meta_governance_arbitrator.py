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
    worst_leakage_category = None
    
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
            
            # Baseline leakage confidence (start with permutation)
            base_confidence = min(100, importance * 150)
            
            # Subtractive Calibration: Low ablation subtracts confidence
            if ablation < 0.05:
                leakage_confidence = base_confidence * 0.35  # Severe contradiction penalty
                category = "A"
            elif ablation < 0.15:
                leakage_confidence = base_confidence * 0.70
                category = "B"
            elif ablation < 0.30:
                leakage_confidence = base_confidence * 0.90
                category = "C"
            else:
                leakage_confidence = base_confidence
                category = "D"
                
            # ARBITRATION CATEGORIES
            if category == "A":
                # CATEGORY A: Strong Predictive Feature
                leaky_issue["severity"] = "LOW"
                leaky_issue["description"] = f"Feature contributes strongly to predictions (importance: {importance:.2f}) but model retains resilience after feature removal (ablation drop: {ablation*100:.1f}%)."
                leaky_issue["suggestion"] = "No immediate action required. Predictive strength stems from redundant structure, not direct proxy leakage."
                leaky_issue["type"] = "feature_redundancy"
                leaky_issue["confidence"] = leakage_confidence
                arbitrated_issues.append(leaky_issue)
                if worst_leakage_category not in ["B", "C", "D"]: worst_leakage_category = "A"
                
            elif category == "B":
                # CATEGORY B: Proxy-like Behavior
                leaky_issue["severity"] = "MEDIUM"
                leaky_issue["description"] = f"Feature exhibits proxy-like predictive behavior (ablation drop: {ablation*100:.1f}%)."
                leaky_issue["suggestion"] = "Further validation recommended before deployment to rule out target encoding."
                leaky_issue["type"] = "proxy_behavior"
                leaky_issue["confidence"] = leakage_confidence
                arbitrated_issues.append(leaky_issue)
                if worst_leakage_category not in ["C", "D"]: worst_leakage_category = "B"
                
            elif category == "C":
                # CATEGORY C: Suspicious Leakage
                has_leakage = True
                leakage_explanation = f"Potential proxy-like behavior observed. Cross-engine agreement exists, but moderate ablation sensitivity ({ablation*100:.1f}%) suggests distributed feature redundancy rather than direct causal leakage."
                leaky_issue["severity"] = "HIGH"
                leaky_issue["description"] = f"Suspicious leakage pattern detected. High ablation impact ({ablation*100:.1f}%) and permutation dominance."
                leaky_issue["suggestion"] = "Conditional deployment restriction. Investigate data generation timeline."
                leaky_issue["confidence"] = leakage_confidence
                arbitrated_issues.append(leaky_issue)
                if worst_leakage_category not in ["D"]: worst_leakage_category = "C"
                
            else:
                # CATEGORY D: Confirmed Leakage
                has_leakage = True
                leakage_explanation = f"Confirmed causal leakage. Feature ablation severely degrades the model ({ablation*100:.1f}% drop)."
                leaky_issue["severity"] = "CRITICAL"
                leaky_issue["description"] = leakage_explanation
                leaky_issue["suggestion"] = "Deployment blocked. Remove feature and retrain."
                leaky_issue["confidence"] = leakage_confidence
                arbitrated_issues.append(leaky_issue)
                worst_leakage_category = "D"
                
        else:
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
        "leakage_confidence": leakage_confidence,
        "worst_leakage_category": worst_leakage_category
    }
