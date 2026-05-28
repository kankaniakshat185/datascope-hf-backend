from typing import Dict, Any

def evaluate_remediation_acceptance(deltas: Dict[str, Any], confidence: str, problem_type: str, issue_type: str) -> Dict[str, Any]:
    """
    Deterministically decides whether to accept, reject, or require manual review for a remediation.
    Returns a verdict and a causal explanation.
    """
    verdict = "REJECTED"
    explanation = "No reliable remediation strategy produced net-positive stability improvement."

    stability_delta = deltas.get("stability_delta", 0)
    
    if problem_type == "classification":
        primary_metric_improved = deltas.get("accuracy_delta", 0) > 0
        primary_metric_name = "accuracy"
        primary_delta_str = f"+{deltas.get('accuracy_delta', 0)*100:.2f}%" if deltas.get("accuracy_delta", 0) > 0 else f"{deltas.get('accuracy_delta', 0)*100:.2f}%"
    else:
        primary_metric_improved = deltas.get("rmse_delta", 0) < 0
        primary_metric_name = "RMSE"
        primary_delta_str = f"{deltas.get('rmse_delta', 0):.2f}"

    # RULE 1: Severe Degradation -> Hard Reject
    if stability_delta < -10:
        return {
            "verdict": "REJECTED",
            "explanation": f"Remediation caused severe stability degradation ({stability_delta:.1f} points). Automatic fix rejected."
        }
        
    # RULE 2: Metric Agreement -> Accept
    if primary_metric_improved and stability_delta >= -2 and confidence in ["HIGH", "MODERATE"]:
        return {
            "verdict": "ACCEPTED",
            "explanation": f"Remediation validated. {primary_metric_name.capitalize()} improved by {primary_delta_str} and stability score changed by {stability_delta:.1f} points. Fix approved."
        }
        
    # RULE 3: Trade-off -> Manual Review
    if (primary_metric_improved and stability_delta < -2) or (not primary_metric_improved and stability_delta > 5):
        return {
            "verdict": "UNCERTAIN",
            "explanation": f"Inconclusive outcomes: {primary_metric_name.capitalize()} changed by {primary_delta_str} but stability shifted by {stability_delta:.1f}. Manual review recommended."
        }
        
    # Default Reject
    return {
        "verdict": "REJECTED",
        "explanation": f"Remediation did not produce measurable improvement ({primary_metric_name} delta: {primary_delta_str}, Stability delta: {stability_delta:.1f})."
    }
