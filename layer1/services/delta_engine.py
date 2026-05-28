from typing import Dict, Any

def compute_metric_deltas(baseline: Dict[str, Any], candidate: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """
    Computes absolute and relative deltas between baseline and candidate metrics.
    """
    if "error" in baseline or "error" in candidate:
        return {"error": "Invalid metrics for delta calculation"}

    deltas = {}
    
    if problem_type == "classification":
        deltas["accuracy_delta"] = candidate.get("accuracy", 0) - baseline.get("accuracy", 0)
        deltas["log_loss_delta"] = candidate.get("log_loss", 0) - baseline.get("log_loss", 0)
    else:
        deltas["rmse_delta"] = candidate.get("rmse", 0) - baseline.get("rmse", 0)
        deltas["r2_delta"] = candidate.get("r2", 0) - baseline.get("r2", 0)
        deltas["mae_delta"] = candidate.get("mae", 0) - baseline.get("mae", 0)

    deltas["stability_delta"] = candidate.get("stability_score", 0) - baseline.get("stability_score", 0)
    deltas["missingness_delta"] = candidate.get("missingness_ratio", 0) - baseline.get("missingness_ratio", 0)

    return deltas

def calculate_confidence(deltas: Dict[str, Any], problem_type: str) -> str:
    """
    Computes confidence (LOW, MODERATE, HIGH) based on metric agreement.
    """
    stability = deltas.get("stability_delta", 0)
    
    if problem_type == "classification":
        acc = deltas.get("accuracy_delta", 0)
        ll = deltas.get("log_loss_delta", 0)
        
        # High confidence: accuracy goes up, loss goes down, stability goes up
        if acc > 0.01 and ll < -0.01 and stability >= 0:
            return "HIGH"
        elif acc > 0 and stability > -5:
            return "MODERATE"
        else:
            return "LOW"
    else:
        rmse = deltas.get("rmse_delta", 0)
        r2 = deltas.get("r2_delta", 0)
        
        if rmse < -0.01 and r2 > 0.01 and stability >= 0:
            return "HIGH"
        elif (rmse < 0 or r2 > 0) and stability > -5:
            return "MODERATE"
        else:
            return "LOW"
