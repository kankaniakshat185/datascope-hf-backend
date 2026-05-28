import pandas as pd
from validators import run_validators
from ml_checks import run_ml_checks
from impact_engine import calculate_impact
from suggestions import format_suggestions

import pandas as pd
from validators import run_validators
from ml_checks import run_ml_checks
from suggestions import format_suggestions
from layer1.services.baseline_snapshot_engine import compute_baseline_metrics
from layer1.services.sandbox_remediation_engine import apply_sandbox_remediation
from layer1.services.delta_engine import compute_metric_deltas, calculate_confidence
from layer1.services.governance_acceptance_engine import evaluate_remediation_acceptance

def run_all_checks(df: pd.DataFrame, target_col: str, custom_rules: list = None):
    """
    Main orchestrator for dataset debugger.
    """
    issues = []

    # 1. Validators
    validation_issues = run_validators(df)
    issues.extend(validation_issues)

    # 2. ML Checks
    ml_issues = run_ml_checks(df, target_col)
    issues.extend(ml_issues)
    
    # 2.5 Custom Rules
    if custom_rules:
        from validators import run_custom_rules
        custom_issues = run_custom_rules(df, custom_rules)
        issues.extend(custom_issues)

    final_results = []

    # Compute the global baseline snapshot BEFORE any remediation
    baseline_metrics = compute_baseline_metrics(df, target_col)
    problem_type = "classification" if "accuracy" in baseline_metrics else "regression"
    
    # Pre-filter raw duplicate ML checks to prevent repeating UI cards
    seen_issues = set()
    unique_issues = []
    for issue in issues:
        # Avoid showing generic OUTLIERS if Advanced Consensus already flags it, but here we just deduplicate by description
        desc = issue.get("description", "")
        if desc not in seen_issues:
            seen_issues.add(desc)
            unique_issues.append(issue)

    final_results = []

    # 3. Sandbox Validation Pipeline (Closed-Loop Governance)
    for issue in unique_issues:
        # Sandbox Remediation
        candidate_df = apply_sandbox_remediation(df, issue)
        
        # Recompute Benchmark
        candidate_metrics = compute_baseline_metrics(candidate_df, target_col)
        
        if "error" in baseline_metrics or "error" in candidate_metrics:
            # Fallback for untestable issues
            formatted_issue = format_suggestions(issue, 0)
            formatted_issue["governance_verdict"] = "UNCERTAIN"
            formatted_issue["governance_explanation"] = "Could not compute reliable baseline."
            final_results.append(formatted_issue)
            continue
            
        # Compute Deltas
        deltas = compute_metric_deltas(baseline_metrics, candidate_metrics, problem_type)
        
        # Calculate Confidence
        confidence = calculate_confidence(deltas, problem_type)
        
        # Governance Acceptance
        decision = evaluate_remediation_acceptance(deltas, confidence, problem_type, issue.get("type", ""))
        
        # Mock old impact calculation for legacy frontend compatibility
        impact_val = 0
        if problem_type == "classification":
            impact_val = deltas.get("accuracy_delta", 0) * 100
            baseline_score = baseline_metrics.get("accuracy", 0) * 100
            after_score = candidate_metrics.get("accuracy", 0) * 100
            metric = "Accuracy"
        else:
            impact_val = -deltas.get("rmse_delta", 0) # Negative RMSE delta is positive impact
            if baseline_metrics.get("rmse", 1) != 0:
                impact_val = (impact_val / baseline_metrics.get("rmse", 1)) * 100
            baseline_score = baseline_metrics.get("rmse", 0)
            after_score = candidate_metrics.get("rmse", 0)
            metric = "RMSE"
            
        formatted_issue = format_suggestions(issue, impact_val)
        
        # Attach Governance Payload
        formatted_issue["baseline_score"] = baseline_score
        formatted_issue["after_score"] = after_score
        formatted_issue["metric"] = metric
        formatted_issue["confidence_score"] = 90 if confidence == "HIGH" else (60 if confidence == "MODERATE" else 30)
        formatted_issue["governance_verdict"] = decision["verdict"]
        formatted_issue["governance_explanation"] = decision["explanation"]
        
        # Handle 0% synthetic improvements
        if abs(impact_val) < 0.1 and decision["verdict"] != "REJECTED":
            formatted_issue["governance_explanation"] += " (Impact was negligible because the issue did not strictly degrade the model's structural distribution)."
        
        # If rejected, override the suggestion to block user from blindly applying it
        if decision["verdict"] == "REJECTED":
            formatted_issue["suggestion"] = f"REMEDIATION BLOCKED: {decision['explanation']}"
            formatted_issue["severity"] = "LOW" # Downgrade severity so it doesn't alarm the user

        final_results.append(formatted_issue)
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    final_results.sort(
        key=lambda x: (x["impact"], severity_order.get(x["severity"], 0)),
        reverse=True
    )

    # Total impact (only positive)
    total_impact = sum([
        item["impact"] for item in final_results
        if isinstance(item["impact"], (int, float)) and item["impact"] > 0
    ])

    return {
        "issues": final_results,
        "total_impact": round(total_impact, 2)
    }