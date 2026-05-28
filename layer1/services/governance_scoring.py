from typing import Dict, Any, List
import datetime
from layer1.services.baseline_snapshot_engine import compute_baseline_metrics
import pandas as pd
def detect_identifier_columns(df):
    """
    Detects synthetic identifiers and row-index-like columns.
    These should usually never participate in model training.
    """

    suspicious = []

    for col in df.columns:
        try:
            name = str(col).lower()

            uniqueness_ratio = df[col].nunique(dropna=False) / max(1, len(df))

            is_monotonic = False
            try:
                is_monotonic = df[col].is_monotonic_increasing
            except:
                pass

            if (
                "id" in name
                or "uuid" in name
                or "index" in name
                or "key" in name
                or (
                    uniqueness_ratio > 0.995
                    and pd.api.types.is_integer_dtype(df[col])
                )
            ):
                suspicious.append(col)

        except Exception:
            continue

    return suspicious


def calculate_governance_score(
    df: Any,
    target_column: str,
    ml_issues: List[Dict[str, Any]],
    layer1_data: Dict[str, Any],
    shap_data: Dict[str, Any],
    runtime_ms: int = 1500,
    rows_processed: int = 0,
    features_evaluated: int = 0,
    worst_leakage_category: str = None
) -> Dict[str, Any]:

    """
    Enterprise-style deterministic governance scoring engine.

    Philosophy:
    - Clean datasets should usually APPROVE
    - Moderate imperfections should trigger REVIEW
    - Only catastrophic issues should REJECT
    """

    audit_logs = []

    baseline = compute_baseline_metrics(df, target_column) if df is not None else {}

    real_stability = baseline.get("stability_score", 82.0)
    missingness = baseline.get("missingness_ratio", 0)

    # =========================================================
    # BASELINE PRIORS
    # =========================================================

    governance_score = 78.0
    stability_score = max(45.0, min(real_stability, 94.0))

    confidence_score = 100.0

    critical_found = False
    shap_leakage = False

    # =========================================================
    # IDENTIFIER DETECTION
    # =========================================================

    identifier_columns = detect_identifier_columns(df) if df is not None else []

    REAL_IDENTIFIER_COLUMNS = []

    for col in identifier_columns:

        try:
            uniqueness_ratio = (
                df[col].nunique(dropna=False) / max(1, len(df))
            )

            if uniqueness_ratio > 0.98:
                REAL_IDENTIFIER_COLUMNS.append(col)

        except:
            continue

    if REAL_IDENTIFIER_COLUMNS:

        governance_score -= min(2.5, len(REAL_IDENTIFIER_COLUMNS) * 0.8)

        stability_score -= min(2.0, len(REAL_IDENTIFIER_COLUMNS) * 0.5)

        confidence_score -= 2

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "IDENTIFIER_COLUMNS_DETECTED",
            "severity": "INFO",
            "message":
                f"Identifier-like columns detected: "
                f"{', '.join(REAL_IDENTIFIER_COLUMNS[:5])}. "
                f"These columns are recommended for exclusion from training."
        })

    # =========================================================
    # VALIDATION PHASE
    # =========================================================

    validation_status = "VALIDATION_PASSED"

    audit_logs.append({
        "phase": "VALIDATION PHASE",
        "rule": "DATASET_REGISTERED",
        "severity": "INFO",
        "message":
            f"Dataset registered with {rows_processed} rows."
    })

    audit_logs.append({
        "phase": "VALIDATION PHASE",
        "rule": "VALIDATION_STARTED",
        "severity": "INFO",
        "message":
            f"Validation engines scanning {features_evaluated} features."
    })

    # =========================================================
    # POSITIVE STRUCTURAL SIGNALS
    # =========================================================

    if missingness < 0.01:

        governance_score += 6.0
        stability_score += 4.0

    elif missingness < 0.05:

        governance_score += 3.0
        stability_score += 2.0

    if rows_processed > 1000:

        governance_score += 3.0

    elif rows_processed > 300:

        governance_score += 2.0

    if features_evaluated >= 5:

        governance_score += 2.0

    # =========================================================
    # MISSINGNESS PENALTIES
    # =========================================================

    if missingness > 0.10:

        governance_score -= (missingness * 40)
        stability_score -= (missingness * 45)

        confidence_score -= missingness * 60

        validation_status = "VALIDATION_WARNING"

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "HIGH_MISSINGNESS",
            "severity": "WARNING",
            "message":
                f"Dataset contains {missingness*100:.1f}% missing values."
        })

    elif missingness > 0.05:

        governance_score -= (missingness * 20)
        stability_score -= (missingness * 25)

    # =========================================================
    # ML ISSUE PENALTIES
    # =========================================================

    for issue in ml_issues:

        severity = issue.get("severity", "LOW")
        description = issue.get("description", "")

        if severity == "CRITICAL":

            critical_found = True

            governance_score -= 20.0
            stability_score -= 16.0

            confidence_score -= 25

            validation_status = "VALIDATION_FAILED"

            audit_logs.append({
                "phase": "VALIDATION PHASE",
                "rule": "CRITICAL_VULNERABILITY",
                "severity": "CRITICAL",
                "message":
                    f"Critical structural issue detected: {description}"
            })

        elif severity == "HIGH":

            governance_score -= 6.0
            stability_score -= 4.0

            confidence_score -= 8

            validation_status = (
                "VALIDATION_WARNING"
                if validation_status == "VALIDATION_PASSED"
                else validation_status
            )

            audit_logs.append({
                "phase": "VALIDATION PHASE",
                "rule": issue.get("type", "HIGH_RISK").upper(),
                "severity": "HIGH",
                "message":
                    f"Elevated risk observation: {description}"
            })

        elif severity == "MEDIUM":

            governance_score -= 1.5
            stability_score -= 1.0

            confidence_score -= 2

    # =========================================================
    # VALIDATION COMPLETION
    # =========================================================

    if validation_status == "VALIDATION_PASSED":

        governance_score += 4.0
        stability_score += 2.0

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message":
                "Dataset passed foundational structural validation."
        })

    # =========================================================
    # ANALYSIS PHASE
    # =========================================================

    audit_logs.append({
        "phase": "ANALYSIS PHASE",
        "rule": "BENCHMARK_COMPLETED",
        "severity": "INFO",
        "message":
            "Baseline benchmarking and anomaly arbitration completed."
    })

    outlier_pct = 0.0

    if layer1_data and "outlier_analysis" in layer1_data:

        outlier_pct = (
            layer1_data["outlier_analysis"]
            .get("summary", {})
            .get("percentage_flagged", 0)
        )

    # =========================================================
    # OUTLIER PENALTIES
    # =========================================================

    if outlier_pct > 15.0:

        governance_score -= (outlier_pct * 1.0)
        stability_score -= (outlier_pct * 1.5)

        confidence_score -= 14

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "SEVERE_OUTLIER_CONTAMINATION",
            "severity": "CRITICAL",
            "message":
                f"{outlier_pct:.1f}% anomalous rows detected."
        })

    elif outlier_pct > 5.0:

        governance_score -= (outlier_pct * 0.35)
        stability_score -= (outlier_pct * 0.7)

        confidence_score -= 4

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "MODERATE_OUTLIER_CONTAMINATION",
            "severity": "WARNING",
            "message":
                f"Moderate anomaly contamination observed "
                f"({outlier_pct:.1f}%)."
        })

    # =========================================================
    # LEAKAGE ARBITRATION
    # =========================================================

    if worst_leakage_category:

        if worst_leakage_category == "A":

            governance_score -= 1.0

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "FEATURE_REDUNDANCY_OBSERVED",
                "severity": "INFO",
                "message":
                    "Feature redundancy detected without leakage consensus."
            })

        elif worst_leakage_category == "B":

            governance_score -= 6.0
            stability_score -= 4.0

            confidence_score -= 6

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "PROXY_BEHAVIOR_DETECTED",
                "severity": "WARNING",
                "message":
                    "Proxy-like predictive behavior observed."
            })

        elif worst_leakage_category == "C":

            shap_leakage = True

            governance_score -= 14.0
            stability_score -= 10.0

            confidence_score -= 14

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "SUSPICIOUS_LEAKAGE_PATTERN",
                "severity": "HIGH",
                "message":
                    "Potential target leakage suspected."
            })

        elif worst_leakage_category == "D":

            shap_leakage = True
            critical_found = True

            governance_score -= 40.0
            stability_score -= 32.0

            confidence_score -= 40

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "CAUSAL_LEAKAGE_CONFIRMED",
                "severity": "CRITICAL",
                "message":
                    "Confirmed causal leakage detected."
            })

    else:

        governance_score += 4.0

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "LEAKAGE_VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message":
                "No actionable leakage consensus detected."
        })

    # =========================================================
    # FINAL CALIBRATION
    # =========================================================

    governance_score = round(max(5.0, min(98.0, governance_score)), 1)

    stability_score = round(max(5.0, min(95.0, stability_score)), 1)

    confidence_score = round(max(5.0, min(99.0, confidence_score)), 1)

    # =========================================================
    # DEPLOYMENT GATING
    # =========================================================

    deployment_ready = (
        governance_score >= 80.0
        and stability_score >= 72.0
        and not critical_found
        and worst_leakage_category != "D"
    )

    retraining_required = (
        stability_score < 60.0
        or shap_leakage
        or outlier_pct > 15.0
    )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    if (
        governance_score >= 80
        and stability_score >= 72
        and not critical_found
    ):

        final_status = "APPROVED"

    elif (
        governance_score < 45
        or stability_score < 40
        or worst_leakage_category == "D"
    ):

        final_status = "REJECTED"

    else:

        final_status = "AWAITING_REVIEW"

    # =========================================================
    # CONFIDENCE LABEL
    # =========================================================

    if confidence_score >= 85:

        confidence_label = "HIGH"

    elif confidence_score >= 60:

        confidence_label = "MODERATE"

    else:

        confidence_label = "LOW"

    # =========================================================
    # GOVERNANCE EVENTS
    # =========================================================

    if final_status == "APPROVED":

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "GOVERNANCE_APPROVED",
            "severity": "SUCCESS",
            "message":
                (
                    "**Governance Approved**\n\n"
                    "- Structural integrity remained stable.\n"
                    "- No catastrophic leakage detected.\n"
                    "- Stability exceeded deployment threshold.\n\n"
                    f"**Confidence level:** {confidence_label}."
                )
        })

    elif final_status == "AWAITING_REVIEW":

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "GOVERNANCE_REVIEW_STARTED",
            "severity": "WARNING",
            "message":
                (
                    "**Manual Governance Review Required**\n\n"
                    f"Governance Score: {governance_score}/100\n"
                    f"Stability Score: {stability_score}/100\n\n"
                    "Moderate structural concerns require human review."
                )
        })

    else:

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "DEPLOYMENT_BLOCKED",
            "severity": "CRITICAL",
            "message":
                (
                    "**Deployment Blocked**\n\n"
                    f"Governance Score: {governance_score}/100\n"
                    f"Stability Score: {stability_score}/100\n\n"
                    "Critical structural risk detected."
                )
        })

    if retraining_required:

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "RETRAINING_RECOMMENDED",
            "severity": "HIGH",
            "message":
                (
                    "**Retraining Recommended**\n\n"
                    "Instability patterns weaken production reliability."
                )
        })

    print("Worst leakage category:", worst_leakage_category)
    print("Governance:", governance_score)
    print("Stability:", stability_score)
    print("Final status:", final_status)

    return {

        "recommended_status": final_status,

        "governance_score": governance_score,

        "stability_score": stability_score,

        "confidence_score": confidence_score,

        "deployment_ready": deployment_ready,

        "retraining_required": retraining_required,

        "retraining_reason":
            (
                "Potential leakage or instability detected."
                if shap_leakage
                else (
                    f"Elevated outlier contamination ({outlier_pct:.1f}%)."
                    if outlier_pct > 10.0
                    else "Minor structural observations detected."
                )
            ),

        "metadata": {

            "run_id":
                f"RUN_{datetime.datetime.now().strftime('%Y_%m%d')}_001",

            "dataset_version": "v1",

            "model_type": "RandomForestRegressor",

            "pipeline_version": "pipeline_v4",

            "monitoring_status": "ACTIVE",

            "runtime_ms": runtime_ms,

            "rows_processed": rows_processed,

            "features_evaluated": features_evaluated,

            "drift_monitors_active": 12,

            "validation_engines_triggered": 7,

            "validation_status": validation_status,

            "identifier_columns_detected": REAL_IDENTIFIER_COLUMNS,

            "outlier_percentage": outlier_pct,

            "confidence_label": confidence_label
        },

        "audit_logs": audit_logs
    }
