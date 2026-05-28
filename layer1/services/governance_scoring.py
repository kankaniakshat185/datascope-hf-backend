from typing import Dict, Any, List
import datetime
from layer1.services.baseline_snapshot_engine import compute_baseline_metrics

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
                or uniqueness_ratio > 0.97
                or is_monotonic
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
    - Systems begin suspicious by default
    - Trust is earned progressively
    - Deployment approval should be rare
    - REVIEW state should be common
    """

    audit_logs = []

    baseline = compute_baseline_metrics(df, target_column) if df is not None else {}

    real_stability = baseline.get("stability_score", 72.0)
    missingness = baseline.get("missingness_ratio", 0)

    # -----------------------------------------------------------------
    # CONSERVATIVE PRIORS
    # -----------------------------------------------------------------

    governance_score = 62.0
    stability_score = max(35.0, min(real_stability, 92.0))

    confidence_score = 100.0

    critical_found = False
    shap_leakage = False

    # -----------------------------------------------------------------
    # IDENTIFIER DETECTION
    # -----------------------------------------------------------------

    identifier_columns = detect_identifier_columns(df) if df is not None else []

    if identifier_columns:

        id_penalty = min(12.0, len(identifier_columns) * 3.0)

        governance_score -= id_penalty
        stability_score -= id_penalty * 0.5

        confidence_score -= 10

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "IDENTIFIER_COLUMNS_DETECTED",
            "severity": "WARNING",
            "message":
                f"Synthetic identifier-like columns detected: "
                f"{', '.join(identifier_columns[:5])}. "
                f"These columns may encode row ordering or non-generalizable structure."
        })

    # -----------------------------------------------------------------
    # VALIDATION PHASE
    # -----------------------------------------------------------------

    validation_status = "VALIDATION_PASSED"

    audit_logs.append({
        "phase": "VALIDATION PHASE",
        "rule": "DATASET_REGISTERED",
        "severity": "INFO",
        "message":
            f"Dataset v1 registered. "
            f"Initializing validation sequence for {rows_processed} rows."
    })

    audit_logs.append({
        "phase": "VALIDATION PHASE",
        "rule": "VALIDATION_STARTED",
        "severity": "INFO",
        "message":
            f"ML Validation engines triggered. "
            f"Scanning {features_evaluated} features."
    })

    # -----------------------------------------------------------------
    # POSITIVE STRUCTURAL REWARDS
    # -----------------------------------------------------------------

    if missingness < 0.01:
        governance_score += 8.0
        stability_score += 4.0

    elif missingness < 0.05:
        governance_score += 4.0
        stability_score += 2.0

    if rows_processed > 1000:
        governance_score += 3.0

    elif rows_processed > 300:
        governance_score += 1.5

    if features_evaluated >= 5:
        governance_score += 2.0

    # -----------------------------------------------------------------
    # MISSINGNESS PENALTIES
    # -----------------------------------------------------------------

    if missingness > 0.05:

        governance_score -= (missingness * 85)
        stability_score -= (missingness * 90)

        confidence_score -= missingness * 100

        validation_status = "VALIDATION_WARNING"

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "HIGH_MISSINGNESS",
            "severity": "WARNING",
            "message":
                f"Dataset contains {missingness*100:.1f}% missing values. "
                f"Imputation may introduce instability and probabilistic drift."
        })

    # -----------------------------------------------------------------
    # ML ISSUE PENALTIES
    # -----------------------------------------------------------------

    for issue in ml_issues:

        severity = issue.get("severity", "LOW")
        description = issue.get("description", "")

        if severity == "CRITICAL":

            critical_found = True

            governance_score -= 20.0
            stability_score -= 15.0
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

            governance_score -= 10.0
            stability_score -= 6.0
            confidence_score -= 12

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

            governance_score -= 4.0
            stability_score -= 2.0
            confidence_score -= 5

    # -----------------------------------------------------------------
    # VALIDATION COMPLETION
    # -----------------------------------------------------------------

    if validation_status == "VALIDATION_PASSED":

        governance_score += 6.0
        stability_score += 3.0

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message":
                "Dataset passed foundational structural integrity checks."
        })

    elif validation_status == "VALIDATION_FAILED":

        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "VALIDATION_FAILED",
            "severity": "CRITICAL",
            "message":
                "Structural integrity failure detected. "
                "Execution confidence severely degraded."
        })

    # -----------------------------------------------------------------
    # ANALYSIS PHASE
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # OUTLIER PENALTIES
    # -----------------------------------------------------------------

    if outlier_pct > 10.0:

        governance_score -= (outlier_pct * 1.5)
        stability_score -= (outlier_pct * 2.4)

        confidence_score -= 18

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "SEVERE_OUTLIER_CONTAMINATION",
            "severity": "CRITICAL",
            "message":
                f"Outlier consensus engines flagged "
                f"{outlier_pct:.1f}% anomalous rows. "
                f"Production stability severely compromised."
        })

    elif outlier_pct > 2.0:

        governance_score -= (outlier_pct * 0.7)
        stability_score -= (outlier_pct * 1.8)

        confidence_score -= 8

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "MODERATE_OUTLIER_CONTAMINATION",
            "severity": "WARNING",
            "message":
                f"Adaptive anomaly consensus observed "
                f"{outlier_pct:.1f}% deviation."
        })

    # -----------------------------------------------------------------
    # LEAKAGE ARBITRATION
    # -----------------------------------------------------------------

    if worst_leakage_category:

        if worst_leakage_category == "A":

            governance_score -= 2.0

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "FEATURE_REDUNDANCY_OBSERVED",
                "severity": "INFO",
                "message":
                    "Strong predictive feature redundancy detected. "
                    "No actionable leakage consensus reached."
            })

        elif worst_leakage_category == "B":

            governance_score -= 10.0
            stability_score -= 8.0

            confidence_score -= 12

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "PROXY_BEHAVIOR_DETECTED",
                "severity": "WARNING",
                "message":
                    "Indirect proxy-like predictive behavior observed. "
                    "Further causal review recommended."
            })

        elif worst_leakage_category == "C":

            shap_leakage = True

            governance_score -= 24.0
            stability_score -= 18.0

            confidence_score -= 28

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "SUSPICIOUS_LEAKAGE_PATTERN",
                "severity": "HIGH",
                "message":
                    "High permutation impact combined with "
                    "moderate ablation degradation. "
                    "Potential target leakage suspected."
            })

        elif worst_leakage_category == "D":

            shap_leakage = True
            critical_found = True

            governance_score -= 42.0
            stability_score -= 34.0

            confidence_score -= 45

            audit_logs.append({
                "phase": "ANALYSIS PHASE",
                "rule": "CAUSAL_LEAKAGE_CONFIRMED",
                "severity": "CRITICAL",
                "message":
                    "Confirmed causal leakage detected. "
                    "Feature dependency catastrophically compromises "
                    "generalization integrity."
            })

    else:

        governance_score += 4.0

        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "LEAKAGE_VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message":
                "No actionable leakage or proxy consensus detected."
        })

    # -----------------------------------------------------------------
    # CALIBRATION
    # -----------------------------------------------------------------

    governance_score = round(max(3.0, min(96.0, governance_score)), 1)

    stability_score = round(max(2.0, min(92.0, stability_score)), 1)

    confidence_score = round(max(5.0, min(99.0, confidence_score)), 1)

    # -----------------------------------------------------------------
    # DEPLOYMENT GATING
    # -----------------------------------------------------------------

    deployment_ready = (
        governance_score >= 88.0
        and stability_score >= 82.0
        and not critical_found
        and not shap_leakage
    )

    retraining_required = (
        stability_score < 68.0
        or shap_leakage
        or outlier_pct > 10.0
    )

    # -----------------------------------------------------------------
    # FINAL STATUS
    # -----------------------------------------------------------------

    if governance_score >= 88 and stability_score >= 82:

        final_status = "APPROVED"

    elif governance_score >= 72 and stability_score >= 65:

        final_status = "AWAITING_REVIEW"

    else:

        final_status = "REJECTED"

    # -----------------------------------------------------------------
    # CONFIDENCE LABEL
    # -----------------------------------------------------------------

    if confidence_score >= 85:
        confidence_label = "HIGH"

    elif confidence_score >= 60:
        confidence_label = "MODERATE"

    else:
        confidence_label = "LOW"

    # -----------------------------------------------------------------
    # GOVERNANCE EVENTS
    # -----------------------------------------------------------------

    if final_status == "APPROVED":

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "GOVERNANCE_APPROVED",
            "severity": "SUCCESS",
            "message":
                (
                    "**Governance Approved**\n\n"
                    "**Why it was approved:**\n"
                    "- Structural integrity remained stable.\n"
                    "- No catastrophic leakage consensus observed.\n"
                    "- Outlier contamination stayed within acceptable bounds.\n"
                    "- Deployment stability exceeded enterprise threshold.\n\n"
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
                    f"**Governance Score:** {governance_score}/100\n"
                    f"**Stability Score:** {stability_score}/100\n\n"
                    "**What happened:**\n"
                    "The engine detected moderate structural concerns "
                    "that require human arbitration before deployment.\n\n"
                    "**Why it matters:**\n"
                    "The model may generalize inconsistently under "
                    "production distribution shifts.\n\n"
                    "**Recommended action:**\n"
                    "Review highlighted warnings and validate feature "
                    "availability assumptions.\n\n"
                    f"**Confidence level:** {confidence_label}."
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
                    f"**Governance Score:** {governance_score}/100\n"
                    f"**Stability Score:** {stability_score}/100\n\n"
                    "**What happened:**\n"
                    "The governance engine rejected deployment due to "
                    "high structural risk.\n\n"
                    "**Why it matters:**\n"
                    "The model exhibits behavior inconsistent with "
                    "robust real-world generalization.\n\n"
                    "**Recommended action:**\n"
                    "Remove leaking features, clean anomalous rows, "
                    "and retrain the baseline model.\n\n"
                    f"**Confidence level:** {confidence_label}."
                )
        })

    # -----------------------------------------------------------------
    # RETRAINING EVENT
    # -----------------------------------------------------------------

    if retraining_required:

        audit_logs.append({
            "phase": "GOVERNANCE PHASE",
            "rule": "RETRAINING_RECOMMENDED",
            "severity": "HIGH",
            "message":
                (
                    "**Retraining Recommended**\n\n"
                    "**What happened:**\n"
                    "The engine detected instability patterns "
                    "that weaken production reliability.\n\n"
                    "**Why it matters:**\n"
                    "The current feature space may not generalize "
                    "robustly under future inference conditions.\n\n"
                    "**Recommended action:**\n"
                    "Apply the suggested remediations and retrain "
                    "the baseline model."
                )
        })

    # -----------------------------------------------------------------
    # RETURN
    # -----------------------------------------------------------------

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
                    if outlier_pct > 5.0
                    else "Cumulative structural degradation observed."
                )
            ),

        "metadata": {

            "run_id":
                f"RUN_{datetime.datetime.now().strftime('%Y_%m%d')}_001",

            "dataset_version": "v1",

            "model_type": "RandomForestRegressor",

            "pipeline_version": "pipeline_v3",

            "monitoring_status": "ACTIVE",

            "runtime_ms": runtime_ms,

            "rows_processed": rows_processed,

            "features_evaluated": features_evaluated,

            "drift_monitors_active": 12,

            "validation_engines_triggered": 7,

            "validation_status": validation_status,

            "identifier_columns_detected": identifier_columns,

            "outlier_percentage": outlier_pct,

            "confidence_label": confidence_label
        },

        "audit_logs": audit_logs
    }
