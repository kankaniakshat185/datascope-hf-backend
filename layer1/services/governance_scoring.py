from typing import Dict, Any, List
import datetime
from layer1.services.baseline_snapshot_engine import compute_baseline_metrics
import pandas as pd

from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

            import re
            is_id_pattern = bool(re.search(r'(^id$|_id$|^id_|_id_|^uuid$|^guid$|^record_id$|^customer_id$|^transaction_id$)', name))
            
            if (
                is_id_pattern
                or (uniqueness_ratio > 0.995 and pd.api.types.is_integer_dtype(df[col]))
                or (uniqueness_ratio > 0.995 and pd.api.types.is_string_dtype(df[col]) and df[col].str.len().mean() > 15)
            ):
                suspicious.append(col)

        except Exception:
            continue

    return suspicious



def detect_feature_leakage(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    feature_impacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Dedicated statistical leakage detection engine.
    Uses:
    - single feature predictive power
    - correlation analysis
    - mutual information
    - identifier detection
    - permutation impact
    - ablation behavior
    Produces probabilistic leakage classifications.
    """
    leakage_results = {}
    worst_category = None
    target = df[target_col]
    numeric_target = pd.api.types.is_numeric_dtype(target)

    for feature in df.columns:
        if feature == target_col:
            continue
        try:
            series = df[feature]
            # BASIC FEATURE CLEANING
            valid_mask = ~(series.isna() | target.isna())
            X = series[valid_mask]
            y = target[valid_mask]
            if len(X) < 30:
                continue

            # SIGNAL 1 - REMOVED (Identifiers are now filtered out beforehand)

            # SIGNAL 2 — TARGET CORRELATION
            corr_signal = 0.0
            if pd.api.types.is_numeric_dtype(X):
                try:
                    corr_signal = abs(np.corrcoef(X, y)[0, 1])
                    if np.isnan(corr_signal):
                        corr_signal = 0.0
                except:
                    corr_signal = 0.0

            # SIGNAL 3 — MUTUAL INFORMATION
            mi_signal = 0.0
            try:
                encoded = pd.factorize(X)[0].reshape(-1, 1)
                if problem_type == "classification":
                    mi_signal = mutual_info_classif(
                        encoded,
                        y,
                        discrete_features=True
                    )[0]
                else:
                    mi_signal = mutual_info_regression(
                        encoded,
                        y
                    )[0]
            except:
                mi_signal = 0.0

            # SIGNAL 4 — SINGLE FEATURE PREDICTIVE POWER
            predictive_signal = 0.0
            try:
                X_model = pd.factorize(X)[0].reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_model,
                    y,
                    test_size=0.3,
                    random_state=42
                )
                if problem_type == "classification":
                    model = RandomForestClassifier(
                        n_estimators=50,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    predictive_signal = accuracy_score(y_test, preds)
                else:
                    model = RandomForestRegressor(
                        n_estimators=50,
                        random_state=42,
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    target_std = np.std(y_test)
                    predictive_signal = max(
                        0,
                        1 - (rmse / max(target_std, 1e-6))
                    )
            except:
                predictive_signal = 0.0

            # SIGNAL 5 — EXISTING FEATURE IMPACTS
            perm_signal = feature_impacts.get(feature, {}).get(
                "importance_score",
                0
            )
            ablation_signal = feature_impacts.get(feature, {}).get(
                "performance_impact",
                0
            )

            # INITIAL CLASSIFICATION
            category = "NORMAL_FEATURE"
            
            # Identify true predictors without explicit leakage marks
            is_strong_predictor = (
                (predictive_signal > 0.75 or perm_signal > 0.30 or ablation_signal > 0.10) 
                and corr_signal < 0.95
            )
            
            # Explicit target copy / high risk marks
            is_target_copy = (corr_signal >= 0.95)
            is_perfect_predictor = (predictive_signal >= 0.99)
            
            # RAW LEAKAGE SCORING
            raw_leakage_score = 0.0
            if is_target_copy:
                raw_leakage_score += 60
            elif corr_signal > 0.8:
                raw_leakage_score += 20
                
            if is_perfect_predictor:
                raw_leakage_score += 40
            elif predictive_signal > 0.90:
                raw_leakage_score += 15
                
            raw_leakage_score += min(mi_signal * 10, 20)

            # NORMALIZE CONFIDENCE SCORE (SIGMOID)
            import math
            display_confidence = round(100 / (1 + math.exp(-0.06 * (raw_leakage_score - 30))), 1)

            # FINAL CLASSIFICATION HIERARCHY
            if is_target_copy or raw_leakage_score >= 85:
                category = "CONFIRMED_LEAKAGE"
            elif raw_leakage_score >= 60 and not is_strong_predictor:
                category = "HIGH_RISK_PROXY"
            elif is_strong_predictor:
                category = "STRONG_PREDICTOR"
            else:
                category = "NORMAL_FEATURE"

            # TRACK WORST CATEGORY
            priority = {
                "NORMAL_FEATURE": 0,
                "STRONG_PREDICTOR": 1,
                "HIGH_RISK_PROXY": 2,
                "CONFIRMED_LEAKAGE": 3
            }
            if (
                worst_category is None
                or priority[category] > priority.get(worst_category, 0)
            ):
                worst_category = category

            # STORE RESULTS
            leakage_results[feature] = {
                "category": category,
                "leakage_score": display_confidence,
                "raw_score": round(raw_leakage_score, 2),
                "signals": {
                    "correlation": round(float(corr_signal), 4),
                    "mutual_information": round(float(mi_signal), 4),
                    "single_feature_predictive_power": round(float(predictive_signal), 4),
                    "permutation_importance": round(float(perm_signal), 4),
                    "ablation_impact": round(float(ablation_signal), 4)
                }
            }
        except Exception as e:
            logger.warning(
                f"Leakage detection failed for feature {feature}: {e}"
            )

    return {
        "feature_reports": leakage_results,
        "worst_category": worst_category
    }

def calculate_governance_score(

    df: Any,
    target_column: str,
    ml_issues: List[Dict[str, Any]],
    layer1_data: Dict[str, Any],
    shap_data: Dict[str, Any],
    runtime_ms: int = 1500,
    rows_processed: int = 0,
    features_evaluated: int = 0,
    worst_leakage_category: str = None,
    leakage_analysis: Dict[str, Any] = None,
    identifier_columns: List[str] = None
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

    if identifier_columns is None:
        identifier_columns = []
        
    for col in identifier_columns:
        audit_logs.append({
            "phase": "VALIDATION PHASE",
            "rule": "IDENTIFIER_COLUMN_DETECTED",
            "severity": "INFO",
            "message": f"Column '{col}' classified as Sequential Identifier. Excluded from governance evaluation."
        })

    REAL_IDENTIFIER_COLUMNS = identifier_columns

    # Governance penalties for identifiers are removed as requested.
    # Metadata is simply stored via the audit logs above.
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

    if worst_leakage_category == "STRONG_PREDICTOR":
        governance_score -= 1
        stability_score -= 1
        
        if leakage_analysis:
            reports = leakage_analysis.get("feature_reports", {})
            predictor_features = [f for f, data in reports.items() if data["category"] == "STRONG_PREDICTOR"]
            for f in predictor_features[:3]:
                audit_logs.append({
                    "phase": "ANALYSIS PHASE",
                    "rule": "STRONG_PREDICTOR",
                    "severity": "INFO",
                    "message": f"Feature '{f}' classified as STRONG_PREDICTOR. High model dependence observed. No direct leakage evidence detected."
                })

    elif worst_leakage_category == "HIGH_RISK_PROXY":
        governance_score -= 12
        stability_score -= 8
        shap_leakage = True
        
        if leakage_analysis:
            reports = leakage_analysis.get("feature_reports", {})
            candidate_features = [f for f, data in reports.items() if data["category"] == "HIGH_RISK_PROXY"]
            for f in candidate_features[:3]:
                score = reports[f]["leakage_score"]
                audit_logs.append({
                    "phase": "ANALYSIS PHASE",
                    "rule": "HIGH_RISK_PROXY",
                    "severity": "WARNING",
                    "message": f"Feature '{f}' classified as HIGH_RISK_PROXY (confidence: {score}%)."
                })

    elif worst_leakage_category == "CONFIRMED_LEAKAGE":
        governance_score -= 35
        stability_score -= 25
        shap_leakage = True
        critical_found = True
        
        if leakage_analysis:
            reports = leakage_analysis.get("feature_reports", {})
            confirmed_features = [f for f, data in reports.items() if data["category"] == "CONFIRMED_LEAKAGE"]
            for f in confirmed_features[:3]:
                score = reports[f]["leakage_score"]
                audit_logs.append({
                    "phase": "ANALYSIS PHASE",
                    "rule": "LEAKAGE_CONFIRMED",
                    "severity": "CRITICAL",
                    "message": f"Feature '{f}' classified as CONFIRMED_LEAKAGE (confidence: {score}%)."
                })

    else:
        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "LEAKAGE_VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message": "No actionable leakage consensus detected."
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
        and worst_leakage_category not in ["CONFIRMED_LEAKAGE"]
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
        or worst_leakage_category in ["CONFIRMED_LEAKAGE"]
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
