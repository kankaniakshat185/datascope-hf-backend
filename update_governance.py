import re

with open("layer1/services/governance_scoring.py", "r") as f:
    content = f.read()

# 1. Add imports
imports = """
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import logging

logger = logging.getLogger(__name__)
"""
content = re.sub(r'import pandas as pd', 'import pandas as pd\n' + imports, content, count=1)

# 2. Add detect_feature_leakage
new_func = """
def detect_feature_leakage(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    feature_impacts: Dict[str, Any]
) -> Dict[str, Any]:
    \"\"\"
    Dedicated statistical leakage detection engine.
    Uses:
    - single feature predictive power
    - correlation analysis
    - mutual information
    - identifier detection
    - permutation impact
    - ablation behavior
    Produces probabilistic leakage classifications.
    \"\"\"
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

            # SIGNAL 1 — IDENTIFIER DETECTION
            unique_ratio = X.nunique() / max(1, len(X))
            is_identifier = (
                unique_ratio > 0.97
                or feature.lower().endswith("id")
                or feature.lower().startswith("id")
                or "uuid" in feature.lower()
            )

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

            # LEAKAGE CONFIDENCE SCORING
            leakage_score = 0.0
            leakage_score += corr_signal * 30
            leakage_score += predictive_signal * 40
            leakage_score += min(mi_signal * 10, 20)

            if perm_signal > 0.35:
                leakage_score += 10
            if ablation_signal > 0.25:
                leakage_score += 10
            if is_identifier:
                leakage_score += 20

            leakage_score = min(leakage_score, 100)

            # FINAL CLASSIFICATION
            if leakage_score >= 85:
                category = "CONFIRMED_LEAKAGE"
            elif leakage_score >= 65:
                category = "HIGH_RISK"
            elif leakage_score >= 45:
                category = "SUSPICIOUS"
            else:
                category = "SAFE"

            # STEP 6 - BLOCK IDENTIFIER FEATURES
            if is_identifier and predictive_signal > 0.8:
                category = "CONFIRMED_LEAKAGE"

            # TRACK WORST CATEGORY
            priority = {
                "SAFE": 0,
                "SUSPICIOUS": 1,
                "HIGH_RISK": 2,
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
                "leakage_score": round(leakage_score, 2),
                "signals": {
                    "identifier": bool(is_identifier),
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
"""
content = re.sub(r'def calculate_governance_score\(', new_func, content, count=1)

# Modify signature of calculate_governance_score
content = re.sub(r'worst_leakage_category: str = None\n\)', 'worst_leakage_category: str = None,\n    leakage_analysis: Dict[str, Any] = None\n)', content)

# 3. Update category checks
old_leakage_checks = r'if worst_leakage_category:.*?(?=# =========================================================\n\s*# FINAL CALIBRATION)'
new_leakage_checks = """if worst_leakage_category == "SUSPICIOUS":
        governance_score -= 8
        stability_score -= 4
        
        if leakage_analysis:
            reports = leakage_analysis.get("feature_reports", {})
            suspicious_features = [f for f, data in reports.items() if data["category"] == "SUSPICIOUS"]
            for f in suspicious_features[:3]:
                score = reports[f]["leakage_score"]
                audit_logs.append({
                    "phase": "ANALYSIS PHASE",
                    "rule": "LEAKAGE_SUSPICIOUS",
                    "severity": "WARNING",
                    "message": f"Feature '{f}' classified as SUSPICIOUS leakage candidate (confidence: {score})."
                })

    elif worst_leakage_category == "HIGH_RISK":
        governance_score -= 22
        stability_score -= 14
        shap_leakage = True
        
        if leakage_analysis:
            reports = leakage_analysis.get("feature_reports", {})
            high_risk_features = [f for f, data in reports.items() if data["category"] == "HIGH_RISK"]
            for f in high_risk_features[:3]:
                score = reports[f]["leakage_score"]
                audit_logs.append({
                    "phase": "ANALYSIS PHASE",
                    "rule": "LEAKAGE_HIGH_RISK",
                    "severity": "CRITICAL",
                    "message": f"Feature '{f}' classified as HIGH_RISK leakage candidate (confidence: {score})."
                })

    elif worst_leakage_category == "CONFIRMED_LEAKAGE":
        governance_score -= 42
        stability_score -= 30
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
                    "message": f"Feature '{f}' classified as CONFIRMED_LEAKAGE candidate (confidence: {score})."
                })

    else:
        audit_logs.append({
            "phase": "ANALYSIS PHASE",
            "rule": "LEAKAGE_VALIDATION_PASSED",
            "severity": "SUCCESS",
            "message": "No actionable leakage consensus detected."
        })

    """
content = re.sub(old_leakage_checks, new_leakage_checks, content, flags=re.DOTALL)

# Update deployment gating
content = re.sub(r'worst_leakage_category != "D"', 'worst_leakage_category not in ["HIGH_RISK", "CONFIRMED_LEAKAGE"]', content)
content = re.sub(r'worst_leakage_category == "D"', 'worst_leakage_category in ["HIGH_RISK", "CONFIRMED_LEAKAGE"]', content)

with open("layer1/services/governance_scoring.py", "w") as f:
    f.write(content)
