import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, accuracy_score
from typing import Dict, Any, Tuple
import logging
from layer1.services.governance_scoring import detect_identifier_columns

logger = logging.getLogger(__name__)

def train_baseline_model(df: pd.DataFrame, target_col: str, problem_type: str):
    """Trains a baseline Random Forest model."""
    identifiers_dict = detect_identifier_columns(df)
    all_identifiers = identifiers_dict["IDENTIFIER_COLUMN_DETECTED"] + identifiers_dict["POTENTIAL_IDENTIFIER"] + identifiers_dict["HIGH_CARDINALITY_TEXT"]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    X = X.drop(columns=all_identifiers, errors="ignore")
    y = df[target_col]
    
    if problem_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        baseline_score = r2_score(y, model.predict(X))
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        baseline_score = accuracy_score(y, model.predict(X))
        
    return model, X, y, baseline_score

from sklearn.model_selection import train_test_split

def compute_feature_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str
) -> Tuple[Dict[str, float], float]:
    """
    Measures true feature importance by removing one feature at a time,
    retraining the model, and evaluating ONLY on unseen test data.

    WHY THIS EXISTS:
    Training and evaluating on the same dataset creates severe evaluation leakage.
    The model memorizes patterns and falsely appears stable even when critical
    features are removed.

    This implementation fixes that by:
    - splitting train/test data,
    - training on train data,
    - evaluating on unseen test data only.

    This produces realistic ablation behavior.
    """

    ablation_scores = {}

    # ---------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # Train baseline model
    # ---------------------------------------------------
    if problem_type == 'regression':

        baseline_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        baseline_model.fit(X_train, y_train)

        baseline_preds = baseline_model.predict(X_test)

        baseline_score = r2_score(y_test, baseline_preds)

    else:

        baseline_model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        baseline_model.fit(X_train, y_train)

        baseline_preds = baseline_model.predict(X_test)

        baseline_score = accuracy_score(y_test, baseline_preds)

    # ---------------------------------------------------
    # Feature Ablation Loop
    # ---------------------------------------------------
    for feature in X.columns:

        X_train_ab = X_train.drop(columns=[feature])
        X_test_ab = X_test.drop(columns=[feature])

        if problem_type == 'regression':

            model = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_ab, y_train)

            preds = model.predict(X_test_ab)

            score = r2_score(y_test, preds)

        else:

            model = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_ab, y_train)

            preds = model.predict(X_test_ab)

            score = accuracy_score(y_test, preds)

        # Positive drop means feature mattered
        drop = baseline_score - score

        # Prevent negative importance noise
        ablation_scores[feature] = max(0, float(drop))

    return ablation_scores, float(baseline_score)

def compute_causal_impact(df: pd.DataFrame, target_col: str, problem_type: str = 'regression') -> Tuple[Dict[str, Any], list]:
    """
    Computes Feature Ablation, Permutation Importance, and variance explained via PDP.
    Returns the feature impacts and generated insights.
    """
    logger.info("Training baseline model for Causal Impact...")
    # Drop rows with NaN in target or completely empty rows
    df = df.dropna(subset=[target_col])
    
    if len(df) < 20 or len(df.select_dtypes(include=[np.number]).columns) < 2:
        return {}, ["Dataset too small or lacks numeric features for causal analysis."]

    model, X, y, _ = train_baseline_model(df, target_col, problem_type)
    
    # 1. Feature Ablation
    logger.info("Computing Feature Ablation using unseen evaluation data...")
    ablation_scores, baseline_score = compute_feature_ablation(
        X,
        y,
        problem_type
    )

    # 2. Permutation Importance
    logger.info("Computing Permutation Importance...")
    scoring_metric = 'r2' if problem_type == 'regression' else 'accuracy'
    
    # ---------------------------------------------------
    # Create unseen evaluation split
    # ---------------------------------------------------
    # IMPORTANT:
    # Permutation importance must be evaluated on unseen data.
    # Using training data causes inflated importance values and
    # unrealistic governance conclusions.

    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Retrain clean evaluation model
    if problem_type == 'regression':

        eval_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    else:

        eval_model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    eval_model.fit(X_train_eval, y_train_eval)

    perm_importance = permutation_importance(
        eval_model,
        X_test_eval,
        y_test_eval,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring=scoring_metric
    )
    perm_scores = {feat: max(0, score) for feat, score in zip(X.columns, perm_importance.importances_mean)}
    
    # 3. Partial Dependence (Marginal Effect Strength)
    # We estimate how much the feature influences the model's marginal prediction.
    logger.info("Computing Partial Dependence Variance...")
    marginal_effect_strength = {}
    pdp_data = {}
    for i, feature in enumerate(X.columns):
        # Calculate PDP
        try:
            pd_results = partial_dependence(model, X, [i], kind='average')
            # For multi-class, pd_results['average'] has shape (n_classes, n_values)
            # For regression / binary, it has shape (1, n_values)
            avg_pd = pd_results['average'][0]
            grid_vals = pd_results['grid_values'][0]
            # Measures how strongly the model's prediction changes
            # as this feature changes across its range.
            #
            # NOTE:
            # This is NOT true statistical variance explained.
            # It is only a proxy for marginal model sensitivity.
            var_exp = np.var(avg_pd)
            marginal_effect_strength[feature] = float(var_exp)
            pdp_data[feature] = {
                "x": grid_vals.tolist(),
                "y": avg_pd.tolist()
            }
        except Exception as e:
            logger.warning(f"Failed to compute PDP for {feature}: {e}")
            marginal_effect_strength[feature] = 0.0
            pdp_data[feature] = {"x": [], "y": []}
            
    # Combine everything into FeatureImpact schema format
    results = {}
    for feature in X.columns:
        results[feature] = {
            "importance_score": float(perm_scores.get(feature, 0.0)),
            "performance_impact": float(ablation_scores.get(feature, 0.0)),
            "marginal_effect_strength": float(marginal_effect_strength.get(feature, 0.0)),
            "pdp_x": pdp_data.get(feature, {}).get("x", []),
            "pdp_y": pdp_data.get(feature, {}).get("y", [])
        }
        
    # Sort features by importance score
    sorted_features = sorted(results.items(), key=lambda x: x[1]['importance_score'], reverse=True)
    
    # Generate insights
    insights = []
    
    # Add deterministic, educational interpretations based on permutations and ablations
    for feature, metrics in sorted_features[:5]:  # Analyze top 5 features
        perm = metrics['importance_score']
        abl = metrics['performance_impact']
        pdp_var = metrics['marginal_effect_strength']
        
        # ---------------------------------------------------
        # Stabilized percentage scaling
        # ---------------------------------------------------
        # Raw percentage scaling can explode when baseline
        # scores are very small.
        #
        # Example:
        # baseline = 0.02
        # impact = 0.03
        #
        # produces absurd 150%+ outputs.
        #
        # We use tanh normalization to create bounded,
        # interpretable governance percentages.

        safe_baseline = max(abs(baseline_score), 0.15)

        normalized_ratio = (
            abl / safe_baseline
        )

        drop_pct = float(
            np.tanh(normalized_ratio) * 100
        )
        
        # Determine explanation based on matrix
        if perm > 0.05 and abl < 0.01:
            insight_text = (
                f"**What happened:** '{feature}' showed high permutation importance but low ablation impact.\n\n"
                f"**Why it matters:** This indicates possible feature redundancy or multicollinearity. The model relies on it, but if removed, another feature seamlessly takes its place.\n\n"
                f"**Recommended action:** Investigate correlation with other top features and consider removing redundant ones to simplify the model."
            )
        elif perm > 0.05 and abl >= 0.01:
            insight_text = (
                f"**What happened:** '{feature}' is structurally critical to model behavior. Removing it causes a {drop_pct:.1f}% drop in performance.\n\n"
                f"**Why it matters:** This feature is driving the model's decisions. If this feature is a proxy for the target (data leakage), the model will fail in production.\n\n"
                f"**Recommended action:** Verify this feature is logically available at the time of prediction. If it is, ensure its data quality is strictly monitored."
            )
        elif perm < 0.01 and abl < 0.01:
            if pdp_var > 0.1:
                insight_text = (
                    f"**What happened:** Model predictions are highly sensitive to changes in '{feature}' (High PDP variability), despite low overall importance.\n\n"
                    f"**Why it matters:** The feature might only affect a very specific, small sub-population of your data, or it might be interacting with other features in complex ways.\n\n"
                    f"**Recommended action:** Review partial dependence plots for this feature to see exactly which ranges trigger drastic prediction changes."
                )
            else:
                insight_text = (
                    f"**What happened:** '{feature}' has weak influence on model predictions (low permutation and ablation).\n\n"
                    f"**Why it matters:** The feature provides almost no predictive signal.\n\n"
                    f"**Recommended action:** Consider dropping this feature to improve training speed and reduce maintenance overhead."
                )
        else:
            insight_text = (
                f"**What happened:** '{feature}' provides moderate predictive value.\n\n"
                f"**Why it matters:** It contributes to the model's accuracy but is not the primary driver.\n\n"
                f"**Recommended action:** Keep the feature unless you are aggressively pruning for simplicity."
            )
            
        insights.append(insight_text)
            
    return results, insights
