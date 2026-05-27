import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, accuracy_score
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def train_baseline_model(df: pd.DataFrame, target_col: str, problem_type: str):
    """Trains a baseline Random Forest model."""
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
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

def compute_feature_ablation(X: pd.DataFrame, y: pd.Series, target_col: str, problem_type: str, baseline_score: float) -> Dict[str, float]:
    """Measures performance drop by removing one feature at a time and retraining."""
    ablation_scores = {}
    
    for feature in X.columns:
        X_ablated = X.drop(columns=[feature])
        
        if problem_type == 'regression':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_ablated, y)
            score = r2_score(y, model.predict(X_ablated))
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_ablated, y)
            score = accuracy_score(y, model.predict(X_ablated))
            
        drop = baseline_score - score
        ablation_scores[feature] = max(0, drop)  # Drop should ideally be positive if feature is important
        
    return ablation_scores

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

    model, X, y, baseline_score = train_baseline_model(df, target_col, problem_type)
    
    # 1. Permutation Importance
    logger.info("Computing Permutation Importance...")
    scoring_metric = 'r2' if problem_type == 'regression' else 'accuracy'
    perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1, scoring=scoring_metric)
    perm_scores = {feat: max(0, score) for feat, score in zip(X.columns, perm_importance.importances_mean)}
    
    # 2. Feature Ablation
    logger.info("Computing Feature Ablation...")
    ablation_scores = compute_feature_ablation(X, y, target_col, problem_type, baseline_score)
    
    # 3. Partial Dependence (Variance Explained)
    # We estimate how much the feature influences the model's marginal prediction.
    logger.info("Computing Partial Dependence Variance...")
    variance_explained = {}
    pdp_data = {}
    for i, feature in enumerate(X.columns):
        # Calculate PDP
        try:
            pd_results = partial_dependence(model, X, [i], kind='average')
            # For multi-class, pd_results['average'] has shape (n_classes, n_values)
            # For regression / binary, it has shape (1, n_values)
            avg_pd = pd_results['average'][0]
            grid_vals = pd_results['grid_values'][0]
            # Variance of the partial dependence line
            var_exp = np.var(avg_pd)
            variance_explained[feature] = float(var_exp)
            pdp_data[feature] = {
                "x": grid_vals.tolist(),
                "y": avg_pd.tolist()
            }
        except Exception as e:
            logger.warning(f"Failed to compute PDP for {feature}: {e}")
            variance_explained[feature] = 0.0
            pdp_data[feature] = {"x": [], "y": []}
            
    # Combine everything into FeatureImpact schema format
    results = {}
    for feature in X.columns:
        results[feature] = {
            "importance_score": float(perm_scores.get(feature, 0.0)),
            "performance_impact": float(ablation_scores.get(feature, 0.0)),
            "variance_explained": float(variance_explained.get(feature, 0.0)),
            "pdp_x": pdp_data.get(feature, {}).get("x", []),
            "pdp_y": pdp_data.get(feature, {}).get("y", [])
        }
        
    # Sort features by importance score
    sorted_features = sorted(results.items(), key=lambda x: x[1]['importance_score'], reverse=True)
    
    # Generate insights
    insights = []
    if sorted_features:
        top_feat, top_metrics = sorted_features[0]
        drop_pct = (top_metrics['performance_impact'] / max(0.0001, baseline_score)) * 100
        insights.append(f"Feature '{top_feat}' is the most critical driver. Removing it decreases performance by {drop_pct:.1f}%.")
        
        if len(sorted_features) > 1:
            second_feat, sec_metrics = sorted_features[1]
            insights.append(f"'{second_feat}' is the second most important feature, explaining significant model variance.")
            
    return results, insights
