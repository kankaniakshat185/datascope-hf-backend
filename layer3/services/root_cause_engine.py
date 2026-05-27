import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)

class RootCauseEngine:
    """
    Root Cause Analysis Engine
    Identifies high-error prediction regions and performs causal divergence analysis
    to explain why a model is failing in specific segments.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _compute_residuals(self, df: pd.DataFrame, target_col: str, model, is_regression: bool) -> np.ndarray:
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y_true = df[target_col].values
        
        # We assume the model is already trained and passed in.
        y_pred = model.predict(X)
        
        if is_regression:
            return np.abs(y_true - y_pred)
        else:
            # For classification, we use 0 for correct, 1 for incorrect as a binary residual.
            # Alternatively, predict_proba can be used for continuous confidence residuals.
            return (y_true != y_pred).astype(float)

    def _segment_errors(self, residuals: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Clusters rows into low, medium, and high error buckets based on residuals."""
        if len(residuals) < n_clusters:
            return np.zeros_like(residuals)
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(residuals.reshape(-1, 1))
        
        # Sort cluster centers so 0 is low error, 2 is high error
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        
        return np.array([mapping[c] for c in clusters])

    def _compare_distributions(self, feature_data: pd.Series, low_error_mask: np.ndarray, high_error_mask: np.ndarray) -> Dict[str, Any]:
        """Compares feature distribution between low error and high error segments."""
        low_dist = feature_data[low_error_mask].dropna()
        high_dist = feature_data[high_error_mask].dropna()
        
        if len(low_dist) == 0 or len(high_dist) == 0:
            return {"divergence": 0.0, "p_value": 1.0, "drift_type": "insufficient_data"}
            
        # Kolmogorov-Smirnov test for distribution divergence
        stat, p_value = ks_2samp(low_dist, high_dist)
        
        # Calculate shifts
        mean_shift = high_dist.mean() - low_dist.mean()
        var_shift = high_dist.var() - low_dist.var()
        
        drift_type = "stable"
        if stat > 0.2 and p_value < 0.05:
            if abs(mean_shift) > low_dist.std():
                drift_type = "mean_shift"
            elif var_shift > low_dist.var():
                drift_type = "variance_inflation"
            else:
                drift_type = "distribution_shape_change"
                
        return {
            "divergence": float(stat),
            "p_value": float(p_value),
            "mean_shift": float(mean_shift),
            "drift_type": drift_type
        }

    def analyze(self, df: pd.DataFrame, target_col: str, model, is_regression: bool = True) -> Dict[str, Any]:
        logger.info("Starting Root Cause Analysis...")
        df = df.dropna(subset=[target_col]).copy()
        
        if len(df) < 50:
            return {"error": "Dataset too small for root cause analysis. Minimum 50 rows required."}
            
        numeric_features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        if numeric_features.empty:
            return {"error": "No numeric features available for analysis."}
            
        residuals = self._compute_residuals(df, target_col, model, is_regression)
        error_clusters = self._segment_errors(residuals)
        
        low_error_mask = error_clusters == 0
        high_error_mask = error_clusters == 2 # 2 is max error cluster
        
        if sum(high_error_mask) == 0 or sum(low_error_mask) == 0:
            return {"error": "Could not identify distinct error segments."}
            
        feature_divergence = {}
        for col in numeric_features.columns:
            div_stats = self._compare_distributions(numeric_features[col], low_error_mask, high_error_mask)
            feature_divergence[col] = div_stats
            
        # Rank causes
        ranked_causes = sorted(
            [(f, stats) for f, stats in feature_divergence.items() if stats["p_value"] < 0.05],
            key=lambda x: x[1]["divergence"],
            reverse=True
        )
        
        causes_output = []
        for feature, stats in ranked_causes:
            impact = min(1.0, stats["divergence"] * 2) # scale up slightly for impact score
            reason = f"Significant {stats['drift_type'].replace('_', ' ')} detected in high-error predictions."
            causes_output.append({
                "feature": feature,
                "reason": reason,
                "impact_score": round(impact, 4),
                "divergence": round(stats["divergence"], 4)
            })
            
        return {
            "status": "success",
            "high_error_ratio": float(sum(high_error_mask) / len(df)),
            "ranked_causes": causes_output,
            "error_distribution": {
                "low_error_mean_residual": float(residuals[low_error_mask].mean()),
                "high_error_mean_residual": float(residuals[high_error_mask].mean())
            }
        }
