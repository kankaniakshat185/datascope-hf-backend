import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.cluster import MiniBatchKMeans
import logging

logger = logging.getLogger(__name__)

class FailureAnalysisEngine:
    """
    'Why Did My Model Fail?' Engine.
    Clusters rows with high errors and extracts descriptive rules for failure regions.
    Generates deterministic remediation recommendations.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _extract_pattern(self, cluster_data: pd.DataFrame, full_data: pd.DataFrame) -> str:
        """Finds the most distinctive feature range defining this cluster."""
        patterns = []
        for col in cluster_data.columns:
            cluster_mean = cluster_data[col].mean()
            full_mean = full_data[col].mean()
            full_std = full_data[col].std()
            
            if full_std == 0:
                continue
                
            z_score = abs(cluster_mean - full_mean) / full_std
            if z_score > 1.5:  # significant deviation
                direction = ">" if cluster_mean > full_mean else "<"
                threshold = round(full_mean + (1.0 if direction == ">" else -1.0) * full_std, 2)
                patterns.append(f"{col} {direction} {threshold}")
                
        if not patterns:
            return "Mixed multi-dimensional anomalies"
            
        return " and ".join(patterns[:2]) # Keep it readable (max 2 conditions)

    def _generate_recommendations(self, pattern: str, cluster_data: pd.DataFrame) -> List[str]:
        remediations = []
        if ">" in pattern or "<" in pattern:
            remediations.append("Consider capping extreme values using IQR or Z-Score filtering.")
            
        # Check for sparse categories or high variance
        if cluster_data.var().mean() > 1000:
            remediations.append("Apply Log or Power Transformations to stabilize high variance regions.")
            
        remediations.append("Retrain with a weighted loss function giving higher penalty to these edge cases.")
        return remediations

    def analyze(self, df: pd.DataFrame, residuals: np.ndarray, threshold_quantile: float = 0.8) -> Dict[str, Any]:
        logger.info("Starting Failure Analysis...")
        numeric_features = df.select_dtypes(include=[np.number])
        
        # Identify failure region (high residuals)
        threshold = np.quantile(residuals, threshold_quantile)
        failure_mask = residuals >= threshold
        failure_data = numeric_features[failure_mask]
        
        if len(failure_data) < 10:
            return {"status": "insufficient_failures", "message": "Not enough failures to cluster reliably."}
            
        # Cluster the failures to find distinct modes of failure
        n_clusters = min(3, len(failure_data) // 10)
        n_clusters = max(1, n_clusters)
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
        failure_clusters = kmeans.fit_predict(failure_data)
        
        failure_profiles = []
        for c in range(n_clusters):
            c_mask = failure_clusters == c
            c_data = failure_data[c_mask]
            
            pattern = self._extract_pattern(c_data, numeric_features)
            recs = self._generate_recommendations(pattern, c_data)
            
            # Confidence based on cluster purity/size
            confidence = min(0.99, len(c_data) / len(failure_data) + 0.3)
            
            severity = "high" if len(c_data) > len(df) * 0.05 else "medium"
            
            failure_profiles.append({
                "cluster_id": c,
                "failure_pattern": f"Rows with {pattern} show elevated prediction instability.",
                "affected_rows": len(c_data),
                "severity": severity,
                "confidence": round(confidence, 2),
                "recommendations": recs
            })
            
        return {
            "status": "success",
            "total_failures_analyzed": len(failure_data),
            "failure_profiles": failure_profiles
        }
