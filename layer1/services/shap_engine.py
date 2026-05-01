import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging
from typing import Dict, Any, Tuple

# Try to import shap, handle if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def auto_select_k(scaled_data: np.ndarray, max_k: int = 5) -> int:
    """Uses silhouette score to find optimal k between 2 and max_k."""
    if len(scaled_data) < 20:
        return 2  # Fallback for tiny datasets
        
    best_k = 3
    best_score = -1
    
    # Cap max_k to dataset size if needed
    max_k = min(max_k, len(scaled_data) - 1)
    if max_k < 2:
        return 1
        
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(scaled_data)
        
        # Silhouette requires at least 2 clusters and less than n_samples
        if len(set(labels)) > 1:
            score = silhouette_score(scaled_data, labels)
            if score > best_score:
                best_score = score
                best_k = k
                
    return best_k

def compute_segmented_shap(df: pd.DataFrame, target_col: str, problem_type: str = 'regression') -> Tuple[Dict[str, Any], list]:
    """
    1. Clusters dataset.
    2. Trains global model.
    3. Computes SHAP values separately per cluster.
    """
    if not SHAP_AVAILABLE:
        logger.error("SHAP library is not installed. Cannot run Segmented SHAP.")
        return {}, ["SHAP library missing. Please install 'shap' package."]

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, target_col]
    
    if len(X) < 20 or len(X.columns) < 2:
        return {}, ["Dataset too small for segmented SHAP analysis."]

    # Sub-sample if dataset is too large to keep API fast
    if len(X) > 5000:
        logger.info(f"Subsampling SHAP data from {len(X)} to 5000 rows for speed.")
        X_sample = X.sample(n=5000, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    # 1. Train Global Model
    if problem_type == 'regression':
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
    model.fit(X_sample, y_sample)
    
    # 2. Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer(X_sample, check_additivity=False)
    
    if isinstance(shap_values_obj.values, list) or len(shap_values_obj.values.shape) == 3:
        shap_vals = shap_values_obj.values[:, :, 1] if len(shap_values_obj.values.shape) == 3 else shap_values_obj.values[1]
    else:
        shap_vals = shap_values_obj.values

    # 3. Cluster on the SHAP values (Behavioral Clustering)
    # This groups users based on how the model explains their predictions
    optimal_k = auto_select_k(shap_vals, max_k=4)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(shap_vals)
        
    # 4. Aggregate by cluster
    clusters_results = {}
    cluster_top_feats = []
    
    for k in range(optimal_k):
        mask = (cluster_labels == k)
        if not np.any(mask):
            continue
            
        cluster_shap_vals = shap_vals[mask]
        mean_abs_shap = np.abs(cluster_shap_vals).mean(axis=0)
        
        feature_importance = {feat: float(val) for feat, val in zip(X_sample.columns, mean_abs_shap)}
        sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feat for feat, val in sorted_feats[:3]]
        
        cluster_name = f"cluster_{k+1}"
        clusters_results[cluster_name] = {
            "top_features": top_features,
            "feature_importance": feature_importance
        }
        
        if top_features:
            cluster_top_feats.append((cluster_name, top_features[0]))
            
    # 5. Generate Insights
    insights = []
    if len(set(f for c, f in cluster_top_feats)) > 1:
        insights.append("Feature importance differs significantly across data segments.")
    else:
        insights.append("The top driving feature remains consistent across all data segments.")
        
    for cluster_name, top_feat in cluster_top_feats:
        insights.append(f"In {cluster_name}, '{top_feat}' is the most dominant feature.")

    return clusters_results, insights
