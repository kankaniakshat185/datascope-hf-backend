import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_z_score(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Computes Z-scores for numeric columns. Returns DataFrame of absolute Z-scores."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame(index=df.index)
    
    mean = numeric_df.mean()
    std = numeric_df.std().replace(0, np.nan)
    
    z_scores = np.abs((numeric_df - mean) / std).fillna(0.0)
    return z_scores.max(axis=1)

def compute_mad_score(df: pd.DataFrame, threshold: float = 3.5) -> pd.Series:
    """Computes Robust Z-score using Median Absolute Deviation (MAD)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.Series(0.0, index=df.index)
    
    median = numeric_df.median()
    diff = np.abs(numeric_df - median)
    mad = diff.median().replace(0, np.nan)
    
    # 0.6745 is the factor for normal distribution
    modified_z_scores = 0.6745 * diff / mad
    modified_z_scores = modified_z_scores.fillna(0.0)
    
    return modified_z_scores.max(axis=1)

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """Runs Isolation Forest and returns binary outlier labels (1=outlier, 0=inlier)."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.empty or len(numeric_df) < 10:
        return pd.Series(0, index=df.index)
        
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = model.fit_predict(numeric_df)
    
    binary_preds = (preds == -1).astype(int)
    result = pd.Series(binary_preds, index=numeric_df.index)
    return result.reindex(df.index, fill_value=0)

def run_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
    """Runs DBSCAN and returns binary outlier labels based on noise points (-1)."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.empty or len(numeric_df) < min_samples:
        return pd.Series(0, index=df.index)
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    preds = model.fit_predict(scaled_data)
    
    binary_preds = (preds == -1).astype(int)
    result = pd.Series(binary_preds, index=numeric_df.index)
    return result.reindex(df.index, fill_value=0)

def normalize_series(series: pd.Series) -> pd.Series:
    """Min-Max normalizes a series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return pd.Series(0.0, index=series.index)

def compute_consensus(df: pd.DataFrame, weights: Dict[str, float] = None, threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs all methods, combines them, and generates the consensus result."""
    if weights is None:
        weights = {
            "z_score": 0.15,
            "mad_score": 0.25,
            "isolation_forest": 0.40,
            "dbscan": 0.20
        }
    
    logger.info("Computing Z-Scores...")
    z_scores = compute_z_score(df)
    
    logger.info("Computing MAD Scores...")
    mad_scores = compute_mad_score(df)
    
    logger.info("Running Isolation Forest...")
    iso_preds = run_isolation_forest(df, contamination=0.05)
    
    logger.info("Running DBSCAN...")
    dbscan_preds = run_dbscan(df, eps=3.0, min_samples=max(5, int(len(df)*0.01)))
    
    norm_z = normalize_series(z_scores)
    norm_mad = normalize_series(mad_scores)
    
    consensus_score = (
        norm_z * weights.get("z_score", 0.0) +
        norm_mad * weights.get("mad_score", 0.0) +
        iso_preds * weights.get("isolation_forest", 0.0) +
        dbscan_preds * weights.get("dbscan", 0.0)
    )
    
    results_df = pd.DataFrame({
        "z_score": z_scores,
        "mad_score": mad_scores,
        "isolation_forest": iso_preds,
        "dbscan": dbscan_preds,
        "consensus_score": consensus_score,
        "is_outlier": consensus_score >= threshold
    }, index=df.index)
    
    summary = {
        "total_outliers": int(results_df["is_outlier"].sum()),
        "percentage_flagged": float(results_df["is_outlier"].mean() * 100),
        "method_flags": {
            "z_score": float((z_scores > 3.0).mean() * 100),
            "mad_score": float((mad_scores > 3.5).mean() * 100),
            "isolation_forest": float(iso_preds.mean() * 100),
            "dbscan": float(dbscan_preds.mean() * 100)
        }
    }
    
    return results_df, summary
