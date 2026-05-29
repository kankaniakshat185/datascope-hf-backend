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

def run_isolation_forest(df: pd.DataFrame) -> pd.Series:
    """Runs Isolation Forest and returns continuous anomaly scores."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    if numeric_df.empty or len(numeric_df) < 10:
        return pd.Series(0.0, index=df.index)
        
    model = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    model.fit(numeric_df)
    
    scores = -model.score_samples(numeric_df)
    result = pd.Series(scores, index=numeric_df.index)
    return result.reindex(df.index, fill_value=0.0)

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

def compute_consensus(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs all methods and enforces true detector agreement (>= 2 votes)."""
    logger.info("Computing Z-Scores...")
    z_scores = compute_z_score(df)
    
    logger.info("Computing MAD Scores...")
    mad_scores = compute_mad_score(df)
    
    logger.info("Running Isolation Forest...")
    iso_scores = run_isolation_forest(df)
    
    logger.info("Running DBSCAN...")
    dbscan_preds = run_dbscan(df, eps=3.0, min_samples=max(5, int(len(df)*0.01)))
    
    norm_iso = normalize_series(iso_scores)
    
    # 1. Generate Votes
    z_vote = z_scores > 3.0
    mad_vote = mad_scores > 3.5
    iso_vote = norm_iso > 0.80
    dbscan_vote = dbscan_preds == 1
    
    # 2. Count Votes
    vote_count = (
        z_vote.astype(int)
        + mad_vote.astype(int)
        + iso_vote.astype(int)
        + dbscan_vote.astype(int)
    )
    
    # 3. True Consensus
    is_outlier = vote_count >= 2
    
    results_df = pd.DataFrame({
        "z_score": z_scores,
        "mad_score": mad_scores,
        "isolation_forest": iso_scores,
        "dbscan": dbscan_preds,
        "consensus_score": vote_count,  # Keep naming for UI compatibility
        "is_outlier": is_outlier,
        "vote_count": vote_count,
        "z_vote": z_vote,
        "mad_vote": mad_vote,
        "iso_vote": iso_vote,
        "dbscan_vote": dbscan_vote
    }, index=df.index)
    
    percentage_flagged = float(is_outlier.mean() * 100)
    
    summary = {
        "total_outliers": int(results_df["is_outlier"].sum()),
        "percentage_flagged": percentage_flagged,
        "method_flags": {
            "z_score": float(z_vote.mean() * 100),
            "mad_score": float(mad_vote.mean() * 100),
            "isolation_forest": float(iso_vote.mean() * 100),
            "dbscan": float(dbscan_vote.mean() * 100)
        }
    }
    
    return results_df, summary
