import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI)."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    # Ensure breakpoints are unique
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
        
    expected_pct, _ = np.histogram(expected, bins=breakpoints)
    actual_pct, _ = np.histogram(actual, bins=breakpoints)
    
    # Avoid division by zero
    expected_pct = np.clip(expected_pct / len(expected), 0.0001, None)
    actual_pct = np.clip(actual_pct / len(actual), 0.0001, None)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def calculate_kl_divergence(expected: np.ndarray, actual: np.ndarray, bins: int = 20) -> float:
    """Calculates Kullback-Leibler Divergence."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Combine data to find common bins
    combined = np.concatenate([expected, actual])
    hist_bins = np.histogram_bin_edges(combined, bins=bins)
    
    p, _ = np.histogram(expected, bins=hist_bins, density=True)
    q, _ = np.histogram(actual, bins=hist_bins, density=True)
    
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return float(entropy(p, q))

def determine_drift_severity(psi: float, ks_pvalue: float, drift_detected: bool) -> str:
    """Determines severity level based on PSI and KS test p-value."""
    if not drift_detected:
        return "low"
        
    if psi > 0.2 or ks_pvalue < 0.01:
        return "high"
    elif psi > 0.1 or ks_pvalue < 0.05:
        return "moderate"
        
    return "low"

def compute_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame, 
                           psi_threshold: float = 0.1, 
                           ks_alpha: float = 0.05) -> Tuple[Dict[str, Any], bool]:
    """
    Computes drift metrics for all matching numeric features between reference and current data.
    """
    features_results = {}
    overall_drift = False
    
    # Get common numeric columns
    ref_numeric = reference_df.select_dtypes(include=[np.number])
    curr_numeric = current_df.select_dtypes(include=[np.number])
    common_cols = set(ref_numeric.columns).intersection(set(curr_numeric.columns))
    
    if not common_cols:
        logger.warning("No common numeric columns found for drift detection.")
        return {}, False
        
    for col in common_cols:
        expected = ref_numeric[col].dropna().values
        actual = curr_numeric[col].dropna().values
        
        if len(expected) == 0 or len(actual) == 0:
            continue
            
        # 1. Population Stability Index (PSI)
        psi_val = calculate_psi(expected, actual)
        
        # 2. Kullback-Leibler Divergence
        kl_div = calculate_kl_divergence(expected, actual)
        
        # 3. Wasserstein Distance (Earth Mover's Distance)
        wasser_dist = wasserstein_distance(expected, actual)
        
        # 4. Kolmogorov-Smirnov Test
        # returns statistic and p-value
        ks_stat, ks_pval = ks_2samp(expected, actual)
        
        # Determine if drift detected (Consensus approach)
        drift_detected = (psi_val >= psi_threshold) or (ks_pval < ks_alpha)
        if drift_detected:
            overall_drift = True
            
        severity = determine_drift_severity(psi_val, ks_pval, drift_detected)
        
        features_results[col] = {
            "psi": float(psi_val),
            "kl_divergence": float(kl_div),
            "wasserstein": float(wasser_dist),
            "ks_statistic": float(ks_stat),
            "drift_detected": bool(drift_detected),
            "severity": severity
        }
        
    return features_results, overall_drift
