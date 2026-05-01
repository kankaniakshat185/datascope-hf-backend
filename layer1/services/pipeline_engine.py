import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Callable
import logging
from layer1.services.outlier_engine import compute_consensus

logger = logging.getLogger(__name__)

def impute_missing(df: pd.DataFrame, strategy: str = "mean", columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Imputes missing values using mean, median, or mode."""
    df_out = df.copy()
    if columns is None:
        columns = df_out.columns.tolist()
        
    rows_affected = 0
    
    for col in columns:
        missing_mask = df_out[col].isnull()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            continue
            
        rows_affected += missing_count
        
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df_out[col]):
            fill_val = df_out[col].mean()
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_out[col]):
            fill_val = df_out[col].median()
        elif strategy == "mode":
            mode_series = df_out[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
        else:
            # Fallback for non-numeric with mean/median
            mode_series = df_out[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
            
        df_out[col] = df_out[col].fillna(fill_val)
        
    log = {
        "step_name": "Impute Missing Values",
        "transformation_applied": f"Replaced missing with {strategy}",
        "rows_affected": int(rows_affected)
    }
    return df_out, log

def remove_outliers(df: pd.DataFrame, threshold: float = 0.5, columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Removes outliers based on the consensus engine."""
    # We pass the relevant numeric columns to the consensus engine
    if columns:
        df_eval = df[columns].copy()
    else:
        df_eval = df.copy()
        
    results_df, summary = compute_consensus(df_eval, threshold=threshold)
    
    # Filter out the rows flagged as outliers
    outlier_mask = results_df["is_outlier"]
    df_out = df[~outlier_mask].copy()
    
    rows_removed = int(outlier_mask.sum())
    
    log = {
        "step_name": "Remove Outliers",
        "transformation_applied": f"Removed rows with consensus score >= {threshold}",
        "rows_affected": rows_removed
    }
    return df_out, log

def encode_categorical(df: pd.DataFrame, strategy: str = "one_hot", columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encodes categorical columns."""
    df_out = df.copy()
    
    if columns is None:
        columns = df_out.select_dtypes(include=['object', 'category']).columns.tolist()
        
    rows_affected = len(df_out) if columns else 0
    
    if strategy == "one_hot":
        df_out = pd.get_dummies(df_out, columns=columns, drop_first=True)
    elif strategy == "label":
        for col in columns:
            df_out[col] = df_out[col].astype('category').cat.codes
            
    log = {
        "step_name": "Encode Categorical",
        "transformation_applied": f"Applied {strategy} encoding to {len(columns)} columns",
        "rows_affected": int(rows_affected) if columns else 0
    }
    return df_out, log

# Registry of available steps
PIPELINE_STEPS = {
    "impute_missing": impute_missing,
    "remove_outliers": remove_outliers,
    "encode_categorical": encode_categorical
}

class DataPipeline:
    """Dynamic pipeline builder for data cleaning."""
    
    def __init__(self):
        self.steps: List[Tuple[Callable, Dict[str, Any]]] = []
        self.logs: List[Dict[str, Any]] = []
        
    def add_step(self, step_name: str, **kwargs):
        """Adds a processing step by name from the registry."""
        if step_name not in PIPELINE_STEPS:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        self.steps.append((PIPELINE_STEPS[step_name], kwargs))
        return self
        
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Executes the pipeline and returns the cleaned dataframe and logs."""
        df_cleaned = df.copy()
        self.logs = []
        
        logger.info(f"Running pipeline with {len(self.steps)} steps...")
        
        for step_func, kwargs in self.steps:
            try:
                df_cleaned, log = step_func(df_cleaned, **kwargs)
                self.logs.append(log)
            except Exception as e:
                logger.error(f"Pipeline step {step_func.__name__} failed: {e}")
                self.logs.append({
                    "step_name": step_func.__name__,
                    "transformation_applied": f"FAILED: {str(e)}",
                    "rows_affected": 0
                })
                # Decide whether to halt or continue. We'll continue for robustness, 
                # but in production you might want to raise.
                
        return df_cleaned, self.logs

def build_and_run_pipeline(df: pd.DataFrame, config: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Helper to execute pipeline from JSON config API.
    Example config: [{"step": "impute_missing", "params": {"strategy": "median"}}]
    """
    pipeline = DataPipeline()
    
    for step_cfg in config:
        step_name = step_cfg.get("step")
        params = step_cfg.get("params", {})
        if step_name:
            pipeline.add_step(step_name, **params)
            
    return pipeline.run(df)
