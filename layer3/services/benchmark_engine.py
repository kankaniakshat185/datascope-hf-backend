import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor, XGBClassifier = None, None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor, LGBMClassifier = None, None

logger = logging.getLogger(__name__)

class BenchmarkEngine:
    """
    Model Benchmarking Engine.
    Evaluates multiple models not just on accuracy, but latency, robustness, and stability.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regression_registry = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
        }
        self.classification_registry = {
            "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=500),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
        }
        
        if XGBRegressor:
            self.regression_registry["XGBoost"] = XGBRegressor(random_state=random_state, n_jobs=-1)
            self.classification_registry["XGBoost"] = XGBClassifier(random_state=random_state, n_jobs=-1)
            
        if LGBMRegressor:
            self.regression_registry["LightGBM"] = LGBMRegressor(random_state=random_state, n_jobs=-1)
            self.classification_registry["LightGBM"] = LGBMClassifier(random_state=random_state, n_jobs=-1)

    def _inject_noise(self, X: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        """Injects Gaussian noise into numerical features to test robustness."""
        X_noisy = X.copy()
        for col in X_noisy.columns:
            std = X_noisy[col].std()
            noise = np.random.normal(0, std * noise_level, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
        return X_noisy

    def run_benchmark(self, df: pd.DataFrame, target_col: str, problem_type: str = "regression") -> Dict[str, Any]:
        logger.info("Starting Benchmark Engine...")
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]
        
        registry = self.regression_registry if problem_type == "regression" else self.classification_registry
        
        results = []
        
        # We simulate a simple train/test split (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_test_noisy = self._inject_noise(X_test, noise_level=0.15)
        
        for name, model in registry.items():
            try:
                # 1. Training Latency
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time_ms = (time.time() - t0) * 1000
                
                # 2. Inference Latency & Baseline Metrics
                t0 = time.time()
                preds = model.predict(X_test)
                inf_time_ms = (time.time() - t0) * 1000 / len(X_test) # ms per sample
                
                # 3. Robustness test
                noisy_preds = model.predict(X_test_noisy)
                
                if problem_type == "regression":
                    base_score = r2_score(y_test, preds)
                    noisy_score = r2_score(y_test, noisy_preds)
                    metrics = {
                        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                        "mae": float(mean_absolute_error(y_test, preds)),
                        "r2": float(base_score)
                    }
                else:
                    base_score = accuracy_score(y_test, preds)
                    noisy_score = accuracy_score(y_test, noisy_preds)
                    metrics = {
                        "accuracy": float(base_score),
                        "f1_score": float(f1_score(y_test, preds, average='weighted'))
                    }
                
                # Calculate performance degradation due to noise
                degradation = max(0, base_score - noisy_score)
                robustness_score = max(0, 1.0 - (degradation / max(0.01, base_score)))
                
                results.append({
                    "model": name,
                    "metrics": metrics,
                    "latency": {
                        "training_ms": round(train_time_ms, 2),
                        "inference_ms_per_sample": round(inf_time_ms, 4)
                    },
                    "robustness_score": round(float(robustness_score), 3)
                })
            except Exception as e:
                logger.warning(f"Model {name} failed benchmark: {e}")
                
        # Sort by primary metric (R2 or Accuracy)
        sort_key = "r2" if problem_type == "regression" else "accuracy"
        results.sort(key=lambda x: x["metrics"][sort_key], reverse=True)
        
        return {
            "status": "success",
            "problem_type": problem_type,
            "leaderboard": results
        }
