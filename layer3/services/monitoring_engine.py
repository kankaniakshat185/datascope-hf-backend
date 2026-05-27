import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

class MonitoringEngine:
    """
    Continuous Monitoring Mode Engine.
    Simulates production ML observability by tracking drift over time.
    Uses a lightweight JSON/SQLite storage layer to keep history.
    """
    def __init__(self, storage_path: str = "./monitoring_db.json"):
        self.storage_path = storage_path
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump({"history": []}, f)

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return data.get("history", [])
        except Exception:
            return []

    def _save_history(self, history: List[Dict[str, Any]]):
        with open(self.storage_path, 'w') as f:
            json.dump({"history": history}, f)

    def calculate_drift_severity(self, feature_divergence: Dict[str, float]) -> str:
        """Determines overall drift severity based on divergence scores."""
        if not feature_divergence:
            return "stable"
            
        max_drift = max(feature_divergence.values())
        if max_drift > 0.3:
            return "severe_drift"
        elif max_drift > 0.15:
            return "moderate_drift"
        return "stable"

    def record_drift(self, dataset_id: str, feature_divergence: Dict[str, float]) -> Dict[str, Any]:
        """
        Records the current drift snapshot and checks if retraining is recommended.
        feature_divergence: dict of feature_name -> divergence_score (e.g. KS-statistic)
        """
        history = self._load_history()
        
        severity = self.calculate_drift_severity(feature_divergence)
        retraining_recommended = severity == "severe_drift"
        
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_id": dataset_id,
            "feature_divergence": feature_divergence,
            "severity": severity,
            "retraining_recommended": retraining_recommended
        }
        
        history.append(snapshot)
        # Keep only last 100 records to prevent file bloat
        if len(history) > 100:
            history = history[-100:]
            
        self._save_history(history)
        return snapshot

    def get_monitoring_dashboard(self, dataset_id: str = None) -> Dict[str, Any]:
        """Generates the monitoring timeline for the dashboard."""
        history = self._load_history()
        
        if dataset_id:
            history = [h for h in history if h.get("dataset_id") == dataset_id]
            
        timeline = []
        for h in history:
            timeline.append({
                "time": h["timestamp"],
                "severity": h["severity"],
                "max_drift": max(h["feature_divergence"].values()) if h["feature_divergence"] else 0
            })
            
        return {
            "status": "success",
            "total_records": len(timeline),
            "timeline": timeline,
            "latest_status": timeline[-1]["severity"] if timeline else "unknown"
        }
