from fastapi import BackgroundTasks
import uuid
import time
from typing import Dict, Any

# In-memory store for async jobs.
# In a true massive-scale production system, this would be Redis.
# But for lightweight HF Spaces ML Observability, memory is perfect.
jobs_store: Dict[str, Any] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "status": "QUEUED",
        "progress": 0,
        "result": None,
        "error": None,
        "stage": "Initializing job..."
    }
    return job_id

def update_job(job_id: str, status: str, progress: int, stage: str = None, result: Any = None, error: str = None):
    if job_id in jobs_store:
        jobs_store[job_id]["status"] = status
        jobs_store[job_id]["progress"] = progress
        if stage:
            jobs_store[job_id]["stage"] = stage
        if result:
            jobs_store[job_id]["result"] = result
        if error:
            jobs_store[job_id]["error"] = error

def get_job(job_id: str) -> Dict[str, Any]:
    return jobs_store.get(job_id, {"error": "Job not found", "status": "FAILED"})

from layer1.services.governance_engine import evaluate_governance_rules

def process_analysis_job(job_id: str, df, target_column, run_checks_func, dict_func, eda_func, shap_func, layer1_func):
    """
    The background task that runs the heavy ML computation.
    """
    try:
        update_job(job_id, "PROCESSING", 10, "Starting ML Validations...")
        issues = run_checks_func(df, target_column)
        
        update_job(job_id, "PROCESSING", 30, "Generating Data Dictionary...")
        dict_data = dict_func(df)
        
        update_job(job_id, "PROCESSING", 50, "Generating EDA Visualizations...")
        eda_data = eda_func(df)
        
        update_job(job_id, "PROCESSING", 70, "Training Segmented SHAP Intelligence...")
        shap_raw = shap_func(df, target_column)
        shap_data = {}
        if isinstance(shap_raw, tuple) and len(shap_raw) == 2:
            shap_data = {"clusters": shap_raw[0], "insights": shap_raw[1]}
        elif isinstance(shap_raw, dict):
            shap_data = shap_raw
        else:
            shap_data = {"error": "Invalid SHAP response"}

        
        update_job(job_id, "PROCESSING", 90, "Finalizing Consensus Governance Metrics...")
        layer1_data = layer1_func(df, target_column)
        
        update_job(job_id, "PROCESSING", 95, "Running Deterministic Governance Rules...")
        # Note: issues is a dict like {"issues": [...], "total_impact": X}
        actual_issues = issues.get("issues", []) if isinstance(issues, dict) else issues
        governance = evaluate_governance_rules(actual_issues, layer1_data, shap_data)

        result = {
            "issues": issues,
            "dictData": dict_data,
            "edaData": eda_data,
            "shapData": shap_data,
            "layer1Data": layer1_data,
            "governance": governance
        }

        update_job(job_id, "COMPLETED", 100, "Done", result=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Job {job_id} failed: {str(e)}")
        update_job(job_id, "FAILED", 0, "Failed", error=str(e))
