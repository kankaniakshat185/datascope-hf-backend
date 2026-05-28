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

def process_analysis_job(job_id: str, df, target_column, run_checks_func):
    """
    The background task that runs the heavy ML computation.
    """
    try:
        update_job(job_id, "PROCESSING", 10, "Starting ML Validations...")
        
        # Simulate minor delay for orchestrated observability
        time.sleep(1) 
        update_job(job_id, "PROCESSING", 30, "Training Baseline & Impact Engines...")
        
        # Run the heavy computation
        issues = run_checks_func(df, target_column)
        
        update_job(job_id, "PROCESSING", 90, "Finalizing Governance Metrics...")
        time.sleep(1)

        update_job(job_id, "COMPLETED", 100, "Done", result={"issues": issues})
    except Exception as e:
        print(f"Job {job_id} failed: {str(e)}")
        update_job(job_id, "FAILED", 0, "Failed", error=str(e))
