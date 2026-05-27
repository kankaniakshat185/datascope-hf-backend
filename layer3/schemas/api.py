from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    runtime_seconds: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BenchmarkRequest(BaseModel):
    target_column: str
    problem_type: str = "regression"

class RootCauseRequest(BaseModel):
    target_column: str
    problem_type: str = "regression"

class MonitoringRequest(BaseModel):
    dataset_id: str
    feature_divergence: Dict[str, float]
