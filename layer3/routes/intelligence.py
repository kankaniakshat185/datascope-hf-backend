from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import pandas as pd
import io
import logging

from ..services.job_manager import job_manager
from ..services.root_cause_engine import RootCauseEngine
from ..services.failure_analysis_engine import FailureAnalysisEngine
from ..services.benchmark_engine import BenchmarkEngine
from ..services.monitoring_engine import MonitoringEngine
from ..schemas.api import JobResponse, JobStatusResponse

router = APIRouter(prefix="/api/v3", tags=["Layer3 Intelligence"])
logger = logging.getLogger(__name__)

root_cause_engine = RootCauseEngine()
failure_analysis_engine = FailureAnalysisEngine()
benchmark_engine = BenchmarkEngine()
monitoring_engine = MonitoringEngine()

# We need a quick baseline model generator for the root cause engine
def _get_baseline_model(df, target_col, is_regression):
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    X = df.drop(columns=[target_col]).select_dtypes(include=['number']).fillna(0)
    y = df[target_col]
    if is_regression:
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

def run_root_cause_analysis(df: pd.DataFrame, target_col: str, problem_type: str):
    is_regression = problem_type.lower() == 'regression'
    
    # Downsample if too large to ensure fast clustering and KS tests
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42).copy()
        
    model = _get_baseline_model(df, target_col, is_regression)
    
    # Run Root Cause
    rc_results = root_cause_engine.analyze(df, target_col, model, is_regression)
    
    # Run Failure Analysis
    residuals = root_cause_engine._compute_residuals(df, target_col, model, is_regression)
    fa_results = failure_analysis_engine.analyze(df, residuals)
    
    return {
        "root_cause_analysis": rc_results,
        "failure_analysis": fa_results
    }

@router.post("/root-cause", response_model=JobResponse)
async def trigger_root_cause_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: str = Form("regression")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        job_id = job_manager.create_job()
        
        background_tasks.add_task(
            job_manager.execute_job,
            job_id,
            run_root_cause_analysis,
            df,
            target_column,
            problem_type
        )
        
        return JobResponse(job_id=job_id, status="queued", message="Root cause analysis job triggered.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/benchmark", response_model=JobResponse)
async def trigger_benchmark(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: str = Form("regression")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        job_id = job_manager.create_job()
        
        background_tasks.add_task(
            job_manager.execute_job,
            job_id,
            benchmark_engine.run_benchmark,
            df,
            target_column,
            problem_type
        )
        
        return JobResponse(job_id=job_id, status="queued", message="Benchmark job triggered.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    status = job_manager.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.post("/monitoring/record")
async def record_monitoring_drift(dataset_id: str, feature_divergence: dict):
    result = monitoring_engine.record_drift(dataset_id, feature_divergence)
    return result

@router.get("/monitoring/{dataset_id}")
async def get_monitoring_dashboard(dataset_id: str):
    return monitoring_engine.get_monitoring_dashboard(dataset_id)
