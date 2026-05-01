from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import io
import json
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import StreamingResponse

from layer1.models.schemas import (
    FullAnalysisResponse, 
    OutlierAnalysisResponse, 
    CausalImpactResponse, 
    SegmentedShapResponse, 
    DriftAnalysisResponse
)
from layer1.services.outlier_engine import compute_consensus
from layer1.services.impact_engine import compute_causal_impact
from layer1.services.shap_engine import compute_segmented_shap
from layer1.services.drift_engine import compute_drift_analysis
from layer1.services.pipeline_engine import build_and_run_pipeline

router = APIRouter(prefix="/api/v2/analytical-engine", tags=["Layer 1 Analytical Engine"])

class PipelineConfigRequest(BaseModel):
    steps: List[Dict[str, Any]] = [
        {"step": "impute_missing", "params": {"strategy": "median"}},
        {"step": "remove_outliers", "params": {"threshold": 0.6}}
    ]

@router.post("/full-analysis", response_model=FullAnalysisResponse)
async def run_full_analysis(
    target_column: str,
    file: UploadFile = File(...),
    reference_file: UploadFile = File(None)
):
    """
    Runs the complete Layer 1 Analytical Engine on a dataset.
    """
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # 1. Outlier Analysis
        outlier_results, outlier_summary = compute_consensus(df)
        
        row_res = {}
        # Limit row results to actual outliers to save bandwidth if needed, or just return first 100
        outliers_only = outlier_results[outlier_results['is_outlier'] == True].head(100)
        for idx, row in outliers_only.iterrows():
            row_res[int(idx)] = {
                "is_outlier": bool(row["is_outlier"]),
                "consensus_score": float(row["consensus_score"]),
                "method_scores": {
                    "z_score": float(row["z_score"]),
                    "mad_score": float(row["mad_score"]),
                    "isolation_forest": int(row["isolation_forest"]),
                    "dbscan": int(row["dbscan"])
                }
            }
            
        outlier_response = {
            "summary": outlier_summary,
            "row_results": row_res
        }
        
        # 2. Causal-Aware Impact Analysis
        impact_features, impact_insights = compute_causal_impact(df, target_column, problem_type='regression')
        impact_response = {
            "features": impact_features,
            "insights": impact_insights
        }
        
        # 3. Drift Analysis (if reference dataset provided)
        drift_response = {"features": {}, "overall_drift_detected": False}
        if reference_file:
            ref_content = await reference_file.read()
            ref_df = pd.read_csv(io.BytesIO(ref_content))
            drift_features, overall_drift = compute_drift_analysis(ref_df, df)
            drift_response = {
                "features": drift_features,
                "overall_drift_detected": overall_drift
            }
            
        # Mocking pipeline logs for the full analysis endpoint without applying changes permanently
        # In a real flow, cleaning might happen before analysis based on user choice.
        _, pipeline_logs = build_and_run_pipeline(df, config=[
            {"step": "impute_missing", "params": {"strategy": "median"}}
        ])
        
        return {
            "outlier_analysis": outlier_response,
            "feature_importance": impact_response,
            "drift_analysis": drift_response,
            "pipeline_log": pipeline_logs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/segmented-shap", response_model=SegmentedShapResponse)
async def run_segmented_shap(target_column: str, file: UploadFile = File(...)):
    """Runs Segmented SHAP separately since it can be computationally heavy."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        clusters, insights = compute_segmented_shap(df, target_column)
        
        return {
            "clusters": clusters,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmented SHAP failed: {str(e)}")

@router.post("/pipeline/run")
async def run_pipeline(
    file: UploadFile = File(...),
    config_json: str = Form(...)
):
    """Runs the dynamic cleaning pipeline and returns a cleaned CSV file."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Parse dynamic config from frontend
        config = json.loads(config_json)
        
        cleaned_df, logs = build_and_run_pipeline(df, config=config)
        
        stream = io.StringIO()
        cleaned_df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=cleaned_{file.filename}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@router.post("/drift", response_model=DriftAnalysisResponse)
async def run_drift_analysis(
    reference_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    """Runs Layer 1 Multi-Metric Drift Detection using raw dataframes."""
    try:
        ref_content = await reference_file.read()
        ref_df = pd.read_csv(io.BytesIO(ref_content))
        
        test_content = await test_file.read()
        test_df = pd.read_csv(io.BytesIO(test_content))
        
        drift_features, overall_drift = compute_drift_analysis(ref_df, test_df)
        
        return {
            "features": drift_features,
            "overall_drift_detected": overall_drift
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")
