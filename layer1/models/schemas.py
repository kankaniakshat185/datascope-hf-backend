from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class OutlierMethodScores(BaseModel):
    z_score: float
    mad_score: float
    isolation_forest: int  # 0 or 1
    dbscan: int  # 0 or 1

class OutlierResult(BaseModel):
    is_outlier: bool
    consensus_score: float
    method_scores: OutlierMethodScores

class OutlierSummary(BaseModel):
    total_outliers: int
    percentage_flagged: float
    method_flags: Dict[str, float]

class OutlierAnalysisResponse(BaseModel):
    summary: OutlierSummary
    row_results: Dict[int, OutlierResult]

class FeatureImpact(BaseModel):
    importance_score: float
    performance_impact: float
    variance_explained: float

class CausalImpactResponse(BaseModel):
    features: Dict[str, FeatureImpact]
    insights: List[str]

class ClusterShap(BaseModel):
    top_features: List[str]
    feature_importance: Dict[str, float]

class SegmentedShapResponse(BaseModel):
    clusters: Dict[str, ClusterShap]
    insights: List[str]

class FeatureDrift(BaseModel):
    psi: float
    kl_divergence: float
    wasserstein: float
    ks_statistic: float
    drift_detected: bool
    severity: str

class DriftAnalysisResponse(BaseModel):
    features: Dict[str, FeatureDrift]
    overall_drift_detected: bool

class PipelineLog(BaseModel):
    step_name: str
    transformation_applied: str
    rows_affected: int

class FullAnalysisResponse(BaseModel):
    outlier_analysis: OutlierAnalysisResponse
    feature_importance: CausalImpactResponse
    drift_analysis: DriftAnalysisResponse
    pipeline_log: List[PipelineLog]
