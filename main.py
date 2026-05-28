from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from debugger import run_all_checks
from layer1.api.router import router as layer1_router
from layer1.services.job_manager import create_job, get_job, process_analysis_job
from layer1.services.shap_engine import compute_segmented_shap

from layer1.services.target_profiler import compute_target_candidates, identify_potential_proxies

# Removed `get_target_column(df)` to enforce HUMAN-IN-THE-LOOP workflow.

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dataset Debugger ML Service")
app.include_router(layer1_router)


# Enable CORS for Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Vercel, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    file_ext = file.filename.split('.')[-1].lower()
    supported_exts = ['csv', 'xlsx', 'xls', 'json', 'parquet']
    
    if file_ext not in supported_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported: {', '.join(supported_exts)}")
    
    contents = await file.read()
    
    if file_ext == 'csv':
        df = pd.read_csv(io.BytesIO(contents))
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(io.BytesIO(contents))
    elif file_ext == 'json':
        df = pd.read_json(io.BytesIO(contents))
    elif file_ext == 'parquet':
        df = pd.read_parquet(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    return df

from layer1.services.outlier_engine import compute_consensus
import re

def check_pii(col_series: pd.Series) -> bool:
    sample = col_series.dropna().astype(str).head(100)
    email_regex = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    ssn_regex = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    phone_regex = re.compile(r"^\+?1?\s*\(?-*\.*(\d{3})\)?\.*-*\s*(\d{3})\.*-*\s*(\d{4})$")
    
    for val in sample:
        if email_regex.match(val) or ssn_regex.match(val) or phone_regex.match(val):
            return True
    return False

def check_format_anomalies(col_series: pd.Series) -> bool:
    sample = col_series.dropna().astype(str).head(100)
    if len(sample) < 10: return False
    
    num_count = sum(1 for val in sample if val.replace('.','',1).isdigit())
    if num_count > len(sample) * 0.8 and num_count < len(sample):
        return True
    return False

def generate_data_dictionary(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    dictionary = []
    
    for col in df.columns:
        col_series = df[col]
        dtype = str(col_series.dtype)
        missing_count = int(col_series.isnull().sum())
        missing_percentage = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0
        unique_count = int(col_series.nunique())
        
        # Convert sample values to native Python types
        sample_values = [v.item() if hasattr(v, 'item') else v for v in col_series.dropna().unique()[:3]]
        
        col_info = {
            "column_name": str(col),
            "data_type": dtype,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "unique_count": unique_count,
            "sample_values": sample_values,
            "pii_warning": False,
            "imbalance_warning": False,
            "format_warning": False
        }
        
        if pd.api.types.is_numeric_dtype(col_series):
            col_info["min"] = float(col_series.min()) if not pd.isna(col_series.min()) else None
            col_info["max"] = float(col_series.max()) if not pd.isna(col_series.max()) else None
            col_info["mean"] = float(col_series.mean()) if not pd.isna(col_series.mean()) else None
            
            # Calculate univariate outlier percentage using the Layer 1 Consensus engine
            try:
                # Run consensus on just this column
                col_df = df[[col]].copy()
                # compute_consensus requires at least 10 rows for isolation forest/dbscan
                if len(col_df.dropna()) > 10:
                    results_df, summary = compute_consensus(col_df)
                    col_info["outlier_percentage"] = round(summary["percentage_flagged"], 2)
                else:
                    col_info["outlier_percentage"] = 0.0
            except Exception as e:
                col_info["outlier_percentage"] = 0.0
        else:
            mode_val = col_series.mode()
            col_info["top_value"] = str(mode_val.iloc[0]) if not mode_val.empty else None
            
            # PII & Format Checks
            col_info["pii_warning"] = check_pii(col_series)
            col_info["format_warning"] = check_format_anomalies(col_series)
            
            # Class Imbalance Check (If < 20 categories, check if mode > 95%)
            if 1 < unique_count <= 20 and not mode_val.empty:
                mode_freq = (col_series == mode_val.iloc[0]).sum()
                if mode_freq / len(col_series.dropna()) > 0.95:
                    col_info["imbalance_warning"] = True
            
        dictionary.append(col_info)
        
    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": dictionary
    }

from layer1.services.outlier_engine import compute_consensus
from layer1.services.impact_engine import compute_causal_impact
from layer1.services.pipeline_engine import build_and_run_pipeline

def run_layer1_full(df: pd.DataFrame, target_column: str) -> dict:
    # 1. Outlier Analysis
    outlier_results, outlier_summary = compute_consensus(df)
    row_res = {}
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
    try:
        is_classification = df[target_column].nunique() < 20 or not pd.api.types.is_numeric_dtype(df[target_column])
        problem_type = 'classification' if is_classification else 'regression'
        impact_features, impact_insights = compute_causal_impact(df, target_column, problem_type=problem_type)
        impact_response = {
            "features": impact_features,
            "insights": impact_insights
        }
    except Exception as e:
        impact_response = {"error": str(e)}

    # Mock pipeline logs
    _, pipeline_logs = build_and_run_pipeline(df, config=[
        {"step": "impute_missing", "params": {"strategy": "median"}}
    ])
    
    return {
        "outlier_analysis": outlier_response,
        "feature_importance": impact_response,
        "pipeline_log": pipeline_logs
    }

@app.post("/analyze_async")
async def analyze_dataset_async(background_tasks: BackgroundTasks, file: UploadFile = File(...), target_column: str = Form(...)):
    """
    DataScope V2 Governance Endpoint
    Instantly returns a job_id and processes the ML validation in the background.
    """
    df = await parse_uploaded_file(file)
    if not target_column or target_column not in df.columns:
        raise HTTPException(status_code=400, detail="A valid target column must be explicitly selected.")
        
    job_id = create_job()
    
    background_tasks.add_task(
        process_analysis_job, 
        job_id, 
        df, 
        target_column, 
        run_all_checks, 
        generate_data_dictionary, 
        generate_eda_data, 
        compute_segmented_shap, 
        run_layer1_full
    )
    
    return {"job_id": job_id, "status": "QUEUED"}

import math

def clean_json_floats(obj):
    if isinstance(obj, dict):
        return {k: clean_json_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if job.get("error") == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return clean_json_floats(job)

@app.post("/profile-dataset")
async def profile_dataset(file: UploadFile = File(...)):
    """
    Step 1: Parse the dataset and suggest ranked target candidates.
    Does NOT start governance analysis.
    """
    try:
        df = await parse_uploaded_file(file)
        candidates = compute_target_candidates(df)
        
        # We also return the raw column names so the frontend can populate a dropdown
        columns = [str(c) for c in df.columns if not str(c).startswith("Unnamed:")]
        
        return {
            "columns": columns,
            "ranked_candidates": candidates,
            "total_rows": len(df),
            "total_columns": len(columns)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...), 
    rules: str = Form(None),
    target_column: str = Form(...),
    prediction_type: str = Form("Auto Detect"),
    excluded_columns: str = Form("[]")
):
    """
    Step 2: Receives the EXPLICITLY confirmed target and settings to begin the governance job.
    """
    try:
        df = await parse_uploaded_file(file)
        
        # Process exclusions
        import json
        exclusions = json.loads(excluded_columns)
        if exclusions:
            df = df.drop(columns=[col for col in exclusions if col in df.columns], errors='ignore')
            
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Confirmed target column '{target_column}' not found in dataset.")

        # Parse rules
        parsed_rules = []
        if rules:
            try:
                parsed_rules = json.loads(rules)
            except Exception:
                pass
        
        # Attempt clean
        df = df.dropna(axis=1, how='all')

        # We pass the confirmed target. (Prediction type can be passed to checks if needed in future)
        results = run_all_checks(df, target_column, parsed_rules)
        return results
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-dictionary")
async def get_data_dictionary(file: UploadFile = File(...)):
    try:
        df = await parse_uploaded_file(file)
        return generate_data_dictionary(df)
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def generate_eda_data(df: pd.DataFrame) -> dict:
    eda_results = {
        "distributions": {},
        "value_counts": {},
        "correlation_matrix": None,
        "outlier_plots": {}
    }
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # 1. Distributions for Numeric Columns (Max 10 bins)
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0 and col_data.nunique() > 1:
            try:
                counts, bin_edges = np.histogram(col_data, bins=10)
                eda_results["distributions"][col] = {
                    "labels": [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(counts))],
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            except Exception:
                pass
                
    # 2. Value Counts for Categorical Columns (Top 10)
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            unique_vals = col_data.nunique()
            # Skip high cardinality columns (e.g. Names, Phone Numbers, IDs)
            # If it has >50 unique values OR is >50% unique, it's not a good category for a bar chart
            if unique_vals > 50 or (unique_vals / len(col_data)) > 0.5:
                continue
                
            val_counts = col_data.value_counts().head(10)
            eda_results["value_counts"][col] = {
                "labels": [str(k) for k in val_counts.index],
                "counts": val_counts.values.tolist()
            }
            
    # 3. Correlation Matrix (for numeric columns only)
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr()
        # Replace NaNs with 0 to avoid JSON serialization errors
        corr_df = corr_df.fillna(0)
        eda_results["correlation_matrix"] = {
            "columns": numeric_cols,
            "matrix": corr_df.values.tolist()
        }
        
    # 4. Outlier Plots (Boxplot stats)
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Cap the number of outliers sent to frontend to prevent huge payloads
            outlier_list = []
            if len(outliers) > 0:
                if len(outliers) > 100:
                    outlier_list = outliers.sample(100).tolist()
                else:
                    outlier_list = outliers.tolist()
                    
            eda_results["outlier_plots"][col] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q1": q1,
                "median": float(col_data.median()),
                "q3": q3,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers": outlier_list
            }
            
    return eda_results

@app.post("/eda")
async def get_eda(file: UploadFile = File(...)):
    try:
        df = await parse_uploaded_file(file)
        return generate_eda_data(df)
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok"}

def generate_shap_values(df: pd.DataFrame, target_col: str) -> dict:
    if not target_col or target_col not in df.columns:
        return {"error": "Target column not found"}
        
    df = df.dropna(subset=[target_col])
    if len(df) == 0:
        return {"error": "Target column contains only NaNs"}
        
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
    # Handle NaNs
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median() if pd.api.types.is_numeric_dtype(df_clean[col]) else 0)
            
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    if len(X) == 0 or X.shape[1] == 0:
        return {"error": "Not enough features for SHAP analysis"}
        
    if len(X) > 500:
        X_sample = X.sample(500, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y
        
    is_classification = df[target_col].nunique() < 10 or not pd.api.types.is_numeric_dtype(df[target_col])
    
    try:
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
        model.fit(X_sample, y_sample)
        
        # Use native scikit-learn feature importances instead of SHAP to save 200MB+ in Vercel Serverless
        vals = model.feature_importances_
            
        feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        
        top_features = feature_importance.head(10)
        
        return {
            "target": target_col,
            "features": top_features['col_name'].tolist(),
            "importance": top_features['feature_importance_vals'].tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/shap")
async def get_shap_values(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        df = await parse_uploaded_file(file)
        if not target_column or target_column not in df.columns:
            raise HTTPException(status_code=400, detail="A valid target column must be explicitly selected.")
        return generate_shap_values(df, target_column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean")
async def clean_dataset(file: UploadFile = File(...)):
    try:
        df = await parse_uploaded_file(file)
        
        # 1. Drop duplicates
        df = df.drop_duplicates()
        
        # 2. Drop columns with > 50% missing values
        missing_ratios = df.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > 0.5].index
        df = df.drop(columns=cols_to_drop)
        
        # 3. Impute remaining missing values and cap outliers
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Impute missing with median
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
                # Cap outliers (1st and 99th percentiles) to reduce extreme noise
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q1, upper=q99)
            else:
                # Impute categorical with mode
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                        
        # Return as CSV stream
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        
        # Safely handle filename
        safe_filename = file.filename if file.filename else "dataset.csv"
        if not safe_filename.endswith(".csv"):
            safe_filename = safe_filename.rsplit(".", 1)[0] + ".csv"
            
        response.headers["Content-Disposition"] = f"attachment; filename=cleaned_{safe_filename}"
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift")
async def detect_drift(
    test_file: UploadFile = File(...), 
    train_distributions: str = Form(...)
):
    import json
    try:
        train_dist = json.loads(train_distributions)
        test_df = await parse_uploaded_file(test_file)
        
        drift_results = []
        
        for col, dist in train_dist.items():
            if col in test_df.columns and pd.api.types.is_numeric_dtype(test_df[col]):
                test_data = test_df[col].dropna()
                if len(test_data) == 0:
                    continue
                    
                bin_edges = dist.get("bin_edges")
                if not bin_edges:
                    continue
                    
                train_counts = dist.get("counts", [])
                
                train_total = sum(train_counts)
                if train_total == 0:
                    continue
                train_pct = np.array(train_counts) / train_total
                
                epsilon = 0.0001
                train_pct = np.where(train_pct == 0, epsilon, train_pct)
                
                test_counts, _ = np.histogram(test_data, bins=bin_edges)
                test_total = sum(test_counts)
                
                if test_total == 0:
                    continue
                test_pct = np.array(test_counts) / test_total
                test_pct = np.where(test_pct == 0, epsilon, test_pct)
                
                psi_values = (test_pct - train_pct) * np.log(test_pct / train_pct)
                psi_total = float(np.sum(psi_values))
                
                if psi_total > 0.1:
                    drift_results.append({
                        "column": col,
                        "psi": psi_total,
                        "severity": "HIGH" if psi_total > 0.2 else "MEDIUM"
                    })
                    
        drift_results.sort(key=lambda x: x["psi"], reverse=True)
        
        return {
            "drift_detected": len(drift_results) > 0,
            "drifted_features": drift_results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
