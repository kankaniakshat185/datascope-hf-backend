# DataScope Python SDK (`datascope`)

The official Python client for **DataScope: The Machine Learning Observability Platform**.

Connect your local Jupyter Notebooks, CI/CD pipelines, and Python IDEs directly to the DataScope platform.

## Installation

```bash
pip install datascope-ml
```

## Quick Start (Dashboard Mode)

Push your local dataset to the cloud and automatically open the interactive results dashboard in your browser.

**Note:** You can generate your free SDK API Key directly from your account on the [DataScope Website](https://datascope-app.vercel.app).

```python
import pandas as pd
import datascope

df = pd.read_csv("my_dataset.csv")

# Initialize client (points to localhost by default, configure for production)
client = datascope.Client(api_key="YOUR_API_KEY")

# Upload & Analyze
client.analyze(
    df, 
    project_name="fraud_detection", 
    target_column="is_fraud",
    prediction_type="classification"
)
```

## Programmatic MLOps Features

The `datascope` package is a fully-fledged programmatic MLOps tool. You can fetch raw JSON metrics and use them in automated pipelines.

### 1. Automated Governance (CI/CD Blocking)
Run rigorous checks and raise an Exception if your dataset is deemed `REJECTED` by the DataScope Governance Engine.
```python
# Will raise RuntimeError if data fails governance
client.assert_ready_for_deployment(df, target_column="is_fraud")

# Or fetch raw results:
results = client.run_governance_checks(df, target_column="is_fraud")
print(results["governance"]["status"])
```

### 2. Async Analysis (For Large Datasets)
```python
job_id = client.start_analysis_job(df, target_column="is_fraud")
status = client.get_job_status(job_id)
```

### 3. Granular ML Microservices
Use the underlying DataScope engines directly in your notebook:

```python
# 1. Automatic Data Cleaning & Imputation
clean_df = client.clean(df)

# 2. Concept Drift Detection (PSI)
drift_report = client.detect_drift(test_df, train_distributions)

# 3. Exploratory Data Analysis (EDA)
eda_stats = client.get_eda(df)

# 4. Feature Importance (SHAP/Random Forest)
features = client.get_feature_importance(df, target_column="is_fraud")

# 5. Data Dictionary (PII warnings, formats, types)
data_dict = client.get_data_dictionary(df)
```
