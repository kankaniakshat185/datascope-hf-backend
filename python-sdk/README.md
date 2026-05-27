# DataScope ML SDK

The official Python SDK for [DataScope](https://your-frontend-domain.com), the premier Machine Learning Observability Platform.

## Installation

```bash
pip install datascope-ml
```

## Quick Start

Seamlessly upload and analyze your Pandas DataFrames directly from your Jupyter Notebook or Python IDE.

```python
import pandas as pd
import datascope

# Load your local dataset
df = pd.read_csv("my_training_data.csv")

# Initialize the DataScope client
client = datascope.Client(api_key="YOUR_API_KEY")

# Upload and generate analysis
# This automatically opens your browser to the results dashboard!
client.analyze(df, project_name="fraud_detection_v2", target_column="is_fraud")
```

## Features
- **Zero Friction:** No need to export CSVs and drag-and-drop into a browser.
- **Privacy First:** Data dictionaries and outlier math can run locally, minimizing data sent over the network.
- **Auto-Browser Launch:** Automatically opens your Vercel-hosted dashboard as soon as the backend completes analysis.
