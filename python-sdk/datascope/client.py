import io
import requests
import webbrowser
import pandas as pd
import json
import time
from typing import Optional, Dict, List, Any

class Client:
    """
    The DataScope ML Client.
    Connects your local Jupyter Notebooks or Python IDEs directly to the DataScope platform.
    """
    
    def __init__(self, api_key: str = None, base_url: str = "http://127.0.0.1:8000", frontend_url: str = "https://datascope-app.vercel.app"):
        """
        Initialize the DataScope client.
        
        Args:
            api_key: Your secret API key (can be generated from the DataScope dashboard).
            base_url: The URL of your FastAPI backend.
            frontend_url: The URL of your Next.js frontend dashboard.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/')
        
    def _get_headers(self):
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        return headers

    def _df_to_csv_buffer(self, df: pd.DataFrame) -> io.StringIO:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer
        
    def analyze(self, df: pd.DataFrame, target_column: str, prediction_type: str = "Auto Detect", project_name: str = "my_dataset"):
        """
        Uploads the dataframe to DataScope, runs full Layer 1 & 2 analytics, 
        and automatically opens the results dashboard in your browser.
        Note: This uses the frontend API route orchestration.
        """
        print(f"🚀 Initializing DataScope audit for project: '{project_name}'")
        print(f"📊 Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': (f"{project_name}.csv", csv_buffer.getvalue(), 'text/csv')}
        data = {'target_column': target_column, 'prediction_type': prediction_type}
        
        print("⏳ Uploading to DataScope engine and computing analytics...")
        try:
            # This calls the next.js api
            response = requests.post(f"{self.frontend_url}/api/upload", files=files, data=data, headers=self._get_headers())
            
            if response.status_code == 200:
                result = response.json()
                job_id = result.get("datasetId") or result.get("id") or result.get("job_id")
                
                if not job_id:
                    print("⚠️ Success, but could not parse job_id from response.")
                    return result
                
                dashboard_url = f"{self.frontend_url}/results/{job_id}"
                print(f"✅ Analysis Complete!")
                print(f"🔗 View your interactive dashboard here: {dashboard_url}")
                webbrowser.open(dashboard_url)
                return dashboard_url
            else:
                print(f"❌ Error analyzing dataset: HTTP {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset (drops duplicates, removes columns with >50% missing values, imputes missing values, and caps outliers).
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        
        response = requests.post(f"{self.base_url}/clean", files=files, headers=self._get_headers())
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))
        else:
            raise Exception(f"Failed to clean dataset: {response.text}")

    def get_eda(self, df: pd.DataFrame) -> dict:
        """
        Returns Exploratory Data Analysis (EDA) stats including distributions, value counts, correlations, and outliers.
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        
        response = requests.post(f"{self.base_url}/eda", files=files, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get EDA: {response.text}")
            
    def get_data_dictionary(self, df: pd.DataFrame) -> dict:
        """
        Returns a data dictionary with column types, missingness, PII warnings, and anomalies.
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        
        response = requests.post(f"{self.base_url}/data-dictionary", files=files, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get data dictionary: {response.text}")

    def profile(self, df: pd.DataFrame) -> dict:
        """
        Profiles the dataset and suggests ranked target column candidates.
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        
        response = requests.post(f"{self.base_url}/profile-dataset", files=files, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to profile dataset: {response.text}")

    def get_feature_importance(self, df: pd.DataFrame, target_column: str) -> dict:
        """
        Returns Random Forest based feature importance (similar to SHAP) for the target column.
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        data = {'target_column': target_column}
        
        response = requests.post(f"{self.base_url}/shap", files=files, data=data, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to compute feature importance: {response.text}")

    def detect_drift(self, test_df: pd.DataFrame, train_distributions: dict) -> dict:
        """
        Detects data drift using Population Stability Index (PSI) between a test dataset and a training distribution.
        """
        csv_buffer = self._df_to_csv_buffer(test_df)
        files = {'test_file': ("test_dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        data = {'train_distributions': json.dumps(train_distributions)}
        
        response = requests.post(f"{self.base_url}/drift", files=files, data=data, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to detect drift: {response.text}")

    def run_governance_checks(self, df: pd.DataFrame, target_column: str, rules: dict = None, excluded_columns: list = None) -> dict:
        """
        Synchronously runs full governance checks on the backend (blocking).
        Returns raw JSON metrics (including APPROVED/REJECTED status).
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        data = {
            'target_column': target_column,
            'prediction_type': 'Auto Detect',
            'excluded_columns': json.dumps(excluded_columns or [])
        }
        if rules:
            data['rules'] = json.dumps(rules)
            
        response = requests.post(f"{self.base_url}/analyze", files=files, data=data, headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to run governance checks: {response.text}")

    def assert_ready_for_deployment(self, df: pd.DataFrame, target_column: str, rules: dict = None, excluded_columns: list = None) -> bool:
        """
        Acts as a CI/CD block. Raises a RuntimeError if the dataset is REJECTED by governance checks.
        """
        print(f"🔒 Running Governance Checks on Target: {target_column}...")
        results = self.run_governance_checks(df, target_column, rules, excluded_columns)
        
        status = results.get("governance", {}).get("status", "UNKNOWN")
        if status == "REJECTED":
            reason = results.get("governance", {}).get("reason", "Dataset failed governance checks.")
            raise RuntimeError(f"Governance Check Failed: {reason} (Status: {status})")
            
        print(f"✅ Governance Check Passed! Status: {status}")
        return True

    def start_analysis_job(self, df: pd.DataFrame, target_column: str, excluded_columns: list = None) -> str:
        """
        Starts an asynchronous analysis job and returns a job_id.
        """
        csv_buffer = self._df_to_csv_buffer(df)
        files = {'file': ("dataset.csv", csv_buffer.getvalue(), 'text/csv')}
        data = {
            'target_column': target_column,
            'prediction_type': 'Auto Detect',
            'excluded_columns': json.dumps(excluded_columns or [])
        }
            
        response = requests.post(f"{self.base_url}/analyze_async", files=files, data=data, headers=self._get_headers())
        if response.status_code == 200:
            return response.json().get("job_id")
        else:
            raise Exception(f"Failed to start async analysis job: {response.text}")

    def get_job_status(self, job_id: str) -> dict:
        """
        Polls the status of an asynchronous analysis job.
        """
        response = requests.get(f"{self.base_url}/job/{job_id}", headers=self._get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get job status: {response.text}")
