import io
import requests
import webbrowser
import pandas as pd
from typing import Optional

class Client:
    """
    The DataScope ML Client.
    Connects your local Jupyter Notebooks or Python IDEs directly to the DataScope platform.
    """
    
    def __init__(self, api_key: str = None, base_url: str = "https://datascope-app.vercel.app"):
        """
        Initialize the DataScope client.
        
        Args:
            api_key: Your secret API key (can be generated from the DataScope dashboard).
            base_url: The URL of your Next.js frontend or FastAPI backend.
                      For production, point this to your Next.js API endpoint.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
    def analyze(self, df: pd.DataFrame, target_column: str, prediction_type: str, project_name: str = "my_dataset"):
        """
        Uploads the dataframe to DataScope, runs full Layer 1 & 2 analytics, 
        and automatically opens the results dashboard in your browser.
        
        Args:
            df: The pandas DataFrame to analyze.
            target_column: The target variable for prediction.
            prediction_type: The type of prediction ('classification', 'regression', or 'Auto Detect').
            project_name: A human-readable name for this dataset.
        """
        print(f"🚀 Initializing DataScope audit for project: '{project_name}'")
        print(f"📊 Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 1. Convert DataFrame to CSV in memory (no local file saving needed!)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # 2. Prepare payload
        files = {
            'file': (f"{project_name}.csv", csv_buffer.getvalue(), 'text/csv')
        }
        data = {
            'target_column': target_column,
            'prediction_type': prediction_type
        }
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
            
        print("⏳ Uploading to DataScope engine and computing analytics...")
        
        try:
            # Point this to the Next.js API route that handles upload orchestration
            # NOTE: For local development, this is likely http://localhost:3000/api/upload
            response = requests.post(f"{self.base_url}/api/upload", files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                # Assuming the Next.js API returns the Prisma database ID of the result
                job_id = result.get("datasetId") or result.get("id") or result.get("job_id")
                
                if not job_id:
                    print("⚠️ Success, but could not parse job_id from response.")
                    return result
                
                # 3. Generate the magic link
                dashboard_url = f"{self.base_url}/results/{job_id}"
                print(f"✅ Analysis Complete!")
                print(f"🔗 View your interactive dashboard here: {dashboard_url}")
                
                # 4. Automatically pop open their browser
                webbrowser.open(dashboard_url)
                
                return dashboard_url
            else:
                print(f"❌ Error analyzing dataset: HTTP {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("Please ensure your DataScope backend is running.")
            return None
