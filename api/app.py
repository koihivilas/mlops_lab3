import os
import logging
import requests
import time
import re
import nltk
from nltk.stem import WordNetLemmatizer
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple lemmatization
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

# Data Drift collection
reference_data = None
current_data = []

# Prometheus Metrics
REQUEST_COUNT = Counter("api_predict_requests_total", "Total count of requests to /predict")
REQUEST_LATENCY = Histogram("api_predict_request_latency_seconds", "Latency of requests to /predict")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reference_data
    
    # Try to load a realistic reference dataset if provided
    ref_path = "reference_dataset.csv"
    if os.path.exists(ref_path):
        logger.info(f"Loading realistic reference data from {ref_path}")
        reference_data = pd.read_csv(ref_path)
    else:
        logger.warning(f"{ref_path} not found. Using a small dummy reference dataset.")
        # Dummy reference data that represents our original training data distribution
        reference_data = pd.DataFrame({
            "text": [
                "My payment failed and I need a refund.",
                "The app crashes when I open the dashboard.",
                "How do I change my password?",
                "Unable to connect to the VPN.",
                "Please send the invoice for last month."
            ]
        })
    yield

app = FastAPI(title="Proxy to Databricks Serving API", lifespan=lifespan)

@app.get("/metrics", summary="Get Prometheus Metrics")
async def get_metrics():
    """
    Returns the Prometheus metrics collected by the API.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)

@app.post("/predict")
async def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")
    endpoint_name = os.getenv("DATABRICKS_ENDPOINT_NAME", "ticket-classifier-endpoint")

    if not host or not token:
        raise HTTPException(status_code=500, detail="Databricks credentials not configured in API.")

    # Save request text for evidently drift calculation
    current_data.append({"text": request.text})

    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    cleaned_text = clean_text(request.text)
    
    # Databricks Python Model format
    payload = {
        "dataframe_records": [{"text": cleaned_text}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Databricks Error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Endpoint Error: {response.text}")
            
        # Parse the JSON response returned by our TicketClassifierWrapper in Databricks
        result = response.json()["predictions"][0]
        
        REQUEST_LATENCY.observe(time.time() - start_time)
        return result
        
    except Exception as e:
        logger.error(f"Error calling Databricks API: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/drift-report", response_class=HTMLResponse)
async def generate_drift_report():
    if not current_data:
        return "<p>No prediction data collected yet.</p>"
    
    if len(current_data) < 5:
        return f"<p>Not enough prediction data collected yet. Minimum 5 requests required to run TF-IDF text drift analysis safely. Current requests: {len(current_data)}</p>"
        
    if reference_data is None or len(reference_data) < 5:
        return "<p>Reference dataset is missing or too small to compute drift.</p>"
    
    import warnings
    warnings.filterwarnings('ignore')
    
    ref_df = reference_data
    curr_df = pd.DataFrame(current_data)
    
    column_mapping = ColumnMapping(
        text_features=["text"]
    )
    
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
        
        report_path = "drift_report.html"
        report.save_html(report_path)
        
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        return HTMLResponse(content=f"<p>Error generating drift report: {str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
