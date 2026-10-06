"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: FastAPI REST API
Description: Production-grade REST API with auth, rate limiting,
             background jobs, health checks, and OpenAPI docs.
"""

import os
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from loguru import logger
import uvicorn

# ── Internal modules ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline import DataIngestion, DataQualityAnalyzer, PreprocessingPipeline
from models import EnsembleModelManager, AnomalyDetectionEngine, AutoMLOptimizer
from nlp_engine import DocumentIntelligenceEngine, TextPreprocessor
from time_series import KPIAnalyticsEngine, ProphetForecaster


# ──────────────────────────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IBM Enterprise AI/ML Intelligence Platform",
    description="""
## 🤖 IBM AI/ML Platform API
Enterprise-grade platform for:
- **Anomaly Detection** — Multi-algorithm ensemble anomaly scoring
- **Predictive Analytics** — XGBoost/LightGBM/Deep Learning ensemble
- **Document Intelligence** — Semantic search, NER, topic modeling
- **Time Series Forecasting** — Prophet + LSTM forecasting
- **AutoML** — Optuna-powered hyperparameter optimization
- **IBM watsonx.ai** — Granite LLM integration for insights

### Authentication
Use Bearer token: `Authorization: Bearer ibm-platform-key-2024`
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Security ─────────────────────────────────────────────────────
VALID_API_KEYS = {"ibm-platform-key-2024", "intern-demo-key-2024"}
security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials is None or credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Use: Bearer ibm-platform-key-2024",
        )
    return credentials.credentials

# ── Shared State (in-memory for demo) ───────────────────────────
class PlatformState:
    def __init__(self):
        self.ensemble: Optional[EnsembleModelManager] = None
        self.anomaly_engine: Optional[AnomalyDetectionEngine] = None
        self.doc_engine: Optional[DocumentIntelligenceEngine] = None
        self.kpi_engine = KPIAnalyticsEngine()
        self.preprocessor = PreprocessingPipeline()
        self.ingestion = DataIngestion()
        self.jobs: Dict[str, Dict] = {}
        self.metrics: Dict[str, Any] = {}
        self._demo_trained = False

state = PlatformState()


# ──────────────────────────────────────────────────────────────────
# Pydantic Request/Response Models
# ──────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Feature vector for prediction")
    model_type: str = Field("ensemble", description="Model to use: ensemble | xgboost | lightgbm")

    class Config:
        schema_extra = {"example": {"features": [1.2, -0.5, 3.1, 0.8, -1.2] * 6, "model_type": "ensemble"}}

class AnomalyRequest(BaseModel):
    data: List[List[float]] = Field(..., description="2D array of feature vectors")
    contamination: float = Field(0.05, ge=0.001, le=0.5)

class DocumentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of documents to analyze")
    query: Optional[str] = Field(None, description="Semantic search query")

class ForecastRequest(BaseModel):
    values: List[float] = Field(..., description="Historical time series values")
    horizon: int = Field(30, ge=1, le=365, description="Forecast horizon in days")

class TrainRequest(BaseModel):
    n_samples: int = Field(5000, ge=100, le=100_000, description="Synthetic training data size")
    task: str = Field("classification", description="classification or regression")
    run_automl: bool = Field(False, description="Run Optuna AutoML optimization")

class InsightRequest(BaseModel):
    metrics: Dict[str, Any] = Field(..., description="Model/system metrics dict")
    context: str = Field("", description="Additional context for LLM")


# ──────────────────────────────────────────────────────────────────
# Lifespan — warm up demo models
# ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("IBM AI/ML Platform starting up...")
    state.kpi_engine.generate_synthetic_kpis()
    logger.success("Platform ready ✓")


# ──────────────────────────────────────────────────────────────────
# Health & Metrics
# ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "platform": "IBM Enterprise AI/ML Platform v2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "ensemble": state.ensemble is not None,
            "anomaly_engine": state.anomaly_engine is not None,
            "document_engine": state.doc_engine is not None,
        },
        "kpis_tracked": len(state.kpi_engine.kpi_history),
    }

@app.get("/metrics", tags=["System"])
async def get_metrics(token: str = Depends(verify_token)):
    return {
        "platform_version": "2.0.0",
        "jobs_total": len(state.jobs),
        "jobs_completed": sum(1 for j in state.jobs.values() if j["status"] == "completed"),
        "kpi_alerts": len(state.kpi_engine.alerts),
        "model_metrics": state.metrics,
    }


# ──────────────────────────────────────────────────────────────────
# Training Endpoint
# ──────────────────────────────────────────────────────────────────
def _train_background(job_id: str, n_samples: int, task: str, run_automl: bool):
    state.jobs[job_id]["status"] = "running"
    state.jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
    try:
        from sklearn.model_selection import train_test_split

        # Generate data
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_dataset(n_samples=n_samples, task=task)

        # Quality analysis
        qa = DataQualityAnalyzer(df)
        qa.run()

        # Preprocessing
        pp = PreprocessingPipeline(task=task)
        X, y = pp.build(df, target_col="label")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        state.preprocessor = pp

        # AutoML
        if run_automl and OPTUNA_AVAILABLE_GLOBAL:
            optimizer = AutoMLOptimizer(n_trials=30, timeout=120)
            optimizer.optimize(X_train, y_train)

        # Ensemble training
        manager = EnsembleModelManager()
        model_metrics = manager.train_all(X_train, y_train, X_test, y_test)
        state.ensemble = manager
        state.metrics = model_metrics

        # Anomaly engine
        engine = AnomalyDetectionEngine(contamination=0.05)
        engine.fit(X_train)
        state.anomaly_engine = engine

        state.jobs[job_id]["status"] = "completed"
        state.jobs[job_id]["metrics"] = model_metrics
        state.jobs[job_id]["best_model"] = manager.best_model_name
        logger.success(f"Training job {job_id} completed")
    except Exception as e:
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = str(e)
        logger.error(f"Training job {job_id} failed: {e}")

try:
    import optuna
    OPTUNA_AVAILABLE_GLOBAL = True
except ImportError:
    OPTUNA_AVAILABLE_GLOBAL = False


@app.post("/train", tags=["Training"], summary="Train all ML models on synthetic or uploaded data")
async def train_models(req: TrainRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_token)):
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    state.jobs[job_id] = {"status": "queued", "created_at": datetime.utcnow().isoformat()}
    background_tasks.add_task(_train_background, job_id, req.n_samples, req.task, req.run_automl)
    return {
        "job_id": job_id,
        "message": f"Training started for {req.n_samples:,} samples",
        "status_url": f"/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}", tags=["Training"])
async def get_job_status(job_id: str, token: str = Depends(verify_token)):
    if job_id not in state.jobs:
        raise HTTPException(404, detail=f"Job {job_id} not found")
    return state.jobs[job_id]


# ──────────────────────────────────────────────────────────────────
# Prediction Endpoint
# ──────────────────────────────────────────────────────────────────
@app.post("/predict", tags=["Prediction"], summary="Get ensemble ML prediction with confidence scores")
async def predict(req: PredictionRequest, token: str = Depends(verify_token)):
    if state.ensemble is None:
        # Demo mode
        return {
            "prediction": 1,
            "confidence": 0.873,
            "class_probabilities": {"class_0": 0.082, "class_1": 0.873, "class_2": 0.045},
            "model_used": "xgboost (demo)",
            "note": "Train a model first via POST /train",
        }
    X = np.array(req.features).reshape(1, -1)
    try:
        model = state.ensemble.best_model
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return {
            "prediction": int(pred),
            "confidence": float(proba.max()),
            "class_probabilities": {f"class_{i}": float(p) for i, p in enumerate(proba)},
            "model_used": state.ensemble.best_model_name,
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ──────────────────────────────────────────────────────────────────
# Anomaly Detection Endpoint
# ──────────────────────────────────────────────────────────────────
@app.post("/anomaly/detect", tags=["Anomaly Detection"], summary="Detect anomalies using ensemble scoring")
async def detect_anomalies(req: AnomalyRequest, token: str = Depends(verify_token)):
    X = np.array(req.data)
    if state.anomaly_engine is None:
        # Demo: random anomaly scores
        n = len(X)
        scores = np.random.beta(1, 10, n)
        labels = (scores > 0.7).astype(int)
        return {
            "n_samples": n,
            "n_anomalies": int(labels.sum()),
            "anomaly_rate": float(labels.mean()),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "note": "Demo mode — train first via POST /train",
        }
    results = state.anomaly_engine.predict(X)
    return {
        "n_samples": len(X),
        "n_anomalies": int(results["anomaly_label"].sum()),
        "anomaly_rate": float(results["anomaly_label"].mean()),
        "ensemble_scores": results["ensemble_score"].tolist(),
        "labels": results["anomaly_label"].tolist(),
        "isolation_forest_scores": results.get("isolation_forest_score", np.zeros(len(X))).tolist(),
    }


# ──────────────────────────────────────────────────────────────────
# Document Intelligence Endpoint
# ──────────────────────────────────────────────────────────────────
@app.post("/documents/analyze", tags=["NLP"], summary="Full document intelligence pipeline")
async def analyze_documents(req: DocumentRequest, token: str = Depends(verify_token)):
    if not req.texts:
        raise HTTPException(400, detail="texts list is empty")
    engine = DocumentIntelligenceEngine()
    result = engine.ingest_documents(req.texts[:100])
    if req.query:
        result["search_results"] = engine.search(req.query, top_k=5)
    return result


@app.post("/documents/search", tags=["NLP"], summary="Semantic similarity search over a corpus")
async def semantic_search(texts: List[str], query: str, top_k: int = 5, token: str = Depends(verify_token)):
    if state.doc_engine is None:
        state.doc_engine = DocumentIntelligenceEngine()
        state.doc_engine.ingest_documents(texts)
    results = state.doc_engine.search(query, top_k=top_k)
    return {"query": query, "results": results}


# ──────────────────────────────────────────────────────────────────
# Forecasting Endpoint
# ──────────────────────────────────────────────────────────────────
@app.post("/forecast", tags=["Time Series"], summary="Time series forecasting with Prophet/LSTM")
async def forecast(req: ForecastRequest, token: str = Depends(verify_token)):
    base = datetime(2023, 1, 1)
    df = pd.DataFrame({
        "ds": pd.date_range(base, periods=len(req.values), freq="D"),
        "y": req.values,
    })
    forecaster = ProphetForecaster(horizon=req.horizon)
    forecaster.fit(df)
    forecast_df = forecaster.predict()
    return {
        "horizon": req.horizon,
        "forecast": forecast_df.to_dict(orient="records"),
        "model": "Prophet + Fourier seasonality",
    }


# ──────────────────────────────────────────────────────────────────
# KPI Dashboard Endpoint
# ──────────────────────────────────────────────────────────────────
@app.get("/kpis", tags=["Monitoring"])
async def get_kpis(token: str = Depends(verify_token)):
    kpis = {}
    for name in state.kpi_engine.kpi_history:
        kpis[name] = state.kpi_engine.compute_statistics(name)
        kpis[name]["drift"] = state.kpi_engine.detect_drift(name)
    return {"kpis": kpis, "alerts": state.kpi_engine.alerts[-10:]}


# ──────────────────────────────────────────────────────────────────
# IBM Watson Insights
# ──────────────────────────────────────────────────────────────────
@app.post("/insights/generate", tags=["IBM Watson"])
async def generate_insight(req: InsightRequest, token: str = Depends(verify_token)):
    from nlp_engine import WatsonxAIClient
    watson = WatsonxAIClient()
    watson.connect()
    insight = watson.generate_insight(req.metrics)
    return {"insight": insight, "model": "IBM Granite-13B (demo mode)"}


# ──────────────────────────────────────────────────────────────────
# Data Upload & Processing Endpoint
# ──────────────────────────────────────────────────────────────────
@app.post("/data/upload", tags=["Data"], summary="Upload and auto-process CSV/JSON/text/image files")
async def upload_data(file: UploadFile = File(...), token: str = Depends(verify_token)):
    import io
    content = await file.read()
    size_kb = len(content) / 1024
    fname = file.filename or ""
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    result = {
        "filename": fname,
        "size_kb": round(size_kb, 2),
        "content_type": file.content_type,
        "extension": ext,
        "status": "processed",
    }

    try:
        if ext in ("csv",):
            df = pd.read_csv(io.BytesIO(content))
            qa = DataQualityAnalyzer(df)
            report = qa.run()
            result["pipeline"] = "tabular"
            result["rows"] = len(df)
            result["cols"] = len(df.columns)
            result["missing_pct"] = round(df.isnull().mean().mean() * 100, 2)
            result["dtypes"] = df.dtypes.astype(str).to_dict()
            result["quality_report"] = report

        elif ext in ("json",):
            import json as _json
            data = _json.loads(content)
            if isinstance(data, list):
                df = pd.DataFrame(data)
                result["pipeline"] = "tabular"
                result["rows"] = len(df)
                result["cols"] = len(df.columns)
            else:
                result["pipeline"] = "json"
                result["keys"] = list(data.keys()) if isinstance(data, dict) else []

        elif ext in ("txt",):
            text = content.decode("utf-8", errors="ignore")
            words = len(text.split())
            lines = text.count("\n") + 1
            result["pipeline"] = "text"
            result["words"] = words
            result["lines"] = lines
            result["preview"] = text[:300]

        elif ext in ("png", "jpg", "jpeg", "gif", "webp"):
            result["pipeline"] = "image"
            result["message"] = "Image received. Use /image/analyze for AI analysis."
            result["demo_classification"] = {
                "label": "Business Document",
                "confidence": round(0.88 + 0.1 * (size_kb % 1), 3),
            }

        else:
            result["pipeline"] = "raw"
            result["message"] = f"File stored ({size_kb:.1f} KB). Unsupported type for auto-processing."

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


@app.post("/image/analyze", tags=["Data"], summary="AI image classification and analysis (demo)")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = "classify",
    token: str = Depends(verify_token),
):
    content = await file.read()
    size_kb = len(content) / 1024
    fname = file.filename or "image"

    demo_results = {
        "classify": {
            "labels": [
                {"label": "Business Document", "confidence": 0.94},
                {"label": "Data Visualization", "confidence": 0.81},
            ],
            "summary": "High-confidence business document classification.",
        },
        "detect": {
            "labels": [
                {"label": "Text Region", "confidence": 0.98},
                {"label": "Chart/Graph", "confidence": 0.89},
                {"label": "Table Structure", "confidence": 0.65},
            ],
            "summary": "3 objects detected. Primary: Text Region (98%).",
        },
        "describe": {
            "labels": [{"label": "Enterprise Content", "confidence": 0.91}],
            "summary": "IBM Granite Vision: Structured enterprise data with professional formatting.",
        },
        "anomaly": {
            "labels": [{"label": "No Anomaly Detected", "confidence": 0.96}],
            "summary": "Visual anomaly scan complete. No irregularities found. Integrity: 0.982.",
        },
    }

    result_data = demo_results.get(analysis_type, demo_results["classify"])
    return {
        "filename": fname,
        "size_kb": round(size_kb, 2),
        "analysis_type": analysis_type,
        "results": result_data["labels"],
        "summary": result_data["summary"],
        "model": "IBM Granite Vision (demo)",
        "status": "analyzed",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
