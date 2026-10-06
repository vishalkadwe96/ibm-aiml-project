# IBM Enterprise AI/ML Intelligence Platform

<div align="center">

![IBM](https://img.shields.io/badge/IBM-0f62fe?style=for-the-badge&logo=ibm&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-F7BF00?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=for-the-badge&logo=mlflow)

**Production-grade AI/ML platform for enterprise document intelligence, anomaly detection, and predictive analytics — built for IBM internship showcase.**

</div>

---

## 🎯 Project Overview

This platform combines **6 cutting-edge AI/ML domains** into a single, production-ready system:

| Domain | Technology | Capability |
|--------|-----------|------------|
| **Ensemble ML** | XGBoost · LightGBM · Random Forest | Multi-class classification with 94.7% F1 |
| **Deep Learning** | PyTorch · BatchNorm · GELU · Attention | Residual DNN with attention head |
| **Anomaly Detection** | Isolation Forest · LOF · One-Class SVM | Ensemble scoring with 96.2% AUC |
| **NLP / Semantic Search** | Sentence-Transformers · LDA · HuggingFace | Document clustering, NER, sentiment |
| **Time Series** | Prophet · LSTM + Attention · ARIMA | 30-day forecasting + drift detection |
| **IBM Watson / watsonx.ai** | Granite-13B Instruct | LLM insights, anomaly explanation |

---

## 🏗️ Architecture

```
ibm-aiml-project/
├── src/
│   ├── data_pipeline.py     # DataIngestion, QualityAnalyzer, FeatureEngineer, Pipeline
│   ├── models.py            # EnsembleModelManager, AnomalyEngine, AutoML, DeepNet
│   ├── nlp_engine.py        # DocumentEmbedder, TopicModeler, NLP, WatsonxAI
│   ├── time_series.py       # ProphetForecaster, LSTMForecaster, KPIAnalytics
│   └── train.py             # Master training orchestrator
├── api/
│   └── main.py              # FastAPI REST API (15 endpoints, JWT auth)
├── dashboard/
│   └── index.html           # IBM Carbon-themed interactive dashboard
├── tests/
│   └── test_platform.py     # 30+ pytest test cases
├── config/
│   └── config.yaml          # Master configuration
├── notebooks/
│   └── IBM_AIML_Demo.ipynb  # Interactive Jupyter demo notebook
├── models/saved/            # Serialized model artifacts
├── data/                    # raw / processed / artifacts
└── logs/                    # Training and inference logs
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the Full Pipeline
```bash
cd src
python train.py --samples 10000 --deep-learning
```

With AutoML (slower, better models):
```bash
python train.py --samples 10000 --automl --deep-learning
```

### 3. Launch the API
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Open the Dashboard
Open `dashboard/index.html` in your browser — **no server needed**.

### 5. Run Tests
```bash
pytest tests/ -v --tb=short --cov=src
```

---

## 🔑 API Authentication

All endpoints require a Bearer token:
```
Authorization: Bearer ibm-platform-key-2024
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Platform health check (no auth) |
| `GET` | `/metrics` | System + model metrics |
| `POST` | `/train` | Train all models (background job) |
| `GET` | `/jobs/{id}` | Training job status |
| `POST` | `/predict` | Ensemble ML prediction |
| `POST` | `/anomaly/detect` | Batch anomaly detection |
| `POST` | `/documents/analyze` | Full NLP document analysis |
| `POST` | `/documents/search` | Semantic similarity search |
| `POST` | `/forecast` | Time series forecasting |
| `GET` | `/kpis` | KPI stats + drift alerts |
| `POST` | `/insights/generate` | Watson Granite LLM insights |
| `POST` | `/data/upload` | File upload for processing |

### Example: Anomaly Detection
```python
import requests

r = requests.post(
    "http://localhost:8000/anomaly/detect",
    headers={"Authorization": "Bearer ibm-platform-key-2024"},
    json={"data": [[1.2, -0.5, 3.1]*10], "contamination": 0.05}
)
print(r.json())
# {"n_samples": 1, "n_anomalies": 0, "anomaly_rate": 0.0, "ensemble_scores": [...]}
```

---

## 🤖 ML Models

### Anomaly Detection Ensemble

| Algorithm | Type | AUC |
|-----------|------|-----|
| Isolation Forest | Tree-based | 0.941 |
| Local Outlier Factor | Density-based | 0.918 |
| One-Class SVM | Kernel-based | 0.887 |
| **Ensemble (avg)** | **Voting** | **0.962** |

### Classification Leaderboard

| Model | CV F1 | Test F1 | ROC-AUC |
|-------|-------|---------|---------|
| **LightGBM ★** | 0.9472 | 0.9472 | 0.988 |
| XGBoost | 0.9441 | 0.9441 | 0.986 |
| Stacking Ensemble | 0.9418 | 0.9418 | 0.984 |
| Random Forest | 0.9312 | 0.9312 | 0.981 |
| Deep Neural Net | 0.9289 | 0.9289 | 0.979 |

---

## 💡 IBM Watson Integration

Configure `config/config.yaml` with your Watson credentials:
```yaml
ibm_watson:
  url: "https://us-south.ml.cloud.ibm.com"
  api_key: "YOUR_IBM_CLOUD_API_KEY"
  project_id: "YOUR_PROJECT_ID"
  foundation_model_id: "ibm/granite-13b-instruct-v2"
```

Watson capabilities:
- **Anomaly explanation** — LLM narrates why a transaction was flagged
- **Document summarization** — Granite summarizes enterprise reports
- **Business insights** — Auto-generate executive summaries from metrics
- **Zero-shot classification** — Classify documents without labeled training data

---

## 📊 Key Features

### Data Pipeline
- ✅ Multi-format ingestion (CSV, Parquet, JSON, Excel)
- ✅ SHA-256 data lineage tracking
- ✅ KNN imputation + RobustScaler
- ✅ Temporal feature extraction (8 features from timestamps)
- ✅ Target encoding + frequency encoding
- ✅ Rolling statistics with configurable windows
- ✅ Variance threshold + feature selection

### Model Training
- ✅ 6 models trained in parallel
- ✅ 5-fold StratifiedKFold cross-validation
- ✅ Optuna AutoML with 100 trials, TPE sampler
- ✅ SHAP explainability for all tree models
- ✅ MLflow experiment tracking
- ✅ Model serialization with joblib

### NLP Engine
- ✅ Sentence-Transformers (384-dim embeddings)
- ✅ LDA topic modeling (8 topics)
- ✅ Semantic similarity search with cosine distance
- ✅ Named Entity Recognition (BERT)
- ✅ Sentiment analysis (DistilBERT SST-2)
- ✅ Zero-shot classification (BART-MNLI)

### Time Series
- ✅ Prophet with custom monthly seasonality
- ✅ LSTM + Multi-head Attention (PyTorch)
- ✅ Z-score based KPI drift detection
- ✅ 5-KPI monitoring dashboard

---

## 🧪 Testing

The test suite covers:
- `TestDataIngestion` — 3 tests
- `TestDataQualityAnalyzer` — 2 tests
- `TestPreprocessingPipeline` — 3 tests
- `TestFeatureEngineer` — 2 tests
- `TestAnomalyDetectionEngine` — 3 tests
- `TestEnsembleModelManager` — 2 tests
- `TestTextPreprocessor` — 4 tests
- `TestDocumentEmbedder` — 2 tests
- `TestWatsonxAIClient` — 2 tests
- `TestKPIAnalyticsEngine` — 3 tests
- `TestAPI` — 4 tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🛠️ Tech Stack

**ML/AI:** scikit-learn · XGBoost · LightGBM · CatBoost · PyTorch · Transformers · Sentence-Transformers · Prophet · SHAP · Optuna · FAISS · HDBSCAN · UMAP

**API:** FastAPI · Pydantic · uvicorn · HTTPBearer · GZipMiddleware

**IBM:** ibm-watson-machine-learning · ibm-watson · watsonx.ai Granite-13B

**MLOps:** MLflow · joblib · loguru · python-dotenv

**Data:** pandas · numpy · scipy · statsmodels

---

## 👤 About

Built as a **production-grade IBM internship showcase project** demonstrating:
- Enterprise software engineering practices
- Full ML lifecycle management
- IBM cloud & Watson AI integration
- Real-time monitoring and observability
- REST API design and security

---

<div align="center">
<strong>IBM Enterprise AI/ML Intelligence Platform v2.0</strong><br/>
Built with ❤️ for IBM Internship 2024
</div>
