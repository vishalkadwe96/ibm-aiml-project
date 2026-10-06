"""
IBM Enterprise AI/ML Platform — Test Suite
Uses pytest with mocking to avoid requiring GPU or Watson credentials.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────
# Data Pipeline Tests
# ──────────────────────────────────────────────────────────────────
class TestDataIngestion:
    def test_synthetic_dataset_shape(self):
        from data_pipeline import DataIngestion
        ingestion = DataIngestion(data_dir="/tmp/test_data")
        df = ingestion.generate_synthetic_dataset(n_samples=500, n_features=25)
        assert len(df) == 500
        assert "label" in df.columns
        assert "is_anomaly" in df.columns
        assert "timestamp" in df.columns

    def test_synthetic_dataset_anomaly_rate(self):
        from data_pipeline import DataIngestion
        ingestion = DataIngestion(data_dir="/tmp/test_data")
        df = ingestion.generate_synthetic_dataset(n_samples=1000)
        rate = df["is_anomaly"].mean()
        assert 0.01 <= rate <= 0.10, f"Anomaly rate {rate} out of expected range"

    def test_file_hash_consistency(self, tmp_path):
        from data_pipeline import DataIngestion
        ingestion = DataIngestion(data_dir=str(tmp_path))
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b,c\n1,2,3\n4,5,6")
        h1 = ingestion._file_hash(str(test_file))
        h2 = ingestion._file_hash(str(test_file))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestDataQualityAnalyzer:
    def setup_method(self):
        from data_pipeline import DataIngestion
        ing = DataIngestion(data_dir="/tmp/test_data")
        self.df = ing.generate_synthetic_dataset(n_samples=500)

    def test_report_shape(self):
        from data_pipeline import DataQualityAnalyzer
        qa = DataQualityAnalyzer(self.df)
        report = qa.run()
        assert "shape" in report
        assert report["shape"]["rows"] == 500

    def test_missing_values_detected(self):
        from data_pipeline import DataQualityAnalyzer
        qa = DataQualityAnalyzer(self.df)
        report = qa.run()
        assert "missing" in report
        # Should detect ~5% missing
        assert len(report["missing"]) > 0


class TestPreprocessingPipeline:
    def setup_method(self):
        from data_pipeline import DataIngestion
        ing = DataIngestion(data_dir="/tmp/test_data")
        self.df = ing.generate_synthetic_dataset(n_samples=300, n_features=25)

    def test_pipeline_output_shape(self):
        from data_pipeline import PreprocessingPipeline
        pp = PreprocessingPipeline(task="classification")
        X, y = pp.build(self.df)
        assert X.ndim == 2
        assert len(y) == len(self.df)
        assert X.shape[0] == len(self.df)

    def test_no_nan_in_output(self):
        from data_pipeline import PreprocessingPipeline
        pp = PreprocessingPipeline(task="classification")
        X, y = pp.build(self.df)
        assert not np.isnan(X).any(), "NaN values found after preprocessing"

    def test_label_encoding(self):
        from data_pipeline import PreprocessingPipeline
        pp = PreprocessingPipeline(task="classification")
        X, y = pp.build(self.df)
        assert y.dtype in [np.int32, np.int64, np.int_]


class TestFeatureEngineer:
    def setup_method(self):
        from data_pipeline import DataIngestion
        ing = DataIngestion(data_dir="/tmp/test_data")
        self.df = ing.generate_synthetic_dataset(n_samples=100)

    def test_temporal_features_added(self):
        from data_pipeline import FeatureEngineer
        fe = FeatureEngineer()
        df2 = fe.add_temporal_features(self.df)
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in df2.columns

    def test_frequency_encoding(self):
        from data_pipeline import FeatureEngineer
        fe = FeatureEngineer()
        df2 = fe.frequency_encode(self.df, ["dept"])
        assert "dept_freq_enc" in df2.columns
        assert df2["dept_freq_enc"].between(0, 1).all()


# ──────────────────────────────────────────────────────────────────
# Model Tests
# ──────────────────────────────────────────────────────────────────
class TestAnomalyDetectionEngine:
    def setup_method(self):
        np.random.seed(42)
        self.X_normal = np.random.normal(0, 1, (200, 10))
        self.X_anomaly = np.random.normal(5, 0.5, (20, 10))
        self.X = np.vstack([self.X_normal, self.X_anomaly])

    def test_fit_predict(self):
        from models import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(contamination=0.08)
        engine.fit(self.X_normal)
        results = engine.predict(self.X)
        assert "anomaly_label" in results
        assert "ensemble_score" in results
        assert len(results["anomaly_label"]) == len(self.X)

    def test_anomaly_labels_binary(self):
        from models import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(contamination=0.05)
        engine.fit(self.X_normal)
        results = engine.predict(self.X)
        labels = results["anomaly_label"]
        assert set(labels).issubset({0, 1})

    def test_detects_obvious_anomalies(self):
        from models import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine(contamination=0.08)
        engine.fit(self.X_normal)
        results = engine.predict(self.X)
        # Anomaly scores for injected anomalies should be higher
        normal_scores = results["ensemble_score"][:200]
        anomaly_scores = results["ensemble_score"][200:]
        assert anomaly_scores.mean() > normal_scores.mean(), "Anomaly scores not higher than normal"


class TestEnsembleModelManager:
    def setup_method(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=15, random_state=42)
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def test_train_all_returns_metrics(self):
        from models import EnsembleModelManager
        manager = EnsembleModelManager()
        metrics = manager.train_all(self.X_train, self.y_train, self.X_test, self.y_test)
        assert len(metrics) >= 3
        for name, m in metrics.items():
            assert "test_f1_weighted" in m
            assert 0.0 <= m["test_f1_weighted"] <= 1.0

    def test_best_model_set(self):
        from models import EnsembleModelManager
        manager = EnsembleModelManager()
        manager.train_all(self.X_train, self.y_train, self.X_test, self.y_test)
        assert manager.best_model is not None
        assert manager.best_model_name in ["xgboost", "lightgbm", "random_forest", "gradient_boosting"]


# ──────────────────────────────────────────────────────────────────
# NLP Tests
# ──────────────────────────────────────────────────────────────────
class TestTextPreprocessor:
    def test_clean_removes_urls(self):
        from nlp_engine import TextPreprocessor
        pp = TextPreprocessor()
        result = pp.clean("Visit https://ibm.com for more info")
        assert "url" in result  # tags lowercased after replacement
        assert "https" not in result

    def test_clean_removes_emails(self):
        from nlp_engine import TextPreprocessor
        pp = TextPreprocessor()
        result = pp.clean("Contact test@ibm.com for details")
        assert "email" in result  # tags lowercased after replacement

    def test_tokenize_removes_stopwords(self):
        from nlp_engine import TextPreprocessor
        pp = TextPreprocessor()
        tokens = pp.tokenize("the quick brown fox is running")
        assert "the" not in tokens
        assert "is" not in tokens

    def test_batch_process(self):
        from nlp_engine import TextPreprocessor
        pp = TextPreprocessor()
        texts = ["Hello world!", "IBM is great", "Machine learning rocks"]
        cleaned = pp.process_batch(texts)
        assert len(cleaned) == 3


class TestDocumentEmbedder:
    def test_tfidf_fallback(self):
        from nlp_engine import DocumentEmbedder
        embedder = DocumentEmbedder.__new__(DocumentEmbedder)
        embedder.model_name = "test"
        embedder.model = None
        embedder.tfidf = None
        embedder.embeddings_ = None
        embedder.corpus_ = None
        embedder._use_sbert = False
        texts = ["IBM AI platform", "Machine learning models", "Deep neural networks"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape[0] == 3

    def test_semantic_search_returns_ranked(self):
        from nlp_engine import DocumentEmbedder
        embedder = DocumentEmbedder.__new__(DocumentEmbedder)
        embedder.model = None
        embedder.tfidf = None
        embedder.embeddings_ = None
        embedder.corpus_ = None
        embedder._use_sbert = False
        texts = ["IBM cloud computing", "Machine learning anomaly", "Deep learning NLP"]
        embedder.embed(texts)
        results = embedder.semantic_search("machine learning", top_k=2)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert results[0]["score"] >= results[1]["score"]


class TestWatsonxAIClient:
    def test_demo_mode_generate(self):
        from nlp_engine import WatsonxAIClient
        client = WatsonxAIClient()
        client.connect()  # no creds → demo mode
        result = client.generate("summarize this document")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_insight(self):
        from nlp_engine import WatsonxAIClient
        client = WatsonxAIClient()
        client.connect()
        insight = client.generate_insight({"accuracy": 0.94, "f1": 0.92})
        assert isinstance(insight, str)


# ──────────────────────────────────────────────────────────────────
# Time Series Tests
# ──────────────────────────────────────────────────────────────────
class TestKPIAnalyticsEngine:
    def test_track_and_statistics(self):
        from time_series import KPIAnalyticsEngine
        engine = KPIAnalyticsEngine()
        vals = [float(x) for x in range(100)]
        engine.track("test_kpi", vals)
        stats = engine.compute_statistics("test_kpi")
        assert abs(stats["mean"] - 49.5) < 0.01
        assert "trend" in stats

    def test_drift_detection(self):
        from time_series import KPIAnalyticsEngine
        engine = KPIAnalyticsEngine()
        # No drift: stable signal
        stable = [1.0 + np.random.normal(0, 0.01) for _ in range(100)]
        engine.track("stable_kpi", stable)
        result = engine.detect_drift("stable_kpi", window=30)
        assert not result["drift_detected"]

        # Drift: sudden jump
        engine.track("drifting_kpi", [1.0] * 50 + [100.0] * 50)
        result2 = engine.detect_drift("drifting_kpi", window=30)
        assert result2["drift_detected"]

    def test_generate_synthetic_kpis(self):
        from time_series import KPIAnalyticsEngine
        engine = KPIAnalyticsEngine()
        kpis = engine.generate_synthetic_kpis(n_days=100)
        assert "model_accuracy" in kpis
        assert len(kpis["model_accuracy"]) == 100


# ──────────────────────────────────────────────────────────────────
# API Tests (with TestClient)
# ──────────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
        try:
            from fastapi.testclient import TestClient
            from main import app
            self.client = TestClient(app, raise_server_exceptions=False)
            self.headers = {"Authorization": "Bearer ibm-platform-key-2024"}
        except Exception:
            self.client = None

    def test_health_check(self):
        if not self.client:
            pytest.skip("API not loadable in test environment")
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_unauthorized_returns_401(self):
        if not self.client:
            pytest.skip("API not loadable in test environment")
        resp = self.client.get("/metrics")
        assert resp.status_code == 401

    def test_predict_demo_mode(self):
        if not self.client:
            pytest.skip("API not loadable in test environment")
        payload = {"features": [0.5] * 30, "model_type": "ensemble"}
        resp = self.client.post("/predict", json=payload, headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data

    def test_anomaly_detect_demo(self):
        if not self.client:
            pytest.skip("API not loadable in test environment")
        payload = {"data": [[0.5] * 20 for _ in range(10)], "contamination": 0.1}
        resp = self.client.post("/anomaly/detect", json=payload, headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "n_samples" in data
        assert data["n_samples"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
