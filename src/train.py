"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: Master Training Script
Description: Full end-to-end training pipeline orchestrator.
             Run with: python src/train.py
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Silence MLflow's noisy git-not-found warnings
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
        level="INFO",
    )
    logger.add("logs/training.log", rotation="10 MB", retention="7 days", level="DEBUG")


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        IBM Enterprise AI/ML Intelligence Platform           ║
║                     Training Pipeline v2.0                   ║
║              Powered by XGBoost · LightGBM · PyTorch         ║
║                   IBM watsonx.ai · MLflow                    ║
╚══════════════════════════════════════════════════════════════╝
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="IBM AI/ML Platform Training")
    parser.add_argument("--samples", type=int, default=10_000, help="Training samples")
    parser.add_argument("--features", type=int, default=30, help="Number of features")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--automl", action="store_true", help="Run AutoML with Optuna")
    parser.add_argument("--deep-learning", action="store_true", help="Train deep learning model")
    parser.add_argument("--skip-anomaly", action="store_true", help="Skip anomaly detection")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP pipeline")
    return parser.parse_args()


def main():
    setup_logging()
    banner()
    args = parse_args()
    t_total = time.time()

    # ── Import modules ──────────────────────────────────────────
    from data_pipeline import DataIngestion, DataQualityAnalyzer, PreprocessingPipeline, FeatureEngineer
    from models import EnsembleModelManager, AnomalyDetectionEngine, AutoMLOptimizer, DeepLearningClassifier
    from nlp_engine import DocumentIntelligenceEngine, WatsonxAIClient
    from time_series import KPIAnalyticsEngine, ProphetForecaster

    results_summary = {}

    # ══════════════════════════════════════════════════════════
    # STEP 1: Data Generation & Quality Analysis
    # ══════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("STEP 1/6 — Data Ingestion & Quality Analysis")
    logger.info("━" * 60)

    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_dataset(
        n_samples=args.samples,
        n_features=args.features,
        task=args.task,
    )

    qa = DataQualityAnalyzer(df)
    qa_report = qa.run()
    qa.print_summary()
    results_summary["data_quality"] = qa_report

    # ══════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ══════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("STEP 2/6 — Feature Engineering")
    logger.info("━" * 60)

    fe = FeatureEngineer()
    df = fe.add_temporal_features(df)
    cat_cols = ["dept", "region", "priority"]
    df = fe.target_encode(df, cat_cols, target="label")
    df = fe.frequency_encode(df, cat_cols)
    logger.success(f"Feature engineering complete: {len(df.columns)} total features")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Preprocessing
    # ══════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("STEP 3/6 — Preprocessing Pipeline")
    logger.info("━" * 60)

    pp = PreprocessingPipeline(task=args.task)
    X, y = pp.build(df, target_col="label")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if args.task == "classification" else None,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42,
        stratify=y_train if args.task == "classification" else None,
    )
    pp.save()
    logger.success(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # ══════════════════════════════════════════════════════════
    # STEP 4: AutoML + Ensemble Training
    # ══════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("STEP 4/6 — Ensemble Model Training")
    logger.info("━" * 60)

    if args.automl:
        logger.info("Running AutoML optimization (Optuna)...")
        automl = AutoMLOptimizer(n_trials=50, timeout=180)
        best_params = automl.optimize(X_train, y_train, model_type="lightgbm")
        results_summary["automl_best_params"] = best_params
        results_summary["automl_best_score"] = automl.best_score

    manager = EnsembleModelManager(experiment_name="IBM-AIML-Internship")
    model_metrics = manager.train_all(X_train, y_train, X_test, y_test)
    results_summary["model_metrics"] = model_metrics

    logger.success("Model Comparison:")
    logger.success("─" * 55)
    for name, m in model_metrics.items():
        logger.success(f"  {name:25s} | F1: {m['test_f1_weighted']:.4f} | Acc: {m['test_accuracy']:.4f}")
    logger.success("─" * 55)
    logger.success(f"  🏆 Best: {manager.best_model_name}")

    manager.save_best()

    # SHAP explanations
    shap_values = manager.explain_with_shap(X_test[:100])
    if shap_values is not None:
        logger.success("SHAP explanations computed for 100 test samples")

    # ══════════════════════════════════════════════════════════
    # STEP 4b: Deep Learning (optional)
    # ══════════════════════════════════════════════════════════
    if args.deep_learning:
        logger.info("Training Deep Learning model (PyTorch)...")
        n_classes = len(np.unique(y_train))
        dl = DeepLearningClassifier(
            input_dim=X_train.shape[1],
            n_classes=n_classes,
            epochs=30,
            lr=1e-3,
            batch_size=256,
        )
        dl.fit(X_train, y_train, X_val, y_val)
        dl_preds = dl.predict(X_test)
        from sklearn.metrics import f1_score, accuracy_score
        dl_f1 = f1_score(y_test, dl_preds, average="weighted")
        dl_acc = accuracy_score(y_test, dl_preds)
        dl.save()
        results_summary["deep_learning"] = {"f1": dl_f1, "accuracy": dl_acc}
        logger.success(f"  Deep Learning | F1: {dl_f1:.4f} | Acc: {dl_acc:.4f}")

    # ══════════════════════════════════════════════════════════
    # STEP 5: Anomaly Detection
    # ══════════════════════════════════════════════════════════
    if not args.skip_anomaly:
        logger.info("━" * 60)
        logger.info("STEP 5/6 — Anomaly Detection Engine")
        logger.info("━" * 60)

        anom_engine = AnomalyDetectionEngine(contamination=0.05)
        anom_engine.fit(X_train)
        anom_results = anom_engine.predict(X_test)
        anom_engine.save()

        # Evaluate against ground truth
        if "is_anomaly" in df.columns:
            from sklearn.metrics import classification_report
            gt = df["is_anomaly"].values[-len(X_test):]
            if len(gt) == len(anom_results["anomaly_label"]):
                from sklearn.metrics import f1_score
                anom_f1 = f1_score(gt, anom_results["anomaly_label"], zero_division=0)
                results_summary["anomaly_f1"] = anom_f1
                logger.success(f"Anomaly Detection F1: {anom_f1:.4f}")

    # ══════════════════════════════════════════════════════════
    # STEP 6: NLP + Watson
    # ══════════════════════════════════════════════════════════
    if not args.skip_nlp:
        logger.info("━" * 60)
        logger.info("STEP 6/6 — NLP & IBM Watson Integration")
        logger.info("━" * 60)

        sample_docs = [
            "IBM AI platform detected critical anomalies in transaction processing systems.",
            "Machine learning model accuracy improved by 12% after hyperparameter optimization.",
            "Enterprise NLP pipeline processed 10,000 financial documents with 94% precision.",
            "Watsonx.ai Granite model generated actionable business insights from raw data.",
            "Anomaly detection engine flagged 47 suspicious transactions for review.",
            "Deep learning classifier achieved 96.3% F1 score on customer churn prediction.",
            "Time series forecasting model predicts 15% increase in system load next month.",
            "SHAP explainability dashboard revealed top features driving fraud predictions.",
        ] * 5  # 40 docs

        doc_engine = DocumentIntelligenceEngine()
        doc_result = doc_engine.ingest_documents(sample_docs)
        doc_engine.topic_modeler.print_topics()
        logger.success(f"Documents: {doc_result['total_documents']} | Clusters: {doc_result['n_clusters']}")

        watson = WatsonxAIClient()
        watson.connect()
        insight = watson.generate_insight(model_metrics.get("xgboost", {}))
        logger.success(f"Watson Insight: {insight[:100]}...")
        results_summary["watson_insight"] = insight

    # ══════════════════════════════════════════════════════════
    # Time Series KPI Analysis
    # ══════════════════════════════════════════════════════════
    kpi_engine = KPIAnalyticsEngine()
    kpi_engine.generate_synthetic_kpis()
    for kpi in ["model_accuracy", "api_latency_ms", "anomaly_rate"]:
        stats = kpi_engine.compute_statistics(kpi)
        drift = kpi_engine.detect_drift(kpi)
        logger.info(f"KPI [{kpi:25s}] mean={stats['mean']:.4f} | drift={drift['drift_detected']}")

    # ══════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════
    Path("data/artifacts").mkdir(parents=True, exist_ok=True)
    results_path = "data/artifacts/training_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    total_time = time.time() - t_total
    logger.success("═" * 60)
    logger.success(f"  ✅ Full pipeline completed in {total_time:.1f}s")
    logger.success(f"  📊 Results saved to {results_path}")
    logger.success(f"  🏆 Best model: {manager.best_model_name}")
    logger.success(f"  🚀 Start API: cd api && python main.py")
    logger.success("═" * 60)


if __name__ == "__main__":
    main()
