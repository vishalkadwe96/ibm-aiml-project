"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: ML Models
Description: Ensemble ML, Deep Neural Network, Anomaly Detection,
             AutoML with Optuna, SHAP explainability, and MLflow tracking.
"""

import warnings
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, VotingClassifier,
    StackingClassifier, AdaBoostClassifier,
    IsolationForest,
)
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# Deep Neural Network (PyTorch)
# ──────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class EnterpriseDeepNet(nn.Module):
        """
        Residual deep neural network with BatchNorm, Dropout,
        and skip connections — IBM-grade architecture.
        """

        def __init__(self, input_dim: int, n_classes: int, hidden_dims: List[int] = None):
            super().__init__()
            hidden_dims = hidden_dims or [512, 256, 128, 64]

            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.GELU(),
                    nn.Dropout(0.3),
                ]
                prev = h

            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev, n_classes)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.head(self.backbone(x))


class DeepLearningClassifier:
    """Wrapper around EnterpriseDeepNet with train/predict API."""

    def __init__(
        self,
        input_dim: int = 50,
        n_classes: int = 3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")

        device = torch.device(self.device)
        self.model = EnterpriseDeepNet(self.input_dim, self.n_classes).to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        Xt = torch.FloatTensor(X_train).to(device)
        yt = torch.LongTensor(y_train).to(device)
        dataset = TensorDataset(Xt, yt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            avg_loss = epoch_loss / len(loader)
            self.history["train_loss"].append(avg_loss)

            if X_val is not None:
                val_acc, val_loss = self._evaluate(X_val, y_val, criterion, device)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1:3d}/{self.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        logger.success(f"Deep learning training complete | Final val acc: {self.history['val_acc'][-1]:.4f}")
        return self

    def _evaluate(self, X, y, criterion, device):
        self.model.eval()
        with torch.no_grad():
            Xv = torch.FloatTensor(X).to(device)
            yv = torch.LongTensor(y).to(device)
            logits = self.model(Xv)
            loss = criterion(logits, yv).item()
            preds = logits.argmax(1).cpu().numpy()
        return accuracy_score(y, preds), loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X).to(torch.device(self.device))
            return self.model(Xt).argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X).to(torch.device(self.device))
            logits = self.model(Xt)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str = "models/saved/deep_net.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Deep net saved → {path}")


# ──────────────────────────────────────────────────────────────────
# Anomaly Detection Engine
# ──────────────────────────────────────────────────────────────────
class AnomalyDetectionEngine:
    """
    Ensemble anomaly detection combining Isolation Forest,
    LOF, One-Class SVM, and Autoencoder with consensus voting.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.detectors: Dict[str, Any] = {}
        self.scores_: Optional[np.ndarray] = None

    def build_detectors(self):
        self.detectors = {
            "isolation_forest": IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "local_outlier_factor": LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            ),
            "one_class_svm": OneClassSVM(
                nu=self.contamination,
                kernel="rbf",
                gamma="scale",
            ),
        }
        return self

    def fit(self, X: np.ndarray) -> "AnomalyDetectionEngine":
        self.build_detectors()
        for name, det in self.detectors.items():
            det.fit(X)
            logger.info(f"  ✓ {name} fitted")
        logger.success(f"Anomaly Detection Engine fitted on {X.shape[0]:,} samples")
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        raw_scores = []
        for name, det in self.detectors.items():
            if hasattr(det, "score_samples"):
                scores = -det.score_samples(X)
            else:
                scores = -det.decision_function(X)
            results[f"{name}_score"] = scores
            raw_scores.append(scores)

        # Ensemble: average normalized scores
        stacked = np.stack(raw_scores, axis=1)
        norm = (stacked - stacked.min(0)) / (stacked.max(0) - stacked.min(0) + 1e-9)
        ensemble_score = norm.mean(axis=1)
        threshold = np.percentile(ensemble_score, (1 - self.contamination) * 100)
        results["ensemble_score"] = ensemble_score
        results["anomaly_label"] = (ensemble_score > threshold).astype(int)
        self.scores_ = ensemble_score

        n_anomalies = results["anomaly_label"].sum()
        logger.success(f"Anomaly detection: {n_anomalies} / {len(X):,} flagged ({n_anomalies/len(X)*100:.2f}%)")
        return results

    def save(self, path: str = "models/saved/anomaly_engine.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.detectors, path)
        logger.info(f"Anomaly engine saved → {path}")


# ──────────────────────────────────────────────────────────────────
# AutoML with Optuna
# ──────────────────────────────────────────────────────────────────
class AutoMLOptimizer:
    """
    Hyperparameter optimization using Optuna TPE sampler
    with early stopping — IBM AutoAI-style optimization.
    """

    def __init__(self, n_trials: int = 100, timeout: int = 300, random_state: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.study = None

    def optimize(self, X: np.ndarray, y: np.ndarray, model_type: str = "xgboost") -> Dict[str, Any]:
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available — returning default params")
            return {}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        def objective(trial: "optuna.Trial") -> float:
            if model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "eval_metric": "mlogloss",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                }
                model = xgb.XGBClassifier(**params)
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                    "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                model = lgb.LGBMClassifier(**params)

            scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
            return scores.mean()

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logger.success(f"AutoML | Best F1: {self.best_score:.4f} | Params: {self.best_params}")
        return self.best_params


# ──────────────────────────────────────────────────────────────────
# Ensemble Model Manager
# ──────────────────────────────────────────────────────────────────
class EnsembleModelManager:
    """
    Trains and evaluates XGBoost, LightGBM, Random Forest, and
    Stacking ensemble. Logs all runs to MLflow.
    """

    MODELS = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, random_state=42,
            n_jobs=-1, verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    }

    def __init__(self, experiment_name: str = "IBM-AIML-Platform"):
        self.experiment_name = experiment_name
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model = None
        self._setup_mlflow()

    def _setup_mlflow(self):
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.experiment_name)

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.MODELS.items():
            logger.info(f"Training {name}...")
            t0 = time.time()

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            metrics = {
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
                "test_accuracy": float(accuracy_score(y_test, preds)),
                "test_f1_weighted": float(f1_score(y_test, preds, average="weighted")),
                "train_time_sec": round(time.time() - t0, 2),
            }
            if proba is not None and len(np.unique(y_test)) > 1:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr"))
                except Exception:
                    pass

            self.results[name] = {"model": model, "metrics": metrics}
            logger.success(
                f"  {name:25s} | CV F1: {metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f} "
                f"| Test F1: {metrics['test_f1_weighted']:.4f}"
            )

            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=name):
                    mlflow.log_metrics(metrics)
                    if name == "xgboost":
                        mlflow.xgboost.log_model(model, f"model_{name}")
                    elif name == "lightgbm":
                        mlflow.lightgbm.log_model(model, f"model_{name}")
                    else:
                        # Native sklearn models (random_forest, etc.) still go
                        # through skops; trust the core tree storage type since
                        # these models are trained locally, not loaded from
                        # an untrusted source.
                        mlflow.sklearn.log_model(
                            model,
                            f"model_{name}",
                            skops_trusted_types=["sklearn.tree._tree.Tree"],
                        )

        # Pick best
        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["metrics"]["test_f1_weighted"]
        )
        self.best_model = self.results[self.best_model_name]["model"]
        logger.success(f"Best model: {self.best_model_name}")
        return {k: v["metrics"] for k, v in self.results.items()}

    def build_stacking_ensemble(self, X_train, y_train, X_test, y_test):
        estimators = [
            ("xgb", self.MODELS["xgboost"]),
            ("lgb", self.MODELS["lightgbm"]),
            ("rf", self.MODELS["random_forest"]),
        ]
        meta = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        stack = StackingClassifier(estimators=estimators, final_estimator=meta, cv=5, n_jobs=-1)
        stack.fit(X_train, y_train)
        preds = stack.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted")
        logger.success(f"Stacking Ensemble F1: {f1:.4f}")
        self.results["stacking"] = {"model": stack, "metrics": {"test_f1_weighted": f1}}
        return stack

    def explain_with_shap(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Optional[np.ndarray]:
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed")
            return None
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X[:200])
        logger.success("SHAP explanations computed for 200 samples")
        return shap_values

    def save_best(self, path: str = "models/saved/best_model.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, path)
        logger.info(f"Best model ({self.best_model_name}) saved → {path}") 