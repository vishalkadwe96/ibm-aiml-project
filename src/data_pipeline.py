"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: Data Pipeline
Description: Enterprise-grade data ingestion, cleaning, feature engineering,
             and preprocessing pipeline with full audit trail.
"""

import os
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    VarianceThreshold, RFE
)
from loguru import logger
import joblib
import yaml

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# Configuration Loader
# ──────────────────────────────────────────────────────────────────
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return cfg


# ──────────────────────────────────────────────────────────────────
# Data Ingestion
# ──────────────────────────────────────────────────────────────────
class DataIngestion:
    """
    Enterprise data ingestion supporting CSV, JSON, Parquet, Excel.
    Computes file hash for data lineage tracking.
    """

    SUPPORTED_FORMATS = {".csv", ".json", ".parquet", ".xlsx", ".xls", ".tsv"}

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_log: List[Dict] = []

    def _file_hash(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def load(self, filepath: str, **kwargs) -> pd.DataFrame:
        path = Path(filepath)
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        readers = {
            ".csv": pd.read_csv,
            ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
            ".json": pd.read_json,
            ".parquet": pd.read_parquet,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
        }
        df = readers[path.suffix](filepath, **kwargs)

        record = {
            "file": str(filepath),
            "sha256": self._file_hash(filepath),
            "rows": len(df),
            "cols": len(df.columns),
            "loaded_at": datetime.utcnow().isoformat(),
        }
        self.ingestion_log.append(record)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {filepath}")
        return df

    def generate_synthetic_dataset(
        self,
        n_samples: int = 10_000,
        n_features: int = 30,
        n_informative: int = 15,
        n_classes: int = 3,
        task: str = "classification",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a rich synthetic dataset for demo/testing."""
        rng = np.random.default_rng(seed)

        if task == "classification":
            from sklearn.datasets import make_classification
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=5,
                n_classes=n_classes,
                random_state=seed,
                class_sep=0.8,
            )
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                noise=0.1,
                random_state=seed,
            )

        cols = [f"feature_{i:02d}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = y

        # ── Inject realism ───────────────────────────────────────
        # Missing values (~5%)
        mask = rng.random(df.shape) < 0.05
        df[mask] = np.nan

        # Categorical features
        df["dept"] = rng.choice(
            ["Engineering", "Finance", "HR", "Sales", "Operations"],
            size=n_samples,
        )
        df["region"] = rng.choice(["NA", "EMEA", "APAC", "LATAM"], size=n_samples)
        df["priority"] = rng.choice(["Low", "Medium", "High", "Critical"], size=n_samples)

        # Timestamps
        base = datetime(2023, 1, 1)
        df["timestamp"] = pd.date_range(base, periods=n_samples, freq="5min")
        df["transaction_id"] = [f"TXN-{i:07d}" for i in range(n_samples)]

        # Inject anomalies
        anomaly_idx = rng.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        df.loc[anomaly_idx, cols[:5]] = rng.uniform(10, 20, (len(anomaly_idx), 5))
        df["is_anomaly"] = 0
        df.loc[anomaly_idx, "is_anomaly"] = 1

        logger.success(
            f"Generated synthetic dataset: {n_samples:,} rows, {len(df.columns)} cols, "
            f"{df['is_anomaly'].sum()} anomalies"
        )
        return df


# ──────────────────────────────────────────────────────────────────
# Data Quality Report
# ──────────────────────────────────────────────────────────────────
class DataQualityAnalyzer:
    """IBM-style data quality profiling and audit."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        df = self.df
        n, m = df.shape

        missing = df.isnull().sum()
        missing_pct = (missing / n * 100).round(2)
        duplicates = df.duplicated().sum()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        self.report = {
            "shape": {"rows": n, "columns": m},
            "missing": {
                col: {"count": int(cnt), "percent": float(missing_pct[col])}
                for col, cnt in missing.items()
                if cnt > 0
            },
            "duplicates": int(duplicates),
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "statistics": df[numeric_cols].describe().to_dict() if numeric_cols else {},
            "cardinality": {c: int(df[c].nunique()) for c in cat_cols},
            "class_distribution": (
                df["label"].value_counts(normalize=True).round(4).to_dict()
                if "label" in df.columns
                else {}
            ),
        }
        logger.info(
            f"Data Quality Report | Rows: {n:,} | Missing cols: {len(self.report['missing'])} | "
            f"Duplicates: {duplicates}"
        )
        return self.report

    def print_summary(self):
        r = self.report or self.run()
        print("\n" + "═" * 60)
        print("  IBM DATA QUALITY REPORT")
        print("═" * 60)
        print(f"  Shape         : {r['shape']['rows']:,} rows × {r['shape']['columns']} cols")
        print(f"  Duplicates    : {r['duplicates']}")
        print(f"  Missing cols  : {len(r['missing'])}")
        if r["missing"]:
            for col, info in list(r["missing"].items())[:8]:
                print(f"    ├─ {col}: {info['percent']:.1f}%")
        print(f"  Numeric cols  : {len(r['numeric_columns'])}")
        print(f"  Categorical   : {len(r['categorical_columns'])}")
        if r["class_distribution"]:
            print(f"  Class dist.   : {r['class_distribution']}")
        print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Advanced feature engineering: temporal features, interaction terms,
    rolling statistics, polynomial features, and target encoding.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._fitted_encoders: Dict[str, Any] = {}

    def add_temporal_features(self, df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
        if ts_col not in df.columns:
            return df
        df = df.copy()
        dt = pd.to_datetime(df[ts_col])
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["is_business_hour"] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 17)).astype(int)
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        logger.debug(f"Added 8 temporal features from '{ts_col}'")
        return df

    def add_rolling_stats(
        self, df: pd.DataFrame, numeric_cols: List[str], windows: List[int] = [5, 10, 30]
    ) -> pd.DataFrame:
        df = df.copy()
        for col in numeric_cols[:10]:  # Cap to prevent explosion
            for w in windows:
                df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
                df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
        logger.debug(f"Added rolling stats for {min(10, len(numeric_cols))} cols × {len(windows)} windows")
        return df

    def add_interaction_features(self, df: pd.DataFrame, top_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        pairs = [(top_cols[i], top_cols[j]) for i in range(len(top_cols)) for j in range(i + 1, len(top_cols))]
        for a, b in pairs[:15]:  # Cap to 15 interactions
            df[f"{a}_x_{b}"] = df[a] * df[b]
            df[f"{a}_div_{b}"] = df[a] / (df[b].replace(0, 1e-9))
        logger.debug(f"Added {min(30, len(pairs)*2)} interaction features")
        return df

    def target_encode(self, df: pd.DataFrame, cat_cols: List[str], target: str) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            if col in df.columns and target in df.columns:
                mapping = df.groupby(col)[target].mean().to_dict()
                self._fitted_encoders[f"target_enc_{col}"] = mapping
                df[f"{col}_target_enc"] = df[col].map(mapping).fillna(df[target].mean())
        return df

    def frequency_encode(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True).to_dict()
                self._fitted_encoders[f"freq_enc_{col}"] = freq
                df[f"{col}_freq_enc"] = df[col].map(freq).fillna(0)
        return df


# ──────────────────────────────────────────────────────────────────
# Full Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────────
class PreprocessingPipeline:
    """
    End-to-end sklearn-compatible preprocessing pipeline with
    imputation → scaling → encoding → feature selection.
    """

    def __init__(self, config: Optional[Dict] = None, task: str = "classification"):
        self.config = config or {}
        self.task = task
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self._label_encoder = LabelEncoder()

    def build(self, df: pd.DataFrame, target_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
        X = df.drop(columns=[target_col, "timestamp", "transaction_id", "is_anomaly"],
                    errors="ignore")
        y_raw = df[target_col].values

        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", RobustScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ], remainder="drop")

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold(threshold=0.01)),
        ])

        if self.task == "classification":
            y = self._label_encoder.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.astype(float)

        X_processed = self.pipeline.fit_transform(X)
        logger.success(
            f"Preprocessing complete | Input: {X.shape} → Output: {X_processed.shape}"
        )
        return X_processed, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=["label", "timestamp", "transaction_id", "is_anomaly"], errors="ignore")
        return self.pipeline.transform(X)

    def save(self, path: str = "models/saved/preprocessor.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Preprocessor saved → {path}")

    def load(self, path: str = "models/saved/preprocessor.joblib"):
        self.pipeline = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return self
