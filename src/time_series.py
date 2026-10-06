"""
IBM Enterprise AI/ML Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module: Time Series & Predictive Analytics
Description: Prophet forecasting, ARIMA, LSTM-based time series prediction,
             and KPI trend analysis for enterprise operations.
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# LSTM Time Series Model
# ──────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class LSTMForecaster(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            attn_out, _ = self.attention(out, out, out)
            return self.fc(attn_out[:, -1, :])


# ──────────────────────────────────────────────────────────────────
# Prophet Forecaster
# ──────────────────────────────────────────────────────────────────
class ProphetForecaster:
    """IBM-grade time series forecasting using Facebook Prophet."""

    def __init__(self, horizon: int = 30, uncertainty_samples: int = 1000):
        self.horizon = horizon
        self.uncertainty_samples = uncertainty_samples
        self.model = None
        self.forecast_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, ds_col: str = "ds", y_col: str = "y") -> "ProphetForecaster":
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not installed — using naive forecast")
            return self
        train = df[[ds_col, y_col]].rename(columns={ds_col: "ds", y_col: "y"})
        self.model = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            uncertainty_samples=self.uncertainty_samples,
        )
        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        self.model.fit(train)
        logger.success(f"Prophet fitted on {len(train)} time steps")
        return self

    def predict(self) -> pd.DataFrame:
        if self.model is None:
            return self._naive_forecast()
        future = self.model.make_future_dataframe(periods=self.horizon, freq="D")
        self.forecast_ = self.model.predict(future)
        return self.forecast_[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(self.horizon)

    def _naive_forecast(self) -> pd.DataFrame:
        base = datetime.utcnow()
        return pd.DataFrame({
            "ds": [base + timedelta(days=i) for i in range(self.horizon)],
            "yhat": np.random.normal(100, 10, self.horizon).cumsum(),
            "yhat_lower": np.zeros(self.horizon),
            "yhat_upper": np.zeros(self.horizon),
        })

    def get_trend_components(self) -> Optional[pd.DataFrame]:
        if self.forecast_ is None:
            return None
        return self.forecast_[["ds", "trend", "weekly", "yearly"]].tail(self.horizon)


# ──────────────────────────────────────────────────────────────────
# KPI Analytics Engine
# ──────────────────────────────────────────────────────────────────
class KPIAnalyticsEngine:
    """Enterprise KPI monitoring, drift detection, and alerting."""

    def __init__(self):
        self.kpi_history: Dict[str, List] = {}
        self.alerts: List[Dict] = []

    def track(self, kpi_name: str, values: List[float], timestamps: Optional[List] = None):
        self.kpi_history[kpi_name] = {
            "values": values,
            "timestamps": timestamps or list(range(len(values))),
        }

    def compute_statistics(self, kpi_name: str) -> Dict[str, float]:
        if kpi_name not in self.kpi_history:
            raise KeyError(f"KPI '{kpi_name}' not tracked")
        vals = np.array(self.kpi_history[kpi_name]["values"])
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "trend": float(np.polyfit(range(len(vals)), vals, 1)[0]),
            "cv": float(vals.std() / (vals.mean() + 1e-9)),
        }

    def detect_drift(self, kpi_name: str, window: int = 30, threshold: float = 2.0) -> Dict[str, Any]:
        vals = np.array(self.kpi_history[kpi_name]["values"])
        if len(vals) < window * 2:
            return {"drift_detected": False, "reason": "Insufficient data"}

        baseline = vals[:window]
        recent = vals[-window:]
        z_score = abs(recent.mean() - baseline.mean()) / (baseline.std() + 1e-9)
        drift_detected = z_score > threshold

        result = {
            "kpi": kpi_name,
            "drift_detected": bool(drift_detected),
            "z_score": float(z_score),
            "baseline_mean": float(baseline.mean()),
            "recent_mean": float(recent.mean()),
            "change_pct": float((recent.mean() - baseline.mean()) / (abs(baseline.mean()) + 1e-9) * 100),
        }
        if drift_detected:
            self.alerts.append({"type": "KPI_DRIFT", "kpi": kpi_name, "details": result})
            logger.warning(f"Drift detected in {kpi_name}: z={z_score:.2f}")
        return result

    def generate_synthetic_kpis(self, n_days: int = 365) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        t = np.arange(n_days)
        base = datetime(2023, 1, 1)
        kpis = {
            "model_accuracy": np.clip(0.92 - 0.00005 * t + rng.normal(0, 0.005, n_days), 0.7, 1.0),
            "api_latency_ms": 120 + 0.02 * t + rng.normal(0, 5, n_days) + 30 * np.sin(2 * np.pi * t / 7),
            "anomaly_rate": np.clip(0.03 + 0.00002 * t + rng.normal(0, 0.002, n_days), 0, 0.2),
            "throughput_rps": 1000 + 50 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 20, n_days),
            "data_drift_score": np.clip(rng.normal(0.1, 0.02, n_days) + np.where(t > 300, 0.15, 0), 0, 1),
        }
        timestamps = [base + timedelta(days=int(i)) for i in t]
        for name, vals in kpis.items():
            self.track(name, vals.tolist(), timestamps)
        return kpis
