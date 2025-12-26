from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Metrics:
    mse: float
    rmse: float
    mae: float
    mape_pct: float
    accuracy_pct: float
    baseline_mse: float
    gain_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mse": float(self.mse),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "mape_pct": float(self.mape_pct),
            "accuracy_pct": float(self.accuracy_pct),
            "baseline_mse": float(self.baseline_mse),
            "gain_loss": float(self.gain_loss),
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """
    y_true/y_pred: shape (N,) returns in *original scale* (not standardised).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape

    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs(err) / denom) * 100.0
    if np.isnan(mape):
        mape = float("nan")

    accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100.0)
    baseline_mse = float(np.mean((0.0 - y_true) ** 2))
    gain_loss = float(max(mse, 1e-12) / max(baseline_mse, 1e-12) - 1.0)

    return Metrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        mape_pct=float(mape),
        accuracy_pct=accuracy,
        baseline_mse=baseline_mse,
        gain_loss=gain_loss,
    )


