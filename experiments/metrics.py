from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Metrics:
    mse: float
    rmse: float
    mae: float
    price_mape_pct: float
    accuracy_pct: float
    baseline_mse: float
    gain_loss: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mse": float(self.mse),
            "rmse": float(self.rmse),
            "mae": float(self.mae),
            "price_mape_pct": float(self.price_mape_pct),
            "accuracy_pct": float(self.accuracy_pct),
            "baseline_mse": float(self.baseline_mse),
            "gain_loss": float(self.gain_loss),
        }


def _price_mape_pct(
    prev_price: np.ndarray,
    y_true_return: np.ndarray,
    y_pred_return: np.ndarray,
) -> float:
    """
    Compute MAPE on *price* reconstructed from predicted returns.

    Using pct_change return definition:
      true_price_t = prev_price_t * (1 + true_return_t)
      pred_price_t = prev_price_t * (1 + pred_return_t)
    """
    prev_price = np.asarray(prev_price).reshape(-1)
    y_true_return = np.asarray(y_true_return).reshape(-1)
    y_pred_return = np.asarray(y_pred_return).reshape(-1)
    assert prev_price.shape == y_true_return.shape == y_pred_return.shape

    true_price = prev_price * (1.0 + y_true_return)
    pred_price = prev_price * (1.0 + y_pred_return)
    denom = np.where(np.abs(true_price) < 1e-12, np.nan, np.abs(true_price))
    mape = np.nanmean(np.abs(pred_price - true_price) / denom) * 100.0
    return float(mape) if not np.isnan(mape) else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prev_price: np.ndarray,
) -> Metrics:
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

    price_mape = _price_mape_pct(prev_price=prev_price, y_true_return=y_true, y_pred_return=y_pred)

    accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100.0)
    baseline_mse = float(np.mean((0.0 - y_true) ** 2))
    gain_loss = float(max(mse, 1e-12) / max(baseline_mse, 1e-12) - 1.0)

    return Metrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        price_mape_pct=float(price_mape),
        accuracy_pct=accuracy,
        baseline_mse=baseline_mse,
        gain_loss=gain_loss,
    )


