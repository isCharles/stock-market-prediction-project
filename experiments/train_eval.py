from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesSliceDataset
from data.transforms import FeatureStandardiser
from experiments.metrics import Metrics, compute_metrics
from models import CNNLSTMModel, LSTMModel


@dataclass(frozen=True)
class FoldResult:
    fold: int
    metrics: Metrics
    pred_df: pd.DataFrame  # index: Date, columns: y_true, y_pred
    history: Dict[str, List[float]]  # train_loss, val_loss


class _ReturnTargetSliceDataset(torch.utils.data.Dataset):
    """
    Same slicing as TimeSeriesSliceDataset, but targets only the Return channel (col 0).
    This avoids broadcasting bugs when we add calendar features (input_size > 1) but still
    predict next-day return (output_size = 1).
    """

    def __init__(self, data: torch.Tensor, train_length: int, target_length: int, *, return_col_idx: int = 0):
        self.data = data
        self.train_length = train_length
        self.target_length = target_length
        self.return_col_idx = return_col_idx

    def __len__(self) -> int:
        return self.data.shape[0] - self.train_length - self.target_length + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.train_length]
        y_full = self.data[idx + self.train_length : idx + self.train_length + self.target_length]
        y = y_full[:, self.return_col_idx : self.return_col_idx + 1]
        return x, y


def _device_from_string(device: str) -> torch.device:
    d = device.lower()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if d == "cpu":
        return torch.device("cpu")
    if d == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError(f"Unknown device: {device}")


def _build_model(model_name: str, input_size: int, pred_horizon: int, hidden_width: int, dropout: float) -> nn.Module:
    if model_name == "LSTM":
        return LSTMModel(
            input_size=input_size,
            hidden_layer_size=hidden_width,
            pred_horizon=pred_horizon,
            dropout=dropout,
            recurrent_pred_horizon=False,
        )
    if model_name == "CNN-LSTM":
        return CNNLSTMModel(
            input_size=input_size,
            hidden_layer_size=hidden_width,
            pred_horizon=pred_horizon,
            conv_channels=32,
            dropout=dropout,
            recurrent_pred_horizon=False,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, reset_hidden=True)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x, reset_hidden=True)
        loss = criterion(pred, y)
        bs = x.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def _predict_next_day_returns(
    model: nn.Module,
    full_features: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    look_back: int,
    pred_horizon: int,
    standardiser: FeatureStandardiser,
    device: torch.device,
    return_col_idx: int = 0,
    eval_last_n: int = 10,
) -> pd.DataFrame:
    """
    Predict next-day Return for the last `eval_last_n` points of the provided window.
    We evaluate on a contiguous tail window, requiring look_back history.
    """
    assert pred_horizon == 1, "Paper pipeline currently targets next-day prediction (pred_horizon=1)."

    data = torch.tensor(full_features, dtype=torch.float32)
    ds = TimeSeriesSliceDataset(data, look_back, pred_horizon)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # We'll only keep the last eval_last_n samples
    n_total = len(ds)
    start_i = max(n_total - eval_last_n, 0)

    y_true_list = []
    y_pred_list = []
    date_list = []
    for i, (x, y) in enumerate(loader):
        if i < start_i:
            continue
        x = x.to(device)
        y = y.to(device)
        pred = model(x, reset_hidden=True)

        # Inverse only the return channel; output is (1, 1, 1)
        pred_ret_std = pred[..., 0].detach().cpu()
        true_ret_std = y[..., 0].detach().cpu()
        pred_ret = standardiser.inverse(pred_ret_std)[0, 0].item()
        true_ret = standardiser.inverse(true_ret_std)[0, 0].item()

        # The target corresponds to date at index look_back + i
        target_date = dates[look_back + i]
        date_list.append(target_date)
        y_true_list.append(true_ret)
        y_pred_list.append(pred_ret)

    out = pd.DataFrame({"y_true": y_true_list, "y_pred": y_pred_list}, index=pd.DatetimeIndex(date_list, name="Date"))
    return out


def run_time_series_cv(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    device: str,
    folds: int,
    look_back: int,
    pred_horizon: int,
    val_size: int,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_width: int,
    dropout: float,
    seed: int = 0,
) -> list[FoldResult]:
    """
    Expanding-window CV using sklearn TimeSeriesSplit with overlap to include look_back.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = _device_from_string(device)

    # Build arrays
    df = df.dropna().copy()
    dates = df.index
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)  # used only for metrics convenience

    # CV split mirrors repo logic: include look_back overlap in "test" block
    tscv = TimeSeriesSplit(n_splits=folds, gap=-look_back, test_size=look_back + val_size)

    results: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        # train_idx, test_idx are indices into X
        X_train = X[train_idx]
        X_test = X[test_idx]  # includes look_back history + val_size evaluation tail
        dates_test = dates[test_idx]

        # Standardize Return only using train split
        ret_train = X_train[:, 0]
        mean = float(np.mean(ret_train))
        std = float(np.std(ret_train) + 1e-12)
        standardiser = FeatureStandardiser(mean, std)

        X_train_std = X_train.copy()
        X_test_std = X_test.copy()
        X_train_std[:, 0] = standardiser.forward(X_train_std[:, 0])
        X_test_std[:, 0] = standardiser.forward(X_test_std[:, 0])

        train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
        test_tensor = torch.tensor(X_test_std, dtype=torch.float32)

        train_ds = _ReturnTargetSliceDataset(train_tensor, look_back, pred_horizon, return_col_idx=0)
        val_ds = _ReturnTargetSliceDataset(test_tensor, look_back, pred_horizon, return_col_idx=0)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = _build_model(
            model_name=model_name,
            input_size=len(feature_cols),
            pred_horizon=pred_horizon,
            hidden_width=hidden_width,
            dropout=dropout,
        ).to(d)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        bad = 0

        for _epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, d)
            val_loss = _eval_loss(model, val_loader, criterion, d)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict on the last val_size points of the test window
        pred_df = _predict_next_day_returns(
            model,
            X_test_std,
            dates_test,
            look_back=look_back,
            pred_horizon=pred_horizon,
            standardiser=standardiser,
            device=d,
            eval_last_n=val_size,
        )

        m = compute_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
        results.append(FoldResult(fold=fold_idx, metrics=m, pred_df=pred_df, history=history))

    return results


