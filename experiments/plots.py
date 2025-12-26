from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL


def _save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_price_with_ma(
    df: pd.DataFrame,
    *,
    ticker: str,
    price_col: str = "Adj Close",
    windows: Iterable[int] = (7, 15, 30),
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[price_col], label=price_col, linewidth=1.5)
    for w in windows:
        ax.plot(df.index, df[price_col].rolling(w).mean(), label=f"MA{w}", linewidth=1.0)
    ax.set_title(f"{ticker} Price + Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(ncol=4, fontsize=9)
    _save_fig(fig, out_png, out_pdf)


def plot_return_distribution(
    returns: pd.Series,
    *,
    ticker: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    r = returns.dropna().astype(float).values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(r, bins=60, density=True, alpha=0.6, label="Return hist")

    # Fit Normal and Student-t
    mu, std = np.mean(r), np.std(r) + 1e-12
    xs = np.linspace(np.quantile(r, 0.01), np.quantile(r, 0.99), 400)
    ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=std), label="Normal fit", linewidth=1.5)

    try:
        df_t, loc_t, scale_t = stats.t.fit(r)
        ax.plot(xs, stats.t.pdf(xs, df_t, loc=loc_t, scale=scale_t), label="t fit", linewidth=1.5)
    except Exception:
        pass

    ax.set_title(f"{ticker} Daily Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    _save_fig(fig, out_png, out_pdf)


def plot_stl_decomposition(
    df: pd.DataFrame,
    *,
    ticker: str,
    price_col: str = "Adj Close",
    period: int = 252,
    out_png: Path,
    out_pdf: Path,
) -> None:
    series = df[price_col].dropna().astype(float)
    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(series.index, series.values, linewidth=1.2)
    axs[0].set_title(f"{ticker} STL Decomposition ({price_col})")
    axs[0].set_ylabel("Observed")

    axs[1].plot(series.index, res.trend, linewidth=1.2)
    axs[1].set_ylabel("Trend")
    axs[2].plot(series.index, res.seasonal, linewidth=1.2)
    axs[2].set_ylabel("Seasonal")
    axs[3].plot(series.index, res.resid, linewidth=1.2)
    axs[3].set_ylabel("Resid")
    axs[3].set_xlabel("Date")

    _save_fig(fig, out_png, out_pdf)


def plot_training_curves(
    history: Dict[str, list[float]],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val_loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    _save_fig(fig, out_png, out_pdf)


def plot_pred_vs_true(
    df_pred: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    y_col_true: str = "y_true",
    y_col_pred: str = "y_pred",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_pred.index, df_pred[y_col_true].values, label="true", linewidth=1.5)
    ax.plot(df_pred.index, df_pred[y_col_pred].values, label="pred", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend()
    _save_fig(fig, out_png, out_pdf)


def plot_residuals(
    df_pred: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
    y_col_true: str = "y_true",
    y_col_pred: str = "y_pred",
) -> None:
    resid = (df_pred[y_col_pred] - df_pred[y_col_true]).astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(df_pred.index, resid.values, linewidth=1.0)
    axs[0].set_title("Residual time series")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Pred - True")

    axs[1].hist(resid.values, bins=50, density=True, alpha=0.7)
    axs[1].set_title("Residual histogram")
    axs[1].set_xlabel("Residual")
    axs[1].set_ylabel("Density")

    fig.suptitle(title)
    _save_fig(fig, out_png, out_pdf)


def plot_metric_bars(
    summary: pd.DataFrame,
    *,
    metrics: list[str],
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    # summary index: model, columns: metric_mean, metric_std
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metrics))
    width = 0.25
    models = list(summary.index)

    for i, model in enumerate(models):
        means = [summary.loc[model, f"{m}_mean"] for m in metrics]
        stds = [summary.loc[model, f"{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=model)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.legend()
    _save_fig(fig, out_png, out_pdf)


def plot_metric_boxplots(
    per_fold: pd.DataFrame,
    *,
    metric: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    # per_fold columns: model, values: metric per fold
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [per_fold[m].dropna().values for m in per_fold.columns]
    ax.boxplot(data, labels=list(per_fold.columns), showmeans=True)
    ax.set_title(f"{title} ({metric})")
    ax.set_ylabel(metric)
    _save_fig(fig, out_png, out_pdf)


