from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _format_date_axis(ax: plt.Axes) -> None:
    """
    Improve readability for dense datetime x-axes.
    """
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _mean_curve(histories: List[Dict[str, List[float]]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad ragged histories with NaN then compute mean±std per epoch.
    """
    seqs = [np.asarray(h.get(key, []), dtype=float) for h in histories]
    max_len = max([len(s) for s in seqs] + [0])
    if max_len == 0:
        return np.array([]), np.array([])
    mat = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        mat[i, : len(s)] = s
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return mean, std


def plot_paper8_loss_curves(
    model_to_fold_histories: Dict[str, List[Dict[str, List[float]]]],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"A_LSTM": "tab:blue", "B_CNNLSTM": "tab:orange", "C_CalendarLSTM": "tab:green"}

    for model, histories in model_to_fold_histories.items():
        c = colors.get(model, None)
        train_m, train_s = _mean_curve(histories, "train_loss")
        val_m, val_s = _mean_curve(histories, "val_loss")
        xs_train = np.arange(1, len(train_m) + 1)
        xs_val = np.arange(1, len(val_m) + 1)

        if len(train_m):
            ax.plot(xs_train, train_m, linestyle="--", color=c, alpha=0.9, label=f"{model} train")
            ax.fill_between(xs_train, train_m - train_s, train_m + train_s, color=c, alpha=0.10)
        if len(val_m):
            ax.plot(xs_val, val_m, linestyle="-", color=c, alpha=0.9, label=f"{model} val")
            ax.fill_between(xs_val, val_m - val_s, val_m + val_s, color=c, alpha=0.10)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, out_png, out_pdf)


def plot_paper8_pred_curve(
    model_to_oof_pred: Dict[str, pd.DataFrame],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    # Assume all models share same y_true for same dates; use first.
    first = next(iter(model_to_oof_pred.values()))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(first.index, first["y_true"].values, color="black", linewidth=1.5, label="true")
    for model, df in model_to_oof_pred.items():
        ax.plot(df.index, df["y_pred"].values, linewidth=1.2, label=model)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, out_png, out_pdf)


def plot_paper8_residuals(
    model_to_oof_pred: Dict[str, pd.DataFrame],
    *,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for model, df in model_to_oof_pred.items():
        resid = (df["y_pred"] - df["y_true"]).astype(float)
        axs[0].plot(df.index, resid.values, linewidth=1.0, alpha=0.8, label=model)
        axs[1].hist(resid.values, bins=60, density=True, alpha=0.35, label=model)
    _format_date_axis(axs[0])
    axs[0].set_title("Residual time series")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Pred - True")
    axs[0].legend(fontsize=8)

    axs[1].set_title("Residual distribution")
    axs[1].set_xlabel("Residual")
    axs[1].set_ylabel("Density")
    axs[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_png, out_pdf)


def plot_paper8_significance_diffs(
    model_to_fold_metrics: Dict[str, pd.DataFrame],
    sig_df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    Plot paired differences d_i = metric(model_a)_i - metric(model_b)_i across folds for each pair.
    """
    models = list(model_to_fold_metrics.keys())
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i], models[j]))

    diffs = []
    labels = []
    ann = []
    for a, b in pairs:
        da = model_to_fold_metrics[a][metric].values.astype(float)
        db = model_to_fold_metrics[b][metric].values.astype(float)
        d = da - db
        diffs.append(d)
        labels.append(f"{a} - {b}")
        row = sig_df[(sig_df["metric"] == metric) & (sig_df["model_a"] == a) & (sig_df["model_b"] == b)]
        if len(row) == 1:
            p = float(row["ttest_pvalue"].iloc[0])
            ann.append(f"p={p:.3g}")
        else:
            ann.append("")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.boxplot(diffs, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(f"Δ {metric} (paired)")
    ax.tick_params(axis="x", rotation=15)
    # annotate p-values above boxes
    for i, txt in enumerate(ann, start=1):
        if txt:
            ax.text(i, np.max(diffs[i - 1]) if len(diffs[i - 1]) else 0.0, txt, ha="center", va="bottom", fontsize=9)
    _save(fig, out_png, out_pdf)


