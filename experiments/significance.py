from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SignificanceResult:
    metric: str
    model_a: str
    model_b: str
    n: int
    mean_diff: float
    ci95_low: float
    ci95_high: float
    ttest_pvalue: float
    wilcoxon_pvalue: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "n": int(self.n),
            "mean_diff": float(self.mean_diff),
            "ci95_low": float(self.ci95_low),
            "ci95_high": float(self.ci95_high),
            "ttest_pvalue": float(self.ttest_pvalue),
            "wilcoxon_pvalue": float(self.wilcoxon_pvalue),
        }


def _bootstrap_ci_mean(
    diffs: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs).reshape(-1)
    n = diffs.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def paired_tests(
    values_a: Iterable[float],
    values_b: Iterable[float],
    *,
    metric: str,
    model_a: str,
    model_b: str,
    bootstrap_ci: bool = True,
) -> SignificanceResult:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired tests require arrays of same shape (per fold).")

    diffs = a - b
    n = int(diffs.shape[0])
    mean_diff = float(np.mean(diffs)) if n > 0 else float("nan")

    if n >= 2:
        t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
        # Wilcoxon requires non-zero diffs; use zero_method='wilcox' by default
        try:
            w_p = float(stats.wilcoxon(diffs).pvalue)
        except Exception:
            w_p = float("nan")
    else:
        t_p = float("nan")
        w_p = float("nan")

    if bootstrap_ci and n > 0:
        ci_low, ci_high = _bootstrap_ci_mean(diffs)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return SignificanceResult(
        metric=metric,
        model_a=model_a,
        model_b=model_b,
        n=n,
        mean_diff=mean_diff,
        ci95_low=ci_low,
        ci95_high=ci_high,
        ttest_pvalue=t_p,
        wilcoxon_pvalue=w_p,
    )


