from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from experiments.io_utils import to_pretty_mean_std


def aggregate_fold_metrics(per_fold: pd.DataFrame) -> pd.DataFrame:
    """
    per_fold: rows=fold, cols=metric names (for one model)
    Returns: one-row df with <metric>_mean, <metric>_std.
    """
    out = {}
    for col in per_fold.columns:
        vals = per_fold[col].astype(float).values
        out[f"{col}_mean"] = float(np.mean(vals))
        out[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return pd.DataFrame([out])


def build_results_tables(
    model_to_fold_metrics: Dict[str, pd.DataFrame],
    *,
    main_metrics: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - summary_stats: index=model, columns like <metric>_mean/_std (all metrics)
      - main_pretty: index=model, columns are main_metrics formatted as mean±std
    """
    rows = []
    for model, df_fold in model_to_fold_metrics.items():
        agg = aggregate_fold_metrics(df_fold)
        agg.index = [model]
        rows.append(agg)
    summary = pd.concat(rows, axis=0)

    pretty = pd.DataFrame(index=summary.index)
    for m in main_metrics:
        pretty[m] = [
            to_pretty_mean_std(summary.loc[model, f"{m}_mean"], summary.loc[model, f"{m}_std"])
            for model in summary.index
        ]
    return summary, pretty


def write_latex_booktabs(table: pd.DataFrame, out_path: Path, caption: str = "", label: str = "") -> None:
    """
    Write a simple LaTeX booktabs table. Intended for small paper tables.
    """
    # Avoid pandas' optional Jinja2 dependency by rendering LaTeX manually.
    cols = list(table.columns)
    lines: list[str] = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{'l' + 'c' * len(cols)}}}")
    lines.append("\\toprule")
    header = " & ".join(["Model"] + [str(c) for c in cols]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for idx, row in table.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append(" & ".join([str(idx)] + vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


