from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    figures_dir: Path
    figures_png_dir: Path
    figures_pdf_dir: Path
    predictions_dir: Path
    stats_dir: Path


def make_run_dirs(runs_root: Path, run_name: str) -> RunPaths:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    figures_dir = run_dir / "figures"
    figures_png_dir = figures_dir / "png"
    figures_pdf_dir = figures_dir / "pdf"
    predictions_dir = run_dir / "predictions"
    stats_dir = run_dir / "stats"

    for p in [figures_dir, figures_png_dir, figures_pdf_dir, predictions_dir, stats_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        figures_dir=figures_dir,
        figures_png_dir=figures_png_dir,
        figures_pdf_dir=figures_pdf_dir,
        predictions_dir=predictions_dir,
        stats_dir=stats_dir,
    )


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def capture_env_txt(path: Path) -> None:
    """Write a compact but sufficient environment capture for reproducibility."""
    lines: list[str] = []
    lines.append(f"timestamp: {datetime.now().isoformat()}")
    lines.append(f"python: {sys.version.replace(os.linesep, ' ')}")
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"executable: {sys.executable}")
    try:
        import torch  # noqa: WPS433

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        lines.append(f"torch.version.cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            lines.append(f"cuda.device_count: {torch.cuda.device_count()}")
            lines.append(f"cuda.device0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        lines.append(f"torch: <failed to import> ({e})")

    lines.append("")
    lines.append("pip freeze:")
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        lines.append(out.strip())
    except Exception as e:
        lines.append(f"<pip freeze failed: {e}>")

    write_text(path, "\n".join(lines) + "\n")


def safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)


def to_pretty_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


