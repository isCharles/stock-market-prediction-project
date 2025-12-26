# Paper Reproduction Pipeline

This repo contains an **end-to-end, reproducible experiment runner** for a course paper:

- Yahoo Finance daily data download + caching
- LSTM + CNN-LSTM baselines
- A lightweight innovation (**calendar features**) as Model C
- Expanding-window time-series CV
- Metrics + prediction CSV exports
- Paper-grade plots (**PNG + PDF**) saved into `runs/<run_name>/figures/`
- Paired significance tests (paired t-test + Wilcoxon) saved into `runs/<run_name>/stats/significance.csv`

## One-command run

```bash
python run_all.py --tickers AAPL MSFT --start 2012-01-01 --epochs 30 --folds 5 --device cuda
```

Outputs go to `runs/PAPER--<timestamp>/`.

## GPU note (RTX 50 series / sm_120)

For RTX 50 series (e.g. RTX 5060), older CUDA builds of PyTorch may not include `sm_120`.
If you see `no kernel image is available`, install the CUDA 12.8 wheel:

```bash
python -m pip install --upgrade --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
```

Then re-run `run_all.py --device cuda`.

## Reproducibility artifacts

Each run records:
- `runs/<run_name>/config.json`
- `runs/<run_name>/env.txt` (python + torch + pip freeze)
- `runs/<run_name>/dataset.md`
- `runs/<run_name>/predictions/*.csv`
- `runs/<run_name>/results__<ticker>.csv` and `.tex`
- `runs/<run_name>/stats/significance.csv`
- `runs/<run_name>/figures/png/*.png` and `figures/pdf/*.pdf`


