from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments.data import DatasetSpec, add_calendar_features, add_return_column, load_or_fetch_daily, dataset_md
from experiments.io_utils import RunPaths, capture_env_txt, make_run_dirs, now_stamp, write_json, write_text
from experiments.plots import (
    plot_metric_bars,
    plot_metric_boxplots,
    plot_pred_vs_true,
    plot_price_with_ma,
    plot_residuals,
    plot_return_distribution,
    plot_stl_decomposition,
    plot_training_curves,
)
from experiments.report import build_results_tables, write_latex_booktabs
from experiments.significance import paired_tests
from experiments.train_eval import run_time_series_cv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducible end-to-end experiment runner (paper pipeline).")
    p.add_argument("--tickers", nargs="+", default=["AAPL"], help="Yahoo Finance tickers, e.g. AAPL MSFT SPY")
    p.add_argument("--start", type=str, default="2012-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default: today")
    p.add_argument("--run-name", type=str, default=None, help="Run name. Default: PAPER--<timestamp>")
    p.add_argument("--force-download", action="store_true", help="Force re-download data even if cached CSV exists")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--lookback", type=int, default=64)
    p.add_argument("--pred-horizon", type=int, default=1)
    p.add_argument(
        "--val-size",
        type=int,
        default=252,
        help="Evaluation window per fold (next-day return points). Recommended: 252 (~1 trading year).",
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-width", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument(
        "--innovation",
        type=str,
        default="calendar",
        choices=["calendar", "none"],
        help="Model C innovation: calendar features (recommended) or none",
    )
    p.add_argument(
        "--plot-set",
        type=str,
        default="paper8",
        choices=["paper8", "full"],
        help="Plot output set. paper8 = the 8 figures needed for the paper; full = per-fold plots.",
    )
    p.add_argument(
        "--save-fold-predictions",
        action="store_true",
        help="Also save per-fold prediction CSVs (in addition to aggregated OOF).",
    )
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    root = Path(__file__).resolve().parent
    run_name = args.run_name or f"PAPER--{now_stamp()}"
    paths: RunPaths = make_run_dirs(root / "runs", run_name)

    # Persist config first
    write_json(paths.run_dir / "config.json", vars(args))
    capture_env_txt(paths.run_dir / "env.txt")

    # Data fetch/cache
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    spec = DatasetSpec(tickers=args.tickers, start=args.start, end=end)
    per_ticker_csv: Dict[str, Path] = {}

    ticker_to_df: Dict[str, pd.DataFrame] = {}
    for t in args.tickers:
        df = load_or_fetch_daily(root, t, args.start, end, force=args.force_download)
        per_ticker_csv[t] = (root / "data" / "processed" / f"{t}_{args.start}_{end}.csv")
        df = add_return_column(df, price_col="Adj Close")
        if args.innovation == "calendar":
            df = add_calendar_features(df, add_dow_onehot=True, add_month_onehot=True)
        df = df.dropna()
        ticker_to_df[t] = df

    write_text(paths.run_dir / "dataset.md", dataset_md(spec, per_ticker_csv))

    # Run experiments per ticker
    for ticker, df in ticker_to_df.items():
        # EDA plots
        plot_price_with_ma(
            df,
            ticker=ticker,
            price_col="Close" if "Adj Close" not in df.columns and "Close" in df.columns else "Adj Close",
            out_png=paths.figures_png_dir / f"{ticker}_price_ma.png",
            out_pdf=paths.figures_pdf_dir / f"{ticker}_price_ma.pdf",
        )
        plot_return_distribution(
            df["Return"],
            ticker=ticker,
            out_png=paths.figures_png_dir / f"{ticker}_return_dist.png",
            out_pdf=paths.figures_pdf_dir / f"{ticker}_return_dist.pdf",
        )
        # STL may fail if too short; guard
        if len(df) > 400 and args.plot_set == "full":
            plot_stl_decomposition(
                df,
                ticker=ticker,
                price_col="Close" if "Adj Close" not in df.columns and "Close" in df.columns else "Adj Close",
                out_png=paths.figures_png_dir / f"{ticker}_stl.png",
                out_pdf=paths.figures_pdf_dir / f"{ticker}_stl.pdf",
            )

        # Feature columns
        base_feature_cols = ["Return"]
        cal_cols = [c for c in df.columns if c.startswith("dow_") or c.startswith("month_")]

        # Models
        models: Dict[str, Dict[str, object]] = {
            "A_LSTM": {"model_name": "LSTM", "feature_cols": base_feature_cols},
            "B_CNNLSTM": {"model_name": "CNN-LSTM", "feature_cols": base_feature_cols},
        }
        if args.innovation == "calendar":
            models["C_CalendarLSTM"] = {
                "model_name": "LSTM",
                "feature_cols": base_feature_cols + cal_cols,
            }

        model_to_fold_metrics: Dict[str, pd.DataFrame] = {}
        model_to_fold_histories: Dict[str, List[Dict[str, List[float]]]] = {}
        model_to_oof_pred: Dict[str, pd.DataFrame] = {}

        for model_tag, cfg in models.items():
            fold_results = run_time_series_cv(
                df,
                feature_cols=cfg["feature_cols"],  # type: ignore[arg-type]
                target_col="Return",
                model_name=cfg["model_name"],  # type: ignore[arg-type]
                device=args.device,
                folds=args.folds,
                look_back=args.lookback,
                pred_horizon=args.pred_horizon,
                val_size=args.val_size,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden_width=args.hidden_width,
                dropout=args.dropout,
                seed=args.seed,
            )

            # Save per-fold predictions and histories
            rows = []
            model_histories = []
            oof_parts: List[pd.DataFrame] = []
            for fr in fold_results:
                rows.append({"fold": fr.fold, **fr.metrics.as_dict()})
                model_histories.append(fr.history)
                oof_parts.append(fr.pred_df)

                if args.save_fold_predictions or args.plot_set == "full":
                    pred_path = paths.predictions_dir / f"{ticker}__{model_tag}__fold{fr.fold}.csv"
                    fr.pred_df.to_csv(pred_path)

                if args.plot_set == "full":
                    # Per-fold pred/true + residual plots
                    plot_pred_vs_true(
                        fr.pred_df,
                        title=f"{ticker} {model_tag} fold{fr.fold}: true vs pred (Return)",
                        out_png=paths.figures_png_dir / f"{ticker}_{model_tag}_fold{fr.fold}_pred.png",
                        out_pdf=paths.figures_pdf_dir / f"{ticker}_{model_tag}_fold{fr.fold}_pred.pdf",
                    )
                    plot_residuals(
                        fr.pred_df,
                        title=f"{ticker} {model_tag} fold{fr.fold}: residual analysis",
                        out_png=paths.figures_png_dir / f"{ticker}_{model_tag}_fold{fr.fold}_resid.png",
                        out_pdf=paths.figures_pdf_dir / f"{ticker}_{model_tag}_fold{fr.fold}_resid.pdf",
                    )
                    plot_training_curves(
                        fr.history,
                        title=f"{ticker} {model_tag} fold{fr.fold}: training curves",
                        out_png=paths.figures_png_dir / f"{ticker}_{model_tag}_fold{fr.fold}_traincurve.png",
                        out_pdf=paths.figures_pdf_dir / f"{ticker}_{model_tag}_fold{fr.fold}_traincurve.pdf",
                    )

            df_fold = pd.DataFrame(rows).set_index("fold")
            model_to_fold_metrics[model_tag] = df_fold
            model_to_fold_histories[model_tag] = model_histories
            oof_df = pd.concat(oof_parts, axis=0).sort_index()
            model_to_oof_pred[model_tag] = oof_df
            oof_path = paths.predictions_dir / f"{ticker}__{model_tag}__oof.csv"
            oof_df.to_csv(oof_path)

        # Summary tables (per ticker)
        # Note: MAPE is computed on reconstructed prices (price-MAPE), not on returns.
        main_metrics = ["mae", "rmse", "price_mape_pct", "accuracy_pct"]
        summary_stats, main_pretty = build_results_tables(model_to_fold_metrics, main_metrics=main_metrics)

        # Save CSV + LaTeX
        out_results_csv = paths.run_dir / f"results__{ticker}.csv"
        summary_stats.to_csv(out_results_csv)

        out_pretty_csv = paths.run_dir / f"results_pretty__{ticker}.csv"
        main_pretty.to_csv(out_pretty_csv)

        write_latex_booktabs(
            main_pretty,
            out_path=paths.run_dir / f"results__{ticker}.tex",
            caption=f"Cross-validation results for {ticker} (mean ± std across folds).",
            label=f"tab:{ticker.lower()}_results",
        )

        # Plots: metric bars + boxplots (RMSE)
        plot_metric_bars(
            summary_stats,
            metrics=["rmse", "mae"],
            title=f"{ticker}: RMSE/MAE comparison (mean ± std)",
            out_png=paths.figures_png_dir / f"{ticker}_metrics_bar.png",
            out_pdf=paths.figures_pdf_dir / f"{ticker}_metrics_bar.pdf",
        )

        rmse_per_fold = pd.DataFrame({m: dfm["rmse"] for m, dfm in model_to_fold_metrics.items()})
        plot_metric_boxplots(
            rmse_per_fold,
            metric="rmse",
            title=f"{ticker}: RMSE across CV folds",
            out_png=paths.figures_png_dir / f"{ticker}_rmse_box.png",
            out_pdf=paths.figures_pdf_dir / f"{ticker}_rmse_box.pdf",
        )

        # Significance tests (paired across folds) on RMSE and MAE
        sig_rows = []
        model_names = list(model_to_fold_metrics.keys())
        for metric in ["rmse", "mae"]:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    a = model_names[i]
                    b = model_names[j]
                    res = paired_tests(
                        model_to_fold_metrics[a][metric].values,
                        model_to_fold_metrics[b][metric].values,
                        metric=metric,
                        model_a=a,
                        model_b=b,
                    )
                    sig_rows.append({"ticker": ticker, **res.as_dict()})

        # Accuracy vs 50% chance: paired tests across folds against a constant baseline (50).
        for model in model_names:
            acc = model_to_fold_metrics[model]["accuracy_pct"].values.astype(float)
            chance = [50.0] * len(acc)
            res = paired_tests(
                acc,
                chance,
                metric="accuracy_pct_vs_50",
                model_a=model,
                model_b="chance50",
            )
            sig_rows.append({"ticker": ticker, **res.as_dict()})

        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(paths.stats_dir / "significance.csv", index=False)

        # Paper8-only plots: aggregate training curve, prediction curve, residuals, significance diff distribution.
        if args.plot_set == "paper8":
            from experiments.paper8 import (
                plot_paper8_loss_curves,
                plot_paper8_pred_curve,
                plot_paper8_residuals,
                plot_paper8_significance_diffs,
            )

            plot_paper8_loss_curves(
                model_to_fold_histories,
                title=f"{ticker}: Train/Val loss vs epoch (mean across folds)",
                out_png=paths.figures_png_dir / f"{ticker}_loss_curves.png",
                out_pdf=paths.figures_pdf_dir / f"{ticker}_loss_curves.pdf",
            )
            plot_paper8_pred_curve(
                model_to_oof_pred,
                title=f"{ticker}: Out-of-fold prediction (Return)",
                out_png=paths.figures_png_dir / f"{ticker}_pred_curve.png",
                out_pdf=paths.figures_pdf_dir / f"{ticker}_pred_curve.pdf",
            )
            plot_paper8_residuals(
                model_to_oof_pred,
                title=f"{ticker}: Residual analysis (OOF)",
                out_png=paths.figures_png_dir / f"{ticker}_residuals.png",
                out_pdf=paths.figures_pdf_dir / f"{ticker}_residuals.pdf",
            )
            plot_paper8_significance_diffs(
                model_to_fold_metrics,
                sig_df,
                metric="rmse",
                title=f"{ticker}: RMSE paired differences across folds",
                out_png=paths.figures_png_dir / f"{ticker}_significance_rmse.png",
                out_pdf=paths.figures_pdf_dir / f"{ticker}_significance_rmse.pdf",
            )

    # One-line success marker
    write_text(paths.run_dir / "DONE.txt", "OK\n")
    print(f"\nAll done. Outputs written to: {paths.run_dir}\n")


if __name__ == "__main__":
    main()


