import argparse
import sys
from datetime import datetime
from warnings import warn

import torch

DEFAULT_ARGS = {
    "project": "stock-prediction-playground",
    "architecture": "CNN-LSTM",
    "ticker": "AAPL",
    "epochs": 1000,
    "batch_size": 64,
    "look_back": 64,
    "pred_horizon": 1,
    "hidden_width": 128,
    "dropout": 0.0,
    "val_size": 10,
    "train_logs": 50,
    "val_logs": 10,
    "wandb": False,
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Training script for stock market prediction"
    )

    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=DEFAULT_ARGS["project"],
        help="Wandb project name",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help='Run name (default: "RUN--<TIMESTAMP>")',
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str.lower,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        default=DEFAULT_ARGS["architecture"],
        choices=["LSTM", "CNN-LSTM"],
        help="Model architecture",
    )
    parser.add_argument(
        "-t",
        "--ticker",
        type=str,
        default=DEFAULT_ARGS["ticker"],
        help="Stock ticker name",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=DEFAULT_ARGS["epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_ARGS["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "-l",
        "--look-back",
        type=int,
        default=DEFAULT_ARGS["look_back"],
        help="Look-back length",
    )
    parser.add_argument(
        "-ph",
        "--pred-horizon",
        type=int,
        default=DEFAULT_ARGS["pred_horizon"],
        help="Prediction horizon size",
    )
    parser.add_argument(
        "-hw",
        "--hidden-width",
        type=int,
        default=DEFAULT_ARGS["hidden_width"],
        help="Hidden width",
    )
    parser.add_argument(
        "-do",
        "--dropout",
        type=float,
        default=DEFAULT_ARGS["dropout"],
        help="Drop-out rate",
    )
    parser.add_argument(
        "-vs",
        "--val-size",
        type=int,
        default=DEFAULT_ARGS["val_size"],
        help="Validation set size",
    )
    parser.add_argument(
        "-tl",
        "--train-logs",
        type=int,
        default=DEFAULT_ARGS["train_logs"],
        help="Total number of training logs",
    )
    parser.add_argument(
        "-vl",
        "--val-logs",
        type=int,
        default=DEFAULT_ARGS["val_logs"],
        help="Total number of validation logs",
    )
    parser.add_argument(
        "-rph",
        "--recurrent-pred-horizon",
        action="store_true",
        help="Toggle recurrent prediction horizon",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Perform exploratory data analysis before experiment starts",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument(
        "--ignore-timestamp", action="store_true", help="Ignore timestamp in run name"
    )
    parser.add_argument("--TOY", action="store_true", help="Use synthetic toy dataset")
    parser.add_argument(
        "--FAST-TOY",
        action="store_true",
        help="Use small synthetic toy dataset for testing",
    )

    args = parser.parse_args()

    # Track which args were explicitly provided on the CLI so that "FAST-TOY"
    # helpers don't override the user's intent (e.g. GPU-only usage).
    argv = sys.argv[1:]

    def _has_any(opts: list[str]) -> bool:
        return any(opt in argv for opt in opts)

    args._cli_device = _has_any(["-d", "--device"])
    args._cli_epochs = _has_any(["-e", "--epochs"])
    args._cli_ignore_timestamp = "--ignore-timestamp" in argv

    if args.TOY or args.FAST_TOY:
        args.ticker = "TOY"
        print(f"================================================")
        print(f"TOY RUN: Using synthetic toy dataset for testing")
        print(f"================================================")

        if args.FAST_TOY:
            from config import overwrite_fast_toy_args

            args = overwrite_fast_toy_args(args)

    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Run 'pip install wandb' to install."
            )

    time_stamp_suffix = f"--{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    if args.name is not None:
        args.name = args.name + ("" if args.ignore_timestamp else time_stamp_suffix)
    else:
        print(
            f"Run name not provided. Using default: {'TOY-' if args.TOY else ''}RUN{time_stamp_suffix}"
        )
        args.name = f"{'TOY-' if args.TOY else ''}RUN{time_stamp_suffix}"

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (-d cuda) but is not available.\n"
                "Install a CUDA-enabled PyTorch build (and ensure NVIDIA drivers are installed), "
                "then retry."
            )
        else:
            args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    if args.recurrent_pred_horizon and args.pred_horizon == 1:
        warn("Recurrent prediction horizon is only supported for pred_horizon > 1")
        args.recurrent_pred_horizon = False

    print(f"Run configuration:")
    print(f"  - Project: {args.project}")
    print(f"  - Run name: {args.name}")
    print(f"  - Device: {args.device.type}")
    print(f"  - Architecture: {args.architecture}")
    print(f"  - Stock ticker: {args.ticker}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Look-back: {args.look_back}")
    print(f"  - Prediction horizon: {args.pred_horizon}")
    print(f"  - Hidden width: {args.hidden_width}")
    print(f"  - Drop-out rate: {args.dropout}")
    print(f"  - Validation set size: {args.val_size}")
    print(f"  - # of train logs: {args.train_logs}")
    print(f"  - # of validation logs: {args.val_logs}")
    print(f"  - Recurrent prediction horizon: {args.recurrent_pred_horizon}")

    return args
