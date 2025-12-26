def overwrite_fast_toy_args(args):
    # NOTE: Do NOT override the user's device choice. This project supports GPU-only usage.
    args.architecture = "LSTM"
    # Respect explicit CLI overrides (e.g. -e/--epochs).
    if not getattr(args, "_cli_epochs", False):
        args.epochs = 10
    args.batch_size = 64
    args.look_back = 5
    args.pred_horizon = 1
    args.hidden_width = 128
    args.dropout = 0.0
    args.val_size = 1
    args.train_logs = 2
    args.val_logs = 2
    args.recurrent_pred_horizon = False
    args.eda = False
    args.wandb = False
    if not getattr(args, "_cli_ignore_timestamp", False):
        args.ignore_timestamp = False
    args.TOY = True
    return args
