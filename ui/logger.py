from abc import ABC, abstractmethod
import os
from pathlib import Path, PosixPath
from typing import Any, Callable, Dict, List, Union
from warnings import warn

import numpy as np
from plotly.graph_objs import Figure

try:
    import wandb

    WANDB_IMPORTED = True
except ImportError:
    warn(
        "`wandb` is not installed. Wandb logging is not available, do not run with `--wandb` flag."
    )
    WANDB_IMPORTED = False


class Logger(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def log_scalars(self, scalars: Dict[str, Any], step: int):
        pass

    @abstractmethod
    def log_plotly(self, figs: Dict[str, Figure], step: int):
        pass


class WandbLogger(Logger):

    def __init__(self):
        if not WANDB_IMPORTED:
            raise ImportError(
                "`wandb` is not installed. Run 'pip install wandb' to install."
            )

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        wandb.log(scalars, step=step)

    def log_plotly(self, figs: Dict[str, Figure], step: int):
        wandb.log(figs, step=step)


class LocalLogger(Logger):

    def __init__(self, run_folder: Union[str, PosixPath]):
        self.run_folder = Path(run_folder)
        self.longest_scalar_name = 20

    def log_scalars(self, scalars: Dict[str, Any], step: int):
        print(f"\nScalars at step {step}:")
        for name, scalar in scalars.items():
            print(f"  - {name:{self.longest_scalar_name}} : {scalar}")

    def log_plotly(self, figs: Dict[str, Figure], step: int):
        disable_plots = os.getenv("STOCKPRED_DISABLE_PLOTS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        save_png = os.getenv("STOCKPRED_SAVE_PNG", "0").lower() in ("1", "true", "yes")

        if disable_plots:
            print(f"\nPlot saving disabled (STOCKPRED_DISABLE_PLOTS=1).")
            return

        print(f"\nSaving figures at step {step}:")
        for name, fig in figs.items():
            name = name.replace("/", "_")
            print(f"  - {name}_{step}.html" + ("/png" if save_png else ""))
            fig.write_html(f"{self.run_folder}/figures/html/{name}_{step}.html")
            if save_png:
                try:
                    fig.write_image(f"{self.run_folder}/figures/png/{name}_{step}.png")
                except Exception as e:
                    warn(
                        f"Failed to save PNG for figure {name} at step {step}: {e}. "
                        f"HTML was still saved. You can retry with STOCKPRED_SAVE_PNG=1 after fixing kaleido."
                    )


class LoggerHandler:

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers
        self.accumulated_scalars = {}
        self.accumulated_figs = {}
        self.latest_step = 0

    def log_apply_scalars(self, name: str, func: Callable, *args, reduce_before=True):
        """
        Apply a function to the logged (reduced) scalars and log to a new scalar name.
        E.g., if scalars "scalar1" and "scalar2" have already been logged, we can apply
        a function such as `lambda x, y: x + y` to get a new scalar "scalar1_plus_scalar2".
        In this example, `name` is expected to be the new scalar name, `func` is expected
        to be a callable of the same number of arguments as the number of scalar names to
        be provided in args. `reduce_before` is a boolean flag to indicate whether to
        apply the reduction of scalars (averaging) before applying the function.

        Note that if `reduce_before` is set to False, the function should expect a list
        input, as the scalars will not have been reduced to a single value.
        """
        number_of_args = len(args)
        number_of_params = func.__code__.co_argcount
        if number_of_args != number_of_params:
            raise ValueError(
                f"Number of arguments ({number_of_args}) does not match number of parameters ({number_of_params})!"
            )

        if reduce_before:
            scalars = self.reduce_scalars(self.accumulated_scalars)
        else:
            scalars = self.accumulated_scalars

        for scalar_name in args:
            if scalar_name not in scalars:
                raise ValueError(
                    f"Scalar {scalar_name} not found among logged scalars."
                )

        new_scalar = func(*[scalars[scalar_name] for scalar_name in args])
        self.add_scalar(name, new_scalar)

    def reduce_scalars(
        self, scalars: Dict[str, Union[List[Any], Any]], inplace: bool = False
    ):
        """
        Reduce the accumulated scalars to a single scalar value.
        """
        if inplace:
            reduced_scalars = scalars
        else:
            reduced_scalars = {}

        for name, scalar in scalars.items():
            if isinstance(scalar, List) and len(scalar) > 0:
                reduced_scalars[name] = sum(scalar) / len(scalar)
            else:
                reduced_scalars[name] = scalar

        if not inplace:
            return reduced_scalars

    def push_scalars(self, step: int):
        """
        Push the accumulated scalars to the loggers.
        """
        if step < self.latest_step:
            warn(f"Step {step} is less than the latest step {self.latest_step}!")

        if self.accumulated_scalars:
            self.reduce_scalars(self.accumulated_scalars, inplace=True)
            for logger in self.loggers:
                logger.log_scalars(self.accumulated_scalars, step)
            self.accumulated_scalars = {}

        # Update latest step
        self.latest_step = step

        # Reset longest scalar name
        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                logger.longest_scalar_name = 20

    def push_plotly(self, step: int):
        """
        Push the accumulated plots to the loggers.
        """
        if step < self.latest_step:
            warn(f"Step {step} is less than the latest step {self.latest_step}!")

        if self.accumulated_figs:
            for logger in self.loggers:
                logger.log_plotly(self.accumulated_figs, step)
            self.accumulated_figs = {}

        # Update latest step
        self.latest_step = step

    def add_scalar(self, name: str, scalar: Any):
        """
        Add scalar to the accumulated scalars to be logged.
        """
        if name in self.accumulated_scalars:  # append to list
            self.accumulated_scalars[name].append(scalar)
        else:
            self.accumulated_scalars[name] = [scalar]

        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                if len(name) > logger.longest_scalar_name:
                    logger.longest_scalar_name = len(name)

    def add_plotly(self, name: str, fig: Figure):
        """
        Add plot to the accumulated Plotly plots to be logged.
        """
        if not isinstance(fig, Figure):
            warn(f"Figure {name} is not a Plotly figure; returning without adding.")
            return

        if name in self.accumulated_figs:
            warn(f"Overwriting figure {name}!")
        self.accumulated_figs[name] = fig
