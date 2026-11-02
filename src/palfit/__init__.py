"""Fit a physical activity level (PAL) to an observed weight trajectory with Hall's adult model."""

from palfit._fit import FitResult, fit_pal
from palfit._model import Trajectory, simulate

__version__ = "0.1.0"
__all__ = ["FitResult", "Trajectory", "__version__", "fit_pal", "simulate"]
