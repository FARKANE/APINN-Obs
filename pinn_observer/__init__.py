"""
PINN Observer Package

A Physics-Informed Neural Network (PINN) based observer implementation
for nonlinear dynamical systems state estimation.
"""

from .models import PINNObserver
from .data_generation import generate_dataset, SYSTEM_MAP
from .residuals import RESIDUAL_MAP, MEAS_IDX
from .train import train
from .plotting import plot_pinn_observer_separate_figures

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "PINNObserver",
    "generate_dataset", 
    "SYSTEM_MAP",
    "RESIDUAL_MAP",
    "MEAS_IDX", 
    "train",
    "plot_pinn_observer_separate_figures"
]
