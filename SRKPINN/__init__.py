"""Symplectic Runge-Kutta Physics-Informed Neural Networks.

This package implements structure-preserving RK-PINNs for canonical Hamiltonian
systems. It complements the existing :mod:`ai4plasma.piml.rk_pinn` module with
symplectic Runge-Kutta tableaus, Hamiltonian system abstractions, and reusable
training utilities for long-horizon dynamics.
"""

from .tableau import ButcherTableau, load_symplectic_tableau
from .systems import HamiltonianSystem, PendulumSystem
from .networks import HamiltonianSRKNet, SRKStepOutput
from .model import HamiltonianSRKPINN
from .callbacks import SRKPINNVisCallback

__all__ = [
    "ButcherTableau",
    "load_symplectic_tableau",
    "HamiltonianSystem",
    "PendulumSystem",
    "HamiltonianSRKNet",
    "SRKStepOutput",
    "HamiltonianSRKPINN",
    "SRKPINNVisCallback",
]
