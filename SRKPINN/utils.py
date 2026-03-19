"""Utility functions for SRKPINN examples and diagnostics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.integrate import solve_ivp

from ai4plasma.config import REAL


def to_numpy(array) -> np.ndarray:
    """Convert a tensor-like object to a NumPy array."""
    if isinstance(array, np.ndarray):
        return array.astype(REAL(), copy=False)
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(REAL(), copy=False)
    return np.asarray(array, dtype=REAL())


def ensure_2d_state_array(states, state_dim: int) -> np.ndarray:
    """Ensure state data has shape ``(N, state_dim)``."""
    array = to_numpy(states)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != state_dim:
        raise ValueError(f"Expected state array of shape (N, {state_dim}), got {array.shape}.")
    return array.astype(REAL(), copy=False)


def default_reference_stepper(
    system,
    states,
    dt: float,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    """Advance each state by one step using a high-accuracy SciPy solver."""
    state_array = ensure_2d_state_array(states, system.state_dim)
    next_states = np.zeros_like(state_array)
    for idx, state in enumerate(state_array):
        solution = solve_ivp(
            system.numpy_rhs,
            (0.0, float(dt)),
            state,
            method=method,
            t_eval=[float(dt)],
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"Reference integration failed: {solution.message}")
        next_states[idx] = solution.y[:, -1]
    return next_states.astype(REAL())


def rollout_reference(
    system,
    initial_state,
    dt: float,
    num_steps: int,
    stepper: Callable | None = None,
) -> np.ndarray:
    """Generate a reference trajectory."""
    state = ensure_2d_state_array(initial_state, system.state_dim)
    if state.shape[0] != 1:
        raise ValueError("rollout_reference expects a single initial state.")
    reference_stepper = stepper
    if reference_stepper is None:
        reference_stepper = lambda states, step_dt: default_reference_stepper(system, states, step_dt)

    trajectory = [state[0].copy()]
    current = state
    for _ in range(num_steps):
        current = reference_stepper(current, dt)
        current = ensure_2d_state_array(current, system.state_dim)
        trajectory.append(current[0].copy())
    return np.asarray(trajectory, dtype=REAL())


def rollout_model(model, initial_state, num_steps: int) -> np.ndarray:
    """Roll out a trained SRKPINN model."""
    state = ensure_2d_state_array(initial_state, model.state_dim)
    if state.shape[0] != 1:
        raise ValueError("rollout_model expects a single initial state.")

    trajectory = [state[0].copy()]
    current = state
    for _ in range(num_steps):
        current = model.predict_step(current)
        current = ensure_2d_state_array(current, model.state_dim)
        trajectory.append(current[0].copy())
    return np.asarray(trajectory, dtype=REAL())


def compute_energy_drift(system, trajectory: np.ndarray) -> np.ndarray:
    """Compute absolute Hamiltonian drift along a trajectory."""
    energy = system.energy_numpy(trajectory)
    return np.abs(energy - energy[0]).astype(REAL())
