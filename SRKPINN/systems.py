"""Hamiltonian systems for SRKPINN benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ai4plasma.config import REAL


class HamiltonianSystem(ABC):
    r"""Abstract base class for canonical Hamiltonian systems.

    Subclasses define the Hamiltonian

    .. math::

       \dot q = \nabla_p H(q, p), \qquad \dot p = -\nabla_q H(q, p),

    together with a NumPy-compatible right-hand side for reference integration.
    """

    def __init__(self, state_dim: int) -> None:
        if state_dim % 2 != 0:
            raise ValueError("Hamiltonian systems require an even state dimension.")
        self.state_dim = int(state_dim)

    @property
    def dim_q(self) -> int:
        """Return the dimensionality of the canonical coordinates."""
        return self.state_dim // 2

    def split_state(self, state):
        """Split a state array into ``(q, p)`` along the last axis."""
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got {state.shape[-1]}."
            )
        return state[..., : self.dim_q], state[..., self.dim_q :]

    def merge_state(self, q, p):
        """Merge canonical coordinates into a single state array."""
        if isinstance(q, torch.Tensor):
            return torch.cat((q, p), dim=-1)
        return np.concatenate((q, p), axis=-1)

    @abstractmethod
    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Evaluate the Hamiltonian on PyTorch tensors."""

    @abstractmethod
    def hamiltonian_numpy(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Evaluate the Hamiltonian on NumPy arrays."""

    @abstractmethod
    def numpy_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the canonical vector field for SciPy integrators."""

    def gradients(self, q: torch.Tensor, p: torch.Tensor):
        """Return ``(dH/dq, dH/dp)`` using autograd."""
        energy = self.hamiltonian(q, p)
        if energy.ndim == 1:
            energy = energy.unsqueeze(-1)
        dH_dq, dH_dp = torch.autograd.grad(
            energy.sum(),
            (q, p),
            retain_graph=True,
            create_graph=True,
        )
        return dH_dq, dH_dp

    def energy_numpy(self, state: np.ndarray) -> np.ndarray:
        """Compute the Hamiltonian on a full state array."""
        q, p = self.split_state(state)
        return self.hamiltonian_numpy(q, p)

    def sample_initial_states(self, num_samples: int, mode: str = "random") -> np.ndarray:
        """Sample initial states for training.

        Subclasses should override this method when they support internal data
        generation.
        """
        raise NotImplementedError("This Hamiltonian system does not provide a sampler.")


class PendulumSystem(HamiltonianSystem):
    r"""Nonlinear pendulum in canonical coordinates.

    The Hamiltonian is

    .. math::

       H(q, p) = \frac{1}{2} p^2 + \omega_0^2 (1 - \cos q).
    """

    def __init__(
        self,
        omega0: float = 1.0,
        q_range: tuple[float, float] = (-np.pi, np.pi),
        p_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__(state_dim=2)
        self.omega0 = float(omega0)
        self.q_range = q_range
        self.p_range = p_range

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        kinetic = 0.5 * torch.sum(p * p, dim=-1, keepdim=True)
        potential = (self.omega0**2) * torch.sum(1.0 - torch.cos(q), dim=-1, keepdim=True)
        return kinetic + potential

    def hamiltonian_numpy(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        kinetic = 0.5 * np.sum(p * p, axis=-1)
        potential = (self.omega0**2) * np.sum(1.0 - np.cos(q), axis=-1)
        return (kinetic + potential).astype(REAL())

    def numpy_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        del t
        q, p = self.split_state(state.reshape(1, -1))
        dq = p
        dp = -(self.omega0**2) * np.sin(q)
        return self.merge_state(dq, dp).reshape(-1)

    def sample_initial_states(self, num_samples: int, mode: str = "random") -> np.ndarray:
        """Sample pendulum states from a bounded box in phase space."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")

        mode = mode.lower()
        if mode == "uniform":
            grid_side = int(np.ceil(np.sqrt(num_samples)))
            q_vals = np.linspace(self.q_range[0], self.q_range[1], grid_side, dtype=REAL())
            p_vals = np.linspace(self.p_range[0], self.p_range[1], grid_side, dtype=REAL())
            qq, pp = np.meshgrid(q_vals, p_vals, indexing="ij")
            states = np.stack((qq.reshape(-1), pp.reshape(-1)), axis=1)
            return states[:num_samples].astype(REAL())
        if mode == "random":
            q = np.random.uniform(self.q_range[0], self.q_range[1], size=(num_samples, 1))
            p = np.random.uniform(self.p_range[0], self.p_range[1], size=(num_samples, 1))
            return np.concatenate((q, p), axis=1).astype(REAL())
        raise ValueError(f"Unsupported sampling mode for PendulumSystem: {mode}")
