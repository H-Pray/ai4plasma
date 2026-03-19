"""Core SRKPINN model implementations."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai4plasma.config import REAL
from ai4plasma.core.network import FNN
from ai4plasma.piml.pinn import PINN
from ai4plasma.utils.common import numpy2torch
from .networks import HamiltonianSRKNet, SRKStepOutput
from .tableau import load_symplectic_tableau
from .utils import default_reference_stepper, ensure_2d_state_array, rollout_model


class HamiltonianSRKPINN(PINN):
    """Symplectic RK-PINN for canonical Hamiltonian systems.

    The model learns a one-step map ``z_n -> z_{n+1}`` while enforcing the stage
    and closure equations of a symplectic implicit Runge-Kutta method.
    """

    def __init__(
        self,
        system,
        dt: float,
        stages: int = 2,
        method: str = "gauss-legendre",
        backbone_net: nn.Module | None = None,
        train_data_size: int = 512,
        sample_mode: str = "uniform",
        loss_weights: dict[str, float] | None = None,
        train_state_sampler: Callable | None = None,
        reference_stepper: Callable | None = None,
        train_data=None,
    ) -> None:
        self.system = system
        self.state_dim = int(system.state_dim)
        self.dim_q = self.state_dim // 2
        self.dt = float(dt)
        self.stages = int(stages)
        self.method = method
        self.sample_mode = sample_mode
        self.train_data_size = int(train_data_size)
        self.loss_weights = {
            "StageDynamics": 1.0,
            "StepClosure": 1.0,
            "InitialOrData": 1.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        self.tableau = load_symplectic_tableau(method=method, stages=stages)
        self.tableau.assert_symplectic()
        self.train_state_sampler = train_state_sampler or getattr(system, "sample_initial_states", None)
        self.reference_stepper = self._build_reference_stepper(reference_stepper)

        self.train_current_np, self.train_next_np = self._prepare_training_pairs(train_data)
        self.train_pair_np = np.concatenate((self.train_current_np, self.train_next_np), axis=1).astype(REAL())
        self.train_pair_tensor = numpy2torch(self.train_pair_np, require_grad=False)

        if backbone_net is None:
            output_dim = self.state_dim * (self.stages + 1)
            backbone_net = FNN(
                layers=[self.state_dim, 128, 128, 128, output_dim],
                act_fun=nn.Tanh(),
            )
        self.backbone_net = backbone_net

        network = HamiltonianSRKNet(backbone_net, state_dim=self.state_dim, stages=self.stages)
        super().__init__(network)
        self.set_loss_func(F.mse_loss)

    def _default_reference_stepper(self, states, dt: float):
        return default_reference_stepper(self.system, states, dt)

    def _build_reference_stepper(self, reference_stepper: Callable | None) -> Callable:
        if reference_stepper is None:
            return self._default_reference_stepper

        def wrapped(states, dt: float):
            try:
                return reference_stepper(states, dt)
            except TypeError:
                return reference_stepper(states)

        return wrapped

    def _call_sampler(self) -> np.ndarray:
        if self.train_state_sampler is None:
            raise ValueError("No train_state_sampler was provided for SRKPINN.")
        try:
            states = self.train_state_sampler(self.train_data_size, mode=self.sample_mode)
        except TypeError:
            states = self.train_state_sampler(self.train_data_size)
        return ensure_2d_state_array(states, self.state_dim)

    def _call_reference_stepper(self, states: np.ndarray) -> np.ndarray:
        next_states = self.reference_stepper(states, self.dt)
        return ensure_2d_state_array(next_states, self.state_dim)

    def _prepare_training_pairs(self, train_data):
        if train_data is None:
            current_states = self._call_sampler()
            next_states = self._call_reference_stepper(current_states)
            return current_states.astype(REAL()), next_states.astype(REAL())

        if isinstance(train_data, (tuple, list)) and len(train_data) == 2:
            current_states = ensure_2d_state_array(train_data[0], self.state_dim)
            next_states = ensure_2d_state_array(train_data[1], self.state_dim)
            if current_states.shape[0] != next_states.shape[0]:
                raise ValueError("Current and next state arrays must contain the same number of samples.")
            return current_states.astype(REAL()), next_states.astype(REAL())

        pair_array = np.asarray(train_data, dtype=REAL())
        if pair_array.ndim != 2 or pair_array.shape[1] != 2 * self.state_dim:
            raise ValueError(
                f"train_data must have shape (N, {2 * self.state_dim}) or be a tuple of state arrays."
            )
        return (
            pair_array[:, : self.state_dim].astype(REAL()),
            pair_array[:, self.state_dim :].astype(REAL()),
        )

    def _split_pairs(self, pair_batch: torch.Tensor):
        current = pair_batch[:, : self.state_dim]
        target = pair_batch[:, self.state_dim :]
        return current, target

    def _forward_from_pairs(self, network: nn.Module, pair_batch: torch.Tensor):
        current_state, target_state = self._split_pairs(pair_batch)
        output = network(current_state)
        return current_state, target_state, output

    def _hamiltonian_gradients(self, output: SRKStepOutput):
        q_flat = output.q_stages.reshape(-1, self.dim_q)
        p_flat = output.p_stages.reshape(-1, self.dim_q)
        dH_dq, dH_dp = self.system.gradients(q_flat, p_flat)
        return dH_dq.reshape_as(output.q_stages), dH_dp.reshape_as(output.p_stages)

    def _stage_dynamics_residual(self, network: nn.Module, pair_batch: torch.Tensor) -> torch.Tensor:
        current_state, _, output = self._forward_from_pairs(network, pair_batch)
        q0, p0 = self.system.split_state(current_state)
        dH_dq, dH_dp = self._hamiltonian_gradients(output)

        stage_q = torch.einsum("ij,bjd->bid", self.A_torch, dH_dp)
        stage_p = torch.einsum("ij,bjd->bid", self.A_torch, dH_dq)

        q_residual = output.q_stages - q0.unsqueeze(1) - self.dt * stage_q
        p_residual = output.p_stages - p0.unsqueeze(1) + self.dt * stage_p
        return torch.cat((q_residual, p_residual), dim=-1)

    def _step_closure_residual(self, network: nn.Module, pair_batch: torch.Tensor) -> torch.Tensor:
        current_state, _, output = self._forward_from_pairs(network, pair_batch)
        q0, p0 = self.system.split_state(current_state)
        dH_dq, dH_dp = self._hamiltonian_gradients(output)

        q_update = torch.einsum("i,bid->bd", self.b_torch, dH_dp)
        p_update = torch.einsum("i,bid->bd", self.b_torch, dH_dq)

        q_residual = output.q_next - q0 - self.dt * q_update
        p_residual = output.p_next - p0 + self.dt * p_update
        return torch.cat((q_residual, p_residual), dim=-1)

    def _data_residual(self, network: nn.Module, pair_batch: torch.Tensor) -> torch.Tensor:
        _, target_state, output = self._forward_from_pairs(network, pair_batch)
        return output.next_state - target_state

    def _define_loss_terms(self):
        self.A_torch = numpy2torch(self.tableau.A.astype(REAL()), require_grad=False)
        self.b_torch = numpy2torch(self.tableau.b.astype(REAL()), require_grad=False)
        self.add_equation(
            "StageDynamics",
            self._stage_dynamics_residual,
            weight=self.loss_weights["StageDynamics"],
            data=self.train_pair_tensor,
        )
        self.add_equation(
            "StepClosure",
            self._step_closure_residual,
            weight=self.loss_weights["StepClosure"],
            data=self.train_pair_tensor,
        )
        self.add_equation(
            "InitialOrData",
            self._data_residual,
            weight=self.loss_weights["InitialOrData"],
            data=self.train_pair_tensor,
        )

    def predict_step(self, initial_state):
        """Predict one step ahead from a state or batch of states."""
        input_array = ensure_2d_state_array(initial_state, self.state_dim)
        input_tensor = numpy2torch(input_array.astype(REAL()), require_grad=False)
        self.network.eval()
        with torch.no_grad():
            output = self.network(input_tensor)
            next_state = output.next_state.detach().cpu().numpy().astype(REAL())
        if np.asarray(initial_state).ndim == 1:
            return next_state[0]
        return next_state

    def rollout(self, initial_state, num_steps: int) -> np.ndarray:
        """Generate a model rollout from a single initial condition."""
        return rollout_model(self, initial_state, num_steps)
