"""Network wrappers for SRKPINN models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SRKStepOutput:
    """Structured outputs for one SRK-PINN time step."""

    q_stages: torch.Tensor
    p_stages: torch.Tensor
    q_next: torch.Tensor
    p_next: torch.Tensor

    @property
    def next_state(self) -> torch.Tensor:
        """Return the concatenated step-end state."""
        return torch.cat((self.q_next, self.p_next), dim=-1)

    @property
    def stage_state(self) -> torch.Tensor:
        """Return the concatenated stage states."""
        return torch.cat((self.q_stages, self.p_stages), dim=-1)


class HamiltonianSRKNet(nn.Module):
    """Wrap a backbone network and expose structured SRK stage outputs.

    Parameters
    ----------
    network : nn.Module
        Backbone mapping ``z_n -> raw_outputs``.
    state_dim : int
        Full canonical state dimension. Must be even.
    stages : int
        Number of RK stages.
    """

    def __init__(self, network: nn.Module, state_dim: int, stages: int) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError("state_dim must be even for canonical Hamiltonian systems.")
        if stages < 1:
            raise ValueError("stages must be a positive integer.")
        self.network = network
        self.state_dim = int(state_dim)
        self.stages = int(stages)
        self.dim_q = self.state_dim // 2
        self.output_dim = self.state_dim * (self.stages + 1)

    def forward(self, state: torch.Tensor) -> SRKStepOutput:
        """Return stage states and the predicted next state."""
        raw = self.network(state)
        if raw.shape[-1] != self.output_dim:
            raise ValueError(
                f"Backbone output dimension mismatch: expected {self.output_dim}, got {raw.shape[-1]}."
            )

        stage_width = self.stages * self.dim_q
        q_stage_flat = raw[:, :stage_width]
        p_stage_flat = raw[:, stage_width : 2 * stage_width]
        next_width_start = 2 * stage_width
        q_next = raw[:, next_width_start : next_width_start + self.dim_q]
        p_next = raw[:, next_width_start + self.dim_q : next_width_start + 2 * self.dim_q]

        return SRKStepOutput(
            q_stages=q_stage_flat.reshape(-1, self.stages, self.dim_q),
            p_stages=p_stage_flat.reshape(-1, self.stages, self.dim_q),
            q_next=q_next,
            p_next=p_next,
        )
