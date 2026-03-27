"""Visualization callbacks for SRKPINN training."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ai4plasma.piml.pinn import VisualizationCallback
from .utils import compute_energy_drift, rollout_model, rollout_reference


class SRKPINNVisCallback(VisualizationCallback):
    """Visualize SRKPINN phase-space and energy-preservation behavior."""

    def __init__(
        self,
        model,
        initial_state,
        num_rollout_steps: int = 200,
        log_freq: int = 100,
        save_history: bool = True,
        history_freq: int | None = None,
    ) -> None:
        super().__init__(name="SRKPINN", log_freq=log_freq)
        self.model = model
        self.initial_state = np.asarray(initial_state, dtype=np.float32).reshape(-1)
        self.num_rollout_steps = int(num_rollout_steps)
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            "epochs": [],
            "losses": [],
            "energy_drift": [],
            "state_error": [],
            "symplectic_error": [],
        }

    def _compute_rollout_metrics(self):
        pred_traj = rollout_model(self.model, self.initial_state, self.num_rollout_steps)
        ref_traj = rollout_reference(
            self.model.system,
            self.initial_state,
            self.model.dt,
            self.num_rollout_steps,
            stepper=self.model.reference_stepper,
        )
        pred_energy_drift = compute_energy_drift(self.model.system, pred_traj)
        ref_energy_drift = compute_energy_drift(self.model.system, ref_traj)
        state_error = np.linalg.norm(pred_traj - ref_traj, axis=1)
        symplectic_residual = self.model.symplectic_map_residual(self.initial_state)
        symplectic_error = float(np.max(np.abs(symplectic_residual)))
        return {
            "pred_traj": pred_traj,
            "ref_traj": ref_traj,
            "pred_energy_drift": pred_energy_drift,
            "ref_energy_drift": ref_energy_drift,
            "state_error": state_error,
            "symplectic_error": symplectic_error,
        }

    def _make_figure(self, epoch: int, metrics: dict, total_loss=None):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        steps = np.arange(metrics["pred_traj"].shape[0])

        ax_phase = axes[0, 0]
        ax_phase.plot(metrics["ref_traj"][:, 0], metrics["ref_traj"][:, 1], "k--", linewidth=2, label="Reference")
        ax_phase.plot(metrics["pred_traj"][:, 0], metrics["pred_traj"][:, 1], "tab:blue", linewidth=2, label="SRKPINN")
        ax_phase.set_xlabel("q")
        ax_phase.set_ylabel("p")
        ax_phase.set_title("Phase Portrait")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend(loc="best")
        ax_phase.text(
            0.03,
            0.05,
            f"Map sympl. err: {metrics['symplectic_error']:.2e}",
            transform=ax_phase.transAxes,
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax_state = axes[0, 1]
        ax_state.plot(steps, metrics["ref_traj"][:, 0], "k--", linewidth=2, label="q ref")
        ax_state.plot(steps, metrics["pred_traj"][:, 0], "tab:blue", linewidth=2, label="q pred")
        ax_state.plot(steps, metrics["ref_traj"][:, 1], color="gray", linestyle="--", linewidth=2, label="p ref")
        ax_state.plot(steps, metrics["pred_traj"][:, 1], color="tab:orange", linewidth=2, label="p pred")
        ax_state.set_xlabel("Step")
        ax_state.set_ylabel("State")
        ax_state.set_title("State Rollout")
        ax_state.grid(True, alpha=0.3)
        ax_state.legend(loc="best")

        ax_energy = axes[1, 0]
        ax_energy.semilogy(steps, metrics["ref_energy_drift"] + 1e-12, "k--", linewidth=2, label="Reference")
        ax_energy.semilogy(
            steps,
            metrics["pred_energy_drift"] + 1e-12,
            "tab:green",
            linewidth=2,
            label="SRKPINN",
        )
        ax_energy.semilogy(
            steps,
            metrics["state_error"] + 1e-12,
            color="tab:red",
            linewidth=2,
            label="State error",
        )
        ax_energy.set_xlabel("Step")
        ax_energy.set_ylabel("Magnitude")
        ax_energy.set_title("Energy Drift and State Error")
        ax_energy.grid(True, alpha=0.3, which="both")
        ax_energy.legend(loc="best")

        ax_loss = axes[1, 1]
        if self.history["epochs"]:
            ax_loss.semilogy(self.history["epochs"], self.history["losses"], color="tab:purple", linewidth=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss")
        ax_loss.grid(True, alpha=0.3, which="both")
        if total_loss is not None:
            if isinstance(total_loss, torch.Tensor):
                total_loss = float(total_loss.item())
            ax_loss.text(
                0.98,
                0.95,
                f"Current loss: {total_loss:.2e}",
                transform=ax_loss.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.suptitle(f"SRKPINN Pendulum Diagnostics - Epoch {epoch}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def visualize(self, network, epoch: int, writer, **kwargs):
        del network, writer
        metrics = self._compute_rollout_metrics()
        total_loss = kwargs.get("total_loss")

        if self.save_history and epoch % self.history_freq == 0:
            self.history["epochs"].append(epoch)
            if total_loss is not None:
                if isinstance(total_loss, torch.Tensor):
                    self.history["losses"].append(float(total_loss.item()))
                else:
                    self.history["losses"].append(float(total_loss))
            else:
                self.history["losses"].append(np.nan)
            self.history["energy_drift"].append(float(metrics["pred_energy_drift"][-1]))
            self.history["state_error"].append(float(metrics["state_error"][-1]))
            self.history["symplectic_error"].append(float(metrics["symplectic_error"]))

        fig = self._make_figure(epoch, metrics, total_loss=total_loss)
        return {"pendulum_rollout": fig}

    def save_final_results(self, save_dir: str, epoch: int | None = None) -> None:
        """Save final diagnostic figures to disk."""
        os.makedirs(save_dir, exist_ok=True)
        metrics = self._compute_rollout_metrics()
        fig = self._make_figure(epoch or 0, metrics)
        fig.savefig(os.path.join(save_dir, "final_panels.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        loss_fig, ax = plt.subplots(figsize=(7, 4.5))
        if self.history["epochs"]:
            ax.semilogy(self.history["epochs"], self.history["losses"], color="tab:purple", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SRKPINN Training Loss")
        ax.grid(True, alpha=0.3, which="both")
        loss_fig.tight_layout()
        loss_fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(loss_fig)
