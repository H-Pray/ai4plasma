"""Reusable pendulum experiment runner for SRKPINN sweeps."""

from __future__ import annotations

import json
import os
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = RUNNER_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import torch.nn as nn

from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.utils.common import Timer, set_seed
from ai4plasma.utils.device import select_best_device
from SRKPINN import HamiltonianSRKPINN, PendulumSystem, SRKPINNVisCallback


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_scheduler_name(name: str | None) -> str:
    if name is None:
        return ""

    normalized = str(name).strip()
    return "" if normalized.lower() in {"", "none"} else normalized


def default_method_for_stages(stages: int) -> str:
    return "implicit-midpoint" if int(stages) == 1 else "gauss-legendre"


def build_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    activations = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    if normalized not in activations:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation '{name}'. Supported activations: {supported}.")
    return activations[normalized]()


def build_layers(state_dim: int, stages: int, hidden_width: int = 128, hidden_depth: int = 3) -> list[int]:
    return [state_dim] + [hidden_width] * hidden_depth + [state_dim * int(stages)]


def resolve_rollout_steps(dt: float, num_rollout_steps: int, rollout_total_time: float | None) -> tuple[int, float]:
    """Resolve rollout steps and effective physical horizon.

    When ``rollout_total_time`` is provided, convert it to a step count using the
    run-specific ``dt``. Otherwise, keep the requested step count.
    """
    if rollout_total_time is None:
        steps = int(num_rollout_steps)
    else:
        steps = max(1, int(round(float(rollout_total_time) / float(dt))))
    return steps, float(dt) * float(steps)


def write_summary_md(path: Path, title: str, config: dict, summary: dict) -> None:
    lines = [
        f"# {title}",
        "",
        "## Run",
        "",
        f"- run_name: `{config['run_name']}`",
        f"- seed: `{config['seed']}`",
        f"- device: `{config['device']}`",
        "",
        "## Configuration",
        "",
        f"- dt: `{config['dt']}`",
        f"- stages: `{config['stages']}`",
        f"- method: `{config['method']}`",
        f"- train_data_size: `{config['train_data_size']}`",
        f"- sample_mode: `{config['sample_mode']}`",
        (
            f"- loss_weights: `StageDynamics={config['loss_weights']['StageDynamics']}`, "
            f"`InitialOrData={config['loss_weights']['InitialOrData']}`"
        ),
        f"- layers: `{config['layers']}`",
        f"- activation: `{config['activation']}`",
        f"- learning_rate: `{config['learning_rate']}`",
        f"- num_epochs: `{config['num_epochs']}`",
        f"- num_rollout_steps: `{config['num_rollout_steps']}`",
        f"- rollout_total_time: `{config['rollout_total_time']}`",
        f"- scheduler: `{config['scheduler_name']}`",
        f"- scheduler milestones: `{config['scheduler_milestones']}`",
        f"- scheduler gamma: `{config['scheduler_gamma']}`",
        "",
        "## Metrics",
        "",
        f"- train_one_step_rmse: `{summary['train_one_step_rmse']:.6e}`",
        f"- monitored_one_step_l2: `{summary['monitored_one_step_l2']:.6e}`",
        f"- rollout_state_error_final: `{summary['rollout_state_error_final']:.6e}`",
        f"- max_rollout_state_error: `{summary['max_rollout_state_error']:.6e}`",
        f"- energy_drift_final: `{summary['energy_drift_final']:.6e}`",
        f"- max_energy_drift: `{summary['max_energy_drift']:.6e}`",
        f"- symplectic_error: `{summary['symplectic_error']:.6e}`",
        f"- training_time_sec: `{summary['training_time_sec']:.2f}`",
        "",
        "## Artifacts",
        "",
        f"- results: `{config['results_dir']}`",
        f"- runs: `{config['log_dir']}`",
        f"- models: `{config['checkpoint_dir']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pendulum_experiment(
    *,
    run_root: Path,
    run_name: str,
    seed: int = 2026,
    dt: float = 0.1,
    stages: int = 2,
    method: str | None = None,
    num_epochs: int = 6000,
    learning_rate: float = 1e-3,
    train_data_size: int = 512,
    sample_mode: str = "uniform",
    loss_weights: dict[str, float] | None = None,
    layers: list[int] | None = None,
    activation: str = "Tanh",
    monitor_state: list[float] | None = None,
    num_rollout_steps: int = 200,
    rollout_total_time: float | None = None,
    log_freq: int = 100,
    history_freq: int | None = None,
    checkpoint_freq: int = 1000,
    scheduler_name: str = "MultiStepLR",
    scheduler_milestones: list[int] | None = None,
    scheduler_gamma: float = 0.5,
    summary_title: str = "Pendulum Experiment Summary",
) -> tuple[dict, dict]:
    timer = Timer()
    set_seed(seed)

    DEVICE.set_device(select_best_device(print_required=True))
    print(DEVICE)

    stages = int(stages)
    method = default_method_for_stages(stages) if method is None else method
    monitor_state = [1.7, 0.0] if monitor_state is None else list(monitor_state)
    history_freq = log_freq if history_freq is None else history_freq
    scheduler_milestones = [2000, 4000] if scheduler_milestones is None else list(scheduler_milestones)
    scheduler_name = normalize_scheduler_name(scheduler_name)
    num_rollout_steps, effective_rollout_total_time = resolve_rollout_steps(
        dt=dt,
        num_rollout_steps=num_rollout_steps,
        rollout_total_time=rollout_total_time,
    )

    system = PendulumSystem(
        omega0=1.0,
        q_range=(-np.pi, np.pi),
        p_range=(-2.0, 2.0),
    )

    if layers is None:
        layers = build_layers(system.state_dim, stages)
    else:
        layers = list(layers)

    expected_output_dim = system.state_dim * stages
    if layers[-1] != expected_output_dim:
        raise ValueError(
            f"The last layer must equal state_dim * stages = {expected_output_dim}, got {layers[-1]}."
        )

    if loss_weights is None:
        loss_weights = {
            "StageDynamics": 1.0,
            "InitialOrData": 2.0,
        }
    else:
        loss_weights = dict(loss_weights)

    backbone_net = FNN(layers=layers, act_fun=build_activation(activation))

    results_dir = run_root / "results"
    log_dir = run_root / "runs"
    checkpoint_dir = run_root / "models"
    for path in (results_dir, log_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "seed": seed,
        "device": str(DEVICE()),
        "dt": float(dt),
        "stages": stages,
        "method": method,
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "train_data_size": int(train_data_size),
        "sample_mode": sample_mode,
        "loss_weights": loss_weights,
        "layers": layers,
        "activation": activation,
        "monitor_state": monitor_state,
        "num_rollout_steps": int(num_rollout_steps),
        "rollout_total_time": float(effective_rollout_total_time),
        "requested_rollout_total_time": (
            None if rollout_total_time is None else float(rollout_total_time)
        ),
        "log_freq": int(log_freq),
        "history_freq": int(history_freq),
        "checkpoint_freq": int(checkpoint_freq),
        "scheduler_name": scheduler_name or "none",
        "scheduler_milestones": scheduler_milestones,
        "scheduler_gamma": float(scheduler_gamma),
        "results_dir": str(results_dir),
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }
    write_json(run_root / "config.json", config)

    model = HamiltonianSRKPINN(
        system=system,
        dt=dt,
        stages=stages,
        method=method,
        backbone_net=backbone_net,
        train_data_size=train_data_size,
        sample_mode=sample_mode,
        loss_weights=loss_weights,
    )

    viz_callback = SRKPINNVisCallback(
        model=model,
        initial_state=np.asarray(monitor_state, dtype=REAL()),
        num_rollout_steps=num_rollout_steps,
        log_freq=log_freq,
        save_history=True,
        history_freq=history_freq,
    )
    model.register_visualization_callback(viz_callback)

    model.create_optimizer("Adam", lr=learning_rate)
    if scheduler_name:
        model.create_lr_scheduler(
            scheduler_name,
            milestones=scheduler_milestones,
            gamma=scheduler_gamma,
        )

    model.train(
        num_epochs=num_epochs,
        print_loss=True,
        print_loss_freq=log_freq,
        tensorboard_logdir=str(log_dir),
        save_final_model=True,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_freq=checkpoint_freq,
    )

    viz_callback.save_final_results(save_dir=str(results_dir), epoch=num_epochs)
    metrics = viz_callback._compute_rollout_metrics()

    train_pred = model.predict_step(model.train_current_np)
    train_target = model.train_next_np
    train_one_step_rmse = float(np.sqrt(np.mean((train_pred - train_target) ** 2)))

    monitor_state_array = np.asarray(monitor_state, dtype=REAL()).reshape(1, -1)
    monitor_pred = model.predict_step(monitor_state_array)[0]
    monitor_ref = model.reference_stepper(monitor_state_array, dt)[0]
    monitored_one_step_l2 = float(np.linalg.norm(monitor_pred - monitor_ref))

    training_time_sec = float((timer.current(print_required=False) - timer.timer_start).total_seconds())
    summary = {
        "train_one_step_rmse": train_one_step_rmse,
        "monitored_one_step_l2": monitored_one_step_l2,
        "rollout_state_error_final": float(metrics["state_error"][-1]),
        "max_rollout_state_error": float(np.max(metrics["state_error"])),
        "energy_drift_final": float(metrics["pred_energy_drift"][-1]),
        "max_energy_drift": float(np.max(metrics["pred_energy_drift"])),
        "symplectic_error": float(metrics["symplectic_error"]),
        "num_rollout_steps": int(num_rollout_steps),
        "rollout_total_time": float(effective_rollout_total_time),
        "training_time_sec": training_time_sec,
    }
    if num_rollout_steps == 200:
        summary["rollout_state_error_200"] = summary["rollout_state_error_final"]
        summary["energy_drift_200"] = summary["energy_drift_final"]

    write_json(run_root / "summary.json", summary)
    write_summary_md(run_root / "summary.md", summary_title, config, summary)

    print("\n" + "=" * 70)
    print(summary_title.upper())
    print("=" * 70)
    print(f"Run directory: {run_root}")
    print(f"Train one-step RMSE: {summary['train_one_step_rmse']:.6e}")
    print(
        f"Rollout state error @ steps={num_rollout_steps}, T={effective_rollout_total_time:.2f}: "
        f"{summary['rollout_state_error_final']:.6e}"
    )
    print(f"Max rollout state error: {summary['max_rollout_state_error']:.6e}")
    print(
        f"Energy drift @ steps={num_rollout_steps}, T={effective_rollout_total_time:.2f}: "
        f"{summary['energy_drift_final']:.6e}"
    )
    print(f"Max energy drift: {summary['max_energy_drift']:.6e}")
    print(f"Symplectic error: {summary['symplectic_error']:.6e}")
    print(f"Training time (s): {summary['training_time_sec']:.2f}")
    print("=" * 70)

    return config, summary
