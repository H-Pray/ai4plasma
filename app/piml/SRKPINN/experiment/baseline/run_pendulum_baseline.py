"""Run the SRKPINN pendulum baseline in the experiment workspace."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(".")

BASELINE_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = BASELINE_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import torch.nn as nn

from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.utils.common import Timer, set_seed
from ai4plasma.utils.device import check_gpu
from SRKPINN import HamiltonianSRKPINN, PendulumSystem, SRKPINNVisCallback


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value


def env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_md(path: Path, config: dict, summary: dict) -> None:
    lines = [
        "# Baseline Summary",
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
        f"- loss_weights: `StageDynamics={config['loss_weights']['StageDynamics']}`, `InitialOrData={config['loss_weights']['InitialOrData']}`",
        f"- layers: `{config['layers']}`",
        f"- activation: `{config['activation']}`",
        f"- learning_rate: `{config['learning_rate']}`",
        f"- num_epochs: `{config['num_epochs']}`",
        f"- scheduler milestones: `{config['scheduler_milestones']}`",
        f"- scheduler gamma: `{config['scheduler_gamma']}`",
        "",
        "## Metrics",
        "",
        f"- train_one_step_rmse: `{summary['train_one_step_rmse']:.6e}`",
        f"- monitored_one_step_l2: `{summary['monitored_one_step_l2']:.6e}`",
        f"- rollout_state_error_200: `{summary['rollout_state_error_200']:.6e}`",
        f"- max_rollout_state_error: `{summary['max_rollout_state_error']:.6e}`",
        f"- energy_drift_200: `{summary['energy_drift_200']:.6e}`",
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


def main() -> None:
    timer = Timer()

    seed = env_int("SRKPINN_BASELINE_SEED", 2026)
    set_seed(seed)

    if check_gpu(print_required=True):
        DEVICE.set_device(0)
    else:
        DEVICE.set_device(-1)
    print(DEVICE)

    run_name = env_str("SRKPINN_BASELINE_RUN_NAME", "baseline_v1")
    dt = env_float("SRKPINN_BASELINE_DT", 0.1)
    stages = env_int("SRKPINN_BASELINE_STAGES", 2)
    method = env_str("SRKPINN_BASELINE_METHOD", "gauss-legendre")
    num_epochs = env_int("SRKPINN_BASELINE_EPOCHS", 6000)
    learning_rate = env_float("SRKPINN_BASELINE_LR", 1e-3)
    train_data_size = env_int("SRKPINN_BASELINE_TRAIN_DATA_SIZE", 512)
    sample_mode = env_str("SRKPINN_BASELINE_SAMPLE_MODE", "uniform")
    monitor_state = env_float_list("SRKPINN_BASELINE_MONITOR_STATE", [1.7, 0.0])
    num_rollout_steps = env_int("SRKPINN_BASELINE_ROLLOUT_STEPS", 200)
    log_freq = env_int("SRKPINN_BASELINE_LOG_FREQ", 100)
    history_freq = env_int("SRKPINN_BASELINE_HISTORY_FREQ", log_freq)
    checkpoint_freq = env_int("SRKPINN_BASELINE_CHECKPOINT_FREQ", 1000)
    scheduler_milestones = env_int_list("SRKPINN_BASELINE_MILESTONES", [2000, 4000])
    scheduler_gamma = env_float("SRKPINN_BASELINE_GAMMA", 0.5)

    system = PendulumSystem(
        omega0=1.0,
        q_range=(-np.pi, np.pi),
        p_range=(-2.0, 2.0),
    )

    layers = [system.state_dim, 128, 128, 128, system.state_dim * stages]
    backbone_net = FNN(layers=layers, act_fun=nn.Tanh())
    loss_weights = {
        "StageDynamics": 1.0,
        "InitialOrData": 2.0,
    }

    run_root = BASELINE_ROOT / run_name
    results_dir = run_root / "results"
    log_dir = run_root / "runs"
    checkpoint_dir = run_root / "models"
    for path in (results_dir, log_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "seed": seed,
        "device": str(DEVICE()),
        "dt": dt,
        "stages": stages,
        "method": method,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "train_data_size": train_data_size,
        "sample_mode": sample_mode,
        "loss_weights": loss_weights,
        "layers": layers,
        "activation": "Tanh",
        "monitor_state": monitor_state,
        "num_rollout_steps": num_rollout_steps,
        "log_freq": log_freq,
        "history_freq": history_freq,
        "checkpoint_freq": checkpoint_freq,
        "scheduler_milestones": scheduler_milestones,
        "scheduler_gamma": scheduler_gamma,
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
    model.create_lr_scheduler("MultiStepLR", milestones=scheduler_milestones, gamma=scheduler_gamma)

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
        "rollout_state_error_200": float(metrics["state_error"][-1]),
        "max_rollout_state_error": float(np.max(metrics["state_error"])),
        "energy_drift_200": float(metrics["pred_energy_drift"][-1]),
        "max_energy_drift": float(np.max(metrics["pred_energy_drift"])),
        "symplectic_error": float(metrics["symplectic_error"]),
        "training_time_sec": training_time_sec,
    }

    write_json(run_root / "summary.json", summary)
    write_summary_md(run_root / "summary.md", config, summary)

    print("\n" + "=" * 70)
    print("SRKPINN BASELINE SUMMARY")
    print("=" * 70)
    print(f"Run directory: {run_root}")
    print(f"Train one-step RMSE: {summary['train_one_step_rmse']:.6e}")
    print(f"Rollout state error @ {num_rollout_steps}: {summary['rollout_state_error_200']:.6e}")
    print(f"Max rollout state error: {summary['max_rollout_state_error']:.6e}")
    print(f"Energy drift @ {num_rollout_steps}: {summary['energy_drift_200']:.6e}")
    print(f"Max energy drift: {summary['max_energy_drift']:.6e}")
    print(f"Symplectic error: {summary['symplectic_error']:.6e}")
    print(f"Training time (s): {summary['training_time_sec']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
