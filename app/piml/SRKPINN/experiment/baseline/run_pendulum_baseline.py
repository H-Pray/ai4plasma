"""Run the SRKPINN pendulum baseline in the experiment workspace."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(".")

BASELINE_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = BASELINE_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from app.piml.SRKPINN.experiment.runner import run_pendulum_experiment


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


def main() -> None:
    seed = env_int("SRKPINN_BASELINE_SEED", 2026)

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

    run_root = BASELINE_ROOT / run_name
    run_pendulum_experiment(
        run_root=run_root,
        run_name=run_name,
        seed=seed,
        dt=dt,
        stages=stages,
        method=method,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        train_data_size=train_data_size,
        sample_mode=sample_mode,
        loss_weights={
            "StageDynamics": 1.0,
            "InitialOrData": 2.0,
        },
        activation="Tanh",
        monitor_state=monitor_state,
        num_rollout_steps=num_rollout_steps,
        log_freq=log_freq,
        history_freq=history_freq,
        checkpoint_freq=checkpoint_freq,
        scheduler_name="MultiStepLR",
        scheduler_milestones=scheduler_milestones,
        scheduler_gamma=scheduler_gamma,
        summary_title="SRKPINN Baseline Summary",
    )


if __name__ == "__main__":
    main()
