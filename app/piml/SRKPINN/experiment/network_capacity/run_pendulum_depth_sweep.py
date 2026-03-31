"""Network-capacity follow-up: depth sweep with fixed best width and optimization settings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(".")

from app.piml.SRKPINN.experiment.runner import build_layers, run_pendulum_experiment


SWEEP_ROOT = Path(__file__).resolve().parent
EVAL_TOTAL_TIME = 20.0
DT = 0.2
STAGES = 3
METHOD = "gauss-legendre"
TRAIN_DATA_SIZE = 512
SAMPLE_MODE = "uniform"
LOSS_WEIGHTS = {
    "StageDynamics": 1.0,
    "InitialOrData": 2.0,
}
LEARNING_RATE = 1e-3
NUM_EPOCHS = 6000
SCHEDULER_NAME = "MultiStepLR"
SCHEDULER_MILESTONES = [2000, 4000]
SCHEDULER_GAMMA = 0.5
HIDDEN_WIDTH = 128
DEFAULT_DEPTHS = [3, 4, 5]
ACTIVATION = "Tanh"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def sanitize_token(value: str) -> str:
    return value.lower().replace(".", "p").replace("-", "m").replace("+", "")


def sanitize_float(value: float) -> str:
    return sanitize_token(f"{value:.0e}" if value < 1e-2 else f"{value:g}")


def scheduler_label() -> str:
    return f"{SCHEDULER_NAME} milestones={SCHEDULER_MILESTONES}, gamma={SCHEDULER_GAMMA}"


def summary_stem(summary_suffix: str) -> str:
    suffix = f"_{sanitize_token(summary_suffix)}" if summary_suffix else ""
    return f"depth_summary_fixed_time_T20{suffix}"


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def build_run_name(
    run_tag: str,
    learning_rate: float,
    num_epochs: int,
    hidden_width: int,
    hidden_depth: int,
    activation: str,
) -> str:
    base_name = (
        f"depth_T20_lr_{sanitize_float(learning_rate)}_ep_{num_epochs}_w_{hidden_width}_d_{hidden_depth}_{sanitize_token(activation)}"
    )
    return base_name if not run_tag else f"{sanitize_token(run_tag)}_{base_name}"


def write_markdown(
    path: Path,
    rows: list[dict],
    best_run: dict | None,
    *,
    learning_rate: float,
    num_epochs: int,
    hidden_width: int,
    activation: str,
) -> None:
    lines = [
        "# Depth Sweep",
        "",
        "## Fixed Configuration",
        "",
        f"- dt: `{DT}`",
        f"- stages: `{STAGES}`",
        f"- method: `{METHOD}`",
        f"- train_data_size: `{TRAIN_DATA_SIZE}`",
        f"- sample_mode: `{SAMPLE_MODE}`",
        (
            f"- loss_weights: `StageDynamics={LOSS_WEIGHTS['StageDynamics']}`, "
            f"`InitialOrData={LOSS_WEIGHTS['InitialOrData']}`"
        ),
        f"- learning_rate: `{learning_rate:.1e}`",
        f"- num_epochs: `{num_epochs}`",
        f"- scheduler: `{scheduler_label()}`",
        f"- hidden_width: `{hidden_width}`",
        f"- activation: `{activation}`",
        f"- fixed rollout horizon: `T_eval={EVAL_TOTAL_TIME}`",
        "",
        "## Ranking Rule",
        "",
        "Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.",
        "",
        "## Runs",
        "",
        "| run_name | status | depth | layers | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['hidden_depth']}` | `{row['layers']}` | "
                f"`{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['hidden_depth']}` | `{row['layers']}` | "
                "`-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |"
            )
            lines.append("")
            lines.append(f"Failure detail for `{row['run_name']}`: `{row['error']}`")
            lines.append("")

    if best_run is not None:
        summary = best_run["summary"]
        lines.extend(
            [
                "## Best Run",
                "",
                f"- run_name: `{best_run['run_name']}`",
                f"- depth: `{best_run['hidden_depth']}`",
                f"- layers: `{best_run['layers']}`",
                f"- rollout_steps: `{summary['num_rollout_steps']}`",
                f"- rollout_total_time: `{summary['rollout_total_time']:.2f}`",
                f"- rollout_state_error_final: `{summary['rollout_state_error_final']:.6e}`",
                f"- max_energy_drift: `{summary['max_energy_drift']:.6e}`",
                f"- train_one_step_rmse: `{summary['train_one_step_rmse']:.6e}`",
                f"- training_time_sec: `{summary['training_time_sec']:.2f}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    seed = env_int("SRKPINN_DEPTH_SEED", 2026)
    learning_rate = env_float("SRKPINN_DEPTH_LR", LEARNING_RATE)
    num_epochs = env_int("SRKPINN_DEPTH_EPOCHS", NUM_EPOCHS)
    hidden_width = env_int("SRKPINN_DEPTH_WIDTH", HIDDEN_WIDTH)
    depths = env_int_list("SRKPINN_DEPTH_VALUES", DEFAULT_DEPTHS)
    activation = env_str("SRKPINN_DEPTH_ACTIVATION", ACTIVATION)
    log_freq = env_int("SRKPINN_DEPTH_LOG_FREQ", 100)
    history_freq = env_int("SRKPINN_DEPTH_HISTORY_FREQ", log_freq)
    checkpoint_freq = env_int("SRKPINN_DEPTH_CHECKPOINT_FREQ", 1000)
    run_tag = env_str("SRKPINN_DEPTH_RUN_TAG", "")
    summary_suffix = env_str("SRKPINN_DEPTH_SUMMARY_SUFFIX", "")

    rows = []

    for hidden_depth in depths:
        layers = build_layers(state_dim=2, stages=STAGES, hidden_width=hidden_width, hidden_depth=hidden_depth)
        run_name = build_run_name(run_tag, learning_rate, num_epochs, hidden_width, hidden_depth, activation)
        run_root = SWEEP_ROOT / run_name

        try:
            config, summary = run_pendulum_experiment(
                run_root=run_root,
                run_name=run_name,
                seed=seed,
                dt=DT,
                stages=STAGES,
                method=METHOD,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                train_data_size=TRAIN_DATA_SIZE,
                sample_mode=SAMPLE_MODE,
                loss_weights=LOSS_WEIGHTS,
                layers=layers,
                activation=activation,
                rollout_total_time=EVAL_TOTAL_TIME,
                log_freq=log_freq,
                history_freq=history_freq,
                checkpoint_freq=checkpoint_freq,
                scheduler_name=SCHEDULER_NAME,
                scheduler_milestones=SCHEDULER_MILESTONES,
                scheduler_gamma=SCHEDULER_GAMMA,
                summary_title="SRKPINN Depth Sweep Summary (Fixed Physical Horizon)",
            )
            rows.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "hidden_width": hidden_width,
                    "hidden_depth": hidden_depth,
                    "activation": activation,
                    "layers": layers,
                    "config": config,
                    "summary": summary,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "hidden_width": hidden_width,
                    "hidden_depth": hidden_depth,
                    "activation": activation,
                    "layers": layers,
                    "error": str(exc),
                }
            )

    completed_rows = [row for row in rows if row["status"] == "completed"]
    best_run = min(completed_rows, key=rank_key) if completed_rows else None

    payload = {
        "seed": seed,
        "dt": DT,
        "stages": STAGES,
        "method": METHOD,
        "train_data_size": TRAIN_DATA_SIZE,
        "sample_mode": SAMPLE_MODE,
        "loss_weights": LOSS_WEIGHTS,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "scheduler_name": SCHEDULER_NAME,
        "scheduler_milestones": SCHEDULER_MILESTONES,
        "scheduler_gamma": SCHEDULER_GAMMA,
        "hidden_width": hidden_width,
        "activation": activation,
        "depths": depths,
        "eval_total_time": EVAL_TOTAL_TIME,
        "run_tag": run_tag,
        "ranking_rule": [
            "rollout_state_error_final",
            "max_energy_drift",
            "train_one_step_rmse",
        ],
        "best_run": best_run["run_name"] if best_run else None,
        "rows": rows,
    }
    stem = summary_stem(summary_suffix)
    (SWEEP_ROOT / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        SWEEP_ROOT / f"{stem}.md",
        rows,
        best_run,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        hidden_width=hidden_width,
        activation=activation,
    )


if __name__ == "__main__":
    main()
