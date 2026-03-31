"""Fourth SRKPINN sweep block: optimization settings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(".")

from app.piml.SRKPINN.experiment.runner import run_pendulum_experiment


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
DEFAULT_LEARNING_RATES = [3e-4, 1e-3, 3e-3]
DEFAULT_EPOCHS = [6000]
DEFAULT_SCHEDULER_NAME = "MultiStepLR"
DEFAULT_SCHEDULER_GAMMA = 0.5


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def default_scheduler_milestones(num_epochs: int) -> list[int]:
    if num_epochs <= 2:
        return [max(1, num_epochs - 1)]

    first = max(1, int(round(num_epochs / 3)))
    second = max(first + 1, int(round(2 * num_epochs / 3)))
    milestones = sorted({milestone for milestone in (first, second) if 0 < milestone < num_epochs})
    return milestones if milestones else [num_epochs - 1]


def scheduler_label(name: str, milestones: list[int], gamma: float) -> str:
    if not name or name.lower() == "none":
        return "none"
    return f"{name} milestones={milestones}, gamma={gamma}"


def sanitize_token(value: str) -> str:
    return value.lower().replace(".", "p").replace("-", "m").replace("+", "")


def sanitize_float(value: float) -> str:
    return sanitize_token(f"{value:.0e}" if value < 1e-2 else f"{value:g}")


def build_run_name(run_tag: str, learning_rate: float, num_epochs: int, scheduler_name: str) -> str:
    base_name = f"opt_T20_lr_{sanitize_float(learning_rate)}_ep_{num_epochs}_{sanitize_token(scheduler_name or 'none')}"
    return base_name if not run_tag else f"{sanitize_token(run_tag)}_{base_name}"


def summary_stem(summary_suffix: str) -> str:
    suffix = f"_{sanitize_token(summary_suffix)}" if summary_suffix else ""
    return f"aggregate_summary_fixed_time_T20{suffix}"


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def write_markdown(path: Path, rows: list[dict], best_run: dict | None) -> None:
    lines = [
        "# Optimization Sweep",
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
        f"- fixed rollout horizon: `T_eval={EVAL_TOTAL_TIME}`",
        "",
        "## Ranking Rule",
        "",
        "Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.",
        "",
        "## Runs",
        "",
        "| run_name | status | learning_rate | num_epochs | scheduler | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['learning_rate']:.1e}` | `{row['num_epochs']}` | "
                f"`{row['scheduler_label']}` | `{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['learning_rate']:.1e}` | `{row['num_epochs']}` | "
                f"`{row['scheduler_label']}` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |"
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
                f"- learning_rate: `{best_run['learning_rate']:.1e}`",
                f"- num_epochs: `{best_run['num_epochs']}`",
                f"- scheduler: `{best_run['scheduler_label']}`",
                f"- rollout_steps: `{summary['num_rollout_steps']}`",
                f"- rollout_total_time: `{summary['rollout_total_time']:.2f}`",
                f"- rollout_state_error_final: `{summary['rollout_state_error_final']:.6e}`",
                f"- max_energy_drift: `{summary['max_energy_drift']:.6e}`",
                f"- training_time_sec: `{summary['training_time_sec']:.2f}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    seed = env_int("SRKPINN_OPT_SEED", 2026)
    learning_rates = env_float_list("SRKPINN_OPT_LR_VALUES", DEFAULT_LEARNING_RATES)
    epoch_values = env_int_list("SRKPINN_OPT_EPOCHS", DEFAULT_EPOCHS)
    scheduler_name = env_str("SRKPINN_OPT_SCHEDULER_NAME", DEFAULT_SCHEDULER_NAME)
    scheduler_gamma = env_float("SRKPINN_OPT_SCHEDULER_GAMMA", DEFAULT_SCHEDULER_GAMMA)
    explicit_milestones = env_int_list("SRKPINN_OPT_SCHEDULER_MILESTONES", [])
    log_freq = env_int("SRKPINN_OPT_LOG_FREQ", 100)
    history_freq = env_int("SRKPINN_OPT_HISTORY_FREQ", log_freq)
    checkpoint_freq = env_int("SRKPINN_OPT_CHECKPOINT_FREQ", 1000)
    run_tag = env_str("SRKPINN_OPT_RUN_TAG", "")
    summary_suffix = env_str("SRKPINN_OPT_SUMMARY_SUFFIX", "")

    rows = []

    for num_epochs in epoch_values:
        scheduler_milestones = (
            explicit_milestones if explicit_milestones else default_scheduler_milestones(num_epochs)
        )
        current_scheduler_label = scheduler_label(
            scheduler_name,
            scheduler_milestones,
            scheduler_gamma,
        )

        for learning_rate in learning_rates:
            run_name = build_run_name(run_tag, learning_rate, num_epochs, scheduler_name)
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
                    rollout_total_time=EVAL_TOTAL_TIME,
                    log_freq=log_freq,
                    history_freq=history_freq,
                    checkpoint_freq=checkpoint_freq,
                    scheduler_name=scheduler_name,
                    scheduler_milestones=scheduler_milestones,
                    scheduler_gamma=scheduler_gamma,
                    summary_title="SRKPINN Optimization Sweep Summary (Fixed Physical Horizon)",
                )
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "completed",
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "scheduler_name": scheduler_name,
                        "scheduler_milestones": scheduler_milestones,
                        "scheduler_gamma": scheduler_gamma,
                        "scheduler_label": current_scheduler_label,
                        "config": config,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "failed",
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "scheduler_name": scheduler_name,
                        "scheduler_milestones": scheduler_milestones,
                        "scheduler_gamma": scheduler_gamma,
                        "scheduler_label": current_scheduler_label,
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
        "eval_total_time": EVAL_TOTAL_TIME,
        "learning_rates": learning_rates,
        "epoch_values": epoch_values,
        "scheduler_name": scheduler_name,
        "scheduler_gamma": scheduler_gamma,
        "scheduler_milestones": explicit_milestones,
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
    write_markdown(SWEEP_ROOT / f"{stem}.md", rows, best_run)


if __name__ == "__main__":
    main()
