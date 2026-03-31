"""Optimization follow-up: scheduler sweep with fixed best learning rate."""

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
LEARNING_RATE = 1e-3
NUM_EPOCHS = 6000
DEFAULT_SCHEDULER_CONFIGS = [
    {
        "name": "none",
        "milestones": [],
        "gamma": 0.5,
    },
    {
        "name": "MultiStepLR",
        "milestones": [2000, 4000],
        "gamma": 0.5,
    },
    {
        "name": "MultiStepLR",
        "milestones": [3000, 5000],
        "gamma": 0.5,
    },
    {
        "name": "MultiStepLR",
        "milestones": [1500, 3000, 4500],
        "gamma": 0.5,
    },
    {
        "name": "MultiStepLR",
        "milestones": [2000, 4000],
        "gamma": 0.1,
    },
]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def sanitize_token(value: str) -> str:
    return value.lower().replace(".", "p").replace("-", "m").replace("+", "")


def sanitize_float(value: float) -> str:
    return sanitize_token(f"{value:.0e}" if value < 1e-2 else f"{value:g}")


def summary_stem(summary_suffix: str) -> str:
    suffix = f"_{sanitize_token(summary_suffix)}" if summary_suffix else ""
    return f"scheduler_summary_fixed_time_T20{suffix}"


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def normalize_scheduler_entry(raw: dict, index: int) -> dict:
    name = str(raw.get("name", "none")).strip() or "none"
    normalized_name = "none" if name.lower() == "none" else name
    milestones = [int(item) for item in raw.get("milestones", [])]
    gamma = float(raw.get("gamma", 0.5))

    if normalized_name == "none":
        token = "none"
        label = "none"
        milestones = []
    else:
        milestone_token = "_".join(str(item) for item in milestones) if milestones else "nomilestones"
        gamma_token = sanitize_token(f"{gamma:g}")
        token = f"{sanitize_token(normalized_name)}_ms_{milestone_token}_g_{gamma_token}"
        label = f"{normalized_name} milestones={milestones}, gamma={gamma}"

    return {
        "index": index,
        "name": normalized_name,
        "milestones": milestones,
        "gamma": gamma,
        "token": token,
        "label": label,
    }


def load_scheduler_configs() -> list[dict]:
    raw = os.environ.get("SRKPINN_SCHED_CONFIGS_JSON")
    entries = DEFAULT_SCHEDULER_CONFIGS if raw is None or raw.strip() == "" else json.loads(raw)
    return [normalize_scheduler_entry(entry, index) for index, entry in enumerate(entries)]


def build_run_name(run_tag: str, scheduler_config: dict) -> str:
    base_name = (
        f"sched_T20_lr_{sanitize_float(LEARNING_RATE)}_ep_{NUM_EPOCHS}_{scheduler_config['token']}"
    )
    return base_name if not run_tag else f"{sanitize_token(run_tag)}_{base_name}"


def write_markdown(path: Path, rows: list[dict], best_run: dict | None) -> None:
    lines = [
        "# Scheduler Sweep",
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
        f"- learning_rate: `{LEARNING_RATE:.1e}`",
        f"- num_epochs: `{NUM_EPOCHS}`",
        f"- fixed rollout horizon: `T_eval={EVAL_TOTAL_TIME}`",
        "",
        "## Ranking Rule",
        "",
        "Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.",
        "",
        "## Runs",
        "",
        "| run_name | status | scheduler | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['scheduler_label']}` | "
                f"`{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['scheduler_label']}` | "
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
                f"- scheduler: `{best_run['scheduler_label']}`",
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
    seed = env_int("SRKPINN_SCHED_SEED", 2026)
    learning_rate = env_float("SRKPINN_SCHED_LR", LEARNING_RATE)
    num_epochs = env_int("SRKPINN_SCHED_EPOCHS", NUM_EPOCHS)
    log_freq = env_int("SRKPINN_SCHED_LOG_FREQ", 100)
    history_freq = env_int("SRKPINN_SCHED_HISTORY_FREQ", log_freq)
    checkpoint_freq = env_int("SRKPINN_SCHED_CHECKPOINT_FREQ", 1000)
    run_tag = env_str("SRKPINN_SCHED_RUN_TAG", "")
    summary_suffix = env_str("SRKPINN_SCHED_SUMMARY_SUFFIX", "")
    scheduler_configs = load_scheduler_configs()

    rows = []

    for scheduler_config in scheduler_configs:
        run_name = build_run_name(run_tag, scheduler_config)
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
                scheduler_name=scheduler_config["name"],
                scheduler_milestones=scheduler_config["milestones"],
                scheduler_gamma=scheduler_config["gamma"],
                summary_title="SRKPINN Scheduler Sweep Summary (Fixed Physical Horizon)",
            )
            rows.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "scheduler_name": scheduler_config["name"],
                    "scheduler_milestones": scheduler_config["milestones"],
                    "scheduler_gamma": scheduler_config["gamma"],
                    "scheduler_label": scheduler_config["label"],
                    "scheduler_token": scheduler_config["token"],
                    "scheduler_index": scheduler_config["index"],
                    "config": config,
                    "summary": summary,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "scheduler_name": scheduler_config["name"],
                    "scheduler_milestones": scheduler_config["milestones"],
                    "scheduler_gamma": scheduler_config["gamma"],
                    "scheduler_label": scheduler_config["label"],
                    "scheduler_token": scheduler_config["token"],
                    "scheduler_index": scheduler_config["index"],
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
        "eval_total_time": EVAL_TOTAL_TIME,
        "scheduler_configs": scheduler_configs,
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
