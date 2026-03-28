"""Third SRKPINN sweep block: loss balancing."""

from __future__ import annotations

import json
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
LOSS_WEIGHTS = [
    {"StageDynamics": 1.0, "InitialOrData": 1.0},
    {"StageDynamics": 1.0, "InitialOrData": 2.0},
    {"StageDynamics": 1.0, "InitialOrData": 5.0},
    {"StageDynamics": 0.5, "InitialOrData": 2.0},
]


def weight_run_name(weights: dict[str, float]) -> str:
    stage = str(weights["StageDynamics"]).replace(".", "p")
    data = str(weights["InitialOrData"]).replace(".", "p")
    return f"loss_T20_stage_{stage}_data_{data}"


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def weight_label(weights: dict[str, float]) -> str:
    return f"StageDynamics={weights['StageDynamics']}, InitialOrData={weights['InitialOrData']}"


def write_markdown(path: Path, rows: list[dict], best_run: dict | None) -> None:
    lines = [
        "# Loss Balancing Sweep",
        "",
        "## Configuration",
        "",
        f"- dt: `{DT}`",
        f"- stages: `{STAGES}`",
        f"- method: `{METHOD}`",
        f"- train_data_size: `{TRAIN_DATA_SIZE}`",
        f"- sample_mode: `{SAMPLE_MODE}`",
        f"- fixed rollout horizon: `T_eval={EVAL_TOTAL_TIME}`",
        "",
        "## Ranking Rule",
        "",
        "Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.",
        "",
        "## Runs",
        "",
        "| run_name | status | loss_weights | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['loss_weights_label']}` | "
                f"`{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['loss_weights_label']}` | "
                "`-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |"
            )
            lines.append("")
            lines.append(f"Failure detail for `{row['run_name']}`: `{row['error']}`")
            lines.append("")

    if best_run is not None:
        summary = best_run["summary"]
        lines.extend(
            [
                "## Best Pair",
                "",
                f"- run_name: `{best_run['run_name']}`",
                f"- loss_weights: `{best_run['loss_weights_label']}`",
                f"- rollout_steps: `{summary['num_rollout_steps']}`",
                f"- rollout_total_time: `{summary['rollout_total_time']:.2f}`",
                f"- rollout_state_error_final: `{summary['rollout_state_error_final']:.6e}`",
                f"- max_energy_drift: `{summary['max_energy_drift']:.6e}`",
                f"- training_time_sec: `{summary['training_time_sec']:.2f}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = []

    for loss_weights in LOSS_WEIGHTS:
        run_name = weight_run_name(loss_weights)
        run_root = SWEEP_ROOT / run_name
        label = weight_label(loss_weights)

        try:
            config, summary = run_pendulum_experiment(
                run_root=run_root,
                run_name=run_name,
                dt=DT,
                stages=STAGES,
                method=METHOD,
                train_data_size=TRAIN_DATA_SIZE,
                sample_mode=SAMPLE_MODE,
                loss_weights=loss_weights,
                rollout_total_time=EVAL_TOTAL_TIME,
                summary_title="SRKPINN Loss Balancing Sweep Summary (Fixed Physical Horizon)",
            )
            rows.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "loss_weights": loss_weights,
                    "loss_weights_label": label,
                    "config": config,
                    "summary": summary,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "loss_weights": loss_weights,
                    "loss_weights_label": label,
                    "error": str(exc),
                }
            )

    completed_rows = [row for row in rows if row["status"] == "completed"]
    best_run = min(completed_rows, key=rank_key) if completed_rows else None

    payload = {
        "dt": DT,
        "stages": STAGES,
        "method": METHOD,
        "train_data_size": TRAIN_DATA_SIZE,
        "sample_mode": SAMPLE_MODE,
        "eval_total_time": EVAL_TOTAL_TIME,
        "loss_weights": LOSS_WEIGHTS,
        "ranking_rule": [
            "rollout_state_error_final",
            "max_energy_drift",
            "train_one_step_rmse",
        ],
        "best_run": best_run["run_name"] if best_run else None,
        "rows": rows,
    }
    (SWEEP_ROOT / "aggregate_summary_fixed_time_T20.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(SWEEP_ROOT / "aggregate_summary_fixed_time_T20.md", rows, best_run)


if __name__ == "__main__":
    main()
