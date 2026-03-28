"""Second SRKPINN sweep block: training data coverage."""

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
TRAIN_DATA_SIZES = [128, 512, 1024, 2048]
SAMPLE_MODES = ["uniform", "random"]


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def write_markdown(path: Path, rows: list[dict], best_run: dict | None) -> None:
    lines = [
        "# Data Coverage Sweep",
        "",
        "## Configuration",
        "",
        f"- dt: `{DT}`",
        f"- stages: `{STAGES}`",
        f"- method: `{METHOD}`",
        f"- fixed rollout horizon: `T_eval={EVAL_TOTAL_TIME}`",
        "",
        "## Ranking Rule",
        "",
        "Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.",
        "",
        "## Runs",
        "",
        "| run_name | status | train_data_size | sample_mode | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['train_data_size']}` | `{row['sample_mode']}` | "
                f"`{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['train_data_size']}` | `{row['sample_mode']}` | "
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
                f"- train_data_size: `{best_run['train_data_size']}`",
                f"- sample_mode: `{best_run['sample_mode']}`",
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

    for sample_mode in SAMPLE_MODES:
        for train_data_size in TRAIN_DATA_SIZES:
            run_name = f"data_T20_size_{train_data_size}_{sample_mode}"
            run_root = SWEEP_ROOT / run_name

            try:
                config, summary = run_pendulum_experiment(
                    run_root=run_root,
                    run_name=run_name,
                    dt=DT,
                    stages=STAGES,
                    method=METHOD,
                    train_data_size=train_data_size,
                    sample_mode=sample_mode,
                    rollout_total_time=EVAL_TOTAL_TIME,
                    summary_title="SRKPINN Data Coverage Sweep Summary (Fixed Physical Horizon)",
                )
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "completed",
                        "train_data_size": train_data_size,
                        "sample_mode": sample_mode,
                        "config": config,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "failed",
                        "train_data_size": train_data_size,
                        "sample_mode": sample_mode,
                        "error": str(exc),
                    }
                )

    completed_rows = [row for row in rows if row["status"] == "completed"]
    best_run = min(completed_rows, key=rank_key) if completed_rows else None

    payload = {
        "dt": DT,
        "stages": STAGES,
        "method": METHOD,
        "eval_total_time": EVAL_TOTAL_TIME,
        "train_data_sizes": TRAIN_DATA_SIZES,
        "sample_modes": SAMPLE_MODES,
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
