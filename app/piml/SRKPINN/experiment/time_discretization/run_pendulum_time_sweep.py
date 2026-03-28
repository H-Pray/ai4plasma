"""First SRKPINN sweep block: pendulum time discretization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(".")

from app.piml.SRKPINN.experiment.runner import default_method_for_stages, run_pendulum_experiment


SWEEP_ROOT = Path(__file__).resolve().parent
EVAL_TOTAL_TIME = 20.0


def sanitize_float(value: float) -> str:
    return str(value).replace(".", "p")


def rank_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["summary"]["rollout_state_error_final"]),
        float(row["summary"]["max_energy_drift"]),
        float(row["summary"]["train_one_step_rmse"]),
    )


def write_markdown(path: Path, rows: list[dict], best_run: dict | None) -> None:
    lines = [
        "# Time Discretization Sweep",
        "",
        "## Ranking Rule",
        "",
        (
            "Primary key: `rollout_state_error_final` at fixed physical horizon "
            f"`T_eval={EVAL_TOTAL_TIME}`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`."
        ),
        "",
        "## Runs",
        "",
        "| run_name | status | dt | stages | method | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row["status"] == "completed":
            summary = row["summary"]
            lines.append(
                "| "
                f"`{row['run_name']}` | `completed` | `{row['dt']}` | `{row['stages']}` | `{row['method']}` | "
                f"`{summary['num_rollout_steps']}` | `{summary['rollout_total_time']:.2f}` | "
                f"`{summary['train_one_step_rmse']:.6e}` | `{summary['rollout_state_error_final']:.6e}` | "
                f"`{summary['max_rollout_state_error']:.6e}` | `{summary['energy_drift_final']:.6e}` | "
                f"`{summary['max_energy_drift']:.6e}` | `{summary['symplectic_error']:.6e}` | "
                f"`{summary['training_time_sec']:.2f}` |"
            )
        else:
            lines.append(
                f"| `{row['run_name']}` | `failed` | `{row['dt']}` | `{row['stages']}` | "
                f"`{row['method']}` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |"
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
                f"- dt: `{best_run['dt']}`",
                f"- stages: `{best_run['stages']}`",
                f"- method: `{best_run['method']}`",
                f"- rollout_steps: `{summary['num_rollout_steps']}`",
                f"- rollout_total_time: `{summary['rollout_total_time']:.2f}`",
                f"- rollout_state_error_final: `{summary['rollout_state_error_final']:.6e}`",
                f"- max_energy_drift: `{summary['max_energy_drift']:.6e}`",
                f"- training_time_sec: `{summary['training_time_sec']:.2f}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dt_values = [0.05, 0.1, 0.2]
    stage_values = [1, 2, 3]
    rows = []

    for dt in dt_values:
        for stages in stage_values:
            method = default_method_for_stages(stages)
            run_name = f"time_T20_dt_{sanitize_float(dt)}_stages_{stages}"
            run_root = SWEEP_ROOT / run_name

            try:
                config, summary = run_pendulum_experiment(
                    run_root=run_root,
                    run_name=run_name,
                    dt=dt,
                    stages=stages,
                    method=method,
                    rollout_total_time=EVAL_TOTAL_TIME,
                    summary_title="SRKPINN Time Sweep Summary (Fixed Physical Horizon)",
                )
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "completed",
                        "dt": dt,
                        "stages": stages,
                        "method": method,
                        "config": config,
                        "summary": summary,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "run_name": run_name,
                        "status": "failed",
                        "dt": dt,
                        "stages": stages,
                        "method": method,
                        "error": str(exc),
                    }
                )

    completed_rows = [row for row in rows if row["status"] == "completed"]
    best_run = min(completed_rows, key=rank_key) if completed_rows else None

    payload = {
        "dt_values": dt_values,
        "stage_values": stage_values,
        "eval_total_time": EVAL_TOTAL_TIME,
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
