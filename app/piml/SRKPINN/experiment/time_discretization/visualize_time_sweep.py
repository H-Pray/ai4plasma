"""Visualize the fixed-time SRKPINN pendulum sweep results."""

from __future__ import annotations

import json
import os
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = RUNNER_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image


SWEEP_ROOT = Path(__file__).resolve().parent
SUMMARY_JSON = SWEEP_ROOT / "aggregate_summary_fixed_time_T20.json"
METRIC_FIG = SWEEP_ROOT / "comparison_metrics_fixed_time_T20.png"
MONTAGE_FIG = SWEEP_ROOT / "comparison_panels_fixed_time_T20.png"


def load_rows() -> tuple[list[float], list[int], list[dict], float]:
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["status"] == "completed"]
    dt_values = sorted({float(row["dt"]) for row in rows})
    stage_values = sorted({int(row["stages"]) for row in rows})
    return dt_values, stage_values, rows, float(payload["eval_total_time"])


def build_metric_matrix(rows: list[dict], dt_values: list[float], stage_values: list[int], key: str) -> np.ndarray:
    matrix = np.full((len(dt_values), len(stage_values)), np.nan, dtype=float)
    for row in rows:
        i = dt_values.index(float(row["dt"]))
        j = stage_values.index(int(row["stages"]))
        matrix[i, j] = float(row["summary"][key])
    return matrix


def format_scientific(value: float) -> str:
    return f"{value:.2e}"


def plot_heatmap(ax, matrix: np.ndarray, dt_values: list[float], stage_values: list[int], title: str, best_idx=None):
    log_matrix = np.log10(matrix)
    image = ax.imshow(log_matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(stage_values)), [str(stage) for stage in stage_values])
    ax.set_yticks(range(len(dt_values)), [str(dt) for dt in dt_values])
    ax.set_xlabel("stages")
    ax.set_ylabel("dt")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = format_scientific(matrix[i, j])
            color = "black" if log_matrix[i, j] > np.nanmean(log_matrix) else "white"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

    if best_idx is not None:
        rect = plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="crimson",
            linewidth=3,
        )
        ax.add_patch(rect)

    cbar = plt.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("log10(value)")


def create_metric_overview(rows: list[dict], dt_values: list[float], stage_values: list[int], eval_total_time: float) -> None:
    rollout_matrix = build_metric_matrix(rows, dt_values, stage_values, "rollout_state_error_final")
    energy_matrix = build_metric_matrix(rows, dt_values, stage_values, "max_energy_drift")
    rmse_matrix = build_metric_matrix(rows, dt_values, stage_values, "train_one_step_rmse")
    time_matrix = build_metric_matrix(rows, dt_values, stage_values, "training_time_sec")

    best_row = min(
        rows,
        key=lambda row: (
            float(row["summary"]["rollout_state_error_final"]),
            float(row["summary"]["max_energy_drift"]),
            float(row["summary"]["train_one_step_rmse"]),
        ),
    )
    best_idx = (
        dt_values.index(float(best_row["dt"])),
        stage_values.index(int(best_row["stages"])),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_heatmap(
        axes[0, 0],
        rollout_matrix,
        dt_values,
        stage_values,
        f"Rollout Error @ T={eval_total_time:g}",
        best_idx=best_idx,
    )
    plot_heatmap(
        axes[0, 1],
        energy_matrix,
        dt_values,
        stage_values,
        "Max Energy Drift",
        best_idx=best_idx,
    )
    plot_heatmap(
        axes[1, 0],
        rmse_matrix,
        dt_values,
        stage_values,
        "Train One-Step RMSE",
        best_idx=best_idx,
    )

    scatter = axes[1, 1]
    for row in rows:
        summary = row["summary"]
        label = f"dt={row['dt']}, s={row['stages']}"
        color = "crimson" if row["run_name"] == best_row["run_name"] else "tab:blue"
        scatter.scatter(
            float(summary["max_energy_drift"]),
            float(summary["rollout_state_error_final"]),
            s=60,
            c=color,
            alpha=0.9,
        )
        scatter.annotate(label, (float(summary["max_energy_drift"]), float(summary["rollout_state_error_final"])), fontsize=8)
    scatter.set_xscale("log")
    scatter.set_yscale("log")
    scatter.set_xlabel("max_energy_drift")
    scatter.set_ylabel("rollout_state_error_final")
    scatter.set_title("Tradeoff: Rollout Error vs Energy Drift")
    scatter.grid(True, alpha=0.3, which="both")
    scatter.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    scatter.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))

    time_text = "\n".join(
        f"dt={dt}, stages={stage}: {time_matrix[i, j]:.2f}s"
        for i, dt in enumerate(dt_values)
        for j, stage in enumerate(stage_values)
    )
    scatter.text(
        0.98,
        0.02,
        f"Best: {best_row['run_name']}\n\nTraining time per run:\n{time_text}",
        transform=scatter.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle("SRKPINN dt/stages Sweep Comparison (Fixed Physical Horizon)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(METRIC_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_panel_montage(rows: list[dict], dt_values: list[float], stage_values: list[int]) -> None:
    rows_sorted = sorted(rows, key=lambda row: (float(row["dt"]), int(row["stages"])))
    images = []
    labels = []
    for row in rows_sorted:
        run_root = SWEEP_ROOT / row["run_name"] / "results" / "final_panels.png"
        image = Image.open(run_root).convert("RGB")
        images.append(image)
        labels.append(f"dt={row['dt']}, stages={row['stages']}")

    target_width = 460
    resized = []
    for image in images:
        ratio = target_width / image.width
        resized.append(image.resize((target_width, int(round(image.height * ratio))), Image.Resampling.LANCZOS))

    cols = len(stage_values)
    rows_count = len(dt_values)
    title_height = 38
    cell_width = target_width
    cell_height = max(image.height for image in resized) + title_height
    canvas = Image.new("RGB", (cols * cell_width, rows_count * cell_height), color="white")

    for idx, image in enumerate(resized):
        row_id = idx // cols
        col_id = idx % cols
        x0 = col_id * cell_width
        y0 = row_id * cell_height
        canvas.paste(image, (x0, y0 + title_height))

    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(canvas)
    ax.axis("off")

    for idx, label in enumerate(labels):
        row_id = idx // cols
        col_id = idx % cols
        x_center = col_id * cell_width + cell_width / 2
        y_text = row_id * cell_height + title_height / 2
        ax.text(
            x_center,
            y_text,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
        )

    fig.suptitle("SRKPINN Final Diagnostic Panels by dt/stages (T=20)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(MONTAGE_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dt_values, stage_values, rows, eval_total_time = load_rows()
    create_metric_overview(rows, dt_values, stage_values, eval_total_time)
    create_panel_montage(rows, dt_values, stage_values)
    print(f"Wrote {METRIC_FIG}")
    print(f"Wrote {MONTAGE_FIG}")


if __name__ == "__main__":
    main()
