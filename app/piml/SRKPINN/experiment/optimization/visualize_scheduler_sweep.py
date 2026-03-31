"""Visualize the fixed-time SRKPINN pendulum scheduler sweep results."""

from __future__ import annotations

import json
import math
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
SUMMARY_SUFFIX = os.environ.get("SRKPINN_SCHED_SUMMARY_SUFFIX", "").strip().lower().replace(".", "p").replace("-", "m")
SUMMARY_STEM = f"scheduler_summary_fixed_time_T20_{SUMMARY_SUFFIX}" if SUMMARY_SUFFIX else "scheduler_summary_fixed_time_T20"
FIGURE_STEM = f"scheduler_fixed_time_T20_{SUMMARY_SUFFIX}" if SUMMARY_SUFFIX else "scheduler_fixed_time_T20"
SUMMARY_JSON = SWEEP_ROOT / f"{SUMMARY_STEM}.json"
METRIC_FIG = SWEEP_ROOT / f"comparison_metrics_{FIGURE_STEM}.png"
MONTAGE_FIG = SWEEP_ROOT / f"comparison_panels_{FIGURE_STEM}.png"


def row_sort_key(row: dict) -> tuple[int, str]:
    return (
        int(row.get("scheduler_index", 0)),
        str(row["run_name"]),
    )


def format_scientific(value: float) -> str:
    return f"{value:.2e}"


def load_rows() -> tuple[list[dict], float]:
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = sorted(
        [row for row in payload["rows"] if row["status"] == "completed"],
        key=row_sort_key,
    )
    return rows, float(payload["eval_total_time"])


def build_row_label(row: dict) -> str:
    if str(row["scheduler_name"]).lower() == "none":
        return "none"

    milestones = ",".join(str(item) for item in row["scheduler_milestones"])
    gamma = f"{float(row['scheduler_gamma']):g}"
    return f"{row['scheduler_name']}\nms=[{milestones}]\ng={gamma}"


def create_metric_overview(rows: list[dict], eval_total_time: float) -> None:
    if not rows:
        raise ValueError(f"No completed rows found in {SUMMARY_JSON}.")

    labels = [build_row_label(row) for row in rows]
    x = np.arange(len(rows))

    best_row = min(
        rows,
        key=lambda row: (
            float(row["summary"]["rollout_state_error_final"]),
            float(row["summary"]["max_energy_drift"]),
            float(row["summary"]["train_one_step_rmse"]),
        ),
    )
    best_colors = ["crimson" if row["run_name"] == best_row["run_name"] else "tab:blue" for row in rows]

    rollout = [float(row["summary"]["rollout_state_error_final"]) for row in rows]
    energy = [float(row["summary"]["max_energy_drift"]) for row in rows]
    rmse = [float(row["summary"]["train_one_step_rmse"]) for row in rows]
    sympl = [float(row["summary"]["symplectic_error"]) for row in rows]
    times = [float(row["summary"]["training_time_sec"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    bars = ax.bar(x, rollout, color=best_colors)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_title(f"Rollout Error @ T={eval_total_time:g}")
    ax.set_ylabel("rollout_state_error_final")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    for bar, value in zip(bars, rollout):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_scientific(value), ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    bars = ax.bar(x, energy, color=["crimson" if row["run_name"] == best_row["run_name"] else "tab:green" for row in rows])
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_title("Max Energy Drift")
    ax.set_ylabel("max_energy_drift")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    for bar, value in zip(bars, energy):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_scientific(value), ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, rmse, marker="o", linewidth=2, color="tab:purple", label="train_one_step_rmse")
    ax.plot(x, sympl, marker="s", linewidth=2, color="tab:orange", label="symplectic_error")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_title("One-Step RMSE and Symplectic Error")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    ax = axes[1, 1]
    for row, rollout_value, energy_value in zip(rows, rollout, energy):
        color = "crimson" if row["run_name"] == best_row["run_name"] else "tab:blue"
        label = build_row_label(row).replace("\n", ", ")
        ax.scatter(energy_value, rollout_value, s=70, c=color, alpha=0.9)
        ax.annotate(label, (energy_value, rollout_value), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max_energy_drift")
    ax.set_ylabel("rollout_state_error_final")
    ax.set_title("Tradeoff: Rollout Error vs Energy Drift")
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    ax.text(
        0.98,
        0.02,
        "Training time per run:\n" + "\n".join(f"{label.replace(chr(10), ', ')}: {t:.2f}s" for label, t in zip(labels, times)),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle("SRKPINN Scheduler Sweep Comparison (Fixed Physical Horizon)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(METRIC_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_panel_montage(rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No completed rows found in {SUMMARY_JSON}.")

    images = []
    labels = []
    for row in rows:
        run_root = SWEEP_ROOT / row["run_name"] / "results" / "final_panels.png"
        image = Image.open(run_root).convert("RGB")
        images.append(image)
        labels.append(build_row_label(row).replace("\n", ", "))

    target_width = 520
    resized = []
    for image in images:
        ratio = target_width / image.width
        resized.append(image.resize((target_width, int(round(image.height * ratio))), Image.Resampling.LANCZOS))

    cols = min(3, len(resized))
    rows_count = int(math.ceil(len(resized) / cols))
    title_height = 44
    cell_width = target_width
    cell_height = max(image.height for image in resized) + title_height
    canvas = Image.new("RGB", (cols * cell_width, rows_count * cell_height), color="white")

    for idx, image in enumerate(resized):
        row_id = idx // cols
        col_id = idx % cols
        x0 = col_id * cell_width
        y0 = row_id * cell_height
        canvas.paste(image, (x0, y0 + title_height))

    fig, ax = plt.subplots(figsize=(18, 6 * rows_count))
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
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    fig.suptitle("SRKPINN Final Diagnostic Panels by Scheduler Setting (T=20)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(MONTAGE_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, eval_total_time = load_rows()
    create_metric_overview(rows, eval_total_time)
    create_panel_montage(rows)
    print(f"Wrote {METRIC_FIG}")
    print(f"Wrote {MONTAGE_FIG}")


if __name__ == "__main__":
    main()
