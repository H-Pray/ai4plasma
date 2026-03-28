"""Visualize the fixed-time SRKPINN pendulum loss balancing sweep results."""

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


def load_rows() -> tuple[list[str], list[dict], float]:
    payload = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["status"] == "completed"]
    labels = [f"({w['StageDynamics']}, {w['InitialOrData']})" for w in payload["loss_weights"]]
    return labels, rows, float(payload["eval_total_time"])


def format_scientific(value: float) -> str:
    return f"{value:.2e}"


def loss_tuple(row: dict) -> tuple[float, float]:
    return (float(row["loss_weights"]["StageDynamics"]), float(row["loss_weights"]["InitialOrData"]))


def create_metric_overview(rows: list[dict], labels: list[str], eval_total_time: float) -> None:
    best_row = min(
        rows,
        key=lambda row: (
            float(row["summary"]["rollout_state_error_final"]),
            float(row["summary"]["max_energy_drift"]),
            float(row["summary"]["train_one_step_rmse"]),
        ),
    )

    x = np.arange(len(rows))
    rollout = [float(row["summary"]["rollout_state_error_final"]) for row in rows]
    energy = [float(row["summary"]["max_energy_drift"]) for row in rows]
    rmse = [float(row["summary"]["train_one_step_rmse"]) for row in rows]
    sympl = [float(row["summary"]["symplectic_error"]) for row in rows]
    times = [float(row["summary"]["training_time_sec"]) for row in rows]
    bar_labels = [
        f"({row['loss_weights']['StageDynamics']}, {row['loss_weights']['InitialOrData']})"
        for row in rows
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    bars = ax.bar(x, rollout, color=["crimson" if row["run_name"] == best_row["run_name"] else "tab:blue" for row in rows])
    ax.set_yscale("log")
    ax.set_xticks(x, bar_labels, rotation=15)
    ax.set_title(f"Rollout Error @ T={eval_total_time:g}")
    ax.set_ylabel("rollout_state_error_final")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    for bar, value in zip(bars, rollout):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_scientific(value), ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    bars = ax.bar(x, energy, color=["crimson" if row["run_name"] == best_row["run_name"] else "tab:green" for row in rows])
    ax.set_yscale("log")
    ax.set_xticks(x, bar_labels, rotation=15)
    ax.set_title("Max Energy Drift")
    ax.set_ylabel("max_energy_drift")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    for bar, value in zip(bars, energy):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_scientific(value), ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, rmse, marker="o", linewidth=2, color="tab:purple", label="train_one_step_rmse")
    ax.plot(x, sympl, marker="s", linewidth=2, color="tab:orange", label="symplectic_error")
    ax.set_yscale("log")
    ax.set_xticks(x, bar_labels, rotation=15)
    ax.set_title("One-Step RMSE and Symplectic Error")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    ax = axes[1, 1]
    for row, t, r, e in zip(rows, times, rollout, energy):
        color = "crimson" if row["run_name"] == best_row["run_name"] else "tab:blue"
        label = f"({row['loss_weights']['StageDynamics']}, {row['loss_weights']['InitialOrData']})"
        ax.scatter(e, r, s=70, c=color, alpha=0.9)
        ax.annotate(label, (e, r), fontsize=8)
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
        "Training time per run:\n" + "\n".join(f"{label}: {t:.2f}s" for label, t in zip(bar_labels, times)),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.suptitle("SRKPINN Loss Balancing Sweep Comparison (Fixed Physical Horizon)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(METRIC_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_panel_montage(rows: list[dict]) -> None:
    images = []
    labels = []
    for row in rows:
        run_root = SWEEP_ROOT / row["run_name"] / "results" / "final_panels.png"
        image = Image.open(run_root).convert("RGB")
        images.append(image)
        labels.append(f"({row['loss_weights']['StageDynamics']}, {row['loss_weights']['InitialOrData']})")

    target_width = 520
    resized = []
    for image in images:
        ratio = target_width / image.width
        resized.append(image.resize((target_width, int(round(image.height * ratio))), Image.Resampling.LANCZOS))

    cols = 2
    rows_count = int(np.ceil(len(resized) / cols))
    title_height = 40
    cell_width = target_width
    cell_height = max(image.height for image in resized) + title_height
    canvas = Image.new("RGB", (cols * cell_width, rows_count * cell_height), color="white")

    for idx, image in enumerate(resized):
        row_id = idx // cols
        col_id = idx % cols
        x0 = col_id * cell_width
        y0 = row_id * cell_height
        canvas.paste(image, (x0, y0 + title_height))

    fig, ax = plt.subplots(figsize=(16, 14))
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

    fig.suptitle("SRKPINN Final Diagnostic Panels by Loss Weights (T=20)", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(MONTAGE_FIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    labels, rows, eval_total_time = load_rows()
    del labels
    create_metric_overview(rows, [], eval_total_time)
    create_panel_montage(rows)
    print(f"Wrote {METRIC_FIG}")
    print(f"Wrote {MONTAGE_FIG}")


if __name__ == "__main__":
    main()
