# Pendulum Baseline

This folder contains the reproducible baseline for the SRKPINN pendulum experiment.

## Entry Point

```bash
python app/piml/SRKPINN/experiment/baseline/run_pendulum_baseline.py
```

## Default Baseline Configuration

- seed: `2026`
- `dt = 0.1`
- `stages = 2`
- method: `gauss-legendre`
- `train_data_size = 512`
- `sample_mode = uniform`
- loss weights: `StageDynamics = 1.0`, `InitialOrData = 2.0`
- layers: `[2, 128, 128, 128, 4]`
- activation: `Tanh`
- optimizer: `Adam(lr=1e-3)`
- scheduler: `MultiStepLR(milestones=[2000, 4000], gamma=0.5)`
- epochs: `6000`

## Environment Overrides

The script keeps the baseline defaults but allows a few overrides for smoke tests or controlled reruns:

- `SRKPINN_BASELINE_RUN_NAME`
- `SRKPINN_BASELINE_SEED`
- `SRKPINN_BASELINE_EPOCHS`
- `SRKPINN_BASELINE_LR`
- `SRKPINN_BASELINE_TRAIN_DATA_SIZE`
- `SRKPINN_BASELINE_DT`
- `SRKPINN_BASELINE_STAGES`
- `SRKPINN_BASELINE_METHOD`
- `SRKPINN_BASELINE_SAMPLE_MODE`
- `SRKPINN_BASELINE_MONITOR_STATE`
- `SRKPINN_BASELINE_ROLLOUT_STEPS`
- `SRKPINN_BASELINE_LOG_FREQ`
- `SRKPINN_BASELINE_HISTORY_FREQ`
- `SRKPINN_BASELINE_CHECKPOINT_FREQ`
- `SRKPINN_BASELINE_MILESTONES`
- `SRKPINN_BASELINE_GAMMA`

## Output Layout

Each run is written to its own subdirectory:

- `results/`: figures and summaries
- `runs/`: TensorBoard events
- `models/`: checkpoints and final model
