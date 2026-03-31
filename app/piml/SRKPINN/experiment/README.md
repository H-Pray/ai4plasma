# SRKPINN Pendulum Experiment Workspace

This directory is the working area for the pendulum SRKPINN experiments.

## Layout

- `TODO.md`: ordered checklist for the sweep plan.
- `SWEEP_LOG.md`: run registry for baseline and later parameter scans.
- `baseline/`: isolated baseline run entrypoint and artifacts.
- `time_discretization/`: first sweep block over `dt` and `stages`.
- `data_coverage/`: second sweep block over `train_data_size` and `sample_mode`.
- `loss_balancing/`: third sweep block over loss weights.
- `optimization/`: fourth sweep block for learning rate and later optimizer settings.
- `network_capacity/`: fifth sweep block for hidden width, depth, and activation.

## Ground Rules

- Keep the baseline fixed to the current pendulum script configuration before scanning anything else.
- Write new experiment outputs under this directory instead of reusing `app/piml/SRKPINN/results`, `runs`, or `models`.
- Compare runs with the same evaluation protocol before deciding that one setting is better.
