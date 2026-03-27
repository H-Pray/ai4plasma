# SRKPINN Pendulum Experiment Workspace

This directory is the working area for the pendulum SRKPINN experiments.

## Layout

- `TODO.md`: ordered checklist for the sweep plan.
- `SWEEP_LOG.md`: run registry for baseline and later parameter scans.
- `baseline/`: isolated baseline run entrypoint and artifacts.

## Ground Rules

- Keep the baseline fixed to the current pendulum script configuration before scanning anything else.
- Write new experiment outputs under this directory instead of reusing `app/piml/SRKPINN/results`, `runs`, or `models`.
- Compare runs with the same evaluation protocol before deciding that one setting is better.
