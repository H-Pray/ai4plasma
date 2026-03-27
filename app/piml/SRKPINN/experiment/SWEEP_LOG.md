# SRKPINN Pendulum Sweep Log

Update one row per finished run. Keep the metric definitions fixed.

| run_id | status | seed | dt | stages | method | train_data_size | sample_mode | loss_weights | layers | act | lr | epochs | scheduler | train_one_step_rmse | rollout_err_200 | max_rollout_err | energy_drift_200 | max_energy_drift | symplectic_err | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_v1` | `prepared` | `2026` | `0.1` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `app/piml/SRKPINN/experiment/baseline/baseline_v1/` |
| `baseline_smoke` | `completed` | `2026` | `0.1` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `5` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `3.626901e-02` | `1.960639e+00` | `2.381527e+00` | `5.816154e-01` | `5.816154e-01` | `3.112137e-03` | `app/piml/SRKPINN/experiment/baseline/baseline_smoke/` |

## Notes

- Start by replacing the `baseline_v1` row with measured numbers once the run finishes.
- Add new rows instead of overwriting previous experiments.
- If a run fails, keep the row and mark the failure in `status`.
