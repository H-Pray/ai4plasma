# SRKPINN Pendulum Sweep Log

Update one row per finished run. Keep the metric definitions fixed.

| run_id | status | seed | dt | stages | method | train_data_size | sample_mode | loss_weights | layers | act | lr | epochs | scheduler | train_one_step_rmse | rollout_err_200 | max_rollout_err | energy_drift_200 | max_energy_drift | symplectic_err | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_v1` | `completed` | `2026` | `0.1` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `2.460166e-04` | `2.127771e-02` | `2.127771e-02` | `2.501249e-03` | `3.078938e-03` | `1.908541e-04` | `app/piml/SRKPINN/experiment/baseline/baseline_v1/` |
| `baseline_smoke` | `completed` | `2026` | `0.1` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `5` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `3.626901e-02` | `1.960639e+00` | `2.381527e+00` | `5.816154e-01` | `5.816154e-01` | `3.112137e-03` | `app/piml/SRKPINN/experiment/baseline/baseline_smoke/` |
| `time_dt_0p05_stages_1` | `completed` | `2026` | `0.05` | `1` | `implicit-midpoint` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 2]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `1.302531e-04` | `5.389350e-03` | `7.294237e-03` | `4.772902e-03` | `5.648017e-03` | `6.002188e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p05_stages_1/` |
| `time_dt_0p05_stages_2` | `completed` | `2026` | `0.05` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `1.212551e-04` | `4.602314e-03` | `5.177666e-03` | `9.444952e-04` | `2.403021e-03` | `1.323223e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p05_stages_2/` |
| `time_dt_0p05_stages_3` | `completed` | `2026` | `0.05` | `3` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 6]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `1.401090e-04` | `1.472667e-02` | `1.472667e-02` | `2.960443e-03` | `3.440738e-03` | `3.993511e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p05_stages_3/` |
| `time_dt_0p1_stages_1` | `completed` | `2026` | `0.1` | `1` | `implicit-midpoint` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 2]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `2.310528e-04` | `8.702857e-02` | `8.702857e-02` | `1.702070e-02` | `1.794302e-02` | `9.006262e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p1_stages_1/` |
| `time_dt_0p1_stages_2` | `completed` | `2026` | `0.1` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `2.460166e-04` | `2.127771e-02` | `2.127771e-02` | `2.501249e-03` | `3.078938e-03` | `1.908541e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p1_stages_2/` |
| `time_dt_0p1_stages_3` | `completed` | `2026` | `0.1` | `3` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 6]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `2.687145e-04` | `6.799262e-03` | `8.960553e-03` | `1.523972e-03` | `2.448559e-03` | `1.263618e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p1_stages_3/` |
| `time_dt_0p2_stages_1` | `completed` | `2026` | `0.2` | `1` | `implicit-midpoint` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 2]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `6.788900e-04` | `3.823207e-01` | `3.823207e-01` | `3.930855e-02` | `3.930855e-02` | `1.309514e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p2_stages_1/` |
| `time_dt_0p2_stages_2` | `completed` | `2026` | `0.2` | `2` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 4]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `5.047846e-04` | `2.928887e-01` | `2.928887e-01` | `4.249656e-02` | `4.249656e-02` | `4.019737e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p2_stages_2/` |
| `time_dt_0p2_stages_3` | `completed` | `2026` | `0.2` | `3` | `gauss-legendre` | `512` | `uniform` | `StageDynamics=1.0, InitialOrData=2.0` | `[2, 128, 128, 128, 6]` | `Tanh` | `1e-3` | `6000` | `MultiStepLR milestones=[2000,4000], gamma=0.5` | `5.028848e-04` | `2.035337e-02` | `2.060925e-02` | `6.219149e-03` | `6.219149e-03` | `4.625320e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_dt_0p2_stages_3/` |

## Notes

- Start by replacing the `baseline_v1` row with measured numbers once the run finishes.
- Add new rows instead of overwriting previous experiments.
- If a run fails, keep the row and mark the failure in `status`.
- Time-discretization ranking rule for the coarse sweep: compare `rollout_err_200` first, then `max_energy_drift`, then `train_one_step_rmse`.
- Best coarse discretization on seed `2026`: `dt=0.05`, `stages=2`, `method=gauss-legendre`.
- The fixed-`200`-step sweep below is retained for history, but it is not the fair cross-`dt` ranking because the physical horizon changes with `dt`.

## Fixed-Time Rerun (`T_eval = 20`)

Use this table for cross-`dt` comparisons after normalizing every rollout to the same physical horizon.

| run_id | status | seed | dt | stages | method | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_err | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `time_T20_dt_0p05_stages_1` | `completed` | `2026` | `0.05` | `1` | `implicit-midpoint` | `400` | `20.0` | `1.302531e-04` | `3.396456e-02` | `3.396456e-02` | `1.319385e-02` | `1.319385e-02` | `6.002188e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p05_stages_1/` |
| `time_T20_dt_0p05_stages_2` | `completed` | `2026` | `0.05` | `2` | `gauss-legendre` | `400` | `20.0` | `1.212551e-04` | `7.065535e-03` | `8.742251e-03` | `7.638931e-04` | `2.822399e-03` | `1.323223e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p05_stages_2/` |
| `time_T20_dt_0p05_stages_3` | `completed` | `2026` | `0.05` | `3` | `gauss-legendre` | `400` | `20.0` | `1.401090e-04` | `3.912516e-02` | `3.912516e-02` | `6.626725e-03` | `6.663918e-03` | `3.993511e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p05_stages_3/` |
| `time_T20_dt_0p1_stages_1` | `completed` | `2026` | `0.1` | `1` | `implicit-midpoint` | `200` | `20.0` | `2.310528e-04` | `8.702857e-02` | `8.702857e-02` | `1.702070e-02` | `1.794302e-02` | `9.006262e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p1_stages_1/` |
| `time_T20_dt_0p1_stages_2` | `completed` | `2026` | `0.1` | `2` | `gauss-legendre` | `200` | `20.0` | `2.460166e-04` | `2.127771e-02` | `2.127771e-02` | `2.501249e-03` | `3.078938e-03` | `1.908541e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p1_stages_2/` |
| `time_T20_dt_0p1_stages_3` | `completed` | `2026` | `0.1` | `3` | `gauss-legendre` | `200` | `20.0` | `2.687145e-04` | `6.799262e-03` | `8.960553e-03` | `1.523972e-03` | `2.448559e-03` | `1.263618e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p1_stages_3/` |
| `time_T20_dt_0p2_stages_1` | `completed` | `2026` | `0.2` | `1` | `implicit-midpoint` | `100` | `20.0` | `6.788900e-04` | `1.032801e-01` | `1.032801e-01` | `1.856148e-02` | `1.896358e-02` | `1.309514e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p2_stages_1/` |
| `time_T20_dt_0p2_stages_2` | `completed` | `2026` | `0.2` | `2` | `gauss-legendre` | `100` | `20.0` | `5.047846e-04` | `6.858048e-02` | `6.858048e-02` | `2.045274e-02` | `2.045274e-02` | `4.019737e-04` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p2_stages_2/` |
| `time_T20_dt_0p2_stages_3` | `completed` | `2026` | `0.2` | `3` | `gauss-legendre` | `100` | `20.0` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `app/piml/SRKPINN/experiment/time_discretization/time_T20_dt_0p2_stages_3/` |

### Fixed-Time Notes

- Fixed-time ranking rule: compare `rollout_err_final` first, then `max_energy_drift`, then `train_one_step_rmse`.
- Best coarse discretization at fixed `T_eval = 20` on seed `2026`: `dt=0.2`, `stages=3`, `method=gauss-legendre`.

## Data Coverage Sweep (`dt=0.2`, `stages=3`, `T_eval=20`)

Hold the best fixed-time discretization fixed while scanning training data coverage.

| run_id | status | seed | train_data_size | sample_mode | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_err | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data_T20_size_128_uniform` | `completed` | `2026` | `128` | `uniform` | `100` | `20.0` | `5.246736e-04` | `1.580316e-02` | `1.580316e-02` | `5.020857e-03` | `6.056905e-03` | `1.213551e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_128_uniform/` |
| `data_T20_size_512_uniform` | `completed` | `2026` | `512` | `uniform` | `100` | `20.0` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_512_uniform/` |
| `data_T20_size_1024_uniform` | `completed` | `2026` | `1024` | `uniform` | `100` | `20.0` | `2.633082e-04` | `2.407603e-02` | `2.407603e-02` | `6.265640e-03` | `6.265640e-03` | `3.892183e-05` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_1024_uniform/` |
| `data_T20_size_2048_uniform` | `completed` | `2026` | `2048` | `uniform` | `100` | `20.0` | `7.981696e-04` | `2.794093e-01` | `2.794093e-01` | `6.925547e-02` | `6.925547e-02` | `2.400875e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_2048_uniform/` |
| `data_T20_size_128_random` | `completed` | `2026` | `128` | `random` | `100` | `20.0` | `5.158094e-04` | `2.765654e-02` | `2.765654e-02` | `6.191492e-03` | `6.191492e-03` | `3.324151e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_128_random/` |
| `data_T20_size_512_random` | `completed` | `2026` | `512` | `random` | `100` | `20.0` | `5.714895e-04` | `1.484200e-02` | `1.484200e-02` | `1.396418e-03` | `1.742840e-03` | `1.543760e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_512_random/` |
| `data_T20_size_1024_random` | `completed` | `2026` | `1024` | `random` | `100` | `20.0` | `5.235816e-04` | `3.300342e-02` | `3.300342e-02` | `6.686926e-03` | `6.686926e-03` | `1.775026e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_1024_random/` |
| `data_T20_size_2048_random` | `completed` | `2026` | `2048` | `random` | `100` | `20.0` | `1.112499e-03` | `2.030560e-01` | `2.030560e-01` | `5.871272e-02` | `5.871272e-02` | `6.459355e-04` | `app/piml/SRKPINN/experiment/data_coverage/data_T20_size_2048_random/` |

### Data Coverage Notes

- Ranking rule: compare `rollout_err_final` first, then `max_energy_drift`, then `train_one_step_rmse`.
- Best coarse data-coverage setting on seed `2026`: `train_data_size=512`, `sample_mode=uniform`.
- `sample_mode=random` at `512` had better energy drift, but noticeably worse rollout error than `uniform`.
- Increasing to `1024` or `2048` did not help this setup and substantially increased runtime.

## Loss Balancing Sweep (`dt=0.2`, `stages=3`, `train_data_size=512`, `sample_mode=uniform`, `T_eval=20`)

Hold the best discretization and data-coverage settings fixed while scanning loss weights.

| run_id | status | seed | loss_weights | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_err | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `loss_T20_stage_1p0_data_1p0` | `completed` | `2026` | `StageDynamics=1.0, InitialOrData=1.0` | `100` | `20.0` | `4.987519e-04` | `6.875421e-03` | `8.519171e-03` | `5.283356e-04` | `1.774549e-03` | `7.951260e-05` | `app/piml/SRKPINN/experiment/loss_balancing/loss_T20_stage_1p0_data_1p0/` |
| `loss_T20_stage_1p0_data_2p0` | `completed` | `2026` | `StageDynamics=1.0, InitialOrData=2.0` | `100` | `20.0` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `app/piml/SRKPINN/experiment/loss_balancing/loss_T20_stage_1p0_data_2p0/` |
| `loss_T20_stage_1p0_data_5p0` | `completed` | `2026` | `StageDynamics=1.0, InitialOrData=5.0` | `100` | `20.0` | `5.732490e-04` | `4.169866e-02` | `4.169866e-02` | `1.002181e-02` | `1.349068e-02` | `1.990795e-05` | `app/piml/SRKPINN/experiment/loss_balancing/loss_T20_stage_1p0_data_5p0/` |
| `loss_T20_stage_0p5_data_2p0` | `completed` | `2026` | `StageDynamics=0.5, InitialOrData=2.0` | `100` | `20.0` | `5.482921e-04` | `9.407885e-02` | `9.407885e-02` | `2.183402e-02` | `2.183402e-02` | `1.378059e-04` | `app/piml/SRKPINN/experiment/loss_balancing/loss_T20_stage_0p5_data_2p0/` |

### Loss Balancing Notes

- Ranking rule: compare `rollout_err_final` first, then `max_energy_drift`, then `train_one_step_rmse`.
- Best coarse loss-weight setting on seed `2026`: `StageDynamics=1.0`, `InitialOrData=2.0`.
- `StageDynamics=1.0`, `InitialOrData=1.0` reduced energy drift, but worsened rollout error relative to the best pair.
- Overweighting data (`1.0/5.0`) or underweighting stage dynamics (`0.5/2.0`) both caused clear long-rollout regressions.
