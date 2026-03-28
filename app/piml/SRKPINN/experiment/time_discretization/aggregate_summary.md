# Time Discretization Sweep

## Ranking Rule

Primary key: `rollout_state_error_200`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | dt | stages | method | train_one_step_rmse | rollout_err_200 | max_rollout_err | energy_drift_200 | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `time_dt_0p05_stages_1` | `completed` | `0.05` | `1` | `implicit-midpoint` | `1.302531e-04` | `5.389350e-03` | `7.294237e-03` | `4.772902e-03` | `5.648017e-03` | `6.002188e-05` | `23.64` |
| `time_dt_0p05_stages_2` | `completed` | `0.05` | `2` | `gauss-legendre` | `1.212551e-04` | `4.602314e-03` | `5.177666e-03` | `9.444952e-04` | `2.403021e-03` | `1.323223e-04` | `24.23` |
| `time_dt_0p05_stages_3` | `completed` | `0.05` | `3` | `gauss-legendre` | `1.401090e-04` | `1.472667e-02` | `1.472667e-02` | `2.960443e-03` | `3.440738e-03` | `3.993511e-05` | `24.45` |
| `time_dt_0p1_stages_1` | `completed` | `0.1` | `1` | `implicit-midpoint` | `2.310528e-04` | `8.702857e-02` | `8.702857e-02` | `1.702070e-02` | `1.794302e-02` | `9.006262e-05` | `23.89` |
| `time_dt_0p1_stages_2` | `completed` | `0.1` | `2` | `gauss-legendre` | `2.460166e-04` | `2.127771e-02` | `2.127771e-02` | `2.501249e-03` | `3.078938e-03` | `1.908541e-04` | `24.62` |
| `time_dt_0p1_stages_3` | `completed` | `0.1` | `3` | `gauss-legendre` | `2.687145e-04` | `6.799262e-03` | `8.960553e-03` | `1.523972e-03` | `2.448559e-03` | `1.263618e-05` | `24.51` |
| `time_dt_0p2_stages_1` | `completed` | `0.2` | `1` | `implicit-midpoint` | `6.788900e-04` | `3.823207e-01` | `3.823207e-01` | `3.930855e-02` | `3.930855e-02` | `1.309514e-04` | `24.28` |
| `time_dt_0p2_stages_2` | `completed` | `0.2` | `2` | `gauss-legendre` | `5.047846e-04` | `2.928887e-01` | `2.928887e-01` | `4.249656e-02` | `4.249656e-02` | `4.019737e-04` | `24.73` |
| `time_dt_0p2_stages_3` | `completed` | `0.2` | `3` | `gauss-legendre` | `5.028848e-04` | `2.035337e-02` | `2.060925e-02` | `6.219149e-03` | `6.219149e-03` | `4.625320e-05` | `24.75` |
## Best Pair

- run_name: `time_dt_0p05_stages_2`
- dt: `0.05`
- stages: `2`
- method: `gauss-legendre`
- rollout_state_error_200: `4.602314e-03`
- max_energy_drift: `2.403021e-03`
- training_time_sec: `24.23`
