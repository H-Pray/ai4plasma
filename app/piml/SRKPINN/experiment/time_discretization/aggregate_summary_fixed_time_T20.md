# Time Discretization Sweep

## Ranking Rule

Primary key: `rollout_state_error_final` at fixed physical horizon `T_eval=20.0`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | dt | stages | method | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `time_T20_dt_0p05_stages_1` | `completed` | `0.05` | `1` | `implicit-midpoint` | `400` | `20.00` | `1.302531e-04` | `3.396456e-02` | `3.396456e-02` | `1.319385e-02` | `1.319385e-02` | `6.002188e-05` | `27.57` |
| `time_T20_dt_0p05_stages_2` | `completed` | `0.05` | `2` | `gauss-legendre` | `400` | `20.00` | `1.212551e-04` | `7.065535e-03` | `8.742251e-03` | `7.638931e-04` | `2.822399e-03` | `1.323223e-04` | `28.51` |
| `time_T20_dt_0p05_stages_3` | `completed` | `0.05` | `3` | `gauss-legendre` | `400` | `20.00` | `1.401090e-04` | `3.912516e-02` | `3.912516e-02` | `6.626725e-03` | `6.663918e-03` | `3.993511e-05` | `28.82` |
| `time_T20_dt_0p1_stages_1` | `completed` | `0.1` | `1` | `implicit-midpoint` | `200` | `20.00` | `2.310528e-04` | `8.702857e-02` | `8.702857e-02` | `1.702070e-02` | `1.794302e-02` | `9.006262e-05` | `23.74` |
| `time_T20_dt_0p1_stages_2` | `completed` | `0.1` | `2` | `gauss-legendre` | `200` | `20.00` | `2.460166e-04` | `2.127771e-02` | `2.127771e-02` | `2.501249e-03` | `3.078938e-03` | `1.908541e-04` | `24.29` |
| `time_T20_dt_0p1_stages_3` | `completed` | `0.1` | `3` | `gauss-legendre` | `200` | `20.00` | `2.687145e-04` | `6.799262e-03` | `8.960553e-03` | `1.523972e-03` | `2.448559e-03` | `1.263618e-05` | `24.58` |
| `time_T20_dt_0p2_stages_1` | `completed` | `0.2` | `1` | `implicit-midpoint` | `100` | `20.00` | `6.788900e-04` | `1.032801e-01` | `1.032801e-01` | `1.856148e-02` | `1.896358e-02` | `1.309514e-04` | `21.93` |
| `time_T20_dt_0p2_stages_2` | `completed` | `0.2` | `2` | `gauss-legendre` | `100` | `20.00` | `5.047846e-04` | `6.858048e-02` | `6.858048e-02` | `2.045274e-02` | `2.045274e-02` | `4.019737e-04` | `22.49` |
| `time_T20_dt_0p2_stages_3` | `completed` | `0.2` | `3` | `gauss-legendre` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.24` |
## Best Pair

- run_name: `time_T20_dt_0p2_stages_3`
- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- training_time_sec: `22.24`
