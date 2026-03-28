# Data Coverage Sweep

## Configuration

- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | train_data_size | sample_mode | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `data_T20_size_128_uniform` | `completed` | `128` | `uniform` | `100` | `20.00` | `5.246736e-04` | `1.580316e-02` | `1.580316e-02` | `5.020857e-03` | `6.056905e-03` | `1.213551e-04` | `19.27` |
| `data_T20_size_512_uniform` | `completed` | `512` | `uniform` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.43` |
| `data_T20_size_1024_uniform` | `completed` | `1024` | `uniform` | `100` | `20.00` | `2.633082e-04` | `2.407603e-02` | `2.407603e-02` | `6.265640e-03` | `6.265640e-03` | `3.892183e-05` | `27.29` |
| `data_T20_size_2048_uniform` | `completed` | `2048` | `uniform` | `100` | `20.00` | `7.981696e-04` | `2.794093e-01` | `2.794093e-01` | `6.925547e-02` | `6.925547e-02` | `2.400875e-04` | `36.50` |
| `data_T20_size_128_random` | `completed` | `128` | `random` | `100` | `20.00` | `5.158094e-04` | `2.765654e-02` | `2.765654e-02` | `6.191492e-03` | `6.191492e-03` | `3.324151e-04` | `19.77` |
| `data_T20_size_512_random` | `completed` | `512` | `random` | `100` | `20.00` | `5.714895e-04` | `1.484200e-02` | `1.484200e-02` | `1.396418e-03` | `1.742840e-03` | `1.543760e-04` | `22.71` |
| `data_T20_size_1024_random` | `completed` | `1024` | `random` | `100` | `20.00` | `5.235816e-04` | `3.300342e-02` | `3.300342e-02` | `6.686926e-03` | `6.686926e-03` | `1.775026e-04` | `27.90` |
| `data_T20_size_2048_random` | `completed` | `2048` | `random` | `100` | `20.00` | `1.112499e-03` | `2.030560e-01` | `2.030560e-01` | `5.871272e-02` | `5.871272e-02` | `6.459355e-04` | `42.34` |
## Best Pair

- run_name: `data_T20_size_512_uniform`
- train_data_size: `512`
- sample_mode: `uniform`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- training_time_sec: `22.43`
