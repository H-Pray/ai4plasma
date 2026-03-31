# Width Sweep

## Fixed Configuration

- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- train_data_size: `512`
- sample_mode: `uniform`
- loss_weights: `StageDynamics=1.0`, `InitialOrData=2.0`
- learning_rate: `1.0e-03`
- num_epochs: `6000`
- scheduler: `MultiStepLR milestones=[2000, 4000], gamma=0.5`
- hidden_depth: `3`
- activation: `Tanh`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | width | layers | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `width_T20_lr_1em03_ep_6000_w_64_d_3_tanh` | `completed` | `64` | `[2, 64, 64, 64, 6]` | `100` | `20.00` | `3.208517e-04` | `1.993498e-02` | `1.993498e-02` | `5.625248e-03` | `5.625248e-03` | `1.501441e-04` | `20.42` |
| `width_T20_lr_1em03_ep_6000_w_128_d_3_tanh` | `completed` | `128` | `[2, 128, 128, 128, 6]` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.08` |
| `width_T20_lr_1em03_ep_6000_w_256_d_3_tanh` | `completed` | `256` | `[2, 256, 256, 256, 6]` | `100` | `20.00` | `5.090443e-04` | `7.593290e-03` | `8.909126e-03` | `1.433611e-03` | `1.433611e-03` | `1.205206e-04` | `27.56` |
## Best Run

- run_name: `width_T20_lr_1em03_ep_6000_w_128_d_3_tanh`
- width: `128`
- layers: `[2, 128, 128, 128, 6]`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- train_one_step_rmse: `5.028848e-04`
- training_time_sec: `22.08`
