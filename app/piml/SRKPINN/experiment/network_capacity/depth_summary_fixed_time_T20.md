# Depth Sweep

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
- hidden_width: `128`
- activation: `Tanh`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | depth | layers | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `depth_T20_lr_1em03_ep_6000_w_128_d_3_tanh` | `completed` | `3` | `[2, 128, 128, 128, 6]` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.36` |
| `depth_T20_lr_1em03_ep_6000_w_128_d_4_tanh` | `completed` | `4` | `[2, 128, 128, 128, 128, 6]` | `100` | `20.00` | `5.678518e-04` | `9.351863e-02` | `9.351863e-02` | `2.440393e-02` | `2.440393e-02` | `1.866221e-04` | `24.46` |
| `depth_T20_lr_1em03_ep_6000_w_128_d_5_tanh` | `completed` | `5` | `[2, 128, 128, 128, 128, 128, 6]` | `100` | `20.00` | `4.309793e-04` | `4.551701e-02` | `4.551701e-02` | `1.366103e-02` | `1.366103e-02` | `1.940727e-04` | `26.56` |
## Best Run

- run_name: `depth_T20_lr_1em03_ep_6000_w_128_d_3_tanh`
- depth: `3`
- layers: `[2, 128, 128, 128, 6]`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- train_one_step_rmse: `5.028848e-04`
- training_time_sec: `22.36`
