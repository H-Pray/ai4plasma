# Activation Sweep

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
- hidden_depth: `3`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | activation | layers | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `act_T20_lr_1em03_ep_6000_w_128_d_3_tanh` | `completed` | `Tanh` | `[2, 128, 128, 128, 6]` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.63` |
| `act_T20_lr_1em03_ep_6000_w_128_d_3_silu` | `completed` | `SiLU` | `[2, 128, 128, 128, 6]` | `100` | `20.00` | `4.236220e-04` | `5.743910e-02` | `5.743910e-02` | `1.459885e-02` | `1.459885e-02` | `1.357973e-03` | `23.02` |
| `act_T20_lr_1em03_ep_6000_w_128_d_3_gelu` | `completed` | `GELU` | `[2, 128, 128, 128, 6]` | `100` | `20.00` | `1.101870e-04` | `1.740040e-02` | `1.740040e-02` | `4.502773e-03` | `4.502773e-03` | `4.286170e-04` | `23.39` |
## Best Run

- run_name: `act_T20_lr_1em03_ep_6000_w_128_d_3_tanh`
- activation: `Tanh`
- layers: `[2, 128, 128, 128, 6]`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- train_one_step_rmse: `5.028848e-04`
- training_time_sec: `22.63`
