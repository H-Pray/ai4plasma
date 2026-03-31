# Optimization Sweep

## Fixed Configuration

- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- train_data_size: `512`
- sample_mode: `uniform`
- loss_weights: `StageDynamics=1.0`, `InitialOrData=2.0`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | learning_rate | num_epochs | scheduler | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `opt_T20_lr_3em04_ep_6000_multisteplr` | `completed` | `3.0e-04` | `6000` | `MultiStepLR milestones=[2000, 4000], gamma=0.5` | `100` | `20.00` | `1.002758e-03` | `4.904904e-03` | `1.136551e-02` | `5.658269e-03` | `7.202148e-03` | `1.558065e-04` | `22.55` |
| `opt_T20_lr_1em03_ep_6000_multisteplr` | `completed` | `1.0e-03` | `6000` | `MultiStepLR milestones=[2000, 4000], gamma=0.5` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.59` |
| `opt_T20_lr_3em03_ep_6000_multisteplr` | `completed` | `3.0e-03` | `6000` | `MultiStepLR milestones=[2000, 4000], gamma=0.5` | `100` | `20.00` | `2.429325e-03` | `5.129830e-01` | `5.129830e-01` | `1.353902e-01` | `1.353902e-01` | `2.056360e-04` | `23.24` |
## Best Run

- run_name: `opt_T20_lr_1em03_ep_6000_multisteplr`
- learning_rate: `1.0e-03`
- num_epochs: `6000`
- scheduler: `MultiStepLR milestones=[2000, 4000], gamma=0.5`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- training_time_sec: `22.59`
