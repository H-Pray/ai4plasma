# Scheduler Sweep

## Fixed Configuration

- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- train_data_size: `512`
- sample_mode: `uniform`
- loss_weights: `StageDynamics=1.0`, `InitialOrData=2.0`
- learning_rate: `1.0e-03`
- num_epochs: `6000`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | scheduler | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sched_T20_lr_1em03_ep_6000_none` | `completed` | `none` | `100` | `20.00` | `1.593060e-03` | `3.448007e-01` | `3.448007e-01` | `1.108930e-01` | `1.108930e-01` | `1.848340e-04` | `22.82` |
| `sched_T20_lr_1em03_ep_6000_multisteplr_ms_2000_4000_g_0p5` | `completed` | `MultiStepLR milestones=[2000, 4000], gamma=0.5` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.43` |
| `sched_T20_lr_1em03_ep_6000_multisteplr_ms_3000_5000_g_0p5` | `completed` | `MultiStepLR milestones=[3000, 5000], gamma=0.5` | `100` | `20.00` | `4.531003e-04` | `3.548978e-03` | `5.375472e-03` | `7.886887e-04` | `2.259731e-03` | `3.159046e-05` | `22.78` |
| `sched_T20_lr_1em03_ep_6000_multisteplr_ms_1500_3000_4500_g_0p5` | `completed` | `MultiStepLR milestones=[1500, 3000, 4500], gamma=0.5` | `100` | `20.00` | `6.510011e-04` | `3.488601e-03` | `6.418322e-03` | `4.249096e-03` | `5.326271e-03` | `1.049042e-05` | `22.61` |
| `sched_T20_lr_1em03_ep_6000_multisteplr_ms_2000_4000_g_0p1` | `completed` | `MultiStepLR milestones=[2000, 4000], gamma=0.1` | `100` | `20.00` | `1.001879e-03` | `7.565309e-03` | `1.354829e-02` | `4.773140e-03` | `5.630016e-03` | `2.026558e-06` | `22.50` |
## Best Run

- run_name: `sched_T20_lr_1em03_ep_6000_multisteplr_ms_2000_4000_g_0p5`
- scheduler: `MultiStepLR milestones=[2000, 4000], gamma=0.5`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- train_one_step_rmse: `5.028848e-04`
- training_time_sec: `22.43`
