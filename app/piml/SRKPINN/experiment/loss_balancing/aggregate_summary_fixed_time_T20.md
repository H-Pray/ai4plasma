# Loss Balancing Sweep

## Configuration

- dt: `0.2`
- stages: `3`
- method: `gauss-legendre`
- train_data_size: `512`
- sample_mode: `uniform`
- fixed rollout horizon: `T_eval=20.0`

## Ranking Rule

Primary key: `rollout_state_error_final`; tie-breakers: `max_energy_drift`, `train_one_step_rmse`.

## Runs

| run_name | status | loss_weights | rollout_steps | rollout_time | train_one_step_rmse | rollout_err_final | max_rollout_err | energy_drift_final | max_energy_drift | symplectic_error | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `loss_T20_stage_1p0_data_1p0` | `completed` | `StageDynamics=1.0, InitialOrData=1.0` | `100` | `20.00` | `4.987519e-04` | `6.875421e-03` | `8.519171e-03` | `5.283356e-04` | `1.774549e-03` | `7.951260e-05` | `22.69` |
| `loss_T20_stage_1p0_data_2p0` | `completed` | `StageDynamics=1.0, InitialOrData=2.0` | `100` | `20.00` | `5.028848e-04` | `2.214470e-03` | `4.014834e-03` | `2.409577e-03` | `3.867388e-03` | `4.625320e-05` | `22.11` |
| `loss_T20_stage_1p0_data_5p0` | `completed` | `StageDynamics=1.0, InitialOrData=5.0` | `100` | `20.00` | `5.732490e-04` | `4.169866e-02` | `4.169866e-02` | `1.002181e-02` | `1.349068e-02` | `1.990795e-05` | `22.27` |
| `loss_T20_stage_0p5_data_2p0` | `completed` | `StageDynamics=0.5, InitialOrData=2.0` | `100` | `20.00` | `5.482921e-04` | `9.407885e-02` | `9.407885e-02` | `2.183402e-02` | `2.183402e-02` | `1.378059e-04` | `22.32` |
## Best Pair

- run_name: `loss_T20_stage_1p0_data_2p0`
- loss_weights: `StageDynamics=1.0, InitialOrData=2.0`
- rollout_steps: `100`
- rollout_total_time: `20.00`
- rollout_state_error_final: `2.214470e-03`
- max_energy_drift: `3.867388e-03`
- training_time_sec: `22.11`
