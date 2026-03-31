# SRKPINN Pendulum Sweep Summary

## Today's Outcome

- Reviewed the prior memory, sweep checklist, and run registry before continuing.
- Completed the first optimization sweep over `learning_rate` with the fixed best settings from the first three sweep blocks.
- Completed the first scheduler sweep with the best learning rate fixed at `1e-3`.
- Completed the first network-capacity pass over width, depth, and activation.
- Generated aggregate optimization artifacts under `app/piml/SRKPINN/experiment/optimization/`.
- Generated aggregate network-capacity artifacts under `app/piml/SRKPINN/experiment/network_capacity/`.
- Updated the active checklist and sweep log with the finished optimization and network-capacity results.

## Optimization Sweep Configuration

- `dt = 0.2`
- `stages = 3`
- method: `gauss-legendre`
- `train_data_size = 512`
- `sample_mode = uniform`
- loss weights:
  `StageDynamics = 1.0`, `InitialOrData = 2.0`
- evaluation horizon: `T_eval = 20`
- scheduler: `MultiStepLR milestones=[2000, 4000], gamma=0.5`
- scanned learning rates: `3e-4`, `1e-3`, `3e-3`
- epochs per run: `6000`

## Optimization Results

| learning_rate | train_one_step_rmse | rollout_err_final | max_energy_drift | training_time_sec | verdict |
| --- | --- | --- | --- | --- | --- |
| `3e-4` | `1.002758e-03` | `4.904904e-03` | `7.202148e-03` | `22.55` | smooth but worse than `1e-3` |
| `1e-3` | `5.028848e-04` | `2.214470e-03` | `3.867388e-03` | `22.59` | best |
| `3e-3` | `2.429325e-03` | `5.129830e-01` | `1.353902e-01` | `23.24` | unstable |

## Scheduler Results

| scheduler | train_one_step_rmse | rollout_err_final | max_energy_drift | training_time_sec | verdict |
| --- | --- | --- | --- | --- | --- |
| `none` | `1.593060e-03` | `3.448007e-01` | `1.108930e-01` | `22.82` | unusable |
| `MultiStepLR [2000,4000], gamma=0.5` | `5.028848e-04` | `2.214470e-03` | `3.867388e-03` | `22.43` | best |
| `MultiStepLR [3000,5000], gamma=0.5` | `4.531003e-04` | `3.548978e-03` | `2.259731e-03` | `22.78` | better drift, worse rollout |
| `MultiStepLR [1500,3000,4500], gamma=0.5` | `6.510011e-04` | `3.488601e-03` | `5.326271e-03` | `22.61` | worse |
| `MultiStepLR [2000,4000], gamma=0.1` | `1.001879e-03` | `7.565309e-03` | `5.630016e-03` | `22.50` | over-decayed |

## Network-Capacity Results

### Width Sweep

| width | train_one_step_rmse | rollout_err_final | max_energy_drift | training_time_sec | verdict |
| --- | --- | --- | --- | --- | --- |
| `64` | `3.208517e-04` | `1.993498e-02` | `5.625248e-03` | `20.42` | underfit long rollout |
| `128` | `5.028848e-04` | `2.214470e-03` | `3.867388e-03` | `22.08` | best |
| `256` | `5.090443e-04` | `7.593290e-03` | `1.433611e-03` | `27.56` | better drift, worse rollout |

### Depth Sweep

| depth | train_one_step_rmse | rollout_err_final | max_energy_drift | training_time_sec | verdict |
| --- | --- | --- | --- | --- | --- |
| `3` | `5.028848e-04` | `2.214470e-03` | `3.867388e-03` | `22.36` | best |
| `4` | `5.678518e-04` | `9.351863e-02` | `2.440393e-02` | `24.46` | unstable rollout |
| `5` | `4.309793e-04` | `4.551701e-02` | `1.366103e-02` | `26.56` | unstable rollout |

### Activation Sweep

| activation | train_one_step_rmse | rollout_err_final | max_energy_drift | training_time_sec | verdict |
| --- | --- | --- | --- | --- | --- |
| `Tanh` | `5.028848e-04` | `2.214470e-03` | `3.867388e-03` | `22.63` | best |
| `SiLU` | `4.236220e-04` | `5.743910e-02` | `1.459885e-02` | `23.02` | worse rollout |
| `GELU` | `1.101870e-04` | `1.740040e-02` | `4.502773e-03` | `23.39` | better one-step, worse rollout |

## Key Findings

- `1e-3` remains the best learning rate by the ranking rule:
  `rollout_err_final`, then `max_energy_drift`, then `train_one_step_rmse`.
- `3e-4` trains cleanly, but the final rollout error and energy drift are both worse than `1e-3`, so extending epochs is not yet justified.
- `3e-3` is far too aggressive for this fixed configuration and should be dropped from follow-up sweeps.
- `MultiStepLR milestones=[2000,4000], gamma=0.5` remains the best scheduler by the same ranking rule.
- `none` causes a severe long-rollout regression even though the training loss still decreases.
- The later schedule `milestones=[3000,5000], gamma=0.5` is the only credible alternative, but it still loses on the primary rollout metric.
- In the full first-pass network-capacity sweep, none of the architecture variants beat the original `[2, 128, 128, 128, 6]` with `Tanh`.
- Smaller, deeper, and smoother-activation models often improved one-step RMSE or energy drift, but those gains did not carry over to the primary long-rollout metric.
- The current best overall setting therefore remains unchanged from the pre-capacity baseline:
  width `128`, depth `3`, activation `Tanh`, `lr=1e-3`, `MultiStepLR milestones=[2000,4000], gamma=0.5`.

## Artifacts

- Optimization aggregate markdown:
  `app/piml/SRKPINN/experiment/optimization/aggregate_summary_fixed_time_T20.md`
- Optimization aggregate JSON:
  `app/piml/SRKPINN/experiment/optimization/aggregate_summary_fixed_time_T20.json`
- Optimization metric overview:
  `app/piml/SRKPINN/experiment/optimization/comparison_metrics_fixed_time_T20.png`
- Optimization panel montage:
  `app/piml/SRKPINN/experiment/optimization/comparison_panels_fixed_time_T20.png`
- Scheduler aggregate markdown:
  `app/piml/SRKPINN/experiment/optimization/scheduler_summary_fixed_time_T20.md`
- Scheduler aggregate JSON:
  `app/piml/SRKPINN/experiment/optimization/scheduler_summary_fixed_time_T20.json`
- Scheduler metric overview:
  `app/piml/SRKPINN/experiment/optimization/comparison_metrics_scheduler_fixed_time_T20.png`
- Scheduler panel montage:
  `app/piml/SRKPINN/experiment/optimization/comparison_panels_scheduler_fixed_time_T20.png`
- Width aggregate markdown:
  `app/piml/SRKPINN/experiment/network_capacity/width_summary_fixed_time_T20.md`
- Width aggregate JSON:
  `app/piml/SRKPINN/experiment/network_capacity/width_summary_fixed_time_T20.json`
- Width metric overview:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_metrics_width_fixed_time_T20.png`
- Width panel montage:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_panels_width_fixed_time_T20.png`
- Depth aggregate markdown:
  `app/piml/SRKPINN/experiment/network_capacity/depth_summary_fixed_time_T20.md`
- Depth aggregate JSON:
  `app/piml/SRKPINN/experiment/network_capacity/depth_summary_fixed_time_T20.json`
- Depth metric overview:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_metrics_depth_fixed_time_T20.png`
- Depth panel montage:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_panels_depth_fixed_time_T20.png`
- Activation aggregate markdown:
  `app/piml/SRKPINN/experiment/network_capacity/activation_summary_fixed_time_T20.md`
- Activation aggregate JSON:
  `app/piml/SRKPINN/experiment/network_capacity/activation_summary_fixed_time_T20.json`
- Activation metric overview:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_metrics_activation_fixed_time_T20.png`
- Activation panel montage:
  `app/piml/SRKPINN/experiment/network_capacity/comparison_panels_activation_fixed_time_T20.png`

## Recommended Next Step

- Freeze the current best optimization setting:
  `learning_rate = 1e-3`, `MultiStepLR milestones=[2000,4000], gamma=0.5`, width `128`, depth `3`, activation `Tanh`.
- Move to validation-oriented follow-up work instead of more coarse sweeps:
  several initial states, multiple random seeds, and longer rollout evaluation for the top 2 or 3 candidates.
- Only revisit longer training (`10000` or `15000` epochs) if one of those validation passes exposes a credible alternate candidate.
