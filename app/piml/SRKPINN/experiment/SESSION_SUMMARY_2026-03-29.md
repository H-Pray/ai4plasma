# SRKPINN Pendulum Sweep Summary

## Today's Outcome

- Built a reusable SRKPINN pendulum experiment runner and standardized artifact export under `app/piml/SRKPINN/experiment/`.
- Added device selection with `CUDA > MPS > CPU` priority so Apple Silicon environments can use `mps` automatically when available.
- Re-ran the `dt × stages` sweep with a fair fixed physical horizon `T_eval = 20` instead of a fixed rollout step count.
- Completed the second sweep over `train_data_size × sample_mode`.
- Completed the third sweep over loss weights.

## Best Configuration So Far

- `dt = 0.2`
- `stages = 3`
- method: `gauss-legendre`
- `train_data_size = 512`
- `sample_mode = uniform`
- loss weights:
  `StageDynamics = 1.0`, `InitialOrData = 2.0`
- evaluation horizon: `T_eval = 20`

## Key Findings

- The original fixed-`200`-step comparison overstated the advantage of smaller `dt`, because different `dt` values were being compared at different physical rollout times.
- Under fixed `T_eval = 20`, the best discretization became `dt = 0.2`, `stages = 3`.
- Data coverage did not improve beyond `train_data_size = 512`; both `1024` and `2048` degraded long-rollout behavior and cost more time.
- `sample_mode = random` at `512` reduced energy drift, but its rollout error remained worse than `uniform`.
- In the loss-weight sweep, the existing pair `StageDynamics = 1.0`, `InitialOrData = 2.0` remained the best by the main ranking rule.

## Artifacts

- Sweep registry:
  `app/piml/SRKPINN/experiment/SWEEP_LOG.md`
- Active checklist:
  `app/piml/SRKPINN/experiment/TODO.md`
- Time discretization visualizations:
  `app/piml/SRKPINN/experiment/time_discretization/`
- Data coverage visualizations:
  `app/piml/SRKPINN/experiment/data_coverage/`
- Loss balancing visualizations:
  `app/piml/SRKPINN/experiment/loss_balancing/`

## Recommended Next Step

- Start the optimization sweep with the fixed configuration above.
- Scan `learning_rate ∈ {3e-4, 1e-3, 3e-3}` first, then extend `num_epochs` only if the lower-rate candidate is clearly more stable.
