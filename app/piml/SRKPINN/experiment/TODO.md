# SRKPINN Pendulum Sweep TODO

## 0. Evaluation Protocol

- [x] Freeze a common evaluation protocol before scanning parameters.
- [x] Record at least these metrics for every run: train one-step RMSE, rollout terminal state error, max rollout state error, terminal energy drift, max energy drift, symplectic residual.
- [ ] Evaluate more than one initial state once the baseline run path is stable.
- [x] Keep the monitored physical rollout horizon fixed when comparing runs across `dt`.
- [x] Use `T_eval = 20` as the fixed rollout horizon for the first cross-`dt` sweep so the baseline remains comparable.

## 1. Baseline

- [x] Create an isolated experiment workspace under `app/piml/SRKPINN/experiment`.
- [x] Add a dedicated baseline run script that mirrors the current pendulum configuration.
- [x] Smoke-test the baseline entrypoint and artifact export path.
- [x] Run the baseline with the current default settings and archive the summary in `experiment/baseline`.
- [x] Treat the finished baseline as the reference row for all later comparisons.

## 2. First Sweep: Time Discretization

- [x] Scan `dt` first.
- [x] Scan `stages` together with `dt`.
- [x] Use `implicit-midpoint` for `stages=1` and `gauss-legendre` for `stages>=2`.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `dt` | `0.05`, `0.1`, `0.2` |
| `stages` | `1`, `2`, `3` |

- [x] Pick the best `dt/stages` pair before changing data or loss weights.

## 3. Second Sweep: Training Data Coverage

- [x] Scan `train_data_size`.
- [x] Compare `sample_mode="uniform"` and `sample_mode="random"`.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `train_data_size` | `128`, `512`, `1024`, `2048` |
| `sample_mode` | `uniform`, `random` |

- [x] Rebuild the dataset for every run because this model generates training pairs once at initialization.
- [x] Keep `dt=0.2`, `stages=3`, `method=gauss-legendre`, `T_eval=20` fixed during the data-coverage sweep.
- [x] Pick the best `train_data_size/sample_mode` pair before moving to loss balancing.

## 4. Third Sweep: Loss Balancing

- [x] Compare `StageDynamics` and `InitialOrData` weights after the best discretization and data settings are fixed.
- [ ] Recommended grid:

| `StageDynamics` | `InitialOrData` |
| --- | --- |
| `1.0` | `1.0` |
| `1.0` | `2.0` |
| `1.0` | `5.0` |
| `0.5` | `2.0` |

- [x] Watch for trade-offs between one-step accuracy and long-rollout behavior.
- [x] Keep `dt=0.2`, `stages=3`, `train_data_size=512`, `sample_mode=uniform`, `T_eval=20` fixed during the loss-balancing sweep.
- [x] Pick the best loss-weight pair before moving to optimization.

## 5. Fourth Sweep: Optimization

- [x] Add a dedicated optimization sweep entrypoint under `experiment/optimization`.
- [x] Keep `dt=0.2`, `stages=3`, `method=gauss-legendre`, `train_data_size=512`, `sample_mode=uniform`, `StageDynamics=1.0`, `InitialOrData=2.0`, `T_eval=20` fixed for the initial optimization sweep.
- [x] Scan learning rate after the first three sweep blocks are settled.
- [ ] Extend epochs if a smaller learning rate is clearly more stable.
- [x] Compare scheduler settings only after learning rate is roughly in place.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `learning_rate` | `3e-4`, `1e-3`, `3e-3` |
| `num_epochs` | `6000`, `10000`, `15000` |

Notes from the first learning-rate pass (`6000` epochs, `MultiStepLR milestones=[2000,4000], gamma=0.5`):

- `1e-3` remained the best setting on the ranking rule.
- `3e-4` trained smoothly but did not beat `1e-3` on rollout error or energy drift, so an epoch extension is not yet justified.
- `3e-3` was clearly unstable in long rollout, so the next step should focus on scheduler comparisons around `lr=1e-3`.

Notes from the first scheduler pass (`lr=1e-3`, `6000` epochs):

- `MultiStepLR milestones=[2000,4000], gamma=0.5` remained the best optimizer schedule on the ranking rule.
- `none` caused a severe long-rollout regression and should not be used for this configuration.
- A later decay schedule (`milestones=[3000,5000], gamma=0.5`) improved energy drift and one-step RMSE, but its rollout error was still worse than the current baseline.
- No scheduler candidate created a clear case for longer training, so the next main block should move to network capacity.

## 6. Fifth Sweep: Network Capacity

- [x] Compare width before trying many deeper variants.
- [x] Compare depth after width settles and before activation.
- [x] Compare activation function only after a reasonable width/depth is found.
- [x] Keep output dimension tied to `2 * stages`.
- [ ] Recommended first pass:

| Parameter | Values |
| --- | --- |
| hidden width | `64`, `128`, `256` |
| hidden depth | `3`, `4`, `5` layers |
| activation | `Tanh`, `SiLU`, `GELU` |

Notes from the first network-capacity pass (`lr=1e-3`, `6000` epochs, `MultiStepLR milestones=[2000,4000], gamma=0.5`):

- Width sweep: `128` remained best on the ranking rule; `64` underfit long rollout, and `256` improved energy drift but worsened rollout error.
- Depth sweep at width `128`: `3` layers remained best; `4` and `5` layers both caused severe long-rollout regressions.
- Activation sweep at width `128`, depth `3`: `Tanh` remained best; `SiLU` and `GELU` both improved one-step RMSE but degraded rollout behavior.
- No network-capacity variant beat the existing architecture `[2, 128, 128, 128, 6]` with `Tanh`.

## 7. Later Work

- [ ] Expand the evaluation suite to several energy levels and near-separatrix states.
- [ ] Revisit `q_range` and `p_range` only after the main sweeps above are done.
- [ ] Run at least 3 seeds for candidate configurations before drawing conclusions.
- [ ] Promote only the top 2 or 3 candidates to longer rollout evaluation.
