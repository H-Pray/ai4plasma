# SRKPINN Pendulum Sweep TODO

## 0. Evaluation Protocol

- [ ] Freeze a common evaluation protocol before scanning parameters.
- [ ] Record at least these metrics for every run: train one-step RMSE, rollout state error at step 200, max rollout state error, energy drift at step 200, max energy drift, symplectic residual.
- [ ] Evaluate more than one initial state once the baseline run path is stable.
- [ ] Keep the monitored rollout length fixed when comparing runs.

## 1. Baseline

- [x] Create an isolated experiment workspace under `app/piml/SRKPINN/experiment`.
- [x] Add a dedicated baseline run script that mirrors the current pendulum configuration.
- [x] Smoke-test the baseline entrypoint and artifact export path.
- [ ] Run the baseline with the current default settings and archive the summary in `experiment/baseline`.
- [ ] Treat the finished baseline as the reference row for all later comparisons.

## 2. First Sweep: Time Discretization

- [ ] Scan `dt` first.
- [ ] Scan `stages` together with `dt`.
- [ ] Use `implicit-midpoint` for `stages=1` and `gauss-legendre` for `stages>=2`.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `dt` | `0.05`, `0.1`, `0.2` |
| `stages` | `1`, `2`, `3` |

- [ ] Pick the best `dt/stages` pair before changing data or loss weights.

## 3. Second Sweep: Training Data Coverage

- [ ] Scan `train_data_size`.
- [ ] Compare `sample_mode="uniform"` and `sample_mode="random"`.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `train_data_size` | `128`, `512`, `1024`, `2048` |
| `sample_mode` | `uniform`, `random` |

- [ ] Rebuild the dataset for every run because this model generates training pairs once at initialization.

## 4. Third Sweep: Loss Balancing

- [ ] Compare `StageDynamics` and `InitialOrData` weights after the best discretization and data settings are fixed.
- [ ] Recommended grid:

| `StageDynamics` | `InitialOrData` |
| --- | --- |
| `1.0` | `1.0` |
| `1.0` | `2.0` |
| `1.0` | `5.0` |
| `0.5` | `2.0` |

- [ ] Watch for trade-offs between one-step accuracy and long-rollout behavior.

## 5. Fourth Sweep: Optimization

- [ ] Scan learning rate after the first three sweep blocks are settled.
- [ ] Extend epochs if a smaller learning rate is clearly more stable.
- [ ] Compare scheduler settings only after learning rate is roughly in place.
- [ ] Recommended coarse grid:

| Parameter | Values |
| --- | --- |
| `learning_rate` | `3e-4`, `1e-3`, `3e-3` |
| `num_epochs` | `6000`, `10000`, `15000` |

## 6. Fifth Sweep: Network Capacity

- [ ] Compare width before trying many deeper variants.
- [ ] Compare activation function only after a reasonable width/depth is found.
- [ ] Keep output dimension tied to `2 * stages`.
- [ ] Recommended first pass:

| Parameter | Values |
| --- | --- |
| hidden width | `64`, `128`, `256` |
| hidden depth | `3`, `4`, `5` layers |
| activation | `Tanh`, `SiLU`, `GELU` |

## 7. Later Work

- [ ] Expand the evaluation suite to several energy levels and near-separatrix states.
- [ ] Revisit `q_range` and `p_range` only after the main sweeps above are done.
- [ ] Run at least 3 seeds for candidate configurations before drawing conclusions.
- [ ] Promote only the top 2 or 3 candidates to longer rollout evaluation.
