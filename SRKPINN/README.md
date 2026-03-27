# SRKPINN

`SRKPINN` is the Hamiltonian-ODE branch of SRK-PINN in this repository. It focuses on
**symplectic Runge-Kutta Physics-Informed Neural Networks** for canonical Hamiltonian
systems, with the current first benchmark being the nonlinear pendulum.

This document is the maintained project note for the current implementation status,
method design, training workflow, numerical behavior, and next steps.

## 1. Project Positioning

`SRKPINN` is intentionally separated from `ai4plasma.piml.rk_pinn`.

The original `ai4plasma.piml.rk_pinn` implementation is a **corona-discharge-specific**
RK-PINN solver:

- it is built around 1D spatial PDE residuals,
- it hardcodes plasma variables such as `Phi` and `Ne`,
- it embeds corona boundary conditions into the network and loss,
- it is not a ready-made baseline for Hamiltonian ODE benchmarks.

`SRKPINN` is instead designed for:

- canonical Hamiltonian systems
- one-step map learning in state space
- symplectic implicit Runge-Kutta structure
- long-horizon structure diagnostics such as energy drift and local map symplecticity

At the current stage, `SRKPINN` is best understood as a **Hamiltonian benchmark branch**
rather than a general replacement for the original RK-PINN module.

## 2. Current Progress

Status as of **March 22, 2026**:

### 2.1 Completed

- A standalone `SRKPINN/` package has been created.
- Symplectic Butcher tableau support is implemented in `tableau.py`.
- Canonical Hamiltonian system abstractions are implemented in `systems.py`.
- A pendulum benchmark system is implemented in `PendulumSystem`.
- The network now predicts **only RK stage states**, not a free end-step state.
- The step-end state is reconstructed by the **hard symplectic RK closure**.
- The old soft `StepClosure` loss has been removed from the active design.
- A local discrete-map symplecticity diagnostic is implemented.
- A complete pendulum training script is available in
  `app/piml/SRKPINN/solve_pendulum_srkpinn.py`.
- Smoke tests and tableau tests are in place.
- A full pendulum training run has already been completed and saved to disk.

### 2.2 Verified

- Training can run to completion for the pendulum example.
- Checkpoints, final model, TensorBoard logs, and plots are generated.
- The saved model has good **one-step** accuracy on random test states.
- The learned one-step map has small **local symplectic residual** in most tested states.
- Long-horizon performance is reasonable in moderate-energy regions.

### 2.3 Not Yet Completed

- A pendulum version of the **original RK-PINN** baseline has not been implemented.
- A formal SRKPINN vs RKPINN benchmark on the same pendulum task has not started yet.
- Current monitoring still focuses on a single rollout seed and a 200-step horizon.
- Difficult regions near the separatrix are not yet handled robustly.
- Energy conservation is only diagnosed, not directly optimized.
- The learned map is not exactly symplectic; only the embedded RK structure is symplectic.

## 3. Package Layout

- `tableau.py`
  - symplectic Butcher tableaus
  - tableau validation
  - numerical symplecticity checks
- `systems.py`
  - `HamiltonianSystem` base class
  - `PendulumSystem` benchmark
- `networks.py`
  - structured network output wrapper
  - `SRKStepOutput`
  - `HamiltonianSRKNet`
- `model.py`
  - `HamiltonianSRKPINN`
  - training-pair construction
  - RK stage residuals
  - hard step closure
  - prediction and rollout APIs
- `callbacks.py`
  - rollout diagnostics
  - phase portrait comparison
  - energy drift tracking
  - local map symplecticity tracking
- `utils.py`
  - reference stepping
  - rollout helpers
  - energy drift computation

## 4. Problem Class

The current implementation targets canonical Hamiltonian systems of the form

```math
\dot q = \nabla_p H(q,p), \qquad
\dot p = -\nabla_q H(q,p).
```

The full state is

```math
z = \begin{bmatrix} q \\ p \end{bmatrix}.
```

The current benchmark problem is the nonlinear pendulum

```math
H(q,p) = \frac{1}{2} p^2 + \omega_0^2 (1 - \cos q).
```

For the pendulum implemented in this repository:

- state dimension: `2`
- canonical coordinate: `q`
- canonical momentum: `p`
- default training box: `q in [-pi, pi]`, `p in [-2, 2]`

## 5. Core Idea of the Current SRK-PINN Design

The most important design decision in the current version is:

> the neural network predicts only the **stage variables**, and the step-end state is
> reconstructed by the **symplectic RK closure formula** instead of being predicted by an
> extra free output head.

This change is central because it removes one major source of artificial dissipation from
the previous draft. The model is now much closer to "learn a stage-consistent symplectic
discretization" rather than "learn a generic next-step regressor with an RK-flavored loss".

## 6. Symplectic RK Structure

For an `s`-stage RK method with coefficients `(A, b, c)`, the canonical symplecticity
condition is

```math
b_i a_{ij} + b_j a_{ji} - b_i b_j = 0.
```

In this package:

- `ButcherTableau.symplectic_residual()` computes the residual matrix
- `ButcherTableau.assert_symplectic()` rejects tableaus that violate the condition
- `load_symplectic_tableau()` currently supports:
  - `gauss-legendre`
  - `gauss`
  - `implicit-midpoint`

At the moment, the practical default is a **2-stage Gauss-Legendre** method.

## 7. Mathematical Formulation of the Learned Step

Given a current state `z_n = (q_n, p_n)`, the model predicts stage states

```math
(Q_1, P_1), \dots, (Q_s, P_s).
```

The symplectic RK stage equations are

```math
Q_i = q_n + \Delta t \sum_{j=1}^s a_{ij} \nabla_p H(Q_j, P_j),
```

```math
P_i = p_n - \Delta t \sum_{j=1}^s a_{ij} \nabla_q H(Q_j, P_j).
```

The step-end state is then reconstructed by the hard closure

```math
q_{n+1} = q_n + \Delta t \sum_{i=1}^s b_i \nabla_p H(Q_i, P_i),
```

```math
p_{n+1} = p_n - \Delta t \sum_{i=1}^s b_i \nabla_q H(Q_i, P_i).
```

The current implementation follows this exactly:

- the network outputs `q_stages` and `p_stages`
- `model._hamiltonian_gradients()` computes `dH/dq` and `dH/dp` at those stages
- `model._construct_step_output()` reconstructs `q_next` and `p_next`

There is no learned free head for `(q_{n+1}, p_{n+1})`.

## 8. Training Data Construction

The current training workflow is **discrete one-step supervised PINN training**.

### 8.1 Current Training Input

The model trains on one-step pairs

```math
(z_n, z_{n+1}^{ref}).
```

The current implementation supports three ways to provide data:

- `train_data=None`
  - sample states internally
  - generate one-step targets with the reference stepper
- `train_data=(current_states, next_states)`
  - directly provide aligned state arrays
- `train_data=array of shape (N, 2 * state_dim)`
  - provide concatenated pairs

### 8.2 Internal Pair Generation

If `train_data` is not supplied:

1. states are sampled from `system.sample_initial_states`
2. one-step targets are generated by `reference_stepper`
3. the model stores the concatenated pair tensor for training

By default, the reference stepper is a high-accuracy SciPy integrator:

- method: `DOP853`
- tolerances:
  - `rtol = 1e-10`
  - `atol = 1e-10`

So the training target is currently not an analytical one-step symplectic map, but a
numerically accurate reference flow over one time step.

## 9. Loss Design

The current model uses two loss terms.

### 9.1 `StageDynamics`

This term enforces the stage equations:

```math
R^Q_i = Q_i - q_n - \Delta t \sum_j a_{ij} \nabla_p H(Q_j, P_j),
```

```math
R^P_i = P_i - p_n + \Delta t \sum_j a_{ij} \nabla_q H(Q_j, P_j).
```

The residuals are concatenated and driven toward zero.

### 9.2 `InitialOrData`

This term compares the hard-constructed end-step state to the one-step reference target:

```math
R^{data} = z_{n+1}^{SRK}(Q,P) - z_{n+1}^{ref}.
```

### 9.3 Total Loss

The total loss is

```math
\mathcal{L}
= w_{stage} \, \mathrm{MSE}(R^{stage})
+ w_{data} \, \mathrm{MSE}(R^{data}).
```

For the current pendulum script:

- `w_stage = 1.0`
- `w_data = 2.0`

### 9.4 Important Change from the Earlier Draft

The current design **does not** use a separate soft `StepClosure` loss term.

That earlier design was weaker because:

- the RK closure only appeared as a soft penalty
- the network still had freedom to predict an inconsistent next state
- that extra freedom could introduce artificial drift

The current design hard-wires the closure directly into the forward map.

## 10. Network Structure

The backbone network is a standard `FNN`, wrapped by `HamiltonianSRKNet`.

For a state dimension `d` and `s` stages:

- raw output size = `d * s`
- the first half of outputs are reshaped into `q_stages`
- the second half are reshaped into `p_stages`

For the pendulum case:

- `state_dim = 2`
- `stages = 2`
- output dimension = `4`

The default example uses:

```text
[2, 128, 128, 128, 4]
```

with `Tanh` activation.

## 11. Current Pendulum Training Script

The main example is:

```bash
python app/piml/SRKPINN/solve_pendulum_srkpinn.py
```

Current configuration in that script:

- system: nonlinear pendulum
- `omega0 = 1.0`
- `dt = 0.1`
- `stages = 2`
- method: `gauss-legendre`
- epochs: `6000`
- learning rate: `1e-3`
- training sample count: `512`
- sampling mode: `uniform`
- learning-rate scheduler:
  - `MultiStepLR`
  - milestones: `[2000, 4000]`
  - gamma: `0.5`

Monitoring configuration:

- monitored initial state: `[1.7, 0.0]`
- rollout length: `200`
- visualization frequency: every `100` epochs
- checkpoint frequency: every `1000` epochs

Artifacts are written to:

- results: `app/piml/SRKPINN/results/pendulum/`
- TensorBoard logs: `app/piml/SRKPINN/runs/pendulum/`
- checkpoints and final model: `app/piml/SRKPINN/models/pendulum/`

## 12. Diagnostics and What They Mean

### 12.1 One-Step Prediction Error

This is the most direct measure of how well the learned map matches the reference
integrator over one time step.

Good one-step error is necessary, but not sufficient, for long-horizon quality.

### 12.2 Rollout State Error

Rollout error measures how prediction error accumulates when the learned one-step map is
reused repeatedly.

This is often the first place where a model that "looks accurate locally" fails.

### 12.3 Energy Drift

The callback computes

```math
|H(z_k) - H(z_0)|.
```

This is only a diagnostic. It is not directly optimized in the current loss.

### 12.4 Local Symplectic Map Residual

For the learned one-step map `F`, the implementation evaluates

```math
J_F(z)^T \Omega J_F(z) - \Omega,
```

where `Omega` is the canonical symplectic matrix.

This does **not** prove the learned map is globally symplectic, but it is a useful local
defect measure.

## 13. Current Numerical Status

The current repository already contains a trained pendulum model and full artifacts:

- `app/piml/SRKPINN/models/pendulum/final_model.pth`
- `app/piml/SRKPINN/models/pendulum/checkpoint_epoch_6000.pth`
- `app/piml/SRKPINN/results/pendulum/final_panels.png`
- `app/piml/SRKPINN/results/pendulum/loss_curve.png`
- `app/piml/SRKPINN/runs/pendulum/events.out.tfevents...`

### 13.1 Offline Evaluation Summary

On **March 22, 2026**, the saved model was evaluated offline from
`app/piml/SRKPINN/models/pendulum/final_model.pth` using:

- 256 random one-step test states
- 64 sampled states for local symplectic residual estimation
- 4 representative rollout seeds
- rollout horizons of 200 and 1000 steps

#### One-step accuracy

| Metric | Value |
| --- | ---: |
| RMSE L2 | `3.72e-4` |
| Mean L2 | `2.53e-4` |
| Median L2 | `2.28e-4` |
| 95th percentile L2 | `5.01e-4` |
| Max L2 | `3.06e-3` |
| Mean relative L2 | `1.52e-4` |
| 95th percentile relative L2 | `4.23e-4` |

Interpretation:

- the current SRK-PINN has very good one-step accuracy on the tested state box
- the model is learning the local one-step map well

#### Local symplecticity defect

| Metric | Value |
| --- | ---: |
| Mean max-abs residual | `2.76e-4` |
| Median max-abs residual | `2.08e-4` |
| 95th percentile max-abs residual | `5.77e-4` |
| Max sampled max-abs residual | `2.60e-3` |

Interpretation:

- the learned map is not exactly symplectic
- but its local symplectic defect is small over most sampled states

### 13.2 Representative Rollout Results

| Case | Horizon | Final state error | Max state error | Final relative energy drift | Max relative energy drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| small-angle `(0.5, 0.0)` | 200 | `4.63e-2` | `4.63e-2` | `4.95e-2` | `4.95e-2` |
| small-angle `(0.5, 0.0)` | 1000 | `1.61e-1` | `1.62e-1` | `2.23e-1` | `2.23e-1` |
| moderate-angle `(1.7, 0.0)` | 200 | `2.13e-2` | `2.13e-2` | `2.22e-3` | `2.73e-3` |
| moderate-angle `(1.7, 0.0)` | 1000 | `2.63e-1` | `3.44e-1` | `1.35e-2` | `1.43e-2` |
| near-separatrix `(3.0, 0.0)` | 200 | `1.89` | `1.89` | `1.69e-3` | `3.25e-3` |
| near-separatrix `(3.0, 0.0)` | 1000 | `1.62e1` | `2.35e1` | `8.01` | `8.01` |
| momentum-dominant `(0.0, 1.5)` | 200 | `2.60e-2` | `2.77e-2` | `2.94e-3` | `3.45e-3` |
| momentum-dominant `(0.0, 1.5)` | 1000 | `3.64e-1` | `3.69e-1` | `1.42e-2` | `1.42e-2` |

### 13.3 What These Results Mean

The current picture is mixed but clear:

- in **moderate-energy regions**, the current SRK-PINN is already promising
- local structure is much better controlled than a generic free-step predictor
- one-step accuracy is strong
- 200-step and 1000-step energy behavior is acceptable for moderate trajectories

However:

- low-energy trajectories still show non-negligible relative energy drift
- trajectories near the **separatrix** are not robust
- near-separatrix rollout error grows catastrophically over long horizons

So the current SRK-PINN is **not yet a uniformly reliable long-horizon Hamiltonian solver**
over the full pendulum phase space.

It is more accurate to say:

> the structure-aware redesign is working, but difficult energy regions are still
> under-trained and under-diagnosed.

## 14. Current Strengths

- Hard symplectic RK closure is in place.
- The implementation cleanly separates:
  - Hamiltonian structure
  - tableau structure
  - neural approximation
  - rollout diagnostics
- One-step map learning is accurate.
- Local symplectic defect is small in most sampled states.
- The code path from training to plotting to saved artifacts is complete.

## 15. Current Weaknesses and Failure Modes

- Training is still based on a single fixed state-box sampler.
- The current monitored rollout seed is only `[1.7, 0]`, which hides difficult regions.
- The training distribution does not deliberately oversample near-separatrix states.
- Energy is not constrained directly.
- The model can still learn a map with small local symplectic defect but poor global
  long-horizon behavior in hard regions.
- Current diagnostics are informative but not yet benchmark-grade.

## 16. Tests and Validation

Two test files are currently present:

- `tests/test_srkpinn_tableau.py`
  - checks that supported tableaus are symplectic
  - checks that an invalid tableau is rejected
- `tests/test_srkpinn_pendulum_smoke.py`
  - checks loss computation
  - checks backpropagation
  - checks `predict_step`
  - checks `symplectic_map_residual`

This is enough for smoke coverage, but not enough for benchmark-grade validation.

Still missing are tests for:

- long-horizon rollout stability
- regression on energy drift statistics
- behavior near the separatrix
- sensitivity to seed and sampling strategy

## 17. Recommended Next Technical Steps

The next improvements should focus on **data coverage and evaluation discipline**, not just
adding more structure terms.

### 17.1 Highest Priority

- Replace the current plain box sampling with **energy-stratified sampling**.
- Add validation rollouts for several representative energy levels.
- Track not just one monitored seed, but a small evaluation suite.
- Scan `dt` and `stages` before redesigning the loss again.

### 17.2 Medium Priority

- Add weak energy-consistency regularization only if sampling improvements are insufficient.
- Add a benchmark harness that exports metrics as machine-readable tables.
- Add near-separatrix focused diagnostics.

### 17.3 Lower Priority

- Partitioned symplectic RK methods
- More Hamiltonian systems
- Better visualization and benchmark automation

## 18. Benchmark Plan Against the Original RK-PINN

The benchmark has **not started yet** because the original `ai4plasma.piml.rk_pinn`
implementation is not a pendulum solver.

To compare fairly on the same pendulum problem, the following must be prepared first.

### 18.1 What Must Be Implemented

- a `PendulumRKPINN` baseline using the original RK-PINN philosophy
- identical pendulum state-space training data for both methods
- identical reference one-step targets
- identical evaluation scripts and metrics

### 18.2 What Must Be Held Constant

- same state domain
- same `dt`
- same RK stage count
- same train/validation/test splits
- same optimizer and scheduler policy
- same epoch budget
- same random seed protocol
- similar parameter count where possible

### 18.3 Metrics That Should Be Reported

- one-step RMSE
- rollout state error at multiple horizons
- absolute and relative energy drift
- local symplectic map residual
- training wall-clock time
- parameter count
- mean and standard deviation over multiple seeds

### 18.4 Important Reproducibility Note

The original `ai4plasma.piml.rk_pinn` currently expects local `ButcherTable/Butcher_*.npy`
files and will attempt a HuggingFace download if they are missing. That is acceptable for
its own workflow, but it is not ideal for a clean pendulum benchmark. The pendulum baseline
should avoid introducing hidden dependency differences between the two methods.

## 19. Current Practical Conclusion

At the present stage, `SRKPINN` can be summarized as follows:

- **implemented**: yes
- **trainable end-to-end on nonlinear pendulum**: yes
- **mathematically better aligned with symplectic structure than the earlier draft**: yes
- **already competitive in moderate-energy long-horizon behavior**: likely yes
- **ready for a rigorous benchmark against original RK-PINN**: not yet
- **ready to be used as the SRK-PINN side of that benchmark after data/eval cleanup**: yes

## 20. How to Run

Train the pendulum example:

```bash
python app/piml/SRKPINN/solve_pendulum_srkpinn.py
```

Run the current tests:

```bash
python -m unittest tests.test_srkpinn_tableau -v
python -m unittest tests.test_srkpinn_pendulum_smoke -v
```

If you are using the project conda environment:

```bash
conda run -n ai4plasma python app/piml/SRKPINN/solve_pendulum_srkpinn.py
```

## 21. Summary in One Sentence

The current SRK-PINN implementation has already crossed the "works as a real Hamiltonian
training pipeline" threshold, but it has not yet crossed the "benchmark-complete and robust
over the full pendulum phase space" threshold.

## 22. Addendum: Structural Review on March 25, 2026

This section records a later design review of:

- the structural difference between the original `ai4plasma.piml.rk_pinn` and `SRKPINN`
- possible ways to build **exact symplectic structure** into the learned one-step map
- how a symplectic RK-PINN should respond to the usual PINN difficulty with complex
  boundary conditions

### 22.1 What the Original RK-PINN Actually Is

The original `ai4plasma.piml.rk_pinn` is **not** a Hamiltonian solver.

It is a corona-discharge-specific PDE solver with the following traits:

- input is the 1D spatial coordinate `r`
- the network outputs stage values of `Phi` and `Ne`
- the electric-potential boundary condition is embedded directly into the network output
- the loss is built from PDE residuals and boundary-condition residuals
- the RK structure is used as a time discretization device, not as a symplectic structure

So its central design is:

- **PDE residual enforcement**
- **geometry/boundary handling**
- **implicit RK temporal discretization**

and not:

- canonical phase-space evolution
- Hamiltonian geometry
- exact or approximate symplectic map construction

### 22.2 What the Current SRKPINN Actually Is

The current `SRKPINN` is a **canonical Hamiltonian one-step map learner**.

Its main design points are:

- the state is `z = (q, p)` in canonical coordinates
- a symplectic implicit RK tableau is loaded and validated
- the network predicts only stage states `(Q_i, P_i)`
- the step-end state `(q_{n+1}, p_{n+1})` is reconstructed by the hard RK closure
- the training loss contains:
  - `StageDynamics`
  - `InitialOrData`

This means the current implementation is already stronger than an earlier draft that used
a soft next-step head, but it still does **not** make the learned map exactly symplectic.

The exact situation is:

- the **RK tableau** is symplectic
- the **closure formula** is hard-wired
- the **stage equations** are still only enforced through loss minimization

So the learned map is currently:

- more structure-aware than a generic RK-flavored regressor
- usually locally close to symplectic
- not guaranteed to be exactly symplectic

### 22.3 Review Conclusion About the Current Design

The current `SRKPINN` should be described as:

- a Hamiltonian benchmark branch
- a stage-based symplectic RK-inspired PINN
- a model with hard symplectic closure but soft stage consistency

It should **not** be described as an implementation that already guarantees an exactly
symplectic learned one-step map.

## 23. Three Possible Routes Toward Exact Symplectic Structure

Three candidate directions were examined for upgrading `SRKPINN` from
"symplectic-RK-structured but not exact-symplectic" to a model whose forward map is
symplectic by construction.

### 23.1 Scheme A: Differentiable Implicit SRK Solve Layer

#### Core idea

Keep the current symplectic RK identity, but stop letting the network directly output the
final stage states.

Instead:

- the network outputs a stage initial guess or solver warm start
- the forward pass includes a differentiable nonlinear solver
- that solver explicitly solves the symplectic RK stage equations for each input state
- the converged stages are then passed through the existing hard closure

In other words, move stage consistency from the loss into the forward map itself.

#### Why this is attractive

- it is the most faithful continuation of the current `SRKPINN` design
- `tableau.py` remains meaningful
- the hard closure remains meaningful
- `HamiltonianSystem.gradients()` remains central
- the resulting map is defined by a numerically solved symplectic RK system rather than by
  a stage residual that merely becomes small

#### What changes would be needed

- replace direct stage prediction with stage guess prediction
- add a batched implicit solver layer
- make `_predict_tensor()` solve for consistent stages before closure
- remove or greatly de-emphasize `StageDynamics` in the training loss
- add solver diagnostics such as iteration counts, failure rate, and residual norms

#### Main risks

- much higher compute cost
- more difficult training dynamics
- solver non-convergence near difficult regions or at large `dt`
- the network may degenerate into "only a warm-start generator" rather than a true map
  learner

#### Overall judgment

This is the **most SRK-faithful** route.

Recommended when the goal is:

- keep the SRK identity
- upgrade soft stage consistency into forward-level structure

Less attractive when the goal is:

- get a quick exact-symplectic prototype with modest engineering effort

### 23.2 Scheme B: Generating Function / Discrete Hamiltonian Parameterization

#### Core idea

Replace stage-state parameterization with a generating-function parameterization of the
one-step map.

The most natural choice for the current canonical setting is a **type-2 generating
function**:

```math
F_2(q_n, P_{n+1}) = q_n^\top P_{n+1} + G_\theta(q_n, P_{n+1}).
```

Then define the step implicitly by:

```math
p_n = \partial_{q_n} F_2,
\qquad
q_{n+1} = \partial_{P_{n+1}} F_2.
```

This gives a symplectic map by construction.

#### Why this is attractive

- mathematically the cleanest route to an exact symplectic map
- highly compatible with the current canonical state-space viewpoint
- especially natural for low-dimensional systems such as the pendulum
- the existing rollout, data generation, and diagnostics can still be reused

#### What changes would be needed

- add a new scalar-output generating-function network
- solve an implicit equation for `P_{n+1}` in the forward pass
- construct `q_{n+1}` from gradients of the generating function
- replace the stage-based loss structure with data-fit plus solver-quality monitoring
- keep this as a **new model class**, not as a small mode switch inside
  `HamiltonianSRKPINN`

#### Main risks

- nonlinear root solving is required in every forward pass
- mixed higher-order derivatives are required
- local chart issues may appear in harder systems
- gauge freedom exists because the generating function value is not directly identified,
  only its derivatives are

#### Overall judgment

This is the **mathematically cleanest** route.

Recommended when the goal is:

- build a truly symplectic canonical map learner
- explore a parallel research branch inside `SRKPINN/`

Less attractive when the goal is:

- preserve the current SRK stage-based identity with minimal conceptual change

### 23.3 Scheme C: Composition of Exact Symplectic Submaps

#### Core idea

Stop using stage states as the primary parameterization and instead build the one-step map
as a composition of symplectic elementary maps.

Typical examples are:

- alternating `q`-shear and `p`-shear updates
- learned splitting maps
- leapfrog- or Yoshida-style compositions
- more general symplectic coupling layers

For separable structures this can be written as repeated updates of the form:

```text
q <- q + grad T_k(p)
p <- p - grad V_k(q)
```

Each submap is symplectic, so the full composition is symplectic.

#### Why this is attractive

- the forward map is structure-preserving by construction
- there is no implicit stage solver in the simplest versions
- engineering cost is lower than Scheme A or Scheme B
- it is especially attractive for the current pendulum benchmark

#### What changes would be needed

- add a new symplectic-map network backend
- likely add a new model class such as `HamiltonianSympMapPINN`
- reuse training pairs, rollout, callback logic, and symplectic diagnostics
- treat this as a parallel model family rather than as a hidden implementation mode of
  `HamiltonianSRKPINN`

#### Main risks

- exact symplecticity does not guarantee low one-step error
- simple separable splitting may be too restrictive for more general Hamiltonians
- this path moves away from the "SRK" identity and toward a broader symplectic-map learner

#### Overall judgment

This is the **fastest way to obtain a forward-structural exact-symplectic prototype**.

Recommended when the goal is:

- get a robust pendulum showcase quickly
- add a parallel exact-symplectic baseline with moderate implementation effort

Less attractive when the goal is:

- preserve the current stage-based SRK interpretation

### 23.4 Comparative Recommendation

The ranking depends on the objective.

If the priority is **preserving SRK identity**, the preferred order is:

1. Scheme A
2. Scheme B
3. Scheme C

If the priority is **getting a practical exact-symplectic prototype quickly**, the preferred
order is:

1. Scheme C
2. Scheme B
3. Scheme A

The most sensible repository strategy is therefore a two-track approach:

- keep the current `HamiltonianSRKPINN` line as the main SRK-style branch
- explore exact-symplectic construction in a parallel branch

A practical staged recommendation is:

1. keep improving the current stage-based `HamiltonianSRKPINN`
2. add a parallel exact-symplectic prototype, with Scheme C being the fastest starting point
3. if a more mathematically canonical branch is desired later, add Scheme B
4. if the goal becomes "strict SRK geometry", revisit Scheme A

## 24. Boundary Conditions: How Symplectic RK-PINN Should Respond

One recurring difficulty of ordinary PINNs is poor behavior near complicated boundary
conditions. For `SRKPINN`, the response is different from a standard PDE PINN because the
problem class is different.

### 24.1 Why the Current Pendulum SRKPINN Largely Avoids This Issue

The current `SRKPINN` does **not** solve a PDE on a complicated spatial domain.

Instead it learns one-step maps in canonical phase space for systems such as the pendulum.
So the current benchmark does not involve:

- irregular geometry boundaries
- corner singularities
- sharp boundary layers
- simultaneous enforcement of high-order PDE residuals and complex BCs

In this sense, the current `SRKPINN` mostly **avoids** the classical PINN boundary problem
instead of solving it head-on.

### 24.2 General Principle

For symplectic Hamiltonian learning, the safest principle is:

- keep the **interior Hamiltonian evolution** structure-preserving
- handle **boundary, event, or constraint effects** with separate structure-aware rules

This is usually better than trying to force one unified PINN loss to learn:

- interior Hamiltonian flow
- complex boundary geometry
- event switching
- and exact or approximate symplectic structure

all at the same time.

### 24.3 Preferred Tactics

#### A. Reparameterize or transform the problem

When possible:

- transform coordinates so the domain becomes simpler
- use periodic representations such as `(sin q, cos q)` for angle variables
- encode known invariants or constraints into the state representation

This is the cleanest way to avoid artificial boundary artifacts.

#### B. Split interior flow from boundary/event handling

For hard walls, impacts, or switching surfaces:

- let the interior update be a symplectic or structure-aware map
- treat collisions or event updates as separate explicit maps

This prevents the model from having to learn a discontinuity as though it were a smooth
Hamiltonian interior flow.

#### C. Use hard constraints when the boundary is simple enough

If the geometry and boundary condition are algebraically manageable, embed them in the
network output, much as the original PDE-style `RK-PINN` embeds a potential boundary
condition.

But this is only suitable for relatively simple cases.

#### D. For constrained Hamiltonian systems, preserve the constraint structure too

If the system includes holonomic constraints or motion on a manifold, then naive projection
after a symplectic step can destroy the geometric property one is trying to preserve.

Better choices are:

- manifold-aware coordinates
- variational or symplectic projection methods
- explicit constrained geometric integrator logic

#### E. Do not expect soft boundary penalties alone to solve the problem

If the boundary mechanism is complex, simply increasing boundary loss weights is usually not
enough.

That tends to produce:

- poor gradient balance
- unstable optimization
- structure-breaking compromises

### 24.4 Practical Interpretation for Future SRKPINN Work

For future applications, the practical hierarchy should be:

1. avoid complicated boundaries by changing variables or problem formulation
2. if avoidance is impossible, separate boundary/event logic from interior Hamiltonian flow
3. only use soft boundary penalties as a last resort
4. do not claim global exact symplecticity if the boundary mechanism itself is non-Hamiltonian

So the real answer is:

- `SRKPINN` does **not** primarily solve the complex-boundary PINN problem
- it is strongest when that problem can be avoided or factored out
- if boundary effects are essential, they should be handled by a separate geometric module,
  not by hoping a standard PINN loss will absorb them

## 25. Validation Targets for Any Exact-Symplectic Upgrade

If any of the schemes in Section 23 are implemented, the validation target should be
stricter than the current one.

At minimum, report:

- one-step prediction error on held-out states
- rollout state error at several horizons
- absolute and relative energy drift
- local symplectic defect `J^T Omega J - Omega`
- failure rate of any nonlinear solver used in the forward pass
- mean solver iterations and worst-case residuals when implicit solves are present

The standard should change from:

- "the map has small local symplectic defect"

to:

- "the map is defined by a symplectic construction, and any remaining defect is dominated by
  solver tolerance or floating-point error"
