# SRKPINN

`SRKPINN` is a standalone package in this repository for **symplectic Runge-Kutta Physics-Informed Neural Networks**. It targets canonical Hamiltonian systems and keeps the time-discretization layer explicitly structure-preserving by using symplectic implicit RK tableaus.

## Scope

- Canonical Hamiltonian systems in the form
  - `dq/dt = dH/dp`
  - `dp/dt = -dH/dq`
- Discrete-time PINN training on one-step state maps
- Symplectic IRK methods with hard tableau constraints
- Long-horizon diagnostics such as phase portrait error and energy drift

This package is intentionally separate from `ai4plasma.piml.rk_pinn`, which is tuned for corona discharge equations and generic implicit RK residuals rather than Hamiltonian structure preservation.

## Package Layout

- `tableau.py`: symplectic Butcher tableaus and symplecticity checks
- `systems.py`: Hamiltonian system abstractions and the nonlinear pendulum benchmark
- `networks.py`: stage-aware network wrapper for SRK-PINN outputs
- `model.py`: `HamiltonianSRKPINN` built on top of `ai4plasma.piml.pinn.PINN`
- `callbacks.py`: visualization callback for phase-space, rollout, and energy diagnostics
- `utils.py`: reference stepping and rollout helpers

## Symplectic RK Constraint

For a Runge-Kutta method with coefficients `(A, b, c)`, symplecticity is enforced by the tableau condition

```text
b_i a_ij + b_j a_ji - b_i b_j = 0
```

`SRKPINN.tableau.ButcherTableau` computes this residual explicitly and rejects non-symplectic tableaus when `assert_symplectic()` is called.

## Model Structure

For one-step training from `z_n = [q_n, p_n]`, the network predicts:

- stage states `(Q_i, P_i)` for all RK stages
- the step-end state `(q_{n+1}, p_{n+1})`

The training loss contains three terms:

- `StageDynamics`: stage consistency with the symplectic RK equations
- `StepClosure`: step-end consistency with the RK weights
- `InitialOrData`: supervised one-step target residual

The first two terms encode the symplectic RK structure. The data term anchors the learned one-step map to reference trajectories.

## First Example

The first benchmark is the **nonlinear pendulum**

```text
H(q, p) = 0.5 p^2 + omega0^2 (1 - cos q)
```

Run the example with:

```bash
python app/piml/SRKPINN/solve_pendulum_srkpinn.py
```

Artifacts are written under `app/piml/SRKPINN/results/pendulum/`.

## Current Limitations

- First version supports Hamiltonian ODE benchmarks only
- Only `gauss-legendre` and `implicit-midpoint` tableaus are included
- No partitioned symplectic RK method is implemented yet
- Energy is monitored as a diagnostic, not enforced as a training constraint
