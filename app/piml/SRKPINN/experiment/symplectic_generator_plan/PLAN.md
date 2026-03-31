# Symplectic Generator Prototype Plan

## Goal

Replace the current soft-constraint symplectic surrogate idea with a structurally symplectic one-step map for the pendulum benchmark.

The first implementation target is a **pendulum-specific 1-DOF Type-2 generating-function prototype** that can be trained with the existing one-step supervised data pairs and compared directly against the current `HamiltonianSRKPINN` branch.

## Locked Decisions

- method family: `Type-2 generating function`
- prototype scope: pendulum only, `1` degree of freedom
- training objective: supervised one-step `(z_n, z_{n+1}^{ref})` fitting
- integration strategy: differentiable implicit one-step solver inside the model
- code organization: add a **parallel model branch**, do not replace current `HamiltonianSRKPINN`
- experiment strategy: run direct A/B comparison against the current best sweep configuration

## Planned Implementation

### 1. New model branch

- add a new pendulum-specific model class for generating-function-based one-step propagation
- keep the existing `HamiltonianSRKPINN` intact as the control branch
- expose compatible methods:
  - `predict_step`
  - `rollout`
  - `symplectic_map_residual`

### 2. Generating-function step map

- network input: current state `(q_n, p_n)` or an equivalent state-derived input chosen consistently
- network output: scalar generating function representation for a Type-2 map
- solve the implicit one-step update inside the model with a differentiable Newton iteration
- recover `(q_{n+1}, p_{n+1})` from the generating-function relations

### 3. Training path

- continue using existing one-step reference pairs
- use a primary supervised one-step loss
- do not reintroduce soft symplecticity constraints as the main mechanism
- keep optimizer/scheduler defaults aligned with the current best experiment for clean A/B comparison

### 4. Experiment and validation

- add a smoke-test entrypoint
- add a baseline-style training script for the new model
- add an A/B experiment using the current best coarse-sweep configuration
- compare:
  - one-step RMSE
  - rollout final error
  - max energy drift
  - symplectic residual

## Acceptance Criteria

- the new model is structurally symplectic by construction rather than by loss weighting
- the full train/evaluate/visualize pipeline runs end-to-end
- the model can be compared directly against the current SRKPINN best run
- the local symplectic residual is reduced to the numerical-solver/autodiff error scale
