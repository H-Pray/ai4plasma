# Symplectic Generator TODO

- [ ] Write down the exact 1-DOF pendulum Type-2 generating-function formulation to be implemented.
- [ ] Decide the concrete network parameterization of the generating function.
- [ ] Implement a pendulum-specific generating-function model branch parallel to `HamiltonianSRKPINN`.
- [ ] Implement a differentiable Newton solver for the implicit one-step update.
- [ ] Expose `predict_step`, `rollout`, and `symplectic_map_residual` on the new model.
- [ ] Reuse the existing experiment and visualization pipeline where possible.
- [ ] Add a smoke test covering forward pass, backward pass, and symplectic residual evaluation.
- [ ] Add a pendulum prototype training entrypoint for the new model.
- [ ] Run an A/B comparison against the current best SRKPINN configuration.
- [ ] Summarize whether the structural symplectic map improves long-rollout behavior.
- [ ] Decide whether to generalize the prototype beyond the 1-DOF pendulum after the first comparison.
