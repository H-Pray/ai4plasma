"""Smoke tests for the Hamiltonian SRKPINN model."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from ai4plasma.core.network import FNN
from SRKPINN import HamiltonianSRKPINN, PendulumSystem


class TestHamiltonianSRKPINNSmoke(unittest.TestCase):
    """Check basic forward, loss, and gradient behavior."""

    def test_calc_loss_backward_and_predict(self):
        torch.manual_seed(0)
        np.random.seed(0)

        current_states = np.array(
            [
                [-1.0, 0.2],
                [-0.5, -0.1],
                [0.5, 0.4],
                [1.0, -0.3],
            ],
            dtype=np.float32,
        )
        next_states = current_states.copy()

        backbone = FNN(layers=[2, 24, 24, 6], act_fun=nn.Tanh())
        model = HamiltonianSRKPINN(
            system=PendulumSystem(),
            dt=0.1,
            stages=2,
            method="gauss-legendre",
            backbone_net=backbone,
            train_data=(current_states, next_states),
        )

        total_loss, loss_dict = model.calc_loss()
        self.assertEqual(set(loss_dict.keys()), {"StageDynamics", "StepClosure", "InitialOrData"})
        self.assertTrue(torch.isfinite(total_loss))

        total_loss.backward()
        grad_norm = 0.0
        for param in model.network.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.norm().item())
        self.assertGreater(grad_norm, 0.0)

        pred = model.predict_step(current_states[0])
        self.assertEqual(pred.shape, (2,))


if __name__ == "__main__":
    unittest.main()
