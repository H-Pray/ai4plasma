"""Tests for device utility helpers."""

import unittest
from unittest.mock import patch

import torch

from ai4plasma.utils.device import select_best_device, torch_device
from app.piml.SRKPINN.experiment.runner import resolve_rollout_steps


class TestDeviceUtils(unittest.TestCase):
    """Validate device selection on CUDA, MPS, and CPU paths."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_select_best_device_prefers_cuda(self, _mock_cuda):
        self.assertEqual(select_best_device(), 0)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_select_best_device_uses_mps(self, _mock_mps, _mock_cuda):
        self.assertEqual(select_best_device(), "mps")

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_select_best_device_falls_back_to_cpu(self, _mock_mps, _mock_cuda):
        self.assertEqual(select_best_device(), -1)

    def test_torch_device_accepts_cpu_string(self):
        self.assertEqual(torch_device("cpu"), torch.device("cpu"))

    def test_resolve_rollout_steps_uses_requested_total_time(self):
        steps, total_time = resolve_rollout_steps(dt=0.05, num_rollout_steps=200, rollout_total_time=20.0)
        self.assertEqual(steps, 400)
        self.assertAlmostEqual(total_time, 20.0)


if __name__ == "__main__":
    unittest.main()
