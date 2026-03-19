"""Unit tests for symplectic Butcher tableaus."""

import unittest

import numpy as np

from SRKPINN.tableau import ButcherTableau, load_symplectic_tableau


class TestSRKPINNTableau(unittest.TestCase):
    """Validate tableau construction and symplecticity checks."""

    def test_gauss_legendre_two_stage_is_symplectic(self):
        tableau = load_symplectic_tableau("gauss-legendre", stages=2)
        residual = tableau.symplectic_residual()
        self.assertTrue(tableau.is_symplectic)
        self.assertLess(float(np.max(np.abs(residual))), 1e-5)

    def test_implicit_midpoint_is_symplectic(self):
        tableau = load_symplectic_tableau("implicit-midpoint", stages=1)
        residual = tableau.symplectic_residual()
        self.assertTrue(tableau.is_symplectic)
        self.assertLess(float(np.max(np.abs(residual))), 1e-6)

    def test_non_symplectic_tableau_is_rejected(self):
        tableau = ButcherTableau(
            A=np.array([[0.0, 0.0], [0.0, 0.0]]),
            b=np.array([0.5, 0.5]),
            c=np.array([0.0, 1.0]),
            name="non-symplectic",
            order=1,
        )
        with self.assertRaises(ValueError):
            tableau.assert_symplectic()


if __name__ == "__main__":
    unittest.main()
