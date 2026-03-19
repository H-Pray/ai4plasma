"""Symplectic Butcher tableaus for structure-preserving RK-PINNs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai4plasma.config import REAL


def _real_array(values) -> np.ndarray:
    """Cast coefficients to the configured floating-point precision."""
    return np.asarray(values, dtype=REAL())


@dataclass
class ButcherTableau:
    """Container for Runge-Kutta coefficients.

    Parameters
    ----------
    A : numpy.ndarray
        Stage coupling matrix of shape ``(s, s)``.
    b : numpy.ndarray
        Weights for the step update, shape ``(s,)``.
    c : numpy.ndarray
        Collocation nodes, shape ``(s,)``.
    name : str
        Human-readable identifier.
    order : int
        Formal order of accuracy.
    is_symplectic : bool, optional
        Precomputed symplectic flag. If omitted, it is inferred numerically.
    """

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    name: str
    order: int
    is_symplectic: bool | None = None

    def __post_init__(self) -> None:
        self.A = _real_array(self.A)
        self.b = _real_array(self.b).reshape(-1)
        self.c = _real_array(self.c).reshape(-1)
        self.validate()
        if self.is_symplectic is None:
            self.is_symplectic = bool(
                np.allclose(self.symplectic_residual(), 0.0, atol=1e-6, rtol=1e-6)
            )

    @property
    def stages(self) -> int:
        """Return the number of RK stages."""
        return int(self.b.shape[0])

    def validate(self) -> "ButcherTableau":
        """Validate tableau dimensions."""
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Butcher tableau A must be a square matrix.")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Butcher tableau A and b must use the same number of stages.")
        if self.c.shape[0] != self.b.shape[0]:
            raise ValueError("Butcher tableau c must have one entry per stage.")
        return self

    def symplectic_residual(self) -> np.ndarray:
        """Return the symplecticity residual matrix.

        The symplectic condition for RK methods is

        .. math::

           b_i a_{ij} + b_j a_{ji} - b_i b_j = 0.
        """

        left = self.b[:, None] * self.A
        right = self.b[None, :] * self.A.T
        outer = np.outer(self.b, self.b)
        return left + right - outer

    def assert_symplectic(self, atol: float = 1e-6, rtol: float = 1e-6) -> None:
        """Raise if the tableau is not symplectic within tolerance."""
        residual = self.symplectic_residual()
        if not np.allclose(residual, 0.0, atol=atol, rtol=rtol):
            max_err = float(np.max(np.abs(residual)))
            raise ValueError(
                f"Tableau '{self.name}' is not symplectic; max residual is {max_err:.3e}."
            )


def _lagrange_basis(nodes: np.ndarray, index: int) -> np.poly1d:
    """Return the Lagrange basis polynomial at ``nodes[index]``."""
    poly = np.poly1d([1.0])
    denom = 1.0
    for node_id, node in enumerate(nodes):
        if node_id == index:
            continue
        poly *= np.poly1d([1.0, -float(node)])
        denom *= float(nodes[index] - node)
    return poly / denom


def _collocation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Construct the collocation matrix for Gauss-Legendre nodes on ``[0, 1]``."""
    stages = len(nodes)
    matrix = np.zeros((stages, stages), dtype=np.float64)
    for col in range(stages):
        basis = _lagrange_basis(nodes, col)
        integral = np.polyint(basis)
        matrix[:, col] = integral(nodes) - integral(0.0)
    return matrix


def get_implicit_midpoint_tableau() -> ButcherTableau:
    """Return the 1-stage implicit midpoint tableau."""
    return ButcherTableau(
        A=np.array([[0.5]], dtype=np.float64),
        b=np.array([1.0], dtype=np.float64),
        c=np.array([0.5], dtype=np.float64),
        name="implicit-midpoint",
        order=2,
        is_symplectic=True,
    )


def get_gauss_legendre_tableau(stages: int) -> ButcherTableau:
    """Return the shifted Gauss-Legendre tableau on ``[0, 1]``."""
    if stages < 1:
        raise ValueError("Gauss-Legendre tableau requires at least one stage.")
    if stages == 1:
        return get_implicit_midpoint_tableau()

    nodes_std, weights_std = np.polynomial.legendre.leggauss(stages)
    c = 0.5 * (nodes_std + 1.0)
    b = 0.5 * weights_std
    A = _collocation_matrix(c)
    tableau = ButcherTableau(
        A=A,
        b=b,
        c=c,
        name=f"gauss-legendre-{stages}",
        order=2 * stages,
    )
    tableau.assert_symplectic()
    return tableau


def load_symplectic_tableau(method: str = "gauss-legendre", stages: int = 2) -> ButcherTableau:
    """Load a supported symplectic tableau.

    Parameters
    ----------
    method : str, default="gauss-legendre"
        Symplectic RK family to use. Supported values are ``"gauss-legendre"``,
        ``"gauss"``, and ``"implicit-midpoint"``.
    stages : int, default=2
        Number of RK stages.
    """

    normalized = method.strip().lower()
    if normalized in {"gauss-legendre", "gauss"}:
        return get_gauss_legendre_tableau(stages)
    if normalized in {"implicit-midpoint", "midpoint"}:
        if stages != 1:
            raise ValueError("Implicit midpoint uses exactly one stage.")
        return get_implicit_midpoint_tableau()
    raise ValueError(f"Unsupported symplectic RK method: {method}")
