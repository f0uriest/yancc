"""Tests for MDKE operators."""

import numpy as np
import pytest

import yancc.trajectories as trajectories
import yancc.trajectories_scipy as trajectories_scipy


@pytest.mark.parametrize("p1", ["1a", "4b"])
@pytest.mark.parametrize("p2", [2, 4])
@pytest.mark.parametrize("E_psi", [1e-3, 1e3])
@pytest.mark.parametrize("nu", [1e-3, 1e3])
@pytest.mark.parametrize("gauge", [True, False])
def test_scipy_operators(p1, p2, E_psi, nu, gauge, field, pitchgrid):
    """Test that scipy sparse matrices are the same as jax jacobians."""
    A1 = trajectories_scipy.mdke(field, pitchgrid, E_psi, nu, p1, p2, gauge=gauge)
    A2 = trajectories.MDKE(field, pitchgrid, E_psi, nu, p1, p2, "atz", gauge=gauge)
    np.testing.assert_allclose(A1.toarray(), A2.as_matrix())
