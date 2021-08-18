from typing import AsyncIterator
import pytest
from phylojax import Array
import phylojax.substitution
import jax.numpy as np
from scipy.linalg import expm
from numpy.testing import assert_allclose
import jax
from phylojax.math import expm


def matexp_transition_probs(q: Array, t: Array):
    return expm(np.expand_dims(t, (-1, -2)) * q)


@pytest.mark.parametrize(
    "subst_model",
    [phylojax.substitution.HKY(np.array([0.23, 0.26, 0.27, 0.24]), np.array(2.0))],
)
@pytest.mark.parametrize(
    "t", [np.array([1.0, 0.2, 0.4]), np.array([[0.1, 0.2], [0.3, 0.6]])]
)
def test_hky_transition_probs(
    subst_model: phylojax.substitution.SubstitutionModel, t: Array
):
    branch_lengths = np.array([1.0, 0.2, 0.4])
    res = subst_model.transition_probs(t)
    expected = matexp_transition_probs(subst_model.q(), t)
    assert_allclose(res, expected, atol=1e-3)
