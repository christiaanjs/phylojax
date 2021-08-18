from jax._src.lax.lax import exp
import pytest
import phylojax.math
import jax.numpy as jnp
from scipy.linalg import expm as _expm
import numpy as np
from numpy.testing import assert_allclose


@pytest.fixture
def test_matrices_2d():
    test_shape = (2, 3, 2, 2)
    test_size = np.prod(test_shape)
    return np.reshape(np.arange(test_size) / test_size, test_shape)


def test_expm_twoIndices(test_matrices_2d):
    expected = np.zeros(test_matrices_2d.shape)
    for i in range(test_matrices_2d.shape[0]):
        for j in range(test_matrices_2d.shape[1]):
            expected[i, j] = _expm(test_matrices_2d[i, j])

    res = phylojax.math.expm(test_matrices_2d)
    assert_allclose(res, expected, rtol=1e-3)
