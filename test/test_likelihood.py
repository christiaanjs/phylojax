from jax._src.lax.lax import exp
from numpy.testing._private.utils import assert_allclose
import pytest
import pathlib
import pickle
import phylojax.likelihood
import phylojax.substitution
import jax.numpy as np


@pytest.fixture
def test_data_dir():
    return pathlib.Path("test") / "data"


@pytest.fixture
def hello_tree(test_data_dir):
    with open(test_data_dir / "hello-tree.pickle", "rb") as f:
        tree = pickle.load(f)
    return tree


@pytest.fixture
def hello_alignment(test_data_dir):
    with open(test_data_dir / "hello-alignment.pickle", "rb") as f:
        alignment = pickle.load(f)
    return alignment


@pytest.fixture
def hello_hky_likelihood(hello_tree, hello_alignment):
    subst_model = phylojax.substitution.HKY(
        np.array([0.23, 0.27, 0.24, 0.26]), np.array(2.0)
    )
    likelihood = phylojax.likelihood.JaxLikelihood(
        hello_tree["topology"],
        hello_alignment["sequences"],
        subst_model,
        pattern_counts=hello_alignment["weights"],
    )
    return likelihood


def test_likelihood(hello_tree, hello_hky_likelihood):
    expected = -88.86355638556158

    res = hello_hky_likelihood.log_likelihood(hello_tree["branch_lengths"])
    assert_allclose(res, expected)


def test_likelihood_blen_vec(hello_tree, hello_hky_likelihood):
    blens = np.stack(
        [hello_tree["branch_lengths"], hello_tree["branch_lengths"] + 1e-3]
    )
    res = hello_hky_likelihood.log_likelihood(blens)
    expected = [
        hello_hky_likelihood.log_likelihood(blens_element) for blens_element in blens
    ]

    assert_allclose(res, expected)
