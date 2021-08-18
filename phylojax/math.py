import jax.numpy as np
from jax.scipy.linalg import expm as _expm

expm = np.vectorize(_expm, signature="(n,n)->(n,n)")


def vec_matmul(a, b):
    return np.einsum("...ij,...jk", a, b)
