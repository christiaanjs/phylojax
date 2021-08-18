import typing as tp
from phylojax import Array
import jax.numpy as np
from jax.scipy.linalg import expm as _expm

expm: tp.Callable[[Array], Array] = np.vectorize(_expm, signature="(n,n)->(n,n)")


def vec_matmul(a: Array, b: Array) -> Array:
    return np.einsum("...ij,...jk", a, b)
