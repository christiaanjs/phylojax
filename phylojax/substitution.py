from abc import abstractmethod
import typing as tp
from phylojax import Array
import jax
import jax.numpy as np
from phylojax.alphabet import A, C, G, T
from phylojax.math import vec_matmul


def to_diagonal(val):
    n = val.shape[-1]
    i, j = np.diag_indices(n)
    return np.zeros(val.shape[:-1] + (n, n)).at[..., i, j].set(val)


def eigen_transition_probs(U: Array, lambd: Array, Vt: Array, t: Array):
    diag_vals = np.exp(np.expand_dims(t, -1) * lambd)
    diag = to_diagonal(diag_vals)
    return vec_matmul(U, vec_matmul(diag, Vt))


class SubstitutionModel:
    def __init__(self, frequencies: Array):
        self.pi = np.array(frequencies)

    @abstractmethod
    def q(self) -> Array:
        pass

    @abstractmethod
    def transition_probs(self, t: Array) -> Array:
        pass


def stack_matrix(mat: tp.Iterable[tp.Iterable[Array]]):
    return np.stack([np.stack(row) for row in mat])


class HKY(SubstitutionModel):
    def __init__(self, frequencies, kappa: Array):
        super().__init__(frequencies)
        self.kappa = np.array(kappa)

    def q(self) -> Array:
        pi = self.pi
        kappa = self.kappa
        return stack_matrix(
            [
                [-(pi[C] + kappa * pi[G] + pi[T]), pi[C], kappa * pi[G], pi[T]],
                [pi[A], -(pi[A] + pi[G] + kappa * pi[T]), pi[G], kappa * pi[T]],
                [kappa * pi[A], pi[C], -(kappa * pi[A] + pi[C] + pi[T]), pi[T]],
                [pi[A], kappa * pi[C], pi[G], -(pi[A] + kappa * pi[C] + pi[G])],
            ]
        )

    def transition_probs(self, t: Array) -> Array:
        pi = self.pi
        kappa = self.kappa

        piY = pi[T] + pi[C]
        piR = pi[A] + pi[G]

        beta = -1 / (2.0 * (piR * piY + kappa * (pi[A] * pi[G] + pi[C] * pi[T])))
        A_R = 1.0 + piR * (kappa - 1)
        A_Y = 1.0 + piY * (kappa - 1)
        lambd = np.stack([0, beta, beta * A_Y, beta * A_R])  # Eigenvalues
        U = stack_matrix(
            [  # Right eigenvectors as columns (rows of transpose)
                [1, 1, 1, 1],
                [1 / piR, -1 / piY, 1 / piR, -1 / piY],
                [0, pi[T] / piY, 0, -pi[C] / piY],
                [pi[G] / piR, 0, -pi[A] / piR, 0],
            ]
        ).T

        Vt = stack_matrix(
            [  # Left eigenvectors as rows
                [pi[A], pi[C], pi[G], pi[T]],
                [pi[A] * piY, -pi[C] * piR, pi[G] * piY, -pi[T] * piR],
                [0, 1, 0, -1],
                [1, 0, -1, 0],
            ]
        )

        return eigen_transition_probs(U, lambd, Vt, t)
