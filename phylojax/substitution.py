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

    def q_norm(self):
        q = self.q()
        normalising_constant = -np.sum(np.diagonal(q) * self.pi)
        return q / normalising_constant


def stack_matrix(mat: tp.Iterable[tp.Iterable[Array]]):
    return np.stack([np.stack(row) for row in mat])


class JC(SubstitutionModel):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        super().__init__(np.full(4, 0.25, dtype=dtype))
        self._q = np.array(
            [
                [-1, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, -1, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, -1, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3, -1],
            ],
            dtype=dtype,
        )
        self._eigenvectors = np.array(
            [
                [1.0, 2.0, 0.0, 0.5],
                [1.0, -2.0, 0.5, 0.0],
                [1.0, 2.0, 0.0, -0.5],
                [1.0, -2.0, -0.5, 0.0],
            ],
            dtype=self.dtype,
        )
        self._eigenvalues = np.array(
            [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333],
            dtype=dtype,
        )
        self._inverse_eigenvectors = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.125, -0.125, 0.125, -0.125],
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0, 0.0],
            ],
            dtype=dtype,
        )

    def q(self):
        return self._q

    def transition_probs(self, t: Array) -> Array:
        return eigen_transition_probs(
            self._eigenvectors, self._eigenvalues, self._inverse_eigenvectors, t
        )


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


class GTR(SubstitutionModel):
    def __init__(self, frequencies, rates):
        super().__init__(frequencies)
        self.rates = np.array(rates)

    def q(self) -> Array:
        pi = self.pi
        rates = self.rates
        return stack_matrix(
            [
                [
                    -(
                        rates[..., 0] * pi[..., 1]
                        + rates[..., 1] * pi[..., 2]
                        + rates[..., 2] * pi[..., 3]
                    ),
                    rates[..., 0] * pi[..., 1],
                    rates[..., 1] * pi[..., 2],
                    rates[..., 2] * pi[..., 3],
                ],
                [
                    rates[..., 0] * pi[..., 0],
                    -(
                        rates[..., 0] * pi[..., 0]
                        + rates[..., 3] * pi[..., 2]
                        + rates[..., 4] * pi[..., 3]
                    ),
                    rates[..., 3] * pi[..., 2],
                    rates[..., 4] * pi[..., 3],
                ],
                [
                    rates[..., 1] * pi[..., 0],
                    rates[..., 3] * pi[..., 1],
                    -(
                        rates[..., 1] * pi[..., 0]
                        + rates[..., 3] * pi[..., 1]
                        + rates[..., 5] * pi[..., 3]
                    ),
                    rates[..., 5] * pi[..., 3],
                ],
                [
                    rates[..., 2] * pi[..., 0],
                    rates[..., 4] * pi[..., 1],
                    rates[..., 5] * pi[..., 2],
                    -(
                        rates[..., 2] * pi[..., 0]
                        + rates[..., 4] * pi[..., 1]
                        + rates[..., 5] * pi[..., 2]
                    ),
                ],
            ]
        )

    def transition_probs(self, t: Array) -> Array:
        frequencies = self.pi
        q_norm = self.q_norm()
        sqrt_frequencies = np.sqrt(frequencies)
        inverse_sqrt_frequencies = 1.0 / sqrt_frequencies
        sqrt_frequencies_diag_matrix = np.diag(sqrt_frequencies)
        inverse_sqrt_frequencies_diag_matrix = np.diag(inverse_sqrt_frequencies)

        symmetric_matrix = (
            sqrt_frequencies_diag_matrix @ q_norm @ inverse_sqrt_frequencies_diag_matrix
        )
        eigenvalues, s_eigenvectors = np.linalg.eigh(symmetric_matrix)
        eigenvectors = inverse_sqrt_frequencies_diag_matrix @ s_eigenvectors
        inverse_eigenvectors = s_eigenvectors.T @ sqrt_frequencies_diag_matrix
        return eigen_transition_probs(
            eigenvectors, eigenvalues, inverse_eigenvectors, t
        )
