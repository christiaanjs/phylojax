import jax.numpy as np
from tensorflow_probability.substrates import jax as tfp_jax


def get_quantile_probabilities(category_count, dtype=np.float64):
    return (2 * np.arange(category_count, dtype=dtype) + 1) / (2 * category_count)


def get_equal_weights(category_count, dtype=np.float64):
    return np.ones(category_count, dtype=dtype) / category_count


def get_discrete_gamma_weights_rates(category_count, site_gamma_shape):
    site_gamma_shape = np.array(site_gamma_shape)
    probs = get_quantile_probabilities(category_count, dtype=site_gamma_shape.dtype)
    dist = tfp_jax.distributions.Gamma(site_gamma_shape, site_gamma_shape)
    rates = dist.quantile(probs)
    weights = get_equal_weights(category_count, dtype=site_gamma_shape.dtype)
    return weights, rates


def get_discrete_weibull_weights_rates(
    category_count,
    site_weibull_concentration,
    site_weibull_scale=None,
):
    probs = get_quantile_probabilities(
        category_count, dtype=site_weibull_concentration.dtype
    )
    if site_weibull_scale is None:
        site_weibull_scale = 1.0

    rates = site_weibull_scale * (-np.log(1 - probs)) ** (
        1 / site_weibull_concentration
    )
    weights = get_equal_weights(category_count, dtype=site_weibull_concentration.dtype)
    return weights, rates
