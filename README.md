# phylojax

Phylogenetic likelihood computation in [JAX](https://github.com/google/jax). Implements nucleotide substitution models and the pruning algorithm for computing likelihoods and their gradients on fixed tree topologies.

Used as one of the benchmark methods in the [TreeFlow paper](https://github.com/christiaanjs/treeflow-paper).

## Features

- **Substitution models**: JC (Jukes–Cantor), HKY, GTR
- **Site rate variation**: discrete Gamma, discrete Weibull
- **Likelihood**: pruning algorithm with support for site pattern compression
- **Gradients**: automatic differentiation via JAX

## Installation

```bash
pip install -e .
```

Requires `jax` (installed automatically) and `tensorflow-probability` (for the JAX substrate, used by the discrete rate variation distributions).

## Usage

```python
import numpy as np
import phylojax.substitution
import phylojax.likelihood
import phylojax.site_rate_variation

# Create a substitution model
subst_model = phylojax.substitution.HKY(
    frequencies=np.array([0.23, 0.27, 0.24, 0.26]),
    kappa=np.array(2.0)
)

# Compute log-likelihood given a tree topology and alignment
likelihood = phylojax.likelihood.JaxLikelihood(
    topology_dict,       # tree topology with postorder traversal
    sequences_encoded,   # one-hot encoded alignment (taxon x pattern x 4)
    subst_model,
    pattern_counts=weights  # site pattern counts
)
log_ll = likelihood.log_likelihood(branch_lengths)

# With discrete Gamma rate variation
weights, rates = phylojax.site_rate_variation.get_discrete_gamma_weights_rates(
    category_count=4, site_gamma_shape=1.0
)
likelihood = phylojax.likelihood.JaxLikelihood(
    topology_dict, sequences_encoded, subst_model,
    pattern_counts=weights,
    category_weights=weights,
    category_rates=rates
)
```

## Tests

```bash
pip install pytest
pytest
```
