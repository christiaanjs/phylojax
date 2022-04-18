import typing as tp
from phylojax import Array

import phylojax.substitution
import jax.numpy as np
import jax


class JaxLikelihood:
    def __init__(
        self,
        topology_dict: tp.Dict[str, Array],
        sequences_encoded: Array,
        substitution_model: phylojax.substitution.SubstitutionModel,
        pattern_counts: tp.Optional[Array] = None,
        category_weights: tp.Optional[Array] = None,
        category_rates: tp.Optional[Array] = None,
    ):
        """
        Parameters
        ----------
        topology_dict
            Dictionary with keys:
            - postorder_node_indices
            - child_indices

        sequences_encoded
            Numpy array of onehot-encoded sequences

        substitution_model

        """

        self.taxon_count = taxon_count = (
            len(topology_dict["postorder_node_indices"]) + 1
        )
        self.node_indices = topology_dict["postorder_node_indices"]
        self.child_indices = topology_dict["child_indices"][self.node_indices]

        self.substitution_model = substitution_model

        self.category_weights = (
            np.ones(1) if category_weights is None else np.array(category_weights)
        )
        self.category_rates = (
            np.ones(1) if category_weights is None else np.array(category_rates)
        )

        category_count = self.category_weights.shape[0]
        sequences_shape = sequences_encoded.shape
        self.leaf_partials = np.broadcast_to(
            np.expand_dims(sequences_encoded, -2),
            sequences_shape[:-1] + (category_count,) + sequences_shape[-1:],
        )

        self.pattern_count = sequences_encoded.shape[-2]
        self.pattern_counts = (
            np.ones([self.pattern_count])
            if pattern_counts is None
            else np.array(pattern_counts)
        )

    def log_likelihood(self, branch_lengths: Array):
        """
        branch_lengths
            Shape: [..., node]
        """
        node_first_blens = np.moveaxis(branch_lengths, -1, 0)  # node, ...
        transition_probs = self.substitution_model.transition_probs(
            self.category_rates * np.expand_dims(node_first_blens, -1)
        )  # node, ..., category, parent char, child char
        child_transition_probs = transition_probs[
            self.child_indices
        ]  # node, child, ..., category, parent char, child char
        partials = np.concatenate(
            [
                self.leaf_partials,
                np.zeros((self.taxon_count - 1,) + self.leaf_partials.shape[1:]),
            ]
        )  # Node, ..., pattern, category, char

        for i in range(self.taxon_count - 1):
            node_index = self.node_indices[i]
            node_child_indices = self.child_indices[i]
            node_child_transition_probs = child_transition_probs[
                i
            ]  # Child, ...,category, parent char, child char
            child_partials = partials[
                node_child_indices
            ]  # child, ..., pattern, category, child char
            parent_child_probs = np.expand_dims(
                node_child_transition_probs, -4
            ) * np.expand_dims(
                child_partials, -2
            )  # child, ..., category, pattern, parent char, child char
            node_partials = np.prod(
                np.sum(
                    parent_child_probs,
                    axis=-1,
                ),
                axis=0,
            )
            partials = jax.ops.index_add(partials, node_index, node_partials)
        root_partials = partials[-1]
        cat_likelihoods = np.sum(self.substitution_model.pi * root_partials, axis=-1)
        site_likelihoods = np.sum(self.category_weights * cat_likelihoods, axis=-1)
        return np.sum(np.log(site_likelihoods) * self.pattern_counts, axis=-1)
