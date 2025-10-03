from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, Float, PyTree

from .components.svm import LinearSVM


class Node(eqx.Module):
    """Abstract base class for nodes in the tree."""


class Leaf(Node):
    """A leaf node in the tree, representing a single class."""

    class_id: int
    num_classes: int

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
        """Returns a one-hot encoded vector for the class.

        Args:
            x: The input data (unused at leaf nodes).

        Returns:
            A one-hot encoded vector representing the leaf's class.
        """
        return jax.nn.one_hot(self.class_id, self.num_classes)


class InternalNode(Node):
    """An internal node in the tree, which splits the data."""

    svm: LinearSVM
    left: Node
    right: Node

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
        """Recursively traverses the tree with soft routing.

        Args:
            x: The input data.

        Returns:
            A probability distribution over the classes.
        """
        svm_output = self.svm(x)
        prob_right = jax.nn.sigmoid(svm_output)

        dist_left = self.left(x)
        dist_right = self.right(x)

        return (1 - prob_right) * dist_left + prob_right * dist_right


class BaseTreeModel(eqx.Module):
    """A tree model with a fixed, hard-coded topology."""

    root: Node
    in_features: int
    num_classes: int

    def __init__(
        self, in_features: int, num_classes: int, *, key: "jax.random.PRNGKey"
    ):
        """Initializes the BaseTreeModel.

        Args:
            in_features: The number of input features.
            num_classes: The number of output classes.
            key: A JAX PRNG key for initializing the SVMs.
        """
        self.in_features = in_features
        self.num_classes = num_classes

        # Need 9 keys for the 9 internal nodes
        keys = jax.random.split(key, 9)

        # Create all leaf nodes
        leaves = [Leaf(i, num_classes) for i in range(num_classes)]

        # Build the tree from the bottom up, following a balanced binary structure
        node_0_1 = InternalNode(
            LinearSVM(in_features, key=keys[0]), leaves[0], leaves[1]
        )
        node_3_4 = InternalNode(
            LinearSVM(in_features, key=keys[1]), leaves[3], leaves[4]
        )
        node_5_6 = InternalNode(
            LinearSVM(in_features, key=keys[2]), leaves[5], leaves[6]
        )
        node_8_9 = InternalNode(
            LinearSVM(in_features, key=keys[3]), leaves[8], leaves[9]
        )

        node_0_1_2 = InternalNode(
            LinearSVM(in_features, key=keys[4]), node_0_1, leaves[2]
        )
        node_5_6_7 = InternalNode(
            LinearSVM(in_features, key=keys[5]), node_5_6, leaves[7]
        )

        node_0_4 = InternalNode(
            LinearSVM(in_features, key=keys[6]), node_0_1_2, node_3_4
        )
        node_5_9 = InternalNode(
            LinearSVM(in_features, key=keys[7]), node_5_6_7, node_8_9
        )

        self.root = InternalNode(LinearSVM(in_features, key=keys[8]), node_0_4, node_5_9)

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
        """Computes the forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The model's output (a probability distribution over classes).
        """
        return self.root(x)


class OvR_SVM_Model(eqx.Module):
    """A One-vs-Rest model using multiple LinearSVMs."""

    svms: list[LinearSVM]

    def __init__(self, in_features: int, num_classes: int, *, key: "jax.random.PRNGKey"):
        """Initializes the OvR_SVM_Model.

        Args:
            in_features: The number of input features.
            num_classes: The number of output classes.
            key: A JAX PRNG key for initializing the SVMs.
        """
        keys = jax.random.split(key, num_classes)
        self.svms = [LinearSVM(in_features, key=k) for k in keys]

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "num_classes"]:
        """Computes the decision function for each class.

        Args:
            x: The input data.

        Returns:
            A vector of decision function outputs, one for each class.
        """
        # Apply each SVM to the input x
        return jnp.stack([svm(x) for svm in self.svms])