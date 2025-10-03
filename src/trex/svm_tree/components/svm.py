import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LinearSVM(eqx.Module):
    """A simple linear Support Vector Machine."""

    weights: Float[Array, "in_features"]
    bias: Float[Array, ""]

    def __init__(self, in_features: int, *, key: "jax.random.PRNGKey"):
        """Initializes the LinearSVM module.

        Args:
            in_features: The number of input features.
            key: A JAX PRNG key used to initialize the weights.
        """
        wkey, bkey = jax.random.split(key)
        self.weights = jax.random.normal(wkey, (in_features,))
        self.bias = jax.random.normal(bkey, ())

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, ""]:
        """Computes the decision function for the SVM.

        Args:
            x: The input data.

        Returns:
            The decision function output.
        """
        return jnp.dot(self.weights, x) + self.bias