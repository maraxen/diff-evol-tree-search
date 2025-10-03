from jax import custom_vjp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@custom_vjp
def straight_through_estimator(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Identity function with a straight-through estimator for the gradient."""
    return x


def ste_fwd(x):
    """Forward pass for the STE."""
    return x, None


def ste_bwd(_, grad):
    """Backward pass for the STE."""
    return (grad,)


straight_through_estimator.defvjp(ste_fwd, ste_bwd)


def hard_decision(
    x: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    """Applies a hard threshold at 0, with a straight-through gradient."""
    zero = x - jax.lax.stop_gradient(x)
    return straight_through_estimator(jnp.heaviside(x, 0.5) + zero)