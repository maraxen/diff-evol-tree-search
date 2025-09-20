"""Functions for creating, manipulating, and calculating losses on phylogenetic trees."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import nn
from jaxtyping import Array, PRNGKeyArray

from trex.utils.types import (
  AdjacencyMatrix,
  BatchEvoSequence,
  BatchOneHotEvoSequence,
  Cost,
  GroundTruthMetadata,
  OneHotEvoSequence,
  SubstitutionMatrix,
)


@partial(jax.jit, static_argnames=("n_nodes",))
def discretize_tree_topology(
  adjacency: AdjacencyMatrix,
  n_nodes: int,
) -> AdjacencyMatrix:
  """Discretize a soft tree topology into a one-hot representation.

  Args:
      adjacency: A soft tree topology (e.g., after softmax).
      n_nodes: The total number of nodes in the tree.

  Returns:
      A one-hot encoded, discrete tree topology.

  """
  max_indices = jnp.argmax(adjacency, axis=1)
  return nn.one_hot(max_indices, n_nodes)


@jax.jit
def update_tree(
  key: PRNGKeyArray,
  params: dict[str, Array],
  temperature: float = 1.0,
) -> AdjacencyMatrix:
  """Update and return a soft tree topology using trainable parameters.

  This function uses the Gumbel-Softmax trick to produce a differentiable
  approximation of a discrete tree structure.

  Args:
      params: A dictionary containing trainable tree parameters under the key 'tree_params'.
      epoch: The current epoch number, used to seed the random key.
      temperature: The temperature for the Gumbel-Softmax.

  Returns:
      An updated soft tree topology.

  """
  tree_params = params["tree_params"]
  n_total_nodes = tree_params.shape[0] + 1
  n_ancestors = tree_params.shape[1]
  n_leaves = n_total_nodes - n_ancestors

  if n_ancestors == 0 or tree_params.shape[0] == 0:
    return jnp.eye(n_total_nodes, dtype=tree_params.dtype)

  gumbel_noise = jax.random.gumbel(key, shape=tree_params.shape)

  perturbed_params = (tree_params + gumbel_noise) / temperature

  masked_params = jnp.full((n_total_nodes, n_total_nodes), -jnp.inf)

  masked_params = masked_params.at[:n_ancestors, n_leaves:].set(
    perturbed_params[:n_ancestors],
  )
  upper_indices = jnp.triu_indices(n_ancestors)
  shifted_rows = upper_indices[0] + n_ancestors
  shifted_cols = upper_indices[1] + n_leaves
  shifted_indices = (shifted_rows, shifted_cols)

  masked_params = masked_params.at[shifted_indices].set(
    perturbed_params[n_ancestors:, :][upper_indices],
  )

  masked_params = masked_params.at[-1, -1].set(1.0)
  return nn.softmax(masked_params, axis=1)


@jax.jit
def update_seq(
  params: dict[str, Array | list[Array]],
  sequences: BatchEvoSequence,
  temperature: float = 1.0,
) -> BatchEvoSequence:
  """Update ancestor sequences using trainable parameters.

  Args:
      params: Dictionary with trainable parameters for each ancestor sequence.
      sequences: The batch of sequences, where leaves are fixed and ancestors are updated.
      temperature: Softmax temperature.

  Returns:
      The updated batch of sequences.

  """
  n_total_nodes = sequences.shape[0]
  n_leaf_nodes = (n_total_nodes + 1) // 2
  updated_ancestors = nn.softmax(jnp.stack(params["ancestors"]) * temperature, axis=-1)
  return sequences.at[n_leaf_nodes:].set(updated_ancestors)


@jax.jit
def enforce_graph_constraints(
  adjacency: AdjacencyMatrix,
  scaling_factor: float,
) -> Cost:
  """Calculate a loss to enforce that the tree is binary.

  This loss penalizes deviations from the constraint that each ancestor node
  should have exactly two children.

  Args:
      adjacency: The soft tree topology.
      scaling_factor: A scaling factor for the loss.

  Returns:
      The calculated constraint-forcing loss.

  """
  n_total_nodes = adjacency.shape[0]
  n_ancestor_nodes = (n_total_nodes - 1) // 2
  ancestor_columns = adjacency[:-1, -n_ancestor_nodes:]

  return jnp.sum(jnp.power(scaling_factor * jnp.abs(jnp.sum(ancestor_columns, axis=0) - 2), 2))


@jax.jit
def compute_surrogate_cost(
  sequences: BatchOneHotEvoSequence,
  adjacency: AdjacencyMatrix,
) -> Cost:
  """Compute a differentiable surrogate for the tree traversal cost.

  Args:
      sequences: A one-hot encoded batch of sequences.
      adjacency: The soft tree topology.

  Returns:
      The surrogate traversal cost.

  """

  def surrogate_cost(
    one_hot_sequence: OneHotEvoSequence,  # Shape: (n_nodes, n_states)
    adjacency: AdjacencyMatrix,
  ) -> Cost:
    """Compute the surrogate cost for a single sequence position."""
    parent_sequence = jnp.matmul(adjacency, one_hot_sequence)
    return jnp.sum(jnp.abs(parent_sequence - one_hot_sequence)) / 2

  costs = jax.vmap(surrogate_cost, in_axes=(1, None))(sequences, adjacency)
  return jnp.sum(costs)


@jax.jit
def compute_cost(
  sequences: BatchOneHotEvoSequence,
  adjacency: AdjacencyMatrix,
  substitution_matrix: SubstitutionMatrix,
) -> Cost:
  """Compute the exact, non-differentiable traversal cost of a tree.

  Args:
      sequences: One-hot encoded sequences.
      adjacency: The tree topology (can be soft, will be discretized).
      substitution_matrix: The substitution matrix (e.g., Hamming distance).

  Returns:
      The exact parsimony score.

  """
  discrete_tree_topology = discretize_tree_topology(adjacency, adjacency.shape[0])
  discrete_sequences = jnp.argmax(sequences, axis=2)

  parent_sequences = jnp.matmul(discrete_tree_topology, discrete_sequences)

  return substitution_matrix[
    jnp.round(parent_sequences).astype(jnp.int8),
    jnp.round(discrete_sequences).astype(jnp.int8),
  ].sum()


def compute_loss(  # noqa: PLR0913
  key: PRNGKeyArray,
  params: dict[str, Array | list[Array]],
  sequences: BatchEvoSequence,
  metadata: GroundTruthMetadata,
  temperature: float,
  *,
  graph_constraint_scale: float = 10.0,
  verbose: bool = False,
  fix_seqs: bool = False,
  fix_tree: bool = False,
) -> (
  Cost
  | tuple[
    Cost,
    ...,
  ]
):
  """Compute the total loss for optimizing tree and/or sequences.

  Args:
      params: Trainable parameters for the tree and sequences.
      sequences: The initial sequences (leaves are fixed).
      metadata: Dictionary containing metadata.
      temperature: Temperature for softmax and loss scaling.
      epoch: The current training epoch.
      verbose: If True, returns a detailed breakdown of the loss components.
      fix_seqs: If True, do not update sequences (only update tree).
      fix_tree: If True, do not update tree (only update sequences).

  Returns:
      The total loss, or a tuple of loss components if verbose is True.

  """
  updated_sequences = sequences if fix_seqs else update_seq(params, sequences, temperature)
  key, update_tree_key = jax.random.split(key)
  updated_tree_topology = (
    jnp.eye(metadata["n_all"]) if fix_tree else update_tree(update_tree_key, params)
  )

  surrogate_cost = compute_surrogate_cost(updated_sequences, updated_tree_topology)
  tree_constraint_loss = enforce_graph_constraints(updated_tree_topology, graph_constraint_scale)
  total_loss = surrogate_cost + temperature * tree_constraint_loss

  if verbose:
    jax.debug.print(
      "Updated sequences min/max: {}/{}",
      jnp.min(updated_sequences),
      jnp.max(updated_sequences),
    )
    jax.debug.print(
      "Updated tree min/max: {}/{}",
      jnp.min(updated_tree_topology),
      jnp.max(updated_tree_topology),
    )
    jax.debug.print("NaN in sequences? {}", jnp.any(jnp.isnan(updated_sequences)))
    jax.debug.print("NaN in tree? {}", jnp.any(jnp.isnan(updated_tree_topology)))
    jax.debug.print("Surrogate cost: {}", surrogate_cost)
    jax.debug.print("Tree constraint loss: {}", tree_constraint_loss)
    jax.debug.print("Total loss: {}", total_loss)

  return total_loss
