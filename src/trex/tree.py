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
  approximation of a discrete tree structure. This version is fully vectorized
  to provide stable gradients.
  """
  tree_params = params["tree_params"]
  n_all_minus_1, n_ancestors = tree_params.shape
  n_total_nodes = n_all_minus_1 + 1
  n_leaves = n_total_nodes - n_ancestors

  if n_ancestors == 0:
    return jnp.eye(n_total_nodes, dtype=tree_params.dtype)

  gumbel_noise = jax.random.gumbel(key, shape=tree_params.shape)
  perturbed_params = (tree_params + gumbel_noise) / temperature

  # Start with a matrix of -infinity, which corresponds to zero probability after softmax
  final_logits = jnp.full((n_total_nodes, n_total_nodes), -jnp.inf)

  # 1. Populate leaf-to-ancestor logits (n_leaves x n_ancestors)
  leaf_logits = perturbed_params[:n_leaves]
  final_logits = final_logits.at[:n_leaves, n_leaves:].set(leaf_logits)

  # 2. Populate ancestor-to-ancestor logits
  # An ancestor i can only be a child of an ancestor j if j > i.
  # This creates an upper-triangular structure for the ancestor block.
  ancestor_logits = perturbed_params[n_leaves:]

  # Create a mask to enforce the acyclic, upper-triangular constraint.
  # The mask shape is (n_ancestors - 1, n_ancestors) to match the logits.
  i = jnp.arange(n_ancestors - 1)[:, None]
  j = jnp.arange(n_ancestors)[None, :]
  # The mask is True where column index j is strictly greater than row index i
  # (relative to the ancestor-only block).
  ancestor_mask = j > i

  # Apply the mask: invalid connections become -inf.
  masked_ancestor_logits = jnp.where(ancestor_mask, ancestor_logits, -jnp.inf)
  final_logits = final_logits.at[n_leaves:-1, n_leaves:].set(masked_ancestor_logits)

  # 3. The root has no parent, so it points to itself before the softmax.
  final_logits = final_logits.at[-1, -1].set(1.0)

  return nn.softmax(final_logits, axis=1)


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
  sequences: jnp.ndarray,  # (n_nodes, seq_len, n_states)
  adjacency: jnp.ndarray,  # (n_nodes, n_nodes)
) -> jnp.ndarray:
  """Compute a differentiable surrogate for the tree traversal cost.

  This version is the direct differentiable analog of the parsimony score,
  """
  diff = sequences[:, jnp.newaxis, ...] - sequences[jnp.newaxis, :, ...]

  squared_distance = jnp.sum(diff**2, axis=(-1, -2))

  return jnp.sum(squared_distance * adjacency) / 2


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
  # Convert inputs to discrete integer representations
  sequences_int = jnp.argmax(sequences, axis=2)
  discrete_adj = discretize_tree_topology(adjacency, adjacency.shape[0])

  # For each node `i`, find the index of its parent
  parent_indices = jnp.argmax(discrete_adj, axis=1)

  # Gather the sequences for all parents
  parent_seqs = sequences_int[parent_indices]

  return substitution_matrix[parent_seqs, sequences_int][:-1, :].sum()


def compute_loss(
  key: PRNGKeyArray,
  params: dict[str, Array | list[Array]],
  sequences: BatchEvoSequence,
  metadata: GroundTruthMetadata,
  temperature: float,
  adjacency: jax.Array,
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
      graph_constraint_scale: Scaling factor for the tree constraint loss.
      verbose: If True, returns a detailed breakdown of the loss components.
      fix_seqs: If True, do not update sequences (only update tree).
      fix_tree: If True, do not update tree (only update sequences).

  Returns:
      The total loss, or a tuple of loss components if verbose is True.

  """
  updated_sequences = sequences if fix_seqs else update_seq(params, sequences, temperature)
  key, update_tree_key = jax.random.split(key)
  updated_tree_topology = adjacency if fix_tree else update_tree(update_tree_key, params)

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
