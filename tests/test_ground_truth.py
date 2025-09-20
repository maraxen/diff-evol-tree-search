"""Unit tests for ground_truth.py using pytest and chex."""

import chex
import jax
import jax.numpy as jnp
import pytest

from trex.ground_truth import generate_groundtruth, mutate


@pytest.mark.parametrize(
  ("n_states", "n_mutations", "batch_size", "seq_length"),
  [(10, 5, 5, 8), (4, 2, 8, 4)],
)
def test_mutate(n_states, n_mutations, batch_size, seq_length):
  """Tests the _mutate function."""
  key = jax.random.PRNGKey(0)
  # Create a sequence with values less than n_states
  seq = jax.random.randint(key, (batch_size, seq_length), 0, n_states)
  mutated = jax.vmap(
    lambda k, s: mutate(k, s, n_states=n_states, n_mutations=n_mutations),
  )(
    jax.random.split(key, batch_size),
    seq,
  )
  chex.assert_shape(mutated, (batch_size, seq_length))

  # Ensure mutated values are in range and not equal to the original
  assert jnp.all((mutated >= 0) & (mutated < n_states))
  assert not jnp.all(mutated == seq)  # At least some mutations should occur
  assert (
    jnp.sum(mutated != seq) == n_mutations * batch_size
  )  # Each sequence should have exactly n_mutations


@pytest.mark.parametrize(
  ("n_leaves", "seq_length", "n_states", "n_mutations"),
  [(4, 100, 20, 10), (8, 50, 4, 5)],
)
def test_generate_groundtruth(n_leaves, seq_length, n_states, n_mutations):
  """Tests the generate_groundtruth function."""
  tree = generate_groundtruth(n_leaves, n_states, n_mutations, seq_length, seed=123)

  # Check shapes
  n_ancestors = n_leaves - 1
  n_all = n_leaves + n_ancestors
  chex.assert_shape(tree.masked_sequences, (n_all, seq_length))
  chex.assert_shape(tree.all_sequences, (n_all, seq_length))
  chex.assert_shape(tree.adjacency, (n_all, n_all))

  # Check adjacency is binary
  assert jnp.all((tree.adjacency == 0) | (tree.adjacency == 1))

  # Check root is connected to the last two nodes before it
  assert jnp.all(tree.adjacency[:, -1][-3:-1] == 1)

  # Check leaves have sequences in masked_sequences
  assert jnp.any(tree.masked_sequences[:n_leaves] != 0)

  # Check ancestors are masked (i.e., are all zeros)
  assert jnp.all(tree.masked_sequences[n_leaves:] == 0)
  assert jnp.all(tree.masked_sequences[n_leaves:] == 0)

  # 1. Check that the root has exactly two children (the sum of its column is 2)
  assert jnp.sum(tree.adjacency[:, -1]) == 2

  # 2. Check that the root is not its own parent (no self-loop)
  assert tree.adjacency[-1, -1] == 0
