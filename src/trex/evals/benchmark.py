"""Benchmarking script.

Compare TREX (with landscape-aware loss)
against Sankoff on datasets generated from an NK fitness landscape.
This version extends the previous benchmark to include a landscape-aware
loss function for TREX, controlled by a hyperparameter λ (lambda).
When λ=0, TREX optimizes only for parsimony (as before).
When λ>0, TREX also considers the fitness of reconstructed ancestral
sequences according to the NK landscape.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import custom_vjp

# Add this with your other imports
from jaxtyping import PRNGKeyArray, PyTree

from trex.nk_model import create_nk_model_landscape, generate_tree_data
from trex.sankoff import run_sankoff
from trex.tree import compute_loss, compute_surrogate_cost, update_seq
from trex.utils.types import (
  EvoSequence,
  OneHotEvoSequence,
)


@custom_vjp
def straight_through_estimator(soft_sequence: OneHotEvoSequence) -> EvoSequence:
  """Apply argmax in the forward pass.

  The custom VJP rule below will handle the backward pass.
  """
  return jnp.argmax(soft_sequence, axis=-1)


def ste_fwd(soft_sequence: OneHotEvoSequence) -> tuple[jax.Array, OneHotEvoSequence]:
  """Forward pass for the straight-through estimator (STE).

  It applies argmax to get discrete outputs but saves the soft input
  for use in the backward pass.
  """
  primal_out = straight_through_estimator(soft_sequence)
  return primal_out, soft_sequence


# Define the backward pass for the VJP
def ste_bwd(residuals: OneHotEvoSequence, grad_primal_out: jax.Array) -> tuple[jax.Array]:
  """Backward pass for the straight-through estimator (STE).

  It takes the incoming gradient from the next layer (grad_primal_out)
  and passes it back to the input of the argmax (soft_sequence).
  """
  soft_sequence = residuals  # Get the input from the forward pass

  # Create a one-hot mask from the winning indices
  indices = jnp.argmax(soft_sequence, axis=-1)
  mask = jax.nn.one_hot(indices, num_classes=soft_sequence.shape[-1])

  # "Scatter" the incoming gradient, placing it at the location of the
  # winning neuron. The gradient needs to be expanded to broadcast correctly.
  # The result is a tuple, as VJPs can have multiple inputs.
  return (jnp.expand_dims(grad_primal_out, axis=-1) * mask,)


# Register the forward and backward passes with the custom VJP
straight_through_estimator.defvjp(ste_fwd, ste_bwd)


def compute_parental_logits(
  parent_sequences: OneHotEvoSequence,
  landscape: PyTree,
) -> jax.Array:
  """Compute 'parental logits' for each site for a batch of parent sequences.

  The logit for a state at site 'i' is the expected fitness at 'i'
  assuming 'i' is in that state, marginalized over the parent's own
  probabilities for the other interacting sites.

  Returns a tensor of shape (num_parents, seq_len, n_states).
  """
  n_parents, seq_len, n_states = parent_sequences.shape
  fitness_tables = landscape["fitness_tables"]  # Shape: (N, 2**(k+1))
  interactions = landscape["interactions"]  # Shape: (N, k)
  k = interactions.shape[1]

  # Vmap the entire calculation over all N sites of the sequence
  def compute_logits_for_one_site(i, parent_seqs):
    # i: index of the site we're computing logits for
    # parent_seqs: all parent sequences, shape (n_parents, seq_len, n_states)

    # 1. Get the k neighbors for site i
    neighbors = interactions[i]

    # 2. Get the parent's soft probabilities for those neighbors
    # Shape: (n_parents, k, n_states)
    neighbor_probs = parent_seqs[:, neighbors, :]

    # 3. Compute the joint probability distribution over the neighbors' states
    # Starts with shape (n_parents, n_states)
    joint_neighbor_probs = neighbor_probs[:, 0, :]
    for j in range(1, k):
      # Outer product using einsum
      next_neighbor = neighbor_probs[:, j, :]
      combined = jnp.einsum("pc,ps->pcs", joint_neighbor_probs, next_neighbor)
      joint_neighbor_probs = combined.reshape(n_parents, -1)

    # 4. Reshape the fitness table for site i
    # The table encodes fitness F(state_i, state_neighbors).
    # We reshape it to (n_states, 2**k) to separate the two dimensions.
    site_fitness_table = fitness_tables[i].reshape(n_states, -1)

    # 5. Compute the marginalized fitness (our logits)
    # This is the matrix-vector product of the fitness table and the neighbor probabilities.
    # It calculates the expected fitness for each state of site 'i'.
    # (n_states, 2**k) @ (n_parents, 2**k)^T -> (n_states, n_parents) -> (n_parents, n_states)
    return jnp.einsum("si,pi->ps", site_fitness_table, joint_neighbor_probs)

  # Use vmap to apply this logic to all N sites in parallel.
  # We are mapping over the integer index of the site.
  all_logits = jax.vmap(compute_logits_for_one_site, in_axes=(0, None))(
    jnp.arange(seq_len),
    parent_sequences,
  )

  # The vmap output is (seq_len, n_parents, n_states), so we transpose it
  return jnp.transpose(all_logits, (1, 0, 2))


def compute_loss_landscape_aware(
  key: PRNGKeyArray,
  params: dict,
  sequences: OneHotEvoSequence,
  n_leaves: int,
  landscape: PyTree,
  adj_matrix: jax.Array,
  temperature: float,
  n_all: int,
  lambda_val: float,
  *,
  fix_tree: bool = True,
) -> jax.Array:
  """Compute surrogate cost + parental guidance cross-entropy cost."""
  updated_sequences = update_seq(params, sequences, temperature)

  # 1. Parsimony Cost (same as before)
  surrogate_cost = compute_surrogate_cost(updated_sequences, adj_matrix)

  # 2. Parental Guidance Fitness Cost
  # Find the parent of each node
  parent_indices = jnp.argmax(adj_matrix, axis=1)

  # Get the soft sequences for all parents and all children
  parent_soft_seqs = updated_sequences[parent_indices]
  child_soft_seqs = updated_sequences

  # Compute the fitness-derived logits from the parents
  parent_logits = compute_parental_logits(parent_soft_seqs, landscape)

  # Calculate cross-entropy loss: -Σ y * log(softmax(x))
  # where y is the child distribution and x is the parent logits.
  log_predictions = jax.nn.log_softmax(parent_logits, axis=-1)
  cross_entropy = -jnp.sum(child_soft_seqs * log_predictions)

  is_root = jnp.arange(n_all) == parent_indices
  fitness_cost = cross_entropy / jnp.sum(~is_root)
  # jax.debug.print("Surrogate Cost: {c1}, Fitness Cost: {c2}", c1=surrogate_cost, c2=fitness_cost)

  return surrogate_cost + lambda_val * fitness_cost


def run_trex_landscape_aware(
  leaf_sequences: jax.Array,
  n_all: int,
  n_leaves: int,
  n_states: int,
  landscape: PyTree,
  lambda_val: float,
  adj_matrix: jax.Array,
  key: PRNGKeyArray,
) -> jax.Array:
  """Run the TREX optimization with the landscape-aware loss."""
  key, subkey = jax.random.split(key)
  params = {
    "tree_params": adj_matrix,
    "ancestors": [
      jax.random.normal(subkey, (leaf_sequences.shape[1], n_states))
      for _ in range(n_all - n_leaves)
    ],
  }

  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
  )
  opt_state = optimizer.init(params)

  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states)
  masked_sequences = jnp.concatenate(
    [leaf_sequences_one_hot, jnp.zeros((n_all - n_leaves, leaf_sequences.shape[1], n_states))],
    axis=0,
  ).astype(jnp.float32)

  loss_and_grad = jax.jit(
    jax.value_and_grad(compute_loss_landscape_aware, argnums=1),
    static_argnames=("fix_tree", "n_all", "n_leaves"),
  )

  # Training loop
  def training_step(
    i: jax.Array,
    carry: tuple[dict, optax.OptState, PRNGKeyArray],
  ) -> tuple[dict, optax.OptState, PRNGKeyArray]:
    params, opt_state, key = carry
    _, grads = loss_and_grad(
      key,
      params,
      masked_sequences,
      n_leaves=n_leaves,  # Pass as static keyword arg
      landscape=landscape,
      temperature=1.0,
      n_all=n_all,
      lambda_val=lambda_val,
      fix_tree=True,
      adj_matrix=adj_matrix,
    )
    # jax.debug.print("Grads: {grads}", grads=grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = cast("dict[str, jax.Array | list[jax.Array]]", (optax.apply_updates(params, updates)))
    return params, opt_state, key

  params, _, _ = jax.lax.fori_loop(
    0,
    10000,
    training_step,
    (params, opt_state, key),
  )

  return jnp.argmax(jnp.stack(params["ancestors"]), axis=-1)


def create_balanced_binary_tree(n_leaves: int) -> jax.Array:
  """Create a balanced binary tree with n_leaves."""
  n_ancestors = n_leaves - 1
  n_total = n_leaves + n_ancestors
  adj = jnp.zeros((n_total, n_total))

  leaf_parents = n_leaves + jnp.arange(n_leaves) // 2
  ancestor_parents = n_leaves + (jnp.arange(n_ancestors - 1) + n_leaves) // 2

  adj = adj.at[jnp.arange(n_leaves), leaf_parents].set(1)
  return adj.at[n_leaves + jnp.arange(n_ancestors - 1), ancestor_parents].set(1)


def run_trex_optimization(
  leaf_sequences: jax.Array,
  n_all: int,
  n_leaves: int,
  n_states: int,
  adj_matrix: jax.Array,
  key: PRNGKeyArray,
) -> jax.Array:
  """Run the trex optimization to reconstruct ancestral sequences."""
  key, subkey = jax.random.split(key)
  params = {
    "tree_params": adj_matrix,
    "ancestors": [
      jax.random.normal(subkey, (leaf_sequences.shape[1], n_states))
      for _ in range(n_all - n_leaves)
    ],
  }

  # Optimizer
  optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
  )
  opt_state = optimizer.init(params)

  # one-hot encode leaf sequences
  leaf_sequences_one_hot = jax.nn.one_hot(leaf_sequences, n_states)
  # create a masked sequence object where the ancestor sequences are zero
  masked_sequences = jnp.concatenate(
    [leaf_sequences_one_hot, jnp.zeros((n_all - n_leaves, leaf_sequences.shape[1], n_states))],
    axis=0,
  ).astype(jnp.float32)

  # metadata
  metadata = {
    "n_all": n_all,
    "n_leaves": n_leaves,
    "n_states": n_states,
  }

  # JIT compile the loss and gradient calculation
  loss_and_grad = jax.jit(
    jax.value_and_grad(compute_loss, argnums=1),
    static_argnames=("fix_tree",),
  )

  # Training loop
  def training_step(
    i: jax.Array,
    carry: tuple[dict, optax.OptState, PRNGKeyArray],
  ) -> tuple[dict, optax.OptState, PRNGKeyArray]:
    params, opt_state, key = carry
    _, grads = loss_and_grad(
      key,
      params,
      masked_sequences,
      metadata,
      temperature=1.0,
      adjacency=adj_matrix,
      fix_tree=True,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = cast(
      "dict[str, jax.Array | list[jax.Array]]",
      (optax.apply_updates(params, updates)),
    )
    return params, opt_state, key

  params, _, _ = jax.lax.fori_loop(
    0,
    10000,
    training_step,
    (params, opt_state, key),
  )
  # Get the reconstructed ancestors
  return jnp.argmax(jnp.stack(params["ancestors"]), axis=-1)


# Modify the benchmark function to take lambdas and return a dict
def benchmark(
  n: int,
  k: int,
  n_leaves: int,
  mutation_rate: float,
  lambda_values: list[float],
  key: PRNGKeyArray,
) -> dict[str, Any]:
  """Benchmarks Sankoff and TREX (for various lambdas) on a single dataset."""
  results = {}

  # 1. Generate a single dataset for this benchmark run
  adj_matrix = create_balanced_binary_tree(n_leaves)
  n_all = adj_matrix.shape[0]
  cost_matrix = jnp.ones((2, 2)) - jnp.eye(2)
  key, subkey = jax.random.split(key)
  landscape = create_nk_model_landscape(n, k, subkey)
  key, subkey = jax.random.split(key)
  root_sequence = jax.random.randint(subkey, (n, 1), 0, 2)
  key, subkey = jax.random.split(key)
  tree_data = generate_tree_data(
    landscape,
    adj_matrix,
    root_sequence,
    mutation_rate,
    subkey,
    coupled_mutation_prob=0.5,
  )
  leaf_sequences = tree_data.all_sequences[:n_leaves].astype(jnp.int32)
  true_ancestors = tree_data.all_sequences[n_leaves:].astype(jnp.int32)

  # 2. Run Sankoff (only once per dataset)
  reconstructed_sankoff, _, _ = run_sankoff(
    adj_matrix,
    cost_matrix,
    leaf_sequences,
    n_all=n_all,
    n_states=2,
    n_leaves=n_leaves,
    return_path=True,
  )
  sankoff_ancestors = reconstructed_sankoff[n_leaves:]
  results["sankoff"] = jnp.mean(true_ancestors == sankoff_ancestors)

  # 3. Run TREX for each lambda value
  results["trex"] = {}
  for lambda_val in lambda_values:
    key, subkey = jax.random.split(key)

    # λ=0 is the original parsimony-only TREX
    if lambda_val == 0.0:
      reconstructed_trex = run_trex_optimization(
        leaf_sequences,
        n_all,
        n_leaves,
        2,
        adj_matrix,
        subkey,
      )
    else:
      reconstructed_trex = run_trex_landscape_aware(
        leaf_sequences,
        n_all,
        n_leaves,
        2,
        landscape,
        lambda_val,
        adj_matrix,
        subkey,
      )

    trex_accuracy = jnp.mean(true_ancestors == reconstructed_trex)
    results["trex"][lambda_val] = trex_accuracy

  return results


# Update the main execution block
if __name__ == "__main__":
  N = 15
  K_values = [1, 2, 5, 10]
  lambda_values = [0.0, 0.3, 3.0]  # Test these lambda values
  n_leaves = 32
  mutation_rate = 0.1
  num_replicates = 2  # Use a smaller number for faster testing

  # Setup storage for results
  all_results = {K: {"sankoff": [], "trex": {L: [] for L in lambda_values}} for K in K_values}

  key = jax.random.PRNGKey(0)
  print(f"Running benchmark for {num_replicates} replicates...")

  for i in range(num_replicates):
    print(f"  Replicate {i + 1}/{num_replicates}")
    for K in K_values:
      key, subkey = jax.random.split(key)
      # Run one full benchmark for a given K
      rep_results = benchmark(N, K, n_leaves, mutation_rate, lambda_values, subkey)

      # Store results
      all_results[K]["sankoff"].append(rep_results["sankoff"])
      for L in lambda_values:
        all_results[K]["trex"][L].append(rep_results["trex"][L])

  # 6. Plot results
  import matplotlib.pyplot as plt
  import numpy as np

  plt.figure(figsize=(10, 7))

  # Plot Sankoff
  sankoff_means = [np.mean(all_results[K]["sankoff"]) for K in K_values]
  sankoff_stds = [np.std(all_results[K]["sankoff"]) for K in K_values]
  plt.errorbar(
    K_values,
    sankoff_means,
    yerr=sankoff_stds,
    label="Sankoff",
    capsize=5,
    marker="o",
    zorder=10,
  )

  # Plot TREX for each lambda
  for L in lambda_values:
    trex_means = [np.mean(all_results[K]["trex"][L]) for K in K_values]
    trex_stds = [np.std(all_results[K]["trex"][L]) for K in K_values]
    label = f"TREX (λ={L})" if L > 0 else "TREX (Parsimony)"
    plt.errorbar(
      K_values,
      trex_means,
      yerr=trex_stds,
      label=label,
      capsize=5,
      marker="o",
      linestyle="--",
    )

  plt.xlabel("K (epistatic interactions)")
  plt.ylabel("Mean Accuracy")
  plt.title(f"Algorithm Comparison (Avg. over {num_replicates} replicates)")
  plt.legend()
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.savefig("benchmark_lambda_comparison.png")
  print("Benchmark complete. Plot saved to benchmark_lambda_comparison.png")
