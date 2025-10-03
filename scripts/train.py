import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from tqdm import tqdm
from typing import Union

from trex.svm_tree.configs import MNISTConfig, LearnableMNISTConfig
from trex.svm_tree.data_utils import get_mnist_dataloaders
from trex.svm_tree.model import OvR_SVM_Model


def huber_hinge_loss(scores, labels, num_classes=10):
    """
    Computes the multiclass Huber Hinge loss for a One-vs-Rest SVM.
    This function works for both single examples and batches.
    """
    # Convert labels to one-hot and then to +1/-1 format
    labels_one_hot = jax.nn.one_hot(labels, num_classes)
    y_true = (labels_one_hot * 2) - 1

    # Calculate the hinge loss component
    loss = jnp.maximum(0, 1 - y_true * scores)

    # Square it for the Huber-like effect and take the mean
    return jnp.mean(loss**2)


def loss_fn(model, x, y):
    """Computes the Huber Hinge loss for a single example."""
    scores = model(x)
    return huber_hinge_loss(scores, y)


def mean_batch_loss(model, x, y):
    """Computes the mean Huber Hinge loss over an entire batch."""
    # Vmap the single-example loss function across the batch.
    losses = jax.vmap(loss_fn, in_axes=(None, 0, 0))(model, x, y)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model, x, y, optimizer, opt_state):
    """Performs a single training step."""
    # Differentiate the mean batch loss function.
    loss, grads = eqx.filter_value_and_grad(mean_batch_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, x, y):
    """Computes the accuracy on a batch of data for an OvR model."""
    pred_scores = jax.vmap(model)(x)
    # The predicted label is the one with the highest score
    pred_labels = jnp.argmax(pred_scores, axis=1)
    return jnp.mean(pred_labels == y)


def main():
from trex.svm_tree.model import BaseTreeModel, LearnableTreeModel


def main(cfg: Union[MNISTConfig, LearnableMNISTConfig]):
    """Main training and evaluation loop."""
    print(f"Running with configuration: {cfg.__class__.__name__}")
    key = jax.random.PRNGKey(cfg.train.seed)

    # Create DataLoaders
    train_loader, test_loader = get_mnist_dataloaders(
        cfg.data.batch_size,
        cfg.data.train_subset_size,
        cfg.data.test_subset_size,
    )

    # Create model and optimizer
    ovr_model_key, _ = jax.random.split(key)
    ovr_model = OvR_SVM_Model(
        in_features=cfg.model.in_features, num_classes=10, key=model_key
    )
    # Create model
    model_key, train_key, eval_key = jax.random.split(key, 3)
    if isinstance(cfg, LearnableMNISTConfig):
        model = LearnableTreeModel(
            in_features=cfg.model.in_features,
            num_classes=10,
            key=model_key,
            sparsity_regularization_strength=cfg.model.sparsity_regularization_strength,
            graph_constraint_scale=cfg.model.graph_constraint_scale,
        )
    else:
        model = BaseTreeModel(
            in_features=cfg.model.in_features, num_classes=10, key=model_key
        )

    # Create optimizer
    optimizer = optax.adam(cfg.train.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Define loss and step functions
    def loss_fn(model, x, y, key):
        """Computes the total loss."""
        # Use a key for the forward pass if the model requires it
        if isinstance(model, LearnableTreeModel):
            keys = jax.random.split(key, x.shape[0])
            pred_y_batched = jax.vmap(model)(x, key=keys)

            # Compute adjacency matrix and topology loss
            adj = model.topology(key)
            topo_loss = model.loss(adj)
        else:
            pred_y_batched = jax.vmap(model)(x)
            topo_loss = 0.0

        # Cross-entropy loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(
            pred_y_batched, y
        ).mean()

        # Combine losses if applicable
        if isinstance(cfg, LearnableMNISTConfig):
            return ce_loss + cfg.train.topology_loss_weight * topo_loss
        return ce_loss

    @eqx.filter_jit
    def train_step(model, x, y, optimizer, opt_state, key):
        """Performs a single training step."""
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, x, y, key):
        """Computes the accuracy on a batch of data."""
        if isinstance(model, LearnableTreeModel):
            keys = jax.random.split(key, x.shape[0])
            pred_y = jax.vmap(model)(x, key=keys)
        else:
            pred_y = jax.vmap(model)(x)

        pred_labels = jnp.argmax(pred_y, axis=1)
        return jnp.mean(pred_labels == y)

    # Training loop
    for epoch in range(cfg.train.num_epochs):
        train_key, epoch_train_key = jax.random.split(train_key)
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}"):
            x_batch, y_batch = batch
            x_batch = x_batch.numpy()
            y_batch = y_batch.numpy()

            model, opt_state, loss = train_step(
                ovr_model, x_batch, y_batch, optimizer, opt_state

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")
        for batch in pbar:
            epoch_train_key, batch_key = jax.random.split(epoch_train_key)
            x, y = batch
            x = x.numpy()
            y = y.numpy()

            model, opt_state, loss = train_step(
                model, x, y, optimizer, opt_state, batch_key
            )
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        eval_key, epoch_eval_key = jax.random.split(eval_key)
        total_accuracy = 0
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.numpy()
            y_batch = y_batch.numpy()
            epoch_eval_key, batch_key = jax.random.split(epoch_eval_key)
            x, y = batch
            x = x.numpy()
            y = y.numpy()
            total_accuracy += eval_step(model, x, y, batch_key)
            total_ovr_accuracy += eval_step(ovr_model, x_batch, y_batch)


        avg_accuracy = total_accuracy / len(test_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")


if __name__ == "__main__":
    # To run the default experiment: `python scripts/train.py`
    # To run the learnable topology experiment: `python scripts/train.py learnable-mnist-config`
    tyro.cli(main)