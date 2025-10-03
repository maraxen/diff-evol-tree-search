import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from trex.svm_tree.configs import MNISTConfig
from trex.svm_tree.data_utils import get_mnist_dataloaders
from trex.svm_tree.model import BaseTreeModel


def loss_fn(model, x, y):
    """Computes the cross-entropy loss."""
    pred_y_batched = jax.vmap(model)(x)
    # y are integer labels, so we can use sparse softmax cross-entropy
    return optax.softmax_cross_entropy_with_integer_labels(
        pred_y_batched, y
    ).mean()


@eqx.filter_jit
def train_step(model, x, y, optimizer, opt_state):
    """Performs a single training step."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, x, y):
    """Computes the accuracy on a batch of data."""
    pred_y = jax.vmap(model)(x)
    pred_labels = jnp.argmax(pred_y, axis=1)
    return jnp.mean(pred_labels == y)


def main():
    """Main training and evaluation loop."""
    cfg = MNISTConfig()
    key = jax.random.PRNGKey(cfg.train.seed)

    # Create DataLoaders
    train_loader, test_loader = get_mnist_dataloaders(
        cfg.data.batch_size,
        cfg.data.train_subset_size,
        cfg.data.test_subset_size,
    )

    # Create model and optimizer
    model_key, _ = jax.random.split(key)
    model = BaseTreeModel(
        in_features=cfg.model.in_features, num_classes=10, key=model_key
    )
    optimizer = optax.adam(cfg.train.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Training loop
    for epoch in range(cfg.train.num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}"):
            x, y = batch
            x = x.numpy()
            y = y.numpy()

            model, opt_state, loss = train_step(model, x, y, optimizer, opt_state)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        total_accuracy = 0
        for batch in test_loader:
            x, y = batch
            x = x.numpy()
            y = y.numpy()
            total_accuracy += eval_step(model, x, y)

        avg_accuracy = total_accuracy / len(test_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

if __name__ == "__main__":
    main()