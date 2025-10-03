import argparse
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from trex.svm_tree.configs import MNISTConfig, ModelType
from trex.svm_tree.data_utils import get_mnist_dataloaders
from trex.svm_tree.model import BaseTreeModel, SingleSVMModel


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
    accuracy = jnp.mean(pred_labels == y)
    return accuracy, pred_labels


def main():
    """Main training and evaluation loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    cfg = MNISTConfig()
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
        cfg.wandb.run_name = f"{cfg.model.model_type.value}_lr_{args.learning_rate}"

    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=dataclasses.asdict(cfg),
        )

    key = jax.random.PRNGKey(cfg.train.seed)

    # Create DataLoaders
    train_loader, test_loader = get_mnist_dataloaders(
        cfg.data.batch_size,
        cfg.data.train_subset_size,
        cfg.data.test_subset_size,
    )

    # Create model and optimizer
    model_key, _ = jax.random.split(key)
    if cfg.model.model_type == ModelType.BASE_TREE:
        model = BaseTreeModel(
            in_features=cfg.model.in_features, num_classes=10, key=model_key
        )
    elif cfg.model.model_type == ModelType.SINGLE_SVM:
        model = SingleSVMModel(
            in_features=cfg.model.in_features, num_classes=10, key=model_key
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")

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
        all_preds = []
        all_labels = []
        for batch in test_loader:
            x, y = batch
            x = x.numpy()
            y_np = y.numpy()
            accuracy, pred_labels = eval_step(model, x, y_np)
            total_accuracy += accuracy
            all_preds.append(pred_labels)
            all_labels.append(y_np)

        avg_accuracy = total_accuracy / len(test_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

        if cfg.wandb.use_wandb:
            wandb.log(
                {"epoch": epoch, "train_loss": avg_loss, "test_accuracy": avg_accuracy}
            )

    # After the last epoch, compute and log the confusion matrix
    if cfg.wandb.use_wandb:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        # Add text annotations.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center')

        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close(fig)

        wandb.finish()


if __name__ == "__main__":
    main()