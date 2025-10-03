import dataclasses
from typing import Union, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
import wandb
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from trex.svm_tree.configs import (
    LearnableHierarchicalSVMConfig,
    LearnableMNISTConfig,
    MNISTConfig,
    ModelType,
)
from trex.svm_tree.data_utils import get_mnist_dataloaders
from trex.svm_tree.model import (
    BaseTreeModel,
    LearnableHierarchicalSVM,
    LearnableTreeModel,
    OvR_SVM_Model,
)


def call_model_with_key(model, x, key):
    """Helper to call a model's __call__ method with a key."""
    return model(x, key=key)


def loss_fn(model, x, y, key, use_topo_loss: bool, topology_loss_weight: float):
    """Computes the total loss for a given model and batch."""
    if isinstance(model, (LearnableTreeModel, LearnableHierarchicalSVM)):
        keys = jax.random.split(key, x.shape[0])
        pred_y_batched = jax.vmap(call_model_with_key, in_axes=(None, 0, 0))(
            model, x, keys
        )
        adj = model.topology(key)
        topo_loss = model.loss(adj)
    else:
        pred_y_batched = jax.vmap(model)(x)
        topo_loss = 0.0

    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        pred_y_batched, y
    ).mean()

    if use_topo_loss:
        return ce_loss + topology_loss_weight * topo_loss
    return ce_loss


@eqx.filter_jit
def train_step(
    model, x, y, optimizer, opt_state, key, use_topo_loss: bool, topology_loss_weight: float
):
    """Performs a single training step."""
    loss_fn_for_grad = lambda m, x, y: loss_fn(
        m, x, y, key, use_topo_loss, topology_loss_weight
    )
    loss, grads = eqx.filter_value_and_grad(loss_fn_for_grad)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, x, y, key):
    """Computes accuracy and predictions on a batch of data."""
    if isinstance(model, (LearnableTreeModel, LearnableHierarchicalSVM)):
        keys = jax.random.split(key, x.shape[0])
        pred_y = jax.vmap(call_model_with_key, in_axes=(None, 0, 0))(model, x, keys)
    else:
        pred_y = jax.vmap(model)(x)

    pred_labels = jnp.argmax(pred_y, axis=1)
    accuracy = jnp.mean(pred_labels == y)
    return accuracy, pred_labels


def main(
    cfg: Union[
        MNISTConfig, LearnableMNISTConfig, LearnableHierarchicalSVMConfig
    ]
):
    """Main training and evaluation loop."""
    if cfg.wandb.use_wandb:
        run_name = cfg.wandb.run_name
        if isinstance(cfg, LearnableHierarchicalSVMConfig):
            run_name = "learnable-hierarchical-svm"
        elif isinstance(cfg, LearnableMNISTConfig):
            run_name = "learnable-tree"
        elif cfg.model.model_type == ModelType.SINGLE_SVM:
            run_name = "single-svm"

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=dataclasses.asdict(cfg),
        )

    key = jax.random.PRNGKey(cfg.train.seed)
    model_key, train_key, eval_key = jax.random.split(key, 3)

    train_loader, test_loader = get_mnist_dataloaders(
        cfg.data.batch_size,
        cfg.data.train_subset_size,
        cfg.data.test_subset_size,
    )

    if isinstance(cfg, LearnableHierarchicalSVMConfig):
        model = LearnableHierarchicalSVM(
            in_features=cfg.model.in_features,
            num_classes=10,
            key=model_key,
            sparsity_regularization_strength=cfg.model.sparsity_regularization_strength,
            graph_constraint_scale=cfg.model.graph_constraint_scale,
        )
    elif isinstance(cfg, LearnableMNISTConfig):
        model = LearnableTreeModel(
            in_features=cfg.model.in_features,
            num_classes=10,
            key=model_key,
            sparsity_regularization_strength=cfg.model.sparsity_regularization_strength,
            graph_constraint_scale=cfg.model.graph_constraint_scale,
        )
    elif isinstance(cfg, MNISTConfig):
        if cfg.model.model_type == ModelType.BASE_TREE:
            model = BaseTreeModel(
                in_features=cfg.model.in_features, num_classes=10, key=model_key
            )
        elif cfg.model.model_type == ModelType.SINGLE_SVM:
            model = OvR_SVM_Model(
                in_features=cfg.model.in_features, num_classes=10, key=model_key
            )
        else:
            raise ValueError(f"Unknown model type: {cfg.model.model_type}")
    else:
        raise TypeError(f"Unknown config type: {type(cfg)}")

    use_topo_loss = isinstance(cfg, (LearnableMNISTConfig, LearnableHierarchicalSVMConfig))
    topology_loss_weight = cfg.train.topology_loss_weight

    optimizer = optax.adam(cfg.train.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(cfg.train.num_epochs):
        train_key, epoch_train_key = jax.random.split(train_key)
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epochs}")

        for x, y in pbar:
            epoch_train_key, batch_key = jax.random.split(epoch_train_key)
            x, y = x.numpy(), y.numpy()
            model, opt_state, loss = train_step(
                model,
                x,
                y,
                optimizer,
                opt_state,
                batch_key,
                use_topo_loss,
                topology_loss_weight,
            )
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        eval_key, epoch_eval_key = jax.random.split(eval_key)
        total_accuracy = 0.0
        all_preds, all_labels = [], []
        for x, y in test_loader:
            epoch_eval_key, batch_key = jax.random.split(epoch_eval_key)
            x, y = x.numpy(), y.numpy()
            accuracy, pred_labels = eval_step(model, x, y, batch_key)
            total_accuracy += accuracy
            all_preds.append(pred_labels)
            all_labels.append(y)

        avg_accuracy = total_accuracy / len(test_loader)
        print(
            f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}"
        )

        if cfg.wandb.use_wandb:
            log_data = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_accuracy": avg_accuracy,
            }
            if isinstance(model, LearnableHierarchicalSVM):
                adj_key, _ = jax.random.split(epoch_eval_key)
                adj = model.topology(adj_key)
                plt.figure(figsize=(10, 10))
                plt.imshow(adj, cmap="hot", interpolation="nearest")
                plt.title(f"Learned Adjacency Matrix - Epoch {epoch+1}")
                log_data["adjacency_matrix"] = wandb.Image(plt)
                plt.close()
            wandb.log(log_data)

    if cfg.wandb.use_wandb:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set(
            xlabel="Predicted",
            ylabel="True",
            title="Confusion Matrix",
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va="center", ha="center")
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close(fig)
        wandb.finish()


if __name__ == "__main__":
    COMMAND_DEFAULTS = {
        "base-tree": MNISTConfig(),
        "single-svm": MNISTConfig(
            model=dataclasses.replace(
                MNISTConfig().model, model_type=ModelType.SINGLE_SVM
            )
        ),
        "learnable-tree": LearnableMNISTConfig(),
        "learnable-hierarchical-svm": LearnableHierarchicalSVMConfig(),
    }
    Subcommands = tyro.extras.subcommand_type_from_defaults(COMMAND_DEFAULTS)

    cfg = tyro.cli(Subcommands, description="Train an SVM tree model on MNIST.")
    main(cfg)