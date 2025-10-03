from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data loading."""

    batch_size: int = 128
    train_subset_size: int = None
    test_subset_size: int = None


@dataclass
class ModelConfig:
    """Configuration for the model."""

    in_features: int = 28 * 28

@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 1e-3
    num_epochs: int = 10
    seed: int = 42


@dataclass
class MNISTConfig:
    """Top-level configuration for the MNIST experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)