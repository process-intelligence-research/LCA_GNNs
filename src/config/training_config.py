"""
Configuration management for GNN and QSPR training.

This module provides a comprehensive configuration system for all training parameters,
supporting different configuration formats (YAML, JSON, dataclass) and providing
sensible defaults for different model types.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""

    type: str = "Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: str = "ReduceLROnPlateau"
    factor: float = 0.9
    patience: int = 10
    min_lr: float = 1e-9
    cooldown: int = 2
    threshold: float = 1e-4


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    epochs: int = 500
    batch_size: int = 20
    val_length: float = 0.10
    k_fold: int = 10
    early_stopping_patience: int = 15
    gradient_clip_norm: float | None = None
    accumulate_grad_batches: int = 1


@dataclass
class DataConfig:
    """Configuration for dataset and data processing."""

    dataset_type: str = "QSPR"  # QSPR, GNN_M, GNN_C, GNN_E
    data_file: str = ""
    data_path: str = ""
    target_tasks: list[str] = field(
        default_factory=list
    )  # Only used for GNN_C/GNN_E single-task mode
    task_mode: str = "simple"  # simple (QSPR/GNN_M), single (GNN_C/GNN_E), multi
    normalize_features: bool = True
    augment_data: bool = False


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = (
        "qspr"  # qspr, GNN_M, GNN_C_single, GNN_C_multi, GNN_E_single, GNN_E_multi
    )
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "ReLU"
    batch_norm: bool = True
    out_feature: int = 1


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging."""

    project_name: str = "GWP"
    entity: str = "qinghegao"
    experiment_prefix: str = ""
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    enable_wandb: bool = False
    log_frequency: int = 10
    save_checkpoints: bool = True
    checkpoint_dir: str = "./trained_models"


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter sweeps."""

    learning_rates: list[float] | dict[str, list[float]] = field(
        default_factory=lambda: [1e-4, 5e-5, 1e-5]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [20])
    hidden_dims: list[int] = field(default_factory=lambda: [128])
    dropouts: list[float] = field(default_factory=lambda: [0.1])
    num_layers_list: list[int] = field(default_factory=lambda: [3])


@dataclass
class SystemConfig:
    """Configuration for system and hardware settings."""

    device: str | None = None  # "auto", "cpu", "cuda", "cuda:0"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    deterministic: bool = False
    seed: int | None = None


@dataclass
class TrainingConfiguration:
    """Master configuration containing all sub-configurations."""

    # Sub-configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device if not specified
        if self.system.device is None or self.system.device == "auto":
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set deterministic behavior if seed is provided
        if self.system.seed is not None:
            self.set_seed()

    def set_seed(self, seed: int | None = None):
        """Set random seeds for reproducibility."""
        if seed is not None:
            self.system.seed = seed

        if self.system.seed is not None:
            import random

            import numpy as np

            random.seed(self.system.seed)
            np.random.seed(self.system.seed)
            torch.manual_seed(self.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.system.seed)
                torch.cuda.manual_seed_all(self.system.seed)

            if self.system.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def get_device(self) -> torch.device:
        """Get the PyTorch device object."""
        return torch.device(self.system.device)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        def _dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_dataclass_to_dict(item) for item in obj)
            else:
                return obj

        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfiguration":
        """Create configuration from dictionary."""

        def _dict_to_dataclass(datacls, data):
            if not isinstance(data, dict):
                return data

            fieldtypes = {f.name: f.type for f in datacls.__dataclass_fields__.values()}
            filtered_data = {}

            for key, value in data.items():
                if key in fieldtypes:
                    field_type = fieldtypes[key]
                    if hasattr(field_type, "__dataclass_fields__"):
                        filtered_data[key] = _dict_to_dataclass(field_type, value)
                    else:
                        filtered_data[key] = value

            return datacls(**filtered_data)

        return _dict_to_dataclass(cls, config_dict)

    def save_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def save_json(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, filepath: str) -> "TrainingConfiguration":
        """Load configuration from YAML file."""
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> "TrainingConfiguration":
        """Load configuration from JSON file."""
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config(
    dataset_type: str, model_type: str, task_mode: str = "single"
) -> TrainingConfiguration:
    """
    Get default configuration for specific dataset and model types.

    Parameters
    ----------
    dataset_type : str
        Type of dataset (QSPR, GNN_M, GNN_C, GNN_E).
    model_type : str
        Type of model (qspr, GNN_M, GNN_C_single, etc.).
    task_mode : str, optional
        Training mode (single, multi). Default "single".

    Returns
    -------
    TrainingConfiguration
        Default configuration for the specified setup.
    """
    config = TrainingConfiguration()

    # Set dataset and model specific defaults
    config.data.dataset_type = dataset_type
    config.model.model_type = model_type
    config.data.task_mode = task_mode

    # Dataset-specific defaults
    if dataset_type == "QSPR":
        # QSPR fixed hyperparameters (no tuning) - specified values
        config.data.data_file = "2023_09_07_QSPR_mol_only_cc.xlsx"
        config.training.epochs = 800  # Maximum number of epochs before stopping
        config.training.batch_size = 20  # Size of the batch for training set
        config.training.val_length = 0.10
        config.training.k_fold = 10
        config.training.early_stopping_patience = (
            15  # Epochs patience for early stopping
        )

        # Fixed optimizer settings
        config.optimizer.learning_rate = 5e-6  # Starting learning rate of the scheduler
        config.optimizer.weight_decay = 0.0

        # Fixed scheduler settings
        config.scheduler.type = "ReduceLROnPlateau"
        config.scheduler.factor = (
            0.9  # Decreasing factor of new learning rate (new = old * factor)
        )
        config.scheduler.patience = 10  # Epochs patience for decreasing learning rate
        config.scheduler.cooldown = 2  # Epochs of cooldown to reprise normal training
        config.scheduler.min_lr = (
            1e-9  # Minimum learning rate achievable by the scheduler
        )

        # Single fixed learning rate (no hyperparameter tuning for QSPR)
        config.hyperparameters.learning_rates = [5e-6]
        config.hyperparameters.batch_sizes = [20]
        config.hyperparameters.dropouts = [0.1]
        config.hyperparameters.hidden_dims = [128]
        config.hyperparameters.num_layers_list = [3]

        config.experiment.experiment_prefix = "QSPR_cc_"
        config.data.target_tasks = []  # QSPR doesn't use target_tasks
        config.data.task_mode = "simple"  # QSPR uses simple training
        config.model.out_feature = 1

    elif dataset_type == "GNN_M":
        config.data.data_file = "2023_09_07_QSPR_mol_only_cc.xlsx"

        # Fixed hyperparameters for GNN_M (no tuning)
        config.training.epochs = 500  # Maximum number of epochs before stopping
        config.training.batch_size = 20  # Size of the batch for training set
        config.training.val_length = 0.10
        config.training.k_fold = 10
        config.training.early_stopping_patience = (
            15  # Epochs patience for early stopping
        )

        # Optimizer settings
        config.optimizer.learning_rate = 1e-4  # Fixed learning rate

        # Scheduler settings
        config.scheduler.factor = (
            0.9  # Decreasing factor of new learning rate (new = old * factor)
        )
        config.scheduler.patience = 10  # Epochs patience for decreasing learning rate
        config.scheduler.cooldown = 2  # Epochs of cooldown to reprise normal training
        config.scheduler.min_lr = (
            1e-9  # Minimum learning rate achievable by the scheduler
        )

        # Single fixed learning rate (no hyperparameter tuning for GNN_M)
        config.hyperparameters.learning_rates = [1e-4]
        config.hyperparameters.batch_sizes = [20]
        config.hyperparameters.dropouts = [0.1]
        config.hyperparameters.hidden_dims = [128]
        config.hyperparameters.num_layers_list = [3]

        config.experiment.experiment_prefix = "2023_09_19_GNN_cc_molonly_"
        config.data.target_tasks = []  # GNN_M doesn't use target_tasks
        config.data.task_mode = "simple"  # GNN_M uses simple training
        config.model.out_feature = 1

    elif dataset_type == "GNN_C":
        config.data.data_file = "2023_09_14_finaldataset_country_combine.xlsx"
        config.training.epochs = 500
        config.training.batch_size = 20
        config.training.val_length = 0.10
        config.training.k_fold = 10
        config.training.early_stopping_patience = 15
        config.scheduler.factor = 0.9
        config.scheduler.patience = 10
        config.scheduler.cooldown = 2
        config.scheduler.min_lr = 1e-9

        if task_mode == "single":
            # Category-specific learning rates for GNN-C single-task models
            # Map abbreviated names from table to actual task names
            config.hyperparameters.learning_rates = {
                "Acid": [1.00e-03],  # AC -> Acid
                "Gwi": [5.00e-05],  # CC -> Gwi (Climate Change/GWP)
                "CTUe": [5.00e-04],  # ECO -> CTUe (Ecotoxicity)
                "ADP_e": [5.00e-04],  # ER -> ADP_e (Energy Resources)
                "Eutro_f": [1.00e-03],  # EUf -> Eutro_f (Eutrophication freshwater)
                "Eutro_m": [5.00e-04],  # EUm -> Eutro_m (Eutrophication marine)
                "Eutro_t": [5.00e-04],  # EUt -> Eutro_t (Eutrophication terrestrial)
                "CTUh": [1.00e-03],  # HT -> CTUh (Human Toxicity)
                "Ionising": [1.00e-04],  # IR -> Ionising (Ionising Radiation)
                "Soil": [1.00e-04],  # LU -> Soil (Land Use/Soil)
                "ADP_f": [1.00e-04],  # MR -> ADP_f (Mineral Resources)
                "ODP": [5.00e-04],  # OD -> ODP (Ozone Depletion)
                "human_health": [
                    5.00e-04
                ],  # PMF -> human_health (Particulate Matter Formation)
                "Photo": [5.00e-05],  # POF -> Photo (Photochemical Oxidation)
                "Water_use": [5.00e-04],  # WU -> Water_use
            }
            config.experiment.experiment_prefix = "2023_10_25_db_GNN_country_"
        else:
            config.hyperparameters.learning_rates = [1e-3]  # Multi-task default
            config.experiment.experiment_prefix = "GNN_country_multitask_"

    elif dataset_type == "GNN_E":
        config.data.data_file = "2023_09_18_finaldataset_energymix_combine.xlsx"
        config.training.epochs = 500
        config.training.batch_size = 20
        config.training.val_length = 0.10
        config.training.k_fold = 10
        config.training.early_stopping_patience = 15
        config.scheduler.factor = 0.9
        config.scheduler.patience = 10
        config.scheduler.cooldown = 2
        config.scheduler.min_lr = 1e-9

        if task_mode == "single":
            # Category-specific learning rates for GNN-E single-task models
            # Map abbreviated names from table to actual task names
            config.hyperparameters.learning_rates = {
                "Acid": [5.00e-04],  # AC -> Acid
                "Gwi": [5.00e-04],  # CC -> Gwi (Climate Change/GWP)
                "CTUe": [5.00e-04],  # ECO -> CTUe (Ecotoxicity)
                "ADP_e": [5.00e-04],  # ER -> ADP_e (Energy Resources)
                "Eutro_f": [5.00e-04],  # EUf -> Eutro_f (Eutrophication freshwater)
                "Eutro_m": [5.00e-04],  # EUm -> Eutro_m (Eutrophication marine)
                "Eutro_t": [5.00e-04],  # EUt -> Eutro_t (Eutrophication terrestrial)
                "CTUh": [5.00e-04],  # HT -> CTUh (Human Toxicity)
                "Ionising": [1.00e-04],  # IR -> Ionising (Ionising Radiation)
                "Soil": [1.00e-04],  # LU -> Soil (Land Use/Soil)
                "ADP_f": [1.00e-04],  # MR -> ADP_f (Mineral Resources)
                "ODP": [5.00e-04],  # OD -> ODP (Ozone Depletion)
                "human_health": [
                    5.00e-04
                ],  # PMF -> human_health (Particulate Matter Formation)
                "Photo": [5.00e-04],  # POF -> Photo (Photochemical Oxidation)
                "Water_use": [5.00e-04],  # WU -> Water_use
            }
            config.experiment.experiment_prefix = "2023_09_25_db_GNN_energy_"
        else:
            config.hyperparameters.learning_rates = [1e-3]  # Multi-task default
            config.experiment.experiment_prefix = "2023_09_18_GNN_energy_multitask_"

    # Model-specific defaults
    if "multi" in model_type:
        config.data.task_mode = "multi"
        config.model.out_feature = 15  # 15 environmental impact categories
    elif dataset_type not in ["QSPR"]:  # Don't override QSPR's out_feature
        config.model.out_feature = 1

    return config


def create_config_template(filepath: str = "./configs/template_config.yaml"):
    """Create a template configuration file with all options documented."""
    config = TrainingConfiguration()

    # Add some example values and documentation
    config.data.dataset_type = "GNN_C"
    config.model.model_type = "GNN_C_single"
    config.data.task_mode = "single"
    config.experiment.tags = ["experiment", "baseline"]
    config.experiment.notes = "Baseline experiment with default parameters"

    config.save_yaml(filepath)

    # Add comments to the YAML file
    with open(filepath) as f:
        content = f.read()

    commented_content = f"""# Training Configuration Template
# This file contains all available configuration options for GNN/QSPR training
# Modify values as needed for your experiments

{content}

# Configuration Guide:
# 
# dataset_type options: QSPR, GNN_M, GNN_C, GNN_E
# model_type options: qspr, GNN_M, GNN_C_single, GNN_C_multi, GNN_E_single, GNN_E_multi
# task_mode options: single, multi
# 
# For hyperparameter sweeps, provide lists of values to test:
# learning_rates: [1e-3, 5e-4, 1e-4]
# batch_sizes: [16, 32, 64]
# 
# Device options: auto, cpu, cuda, cuda:0, cuda:1, etc.
"""

    with open(filepath, "w") as f:
        f.write(commented_content)

    print(f"Configuration template created at: {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Creating configuration template...")
    create_config_template()

    # Example of creating and using configs
    print("\nExample configurations:")

    # Create default config for GNN_C single-task
    config = get_default_config("GNN_C", "GNN_C_single", "single")
    print(f"GNN_C single-task config: {config.data.data_file}")

    # Save and load example
    config.save_yaml("./configs/example_config.yaml")
    loaded_config = TrainingConfiguration.from_yaml("./configs/example_config.yaml")
    print(f"Loaded config device: {loaded_config.get_device()}")
