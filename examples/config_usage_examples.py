"""
Example Usage of the New Configuration System for GNN Training.

This script demonstrates how to use the new configuration-based approach
for training GNN and QSPR models.
"""

import torch

from src.config.training_config import TrainingConfiguration
from src.engines.training_engines import TrainingEngine
from src.trainer.trainer import create_trainer_from_config


def example_1_using_default_config():
    """Example 1: Using default configuration with modifications."""
    print("Example 1: Using default configuration")

    # Create default configuration
    config = TrainingConfiguration()

    # Modify specific parameters
    config.data.dataset_type = "GNN_C"
    config.data.path = "./data"
    config.data.data_file = "molecular_data.xlsx"
    config.model.model_type = "GNN_C_single"
    config.training.epochs = 200
    config.training.batch_size = 32
    config.optimizer.learning_rate = 0.001

    # Create training engine
    engine = TrainingEngine(config=config)

    print(f"Training configuration created for {config.data.dataset_type}")
    print(f"Model type: {config.model.model_type}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print(f"Batch size: {config.training.batch_size}")


def example_2_loading_from_yaml():
    """Example 2: Loading configuration from YAML file."""
    print("\nExample 2: Loading configuration from YAML")

    # Load configuration from YAML file
    config = TrainingConfiguration.from_yaml("configs/example_training_config.yaml")

    # Create training engine
    engine = TrainingEngine(config=config)

    print("Configuration loaded from YAML")
    print(f"Dataset type: {config.data.dataset_type}")
    print(f"Experiment project: {config.experiment.project_name}")


def example_3_factory_functions():
    """Example 3: Using factory functions for specific model types."""
    print("\nExample 3: Using factory functions")

    # Create configuration for different model types
    qspr_config = TrainingConfiguration.for_qspr_model()
    gnn_m_config = TrainingConfiguration.for_gnn_m_model()
    gnn_c_config = TrainingConfiguration.for_gnn_c_model()

    print(f"QSPR config - Model: {qspr_config.model.model_type}")
    print(f"GNN_M config - Model: {gnn_m_config.model.model_type}")
    print(f"GNN_C config - Model: {gnn_c_config.model.model_type}")


def example_4_trainer_integration():
    """Example 4: Using configuration with the Trainer class."""
    print("\nExample 4: Trainer integration")

    # Create configuration
    config = TrainingConfiguration.for_gnn_c_model()
    config.training.epochs = 50  # Shorter training for example
    config.optimizer.learning_rate = 0.001

    # Create a dummy model (in real usage, you'd use actual model)
    dummy_model = torch.nn.Linear(10, 1)  # Placeholder

    # Create trainer using configuration
    trainer = create_trainer_from_config(dummy_model, config)

    print("Trainer created with configuration")
    print(f"Optimizer type: {type(trainer.optimizer).__name__}")
    print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"Device: {trainer.device}")


def example_5_backward_compatibility():
    """Example 5: Backward compatibility with legacy parameters."""
    print("\nExample 5: Backward compatibility")

    # Create training engine using legacy parameters (still supported)
    engine = TrainingEngine.from_legacy_params(
        path="./data",
        dataset_type="GNN_C",
        model_type="GNN_C_single",
        data_file="data.xlsx",
        k_fold=5,
        batch_size=16,
        epochs=100,
    )

    print("Legacy parameters converted to configuration")
    print(f"K-fold: {engine.config.training.k_fold}")
    print(f"Batch size: {engine.config.training.batch_size}")


def example_6_hyperparameter_search():
    """Example 6: Configuration for hyperparameter search."""
    print("\nExample 6: Hyperparameter search setup")

    config = TrainingConfiguration()

    # Set up hyperparameter search space
    config.hyperparameter.learning_rates = [0.01, 0.001, 0.0001]
    config.hyperparameter.batch_sizes = [16, 32, 64]
    config.hyperparameter.hidden_dims = [64, 128, 256]

    print("Hyperparameter search configured:")
    print(f"Learning rates: {config.hyperparameter.learning_rates}")
    print(f"Batch sizes: {config.hyperparameter.batch_sizes}")
    print(f"Hidden dimensions: {config.hyperparameter.hidden_dims}")


def example_7_save_and_load_config():
    """Example 7: Saving and loading configuration."""
    print("\nExample 7: Save and load configuration")

    # Create and modify configuration
    config = TrainingConfiguration.for_gnn_c_model()
    config.training.epochs = 300
    config.optimizer.learning_rate = 0.0005
    config.experiment.project_name = "MyExperiment"

    # Save to YAML
    config.to_yaml("configs/my_experiment_config.yaml")
    print("Configuration saved to YAML file")

    # Load back from YAML
    loaded_config = TrainingConfiguration.from_yaml("configs/my_experiment_config.yaml")
    print(f"Configuration loaded - Project: {loaded_config.experiment.project_name}")

    # Save to JSON
    config.to_json("configs/my_experiment_config.json")
    print("Configuration also saved to JSON file")


if __name__ == "__main__":
    print("=== Training Configuration System Examples ===\n")

    try:
        example_1_using_default_config()
        # example_2_loading_from_yaml()  # Uncomment if YAML file exists
        example_3_factory_functions()
        example_4_trainer_integration()
        example_5_backward_compatibility()
        example_6_hyperparameter_search()
        example_7_save_and_load_config()

        print("\n=== All examples completed successfully! ===")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: Some examples may require actual model classes to be available.")


"""
Key Benefits of the New Configuration System:

1. **Centralized Parameter Management**: All training parameters are organized
   in one place instead of being scattered throughout the codebase.

2. **Type Safety**: Dataclass-based approach provides automatic type checking
   and IDE support with autocomplete.

3. **Serialization Support**: Easy saving/loading of configurations as YAML 
   or JSON files for reproducibility.

4. **Factory Functions**: Pre-configured setups for different model types
   (QSPR, GNN_M, GNN_C, GNN_E).

5. **Backward Compatibility**: Legacy function signatures still work through
   automatic conversion to the new configuration system.

6. **Validation**: Built-in validation ensures parameters are sensible
   (e.g., positive learning rates, valid device names).

7. **Hierarchical Organization**: Parameters are logically grouped into
   categories (optimizer, scheduler, training, data, model, experiment,
   hyperparameter, system).

8. **Experiment Tracking**: Built-in support for Weights & Biases configuration
   and experiment management.

Usage Patterns:

- For new projects: Use TrainingConfiguration directly
- For existing code: Use factory functions or legacy compatibility functions
- For experiments: Create configuration files and load them
- For hyperparameter search: Use the hyperparameter configuration section
- For reproducibility: Save configurations alongside model checkpoints
"""
