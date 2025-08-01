"""
Test script for final model training.

This script tests the final model training functions that use the entire dataset
without k-fold cross validation for production model training. It includes:

- Final QSPR model training
- Final GNN_M model training
- Final single-task GNN_C training for all 15 environmental impact categories
- Final single-task GNN_E training for all 15 environmental impact categories
- Final multi-task model training

The script uses 500 epochs and batch size 20 for comprehensive training of
production-ready models with category-specific optimized learning rates.
"""

import logging
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.training_engines import TrainingEngine
from src.scripts import prepare_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_final_qspr() -> torch.nn.Module:
    """Test final QSPR model training."""
    logger.info("Testing final QSPR model training...")

    # Prepare data if needed
    try:
        prepare_data(
            data_path="./data/raw",
            dataset_type="QSPR",
            data_file="2023_09_07_QSPR_mol_only_cc.xlsx",
            output_dir="./data/processed",
        )
    except Exception as e:
        logger.info(f"Data preparation result: {e}")

    # Create training engine - it will handle dataset loading internally
    trainer = TrainingEngine(
        path="./",
        dataset_type="QSPR",
        model_type="qspr",
        data_file="2023_09_07_QSPR_mol_only_cc.xlsx",
        project_name="test_final_models",
        entity="test",
        epochs=800,  # Reduced for testing
        batch_size=20,
        val_length=0.1,
        k_fold=1,
        enable_wandb=False,
        experiment_prefix="test_final_",
        device="cpu",  # Force CPU to avoid CUDA issues
    )

    # Train final model
    final_model = trainer.train_final_qspr(learning_rate=5e-6)
    logger.info(f"Final QSPR model training completed. Model type: {type(final_model)}")

    return final_model


def test_final_gnn_m() -> torch.nn.Module:
    """Test final GNN_M model training."""
    logger.info("Testing final GNN_M model training...")

    # Prepare data if needed
    try:
        prepare_data(
            data_path="./data/raw",
            dataset_type="GNN_M",
            data_file="2023_09_07_QSPR_mol_only_cc.xlsx",
            output_dir="./data/processed",
        )
    except Exception as e:
        logger.info(f"Data preparation result: {e}")

    # Create training engine
    trainer = TrainingEngine(
        path="./",
        dataset_type="GNN_M",
        model_type="GNN_M",
        data_file="2023_09_07_GNN_M_CC.xlsx",
        project_name="test_final_models",
        entity="test",
        epochs=500,  # Reduced for testing
        batch_size=20,
        val_length=0.1,
        k_fold=1,
        enable_wandb=False,
        experiment_prefix="test_final_",
    )

    # Train final model
    final_model = trainer.train_final_gnn_m(learning_rate=1e-4)
    logger.info(
        f"Final GNN_M model training completed. Model type: {type(final_model)}"
    )

    return final_model


def test_final_single_task_gnn_c() -> dict[str, torch.nn.Module]:
    """Test final single-task model training with category-specific parameters.

    Trains production-ready models for all 15 environmental impact categories using:
    - 500 epochs for comprehensive training
    - Batch size 20 for optimal memory usage
    - Category-specific optimized learning rates
    - CPU training for compatibility

    Returns
    -------
    dict
        Dictionary mapping category names to trained PyTorch models
    """
    logger.info(
        "Testing final single-task model training with category-specific learning rates..."
    )

    # Prepare data if needed
    try:
        prepare_data(
            data_path="./data/raw",
            dataset_type="GNN_C",
            data_file="2023_09_14_finaldataset_country_combine.xlsx",
            output_dir="./data/processed",
        )
    except Exception as e:
        logger.info(f"Data preparation result: {e}")

    # Create training engine
    trainer = TrainingEngine(
        path="./",
        dataset_type="GNN_C",
        model_type="GNN_C_single",
        data_file="2023_09_14_finaldataset_country_combine.xlsx",
        project_name="test_final_models",
        entity="test",
        epochs=500,  # Full training epochs
        batch_size=20,  # Updated batch size
        val_length=0.1,
        k_fold=1,
        enable_wandb=False,
        experiment_prefix="test_final_",
        device="cpu",  # Force CPU to avoid CUDA issues
    )

    # Get category-specific learning rates from configuration
    from src.config.training_config import get_default_config

    config = get_default_config("GNN_C", "GNN_C_single", "single")
    category_learning_rates = config.hyperparameters.learning_rates

    logger.info(
        f"Available category-specific learning rates: {len(category_learning_rates)} categories"
    )
    logger.info(
        "Starting final model training for all 15 environmental impact categories..."
    )

    # Test training final models for all 15 environmental impact categories
    target_tasks_to_test = [
        "Acid",  # Acidification
        "Gwi",  # Climate change (Global warming impact)
        "CTUe",  # Ecotoxicity (freshwater)
        "ADP_f",  # Fossil resource scarcity
        "Eutro_f",  # Eutrophication (freshwater)
        "Eutro_m",  # Eutrophication (marine)
        "Eutro_t",  # Eutrophication (terrestrial)
        "CTUh",  # Human toxicity (cancer)
        "Ionising",  # Ionising radiation
        "Soil",  # Land use
        "ADP_e",  # Mineral resource scarcity
        "ODP",  # Ozone depletion
        "human_health",  # Human health (non-cancer)
        "Photo",  # Photochemical ozone formation
        "Water_use",  # Water consumption
    ]
    trained_models = {}

    for i, target_task in enumerate(target_tasks_to_test, 1):
        if target_task in category_learning_rates:
            # Use the category-specific learning rate
            optimal_lr = category_learning_rates[target_task][0]
            logger.info(
                f"[{i}/{len(target_tasks_to_test)}] Training final model for '{target_task}' "
                f"with optimal LR: {optimal_lr:.2e}"
            )

            # Train final model for this specific task
            final_model = trainer.train_final_single_task(
                learning_rate=optimal_lr, target_task=target_task
            )

            trained_models[target_task] = final_model
            logger.info(
                f"✓ [{i}/{len(target_tasks_to_test)}] Final model for '{target_task}' completed. "
                f"Model type: {type(final_model)}"
            )
        else:
            logger.warning(
                f"[{i}/{len(target_tasks_to_test)}] No optimal learning rate found for task "
                f"'{target_task}', skipping..."
            )

    logger.info(
        f"Final single-task model training completed for {len(trained_models)} tasks"
    )
    logger.info(f"Successfully trained models for: {list(trained_models.keys())}")
    return trained_models


def test_final_single_task_gnn_e() -> dict[str, torch.nn.Module]:
    """Test final GNN_E single-task model training with category-specific parameters.

    Trains production-ready GNN_E models for all 15 environmental impact categories using:
    - 500 epochs for comprehensive training
    - Batch size 20 for optimal memory usage
    - Category-specific optimized learning rates
    - CPU training for compatibility

    Returns
    -------
    dict
        Dictionary mapping category names to trained PyTorch models
    """
    logger.info("Testing final GNN_E single-task model training...")

    # Prepare data if needed
    try:
        prepare_data(
            data_path="./data/raw",
            dataset_type="GNN_E",
            data_file="2023_09_18_finaldataset_energymix_combine.xlsx",
            output_dir="./data/processed",
        )
    except Exception as e:
        logger.info(f"Data preparation result: {e}")

    # Create training engine for GNN_E
    trainer = TrainingEngine(
        path="./",
        dataset_type="GNN_E",
        model_type="GNN_E_single",
        data_file="2023_09_18_finaldataset_energymix_combine.xlsx",
        project_name="test_final_models",
        entity="test",
        epochs=500,  # Full training epochs
        batch_size=20,  # Updated batch size
        val_length=0.1,
        k_fold=1,
        enable_wandb=False,
        experiment_prefix="test_final_gnn_e_",
        device="cpu",  # Force CPU to avoid CUDA issues
    )

    # Get category-specific learning rates for GNN_E
    from src.config.training_config import get_default_config

    config = get_default_config("GNN_E", "GNN_E_single", "single")
    category_learning_rates = config.hyperparameters.learning_rates

    logger.info(
        f"GNN_E category-specific learning rates: {len(category_learning_rates)} categories"
    )
    logger.info(
        "Starting final GNN_E model training for all 15 environmental impact categories..."
    )

    # Test training final models for all 15 environmental impact categories
    target_tasks_to_test = [
        "Acid",  # Acidification
        "Gwi",  # Climate change (Global warming impact)
        "CTUe",  # Ecotoxicity (freshwater)
        "ADP_f",  # Fossil resource scarcity
        "Eutro_f",  # Eutrophication (freshwater)
        "Eutro_m",  # Eutrophication (marine)
        "Eutro_t",  # Eutrophication (terrestrial)
        "CTUh",  # Human toxicity (cancer)
        "Ionising",  # Ionising radiation
        "Soil",  # Land use
        "ADP_e",  # Mineral resource scarcity
        "ODP",  # Ozone depletion
        "human_health",  # Human health (non-cancer)
        "Photo",  # Photochemical ozone formation
        "Water_use",  # Water consumption
    ]
    trained_models = {}

    for i, target_task in enumerate(target_tasks_to_test, 1):
        if target_task in category_learning_rates:
            # Use the GNN_E-specific learning rate
            optimal_lr = category_learning_rates[target_task][0]
            logger.info(
                f"[{i}/{len(target_tasks_to_test)}] Training GNN_E final model for '{target_task}' "
                f"with optimal LR: {optimal_lr:.2e}"
            )

            # Train final model for this specific task
            final_model = trainer.train_final_single_task(
                learning_rate=optimal_lr, target_task=target_task
            )

            trained_models[target_task] = final_model
            logger.info(
                f"✓ [{i}/{len(target_tasks_to_test)}] GNN_E final model for '{target_task}' "
                f"completed. Model type: {type(final_model)}"
            )
        else:
            logger.warning(
                f"[{i}/{len(target_tasks_to_test)}] No optimal learning rate found for GNN_E task "
                f"'{target_task}', skipping..."
            )

    logger.info(
        f"Final GNN_E single-task model training completed for {len(trained_models)} tasks"
    )
    logger.info(f"Successfully trained GNN_E models for: {list(trained_models.keys())}")
    return trained_models


def test_final_multi_task() -> torch.nn.Module:
    """Test final multi-task model training."""
    logger.info("Testing final multi-task model training...")

    # Prepare data if needed
    try:
        prepare_data(
            data_path="./data/raw",
            dataset_type="GNN_E",
            data_file="2023_09_18_finaldataset_energymix_combine.xlsx",
            output_dir="./data/processed",
        )
    except Exception as e:
        logger.info(f"Data preparation result: {e}")

    # Create training engine
    trainer = TrainingEngine(
        path="./",
        dataset_type="GNN_E",
        model_type="GNN_E_multi",
        data_file="2023_09_18_finaldataset_energymix_combine.xlsx",
        project_name="test_final_models",
        entity="test",
        epochs=500,  # Reduced for testing
        batch_size=20,
        val_length=0.1,
        k_fold=1,
        enable_wandb=False,
        experiment_prefix="test_final_",
    )

    # Train final model
    final_model = trainer.train_final_multi_task(learning_rate=1e-3)
    logger.info(
        f"Final multi-task model training completed. Model type: {type(final_model)}"
    )

    return final_model


if __name__ == "__main__":
    logger.info(
        "Starting final model training tests for all 15 environmental impact categories..."
    )

    try:
        # Test final QSPR
        qspr_model = test_final_qspr()

        # Test final GNN_M
        gnn_m_model = test_final_gnn_m()

        # Test final single-task training with category-specific parameters for all 15 categories
        logger.info(
            "\n=== Testing GNN_C Final Single-Task Training (All 15 Categories) ==="
        )
        gnn_c_models = test_final_single_task_gnn_c()
        logger.info(f"Trained {len(gnn_c_models)} GNN_C final models")

        # Test final GNN_E single-task training with category-specific parameters
        # for all 15 categories
        logger.info(
            "\n=== Testing GNN_E Final Single-Task Training (All 15 Categories) ==="
        )
        gnn_e_models = test_final_single_task_gnn_e()
        logger.info(f"Trained {len(gnn_e_models)} GNN_E final models")

        # Test final multi-task
        logger.info("\n=== Testing Multi-Task Training ===")
        multi_model = test_final_multi_task()

        logger.info("\n=== Final Model Training Summary ===")
        logger.info(f"✓ GNN_C single-task models: {list(gnn_c_models.keys())}")
        logger.info(f"✓ GNN_E single-task models: {list(gnn_e_models.keys())}")
        logger.info("✓ Multi-task model: completed")
        logger.info(
            f"\nTotal models trained: {len(gnn_c_models) + len(gnn_e_models) + 1}"
        )
        logger.info("All final model training tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
