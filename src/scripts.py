"""
A module containing the top level scripts for data processing, model training, testing etc.

This module provides a complete pipeline for GNN and QSPR model training and evaluation,
integrating all components of the system including configuration management, data processing,
model training, and prediction.

Functions:
    prepare_data - Transform raw data into processed datasets
    train_model - Train models using the configuration system
    predict_model - Make predictions using trained models (simplified wrapper)

Note:
    For advanced prediction features like single molecule inference, use functions from
    src.engines.predict_engines directly:
    - predict_single_molecule: Make predictions for individual molecules
    - predict_single_task_model: Make predictions using single-task models
    - predict_multi_task_model: Make predictions using multi-task models
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Example usage
import argparse

from src.config.training_config import TrainingConfiguration
from src.data_processing.make_dataset import (
    GNN_C_dataset,
    GNN_E_dataset,
    GNN_M_dataset,
    QSPR_dataset,
)
from src.engines.predict_engines import predict_model as engine_predict_model
from src.engines.training_engines import TrainingEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_data(
    data_path: str,
    dataset_type: str = "GNN_C",
    data_file: str = "data.xlsx",
    output_dir: str = "./data/processed",
) -> dict[str, str]:
    """
    Transform raw data into processed datasets ready for training.

    Parameters
    ----------
    data_path : str
        Path to the directory containing raw data files.
    dataset_type : str, optional
        Type of dataset to prepare ("QSPR", "GNN_M", "GNN_C", "GNN_E").
    data_file : str, optional
        Name of the input data file.
    output_dir : str, optional
        Directory to save processed data (used for determining root structure).

    Returns
    -------
    Dict[str, str]
        Dictionary containing paths to processed data files.
    """
    logger.info(f"Preparing {dataset_type} data from {data_path}/{data_file}")

    # Ensure proper path handling
    data_path = os.path.abspath(data_path)
    output_dir = os.path.abspath(output_dir)

    input_file = os.path.join(data_path, data_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input data file not found: {input_file}")

    # Read the Excel file as a pandas DataFrame
    logger.info(f"Loading data from {input_file}")
    try:
        dataexcel = pd.read_excel(input_file)
        logger.info(
            f"Loaded {len(dataexcel)} rows and {len(dataexcel.columns)} columns"
        )
    except Exception as e:
        raise ValueError(f"Error reading Excel file {input_file}: {str(e)}")

    # For PyTorch Geometric datasets, we need to provide the correct root directory
    # If data_path is "./data/raw", then root should be "./data" so that:
    # - Raw files are expected in "./data/raw/"
    # - Processed files are saved to "./data/processed/"

    # Extract the parent directory of data_path to use as root
    if (
        data_path.endswith("raw")
        or data_path.endswith("raw/")
        or data_path.endswith("raw\\")
    ):
        # If data_path points to a "raw" directory, use its parent as root
        dataset_root = os.path.dirname(data_path.rstrip("/\\"))
    else:
        # Otherwise, assume data_path is the base directory and use it as root
        dataset_root = data_path

    logger.info(f"Using dataset root: {dataset_root}")

    # Dataset-specific processing
    if dataset_type == "QSPR":
        # QSPR data processing
        dataset = QSPR_dataset(root=dataset_root, dataexcel=dataexcel)

    elif dataset_type == "GNN_M":
        # GNN Molecular data processing
        dataset = GNN_M_dataset(root=dataset_root, dataexcel=dataexcel)

    elif dataset_type == "GNN_C":
        # GNN Country data processing
        dataset = GNN_C_dataset(root=dataset_root, dataexcel=dataexcel)

    elif dataset_type == "GNN_E":
        # GNN Energy data processing
        dataset = GNN_E_dataset(root=dataset_root, dataexcel=dataexcel)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Get the actual path where PyTorch Geometric saved the dataset
    processed_file = dataset.processed_paths[0]

    logger.info(f"Data preparation completed. Processed {len(dataset)} samples.")
    logger.info(f"Processed data saved to: {processed_file}")

    return {
        "processed_data_path": processed_file,
        "dataset_size": len(dataset),
        "dataset_type": dataset_type,
    }


def train_model(
    config_path: str = None,
    config: TrainingConfiguration = None,
    data_path: str = "./data",
    save_results: bool = True,
    results_dir: str = "./results",
) -> dict[str, any]:
    """
    Train models using the configuration system.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration YAML file. If None, uses provided config.
    config : TrainingConfiguration, optional
        Configuration object. If None, loads from config_path.
    data_path : str, optional
        Path to data directory.
    save_results : bool, optional
        Whether to save training results.
    results_dir : str, optional
        Directory to save results.

    Returns
    -------
    Dict[str, any]
        Dictionary containing training results and metrics.
    """
    logger.info("Starting model training...")

    # Load or validate configuration
    if config is None:
        if config_path is None:
            raise ValueError("Either config_path or config must be provided")
        config = TrainingConfiguration.from_yaml(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
    else:
        logger.info("Using provided configuration")

    # Update data path in config if provided
    if data_path:
        config.data.path = data_path

    # Create results directory
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    # Create training engine with individual parameters
    engine = TrainingEngine(
        path=config.data.path,
        dataset_type=config.data.dataset_type,
        model_type=config.model.model_type,
        data_file=config.data.data_file,
        project_name=config.experiment.project_name,
        entity=config.experiment.entity,
        experiment_prefix=config.experiment.experiment_prefix,
        k_fold=config.training.k_fold,
        batch_size=config.training.batch_size,
        val_length=config.training.val_length,
        epochs=config.training.epochs,
        enable_wandb=config.experiment.enable_wandb,
        device=str(config.get_device()),
    )

    logger.info(
        f"Training {config.model.model_type} on {config.data.dataset_type} dataset"
    )
    logger.info(f"Learning rates: {config.hyperparameters.learning_rates}")
    logger.info(f"K-fold cross-validation: {config.training.k_fold} folds")

    # Determine training mode and execute appropriate training
    if config.data.dataset_type in ["QSPR", "GNN_M"]:
        # QSPR and GNN_M always use simple training (single output)
        logger.info("Running simple training (QSPR/GNN_M style)")
        if config.data.dataset_type == "QSPR":
            engine.train_qspr()
        else:  # GNN_M
            engine.train_gnn_m()

    elif config.data.task_mode == "single":
        target_tasks = config.data.target_tasks
        if not target_tasks:
            # Default environmental impact categories
            target_tasks = [
                "Acid",
                "Gwi",
                "CTUe",
                "ADP_f",
                "Eutro_f",
                "Eutro_m",
                "Eutro_t",
                "CTUh",
                "Ionising",
                "Soil",
                "ADP_e",
                "ODP",
                "human_health",
                "Photo",
                "Water_use",
            ]

        logger.info(f"Running single-task training for {len(target_tasks)} tasks")
        engine.train_single_task(config.hyperparameters.learning_rates, target_tasks)

    elif config.data.task_mode == "multi":
        logger.info(f"Running multi-task training for {config.data.dataset_type}")
        if config.data.dataset_type == "GNN_C":
            logger.info("Training GNN_C multi-task model for country-specific impacts")
        elif config.data.dataset_type == "GNN_E":
            logger.info("Training GNN_E multi-task model for energy mix impacts")
        engine.train_multi_task(config.hyperparameters.learning_rates)

    else:
        logger.info("Running simple training (fallback)")
        if config.data.dataset_type == "QSPR":
            engine.train_qspr()
        elif config.data.dataset_type == "GNN_M":
            engine.train_gnn_m()
        else:
            # Default to single task for other types
            target_tasks = config.data.target_tasks or ["default_task"]
            engine.train_single_task(
                config.hyperparameters.learning_rates, target_tasks
            )

    # Collect results
    results = {
        "config": config,
        "dataset_type": config.data.dataset_type,
        "model_type": config.model.model_type,
        "task_mode": config.data.task_mode,
        "learning_rates_tested": config.hyperparameters.learning_rates,
        "k_fold": config.training.k_fold,
        "epochs": config.training.epochs,
        "device": str(config.get_device()),
    }

    # Save configuration used for training
    if save_results:
        config_save_path = os.path.join(results_dir, "training_config.yaml")
        config.save_yaml(config_save_path)
        results["config_saved_to"] = config_save_path
        logger.info(f"Training configuration saved to: {config_save_path}")

    logger.info("Model training completed!")
    return results


def predict_model(
    model_path: str,
    data_path: str,
    dataset_type: str = None,
    model_type: str = None,
    target_task: str = None,
    config_path: str = None,
    output_path: str = "./predictions.csv",
    batch_size: int = 32,
) -> dict[str, any]:
    """
    Make predictions using trained models (simplified interface).

    This is a simplified wrapper around the predict_engines functions for easy use.
    For single molecule prediction, use predict_single_molecule from predict_engines.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (.pth).
    data_path : str
        Path to the data file or directory for prediction.
    dataset_type : str, optional
        Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").
        If None, will try to infer from model path.
    model_type : str, optional
        Type of model ("qspr", "GNN_M", "GNN_C_single", etc.).
        If None, will try to infer from model path.
    target_task : str, optional
        Target task name for single-task models (e.g., "Acid", "Gwi").
        Only needed for single-task models.
    config_path : str, optional
        Path to configuration file used during training.
    output_path : str, optional
        Path to save prediction results.
    batch_size : int, optional
        Batch size for prediction.

    Returns
    -------
    Dict[str, any]
        Dictionary containing prediction results and statistics.
    """
    logger.info("Starting prediction using scripts.predict_model wrapper")

    # Call the engine function with all parameters
    return engine_predict_model(
        model_path=model_path,
        data_path=data_path,
        dataset_type=dataset_type,
        model_type=model_type,
        target_task=target_task,
        config_path=config_path,
        output_path=output_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Training Pipeline")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "predict"],
        required=True,
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to data directory or file"
    )
    parser.add_argument("--config-path", help="Path to configuration file")
    parser.add_argument(
        "--model-path", help="Path to trained model file (for prediction mode)"
    )
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--output-path", help="Output file path (for prediction mode)")
    parser.add_argument(
        "--dataset-type",
        choices=["QSPR", "GNN_M", "GNN_C", "GNN_E"],
        help="Dataset type",
    )
    parser.add_argument("--model-type", help="Model type")
    parser.add_argument("--target-task", help="Target task for single-task models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if args.mode == "prepare":
        results = prepare_data(args.data_path, output_dir=args.output_dir)

    elif args.mode == "train":
        results = train_model(
            config_path=args.config_path,
            data_path=args.data_path,
            results_dir=args.output_dir,
        )

    elif args.mode == "predict":
        if not args.model_path:
            logger.error("--model-path is required for prediction mode")
            sys.exit(1)

        output_path = args.output_path or "./predictions.csv"

        results = predict_model(
            model_path=args.model_path,
            data_path=args.data_path,
            dataset_type=args.dataset_type,
            model_type=args.model_type,
            target_task=args.target_task,
            config_path=args.config_path,
            output_path=output_path,
            batch_size=args.batch_size,
        )

    logger.info("Pipeline completed successfully!")
    logger.info(f"Results: {results}")
