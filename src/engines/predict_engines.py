# For nice result table
"""
Prediction engines for LCA GNN models.

This module provides prediction functionality for Life Cycle Assessment (LCA) Graph Neural Networks.
It includes functions for both batch predictions and single molecule predictions, with automatic
denormalization of results using impact statistics.

Main Functions:
--------------
predict_model():
    Main batch prediction function for all model types. Handles datasets and saves results to CSV.

predict_single_molecule():
    Main single molecule prediction function. Handles both single-task and multi-task models.
    Use multitask=True for multi-task predictions.

Convenience Functions:
---------------------
predict_single_task_model(): Wrapper for single-task batch predictions
predict_multi_task_model(): Wrapper for multi-task batch predictions
predict_single_molecule_multitask(): Wrapper for multi-task single molecule predictions

Legacy Functions:
----------------
prediction(): Original unified prediction function (used internally)
prediction_single(): Original single-task prediction function (used internally)
prediction_multi(): Original multi-task prediction function (used internally)

Features:
---------
- Automatic model type inference from file paths
- Automatic denormalization using impact statistics from impact_stats.json
- Support for GNN_C, GNN_E, and GNN_M models for single molecule inference
- QSPR models only supported for batch prediction (requires pre-computed descriptors)
- Comprehensive error handling and logging
- Backward compatibility with existing code

Usage Examples:
--------------
# Single molecule prediction (single-task)
results = predict_single_molecule(
    model_path="path/to/model.pth",
    smiles="CCO",
    country_name="Germany",
    dataset_type="GNN_C",
    target_task="Acid"
)

# Single molecule prediction (multi-task)
results = predict_single_molecule(
    model_path="path/to/model.pth",
    smiles="CCO",
    country_name="Germany",
    dataset_type="GNN_C",
    multitask=True
)

# Batch predictions
results = predict_model(
    model_path="path/to/model.pth",
    data_path="path/to/data.xlsx",
    dataset_type="GNN_C",
    model_type="GNN_C_multi"
)
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.loader.dataloader
from torch_geometric.loader import DataLoader

from src.data_processing.make_dataset import BaseMolecularDataset

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.training_config import TrainingConfiguration
from src.data_processing.make_dataset import (
    GNN_C_dataset,
    GNN_E_dataset,
    GNN_M_dataset,
    QSPR_dataset,
)
from src.models.models import (
    GNN_M,
    GNN_C_multi,
    GNN_C_single,
    GNN_E_multi,
    GNN_E_single,
    qspr,
)

# Suppress pandas warnings about DataFrame creation
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Shape of passed values.*")
warnings.filterwarnings("ignore", message=".*indices imply.*")

# Set up logging
logger = logging.getLogger(__name__)


def _load_impact_statistics():
    """
    Load impact statistics (mean and std) from JSON file.

    Returns
    -------
    dict
        Dictionary containing mean and std for each impact category.
    """
    # Get the path to the data directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "..", "data", "raw")
    stats_path = os.path.join(data_dir, "impact_stats.json")

    try:
        with open(stats_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Impact statistics file not found at {stats_path}")
        return {}


def _denormalize_predictions(
    predictions: np.ndarray, column_names: list[str], impact_stats: dict
) -> np.ndarray:
    """
    Denormalize predictions using impact statistics.

    Parameters
    ----------
    predictions : np.ndarray
        Normalized predictions from the model.
    column_names : list[str]
        Names of the impact categories corresponding to predictions.
    impact_stats : dict
        Dictionary containing mean and std for each impact category.

    Returns
    -------
    np.ndarray
        Denormalized predictions.
    """
    # Map column names to impact category keys
    category_mapping = {
        "Acid": "AC",
        "Gwi": "CC",
        "CTUe": "ECO",
        "ADP_f": "ER",
        "Eutro_f": "EUf",
        "Eutro_m": "EUm",
        "Eutro_t": "EUt",
        "CTUh": "HT",
        "Ionising": "IR",
        "Soil": "LU",
        "ADP_e": "MR",
        "ODP": "OD",
        "human_health": "PMF",
        "Photo": "POF",
        "Water_use": "WU",
    }

    denormalized = predictions.copy()

    for i, col_name in enumerate(column_names):
        if col_name in category_mapping:
            category_key = category_mapping[col_name]
            if category_key in impact_stats:
                mean = impact_stats[category_key]["mean"]
                std = impact_stats[category_key]["std"]
                # Denormalize: original = normalized * std + mean
                denormalized[i] = predictions[i] * std + mean
            else:
                logger.warning(f"No statistics found for category: {category_key}")
        else:
            logger.warning(f"Unknown column name: {col_name}")

    return denormalized


def _denormalize_single_prediction(
    prediction: float, column_name: str, impact_stats: dict
) -> float:
    """
    Denormalize a single prediction using impact statistics.

    Parameters
    ----------
    prediction : float
        Normalized prediction from the model.
    column_name : str
        Name of the impact category corresponding to the prediction.
    impact_stats : dict
        Dictionary containing mean and std for each impact category.

    Returns
    -------
    float
        Denormalized prediction.
    """
    # Map column names to impact category keys
    category_mapping = {
        "Acid": "AC",
        "Gwi": "CC",
        "CTUe": "ECO",
        "ADP_f": "ER",
        "Eutro_f": "EUf",
        "Eutro_m": "EUm",
        "Eutro_t": "EUt",
        "CTUh": "HT",
        "Ionising": "IR",
        "Soil": "LU",
        "ADP_e": "MR",
        "ODP": "OD",
        "human_health": "PMF",
        "Photo": "POF",
        "Water_use": "WU",
    }

    if column_name in category_mapping:
        category_key = category_mapping[column_name]
        if category_key in impact_stats:
            mean = impact_stats[category_key]["mean"]
            std = impact_stats[category_key]["std"]
            # Denormalize: original = normalized * std + mean
            return prediction * std + mean
        else:
            logger.warning(f"No statistics found for category: {category_key}")
    else:
        logger.warning(f"Unknown column name: {column_name}")

    return prediction


def prediction(
    model: torch.nn.Module,
    test_loader: torch_geometric.loader.DataLoader,
    device: torch.device,
    results,
    number,
    mean,
    std,
    mode="single",
):
    """
    Unified prediction function for both single and multi-task models.

    Args:
        model: your trained model
        test_loader: DataLoader with the test set
        device: for GPU acceleration
        results: results container (dict for single, DataFrame for multi)
        number: fold number
        mean: mean values for denormalization
        std: std values for denormalization
        mode: "single" for single-task, "multi" for multi-task

    Returns
    -------
        Updated results (dict or DataFrame)
    """
    model.eval()

    if mode == "single":
        # Original prediction_single logic
        predict = []
        real = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                tmp_pred_data = model(batch).squeeze()
                tmp_real_data = batch.y
                # save 15 impact categories not sub

                predict.append((tmp_pred_data * std + mean).item())
                real.append((tmp_real_data * std + mean).item())

            results["pre_" + str(number)] = predict
            results["real_" + str(number)] = real
        return results

    elif mode == "multi":
        # Original prediction_multi logic
        empty_array = np.empty((0, 30))
        with torch.no_grad():
            for batch in test_loader:
                combined_list = []
                batch = batch.to(device)
                tmp_pred_data = (model(batch)[0] * std + mean).cpu().numpy()
                tmp_real_data = (batch.y[0] * std + mean).cpu().numpy()
                for i in range(len(tmp_pred_data)):
                    combined_list.append(tmp_pred_data[i])
                    combined_list.append(tmp_real_data[i])
                empty_array = np.vstack((empty_array, np.array(combined_list)))

        # Create appropriate column names if results is empty
        if results.empty:
            # Create column names for 15 environmental impact categories (pred and real pairs)
            impact_categories = [
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
            columns = []
            for category in impact_categories:
                columns.extend([f"pred_{category}", f"real_{category}"])

            # Ensure we have the right number of columns for the data
            actual_cols = empty_array.shape[1] if empty_array.size > 0 else 30
            if len(columns) != actual_cols:
                # Fallback: create generic column names if mismatch
                columns = [f"col_{i}" for i in range(actual_cols)]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = pd.DataFrame(empty_array, columns=columns[:actual_cols])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Ensure column count matches
                if empty_array.shape[1] == len(results.columns):
                    new_df = pd.DataFrame(empty_array, columns=results.columns)
                else:
                    # Handle column mismatch - truncate or pad as needed
                    if empty_array.shape[1] < len(results.columns):
                        # Pad with zeros if fewer columns
                        padding = np.zeros(
                            (
                                empty_array.shape[0],
                                len(results.columns) - empty_array.shape[1],
                            )
                        )
                        empty_array = np.hstack([empty_array, padding])
                    else:
                        # Truncate if more columns
                        empty_array = empty_array[:, : len(results.columns)]
                    new_df = pd.DataFrame(empty_array, columns=results.columns)
                results = pd.concat([results, new_df], ignore_index=True)
        return results

    else:
        raise ValueError("mode must be 'single' or 'multi'")


def prediction_single(
    model: torch.nn.Module,
    test_loader: torch_geometric.loader.DataLoader,
    device: torch.device,
    results,
    number,
    mean,
    std,
) -> pd.DataFrame:
    """
    Evaluates on test set, never seen by the model. Outputs a pandas Dataframe with all info for each entry.

    Args:
        model: your trained model
        test_loader: DataLoader with the test set
        device: for GPU acceleration
        config: contain .mean and .std of target values, to bring them back to how they were in the dataset before
                normalization

    Returns
    -------
        a Dataframe object with Name, country, type and ids, real value and prediction

    """
    return prediction(
        model, test_loader, device, results, number, mean, std, mode="single"
    )


def prediction_multi(
    model: torch.nn.Module,
    test_loader: torch_geometric.loader.DataLoader,
    device: torch.device,
    results,
    number,
    mean,
    std,
) -> pd.DataFrame:
    """
    Evaluates on test set, never seen by the model. Outputs a pandas Dataframe with all info for each entry.

    Args:
        model: your trained model
        test_loader: DataLoader with the test set
        device: for GPU acceleration
        config: contain .mean and .std of target values, to bring them back to how they were in the dataset before
                normalization

    Returns
    -------
        a Dataframe object with Name, country, type and ids, real value and prediction

    """
    return prediction(
        model, test_loader, device, results, number, mean, std, mode="multi"
    )


# Enhanced prediction functions moved from scripts.py


def predict_model(
    model_path: str,
    data_path: str,
    dataset_type: str = None,
    model_type: str = None,
    target_task: str = None,
    config_path: str = None,
    config: TrainingConfiguration = None,
    output_path: str = "./predictions.csv",
    batch_size: int = 32,
) -> dict[str, any]:
    """
    Make predictions using trained models.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (.pth).
    data_path : str
        Path to the data file or directory for prediction.
    dataset_type : str, optional
        Type of dataset ("GNN_M", "GNN_C", "GNN_E").
        If None, will try to infer from model path.
    model_type : str, optional
        Type of model ("GNN_M", "GNN_C_single", etc.).
        If None, will try to infer from model path.
    target_task : str, optional
        Target task name for single-task models (e.g., "Acid", "Gwi").
        Only needed for single-task models.
    config_path : str, optional
        Path to configuration file used during training.
    config : TrainingConfiguration, optional
        Configuration object used during training.
    output_path : str, optional
        Path to save prediction results.
    batch_size : int, optional
        Batch size for prediction.

    Returns
    -------
    Dict[str, any]
        Dictionary containing prediction results and statistics.
    """
    logger.info(f"Loading model from: {model_path}")

    # Try to infer model details from path if not provided
    if dataset_type is None or model_type is None:
        inferred_info = _infer_model_info_from_path(model_path)
        dataset_type = dataset_type or inferred_info.get("dataset_type")
        model_type = model_type or inferred_info.get("model_type")
        target_task = target_task or inferred_info.get("target_task")

    logger.info(f"Using dataset_type: {dataset_type}, model_type: {model_type}")
    if target_task:
        logger.info(f"Target task: {target_task}")

    # Read the Excel file for batch processing
    if not data_path.endswith(".xlsx"):
        raise ValueError("Batch prediction currently only supports Excel files (.xlsx)")

    logger.info(f"Reading input data from: {data_path}")
    input_df = pd.read_excel(data_path)

    # Check required columns
    if "SMILES" not in input_df.columns:
        available_cols = list(input_df.columns)
        raise ValueError(
            f"Input Excel file must contain 'SMILES' column. Available columns: {available_cols}"
        )

    logger.info(f"Found {len(input_df)} molecules to process")

    # Determine if this is a multitask model
    is_multitask = "multi" in model_type.lower() if model_type else False

    # Process molecules one by one
    all_results = []
    all_predictions = []
    column_names = None

    for idx, row in input_df.iterrows():
        smiles = row["SMILES"]
        country_name = row.get("country_name", None)

        try:
            # Use single molecule prediction for each row
            result = predict_single_molecule(
                model_path=model_path,
                smiles=smiles,
                country_name=country_name,
                dataset_type=dataset_type,
                model_type=model_type,
                target_task=target_task,
                config_path=config_path,
                config=config,
                multitask=is_multitask,
            )

            # Extract predictions as array
            if result and "predictions" in result:
                pred_array = [
                    result["predictions"][col] for col in result["column_names"]
                ]
                all_predictions.append(pred_array)

                if column_names is None:
                    column_names = result["column_names"]

                # Store additional info
                result_row = {
                    "sample_id": idx,
                    "smiles": smiles,
                    "country_name": country_name,
                    **result["predictions"],
                }
                all_results.append(result_row)

            else:
                logger.warning(f"Failed to predict for molecule {idx}: {smiles}")
                # Add placeholder values
                if column_names is None:
                    column_names = [target_task] if target_task else ["prediction"]
                all_predictions.append([np.nan] * len(column_names))

                result_row = {
                    "sample_id": idx,
                    "smiles": smiles,
                    "country_name": country_name,
                    **{col: np.nan for col in column_names},
                }
                all_results.append(result_row)

        except Exception as e:
            logger.error(f"Error processing molecule {idx} ({smiles}): {e}")
            # Add placeholder values for failed predictions
            if column_names is None:
                column_names = [target_task] if target_task else ["prediction"]
            all_predictions.append([np.nan] * len(column_names))

            result_row = {
                "sample_id": idx,
                "smiles": smiles,
                "country_name": country_name,
                **{col: np.nan for col in column_names},
            }
            all_results.append(result_row)

    # Convert to arrays
    all_predictions = np.array(all_predictions)

    # Save predictions
    pred_df = pd.DataFrame(all_results)
    pred_df.to_csv(output_path, index=False)

    results = {
        "model_path": model_path,
        "data_path": data_path,
        "dataset_type": dataset_type,
        "model_type": model_type,
        "target_task": target_task,
        "num_samples": len(input_df),
        "num_features": all_predictions.shape[1],
        "predictions_saved_to": output_path,
        "column_names": column_names,
        "prediction_stats": {
            "mean": np.nanmean(all_predictions, axis=0).tolist(),
            "std": np.nanstd(all_predictions, axis=0).tolist(),
            "min": np.nanmin(all_predictions, axis=0).tolist(),
            "max": np.nanmax(all_predictions, axis=0).tolist(),
        },
    }

    logger.info(f"Predictions saved to: {output_path}")
    logger.info(
        f"Predicted {len(input_df)} samples with {all_predictions.shape[1]} features each"
    )

    return results


# Removed individual functions - functionality moved to predict_model


def predict_single_molecule(
    model_path: str,
    smiles: str,
    country_name: str = None,
    energy_mix: dict = None,
    dataset_type: str = "GNN_C",
    model_type: str = None,
    target_task: str = None,
    config_path: str = None,
    config: TrainingConfiguration = None,
    multitask: bool = False,
) -> dict[str, any]:
    """
    Make predictions for a single molecule.

    Parameters
    ----------
    model_path : str
        Path to the trained model file (.pth).
    smiles : str
        SMILES string of the molecule.
    country_name : str, optional
        Country name for GNN_C and GNN_E models. Required for GNN_C models.
        For GNN_E models, either country_name or energy_mix must be provided.
        When country_name is provided for GNN_E models, the energy mix will be
        automatically retrieved from data/raw/energy_mapping.json.
        Available countries are listed in data/raw/energy_mapping.json and
        data/raw/country_mapping.json. Examples: "Germany", "United States",
        "Japan", "China", etc.
    energy_mix : dict, optional
        Custom energy mix dictionary for GNN_E models. If not provided and
        country_name is given for GNN_E models, energy mix will be automatically
        retrieved by country name. Expected keys when providing custom energy mix:
        ["Coal, peat and oil shale", "Crude, NGL and feedstocks",
         "Oil products", "Natural gas", "Renewables and waste",
         "Electricity", "Heat"]
    dataset_type : str, optional
        Type of dataset ("GNN_M", "GNN_C", "GNN_E").
    model_type : str, optional
        Type of model ("GNN_M", "GNN_C_single", etc.).
    target_task : str, optional
        Target task name for single-task models. Available tasks:
        - "Acid": Acidification potential
        - "Gwi": Global warming impact
        - "CTUe": Ecotoxicity potential
        - "ADP_f": Abiotic depletion potential (fossil fuels)
        - "Eutro_f": Eutrophication potential (freshwater)
        - "Eutro_m": Eutrophication potential (marine)
        - "Eutro_t": Eutrophication potential (terrestrial)
        - "CTUh": Human toxicity potential
        - "Ionising": Ionising radiation potential
        - "Soil": Land use potential
        - "ADP_e": Abiotic depletion potential (elements)
        - "ODP": Ozone depletion potential
        - "human_health": Particulate matter formation potential
        - "Photo": Photochemical ozone formation potential
        - "Water_use": Water use potential
    config_path : str, optional
        Path to configuration file used during training.
    config : TrainingConfiguration, optional
        Configuration object used during training.
    multitask : bool, optional
        If True, forces multitask model inference regardless of model_type.

    Returns
    -------
    Dict[str, any]
        Dictionary containing prediction results.
    """
    logger.info(f"Making single molecule prediction for SMILES: {smiles}")

    # Try to infer model details from path if not provided
    if dataset_type is None or model_type is None:
        inferred_info = _infer_model_info_from_path(model_path)
        dataset_type = dataset_type or inferred_info.get("dataset_type")
        model_type = model_type or inferred_info.get("model_type")
        target_task = target_task or inferred_info.get("target_task")

    # QSPR models are not supported for single molecule inference
    # They require pre-computed molecular descriptors
    if dataset_type == "QSPR":
        raise ValueError(
            "QSPR models are not supported for single molecule inference. "
            "QSPR models require pre-computed molecular descriptors which cannot be "
            "generated from SMILES strings alone. Use batch prediction with "
            "pre-computed descriptor data instead."
        )

    # Override model_type if multitask is explicitly requested
    if multitask:
        model_type = f"{dataset_type}_multi"

    logger.info(f"Using dataset_type: {dataset_type}, model_type: {model_type}")
    if target_task:
        logger.info(f"Target task: {target_task}")

    # For GNN_M models, country and energy data are not used
    if dataset_type in ["GNN_M"]:
        actual_country_name = None
        actual_energy_mix = None
        logger.info(
            f"Setting country_name and energy_mix to None for {dataset_type} model"
        )
    else:
        actual_country_name = country_name
        actual_energy_mix = energy_mix

    # Validate that required parameters are provided for specific model types
    if dataset_type == "GNN_C" and actual_country_name is None:
        raise ValueError("country_name is required for GNN_C models")
    elif (
        dataset_type == "GNN_E"
        and actual_country_name is None
        and actual_energy_mix is None
    ):
        raise ValueError(
            "Either country_name or energy_mix is required for GNN_E models. "
            "Recommend using country_name for automatic energy mix lookup."
        )

    # Load configuration if provided, otherwise use defaults
    if config is None and config_path is not None:
        config = TrainingConfiguration.from_yaml(config_path)
    elif config is None:
        # Create minimal config for inference
        config = _create_inference_config(dataset_type, model_type, 1)

    # Load model
    device = config.get_device()
    model = _load_model_for_inference(model_type, dataset_type, model_path, device)

    # Create data object for single molecule
    data = BaseMolecularDataset.create_single_molecule_data_static(
        smiles, actual_country_name, actual_energy_mix, dataset_type
    )

    data_loader = DataLoader([data], batch_size=1, shuffle=False)

    # Make prediction
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)

            # Handle different output formats
            if isinstance(pred, torch.Tensor):
                prediction = pred.cpu().numpy().flatten()
            else:
                # Handle tuple outputs (like multi-task models)
                prediction = pred[0].cpu().numpy().flatten()
            break  # Only process the single batch

    # Create meaningful column names based on model type and target
    column_names = _get_prediction_column_names(
        model_type, target_task, len(prediction)
    )

    # Load impact statistics and denormalize predictions
    impact_stats = _load_impact_statistics()
    if impact_stats:
        if len(prediction) == 1:
            # Single task model
            denormalized_prediction = _denormalize_single_prediction(
                prediction[0], column_names[0], impact_stats
            )
            prediction = np.array([denormalized_prediction])
        else:
            # Multi-task model
            prediction = _denormalize_predictions(
                prediction, column_names, impact_stats
            )

    # Create results dictionary
    results = {
        "smiles": smiles,
        "country_name": actual_country_name,
        "energy_mix": actual_energy_mix,
        "dataset_type": dataset_type,
        "model_type": model_type,
        "target_task": target_task if not multitask else "multitask",
        "predictions": dict(zip(column_names, prediction.tolist())),
        "prediction_array": prediction.tolist(),
        "column_names": column_names,
        "denormalized": bool(impact_stats),  # Indicate if denormalization was applied
        "num_categories": len(column_names),
    }

    logger.info(f"Single molecule prediction completed: {results['predictions']}")
    return results


# Convenience functions for backward compatibility and easier use


def predict_single_task_model(
    model_path: str,
    data_path: str,
    target_task: str,
    dataset_type: str = "GNN_C",
    output_path: str = None,
    batch_size: int = 32,
) -> dict[str, any]:
    """
    Convenience function for single-task batch predictions.

    This is a wrapper around predict_model for single-task models.
    """
    if output_path is None:
        output_path = f"./predictions_{dataset_type}_{target_task}.csv"

    model_type = f"{dataset_type}_single"

    return predict_model(
        model_path=model_path,
        data_path=data_path,
        dataset_type=dataset_type,
        model_type=model_type,
        target_task=target_task,
        output_path=output_path,
        batch_size=batch_size,
    )


def predict_multi_task_model(
    model_path: str,
    data_path: str,
    dataset_type: str = "GNN_C",
    output_path: str = None,
    batch_size: int = 32,
) -> dict[str, any]:
    """
    Convenience function for multi-task batch predictions.

    This is a wrapper around predict_model for multi-task models.
    """
    if output_path is None:
        output_path = f"./predictions_{dataset_type}_multi_task.csv"

    model_type = f"{dataset_type}_multi"

    return predict_model(
        model_path=model_path,
        data_path=data_path,
        dataset_type=dataset_type,
        model_type=model_type,
        output_path=output_path,
        batch_size=batch_size,
    )


def predict_single_molecule_multitask(
    model_path: str,
    smiles: str,
    country_name: str = None,
    energy_mix: dict = None,
    dataset_type: str = "GNN_C",
    config_path: str = None,
    config: TrainingConfiguration = None,
) -> dict[str, any]:
    """
    Convenience function for multitask single molecule predictions.

    This is a wrapper around predict_single_molecule with multitask=True.
    """
    return predict_single_molecule(
        model_path=model_path,
        smiles=smiles,
        country_name=country_name,
        energy_mix=energy_mix,
        dataset_type=dataset_type,
        config_path=config_path,
        config=config,
        multitask=True,
    )


# Utility functions (private)


def _infer_model_info_from_path(model_path: str) -> dict[str, str]:
    """Infer model information from the file path."""
    path_lower = model_path.lower()
    info = {}

    # Infer dataset type
    if "qspr" in path_lower:
        info["dataset_type"] = "QSPR"
        info["model_type"] = "qspr"
    elif "gnn_m" in path_lower:
        info["dataset_type"] = "GNN_M"
        info["model_type"] = "GNN_M"
    elif "gnn_c" in path_lower:
        info["dataset_type"] = "GNN_C"
        if "multi" in path_lower:
            info["model_type"] = "GNN_C_multi"
        else:
            info["model_type"] = "GNN_C_single"
    elif "gnn_e" in path_lower:
        info["dataset_type"] = "GNN_E"
        if "multi" in path_lower:
            info["model_type"] = "GNN_E_multi"
        else:
            info["model_type"] = "GNN_E_single"

    # Try to infer target task for single-task models
    if "single" in info.get("model_type", ""):
        # Look for task names in the path
        task_names = [
            "acid",
            "gwi",
            "ctue",
            "adp_f",
            "eutro_f",
            "eutro_m",
            "eutro_t",
            "ctuh",
            "ionising",
            "soil",
            "adp_e",
            "odp",
            "human_health",
            "photo",
            "water_use",
        ]
        for task in task_names:
            if task in path_lower:
                info["target_task"] = task.capitalize()
                break

    return info


def _create_inference_config(
    dataset_type: str, model_type: str, batch_size: int
) -> TrainingConfiguration:
    """Create a minimal configuration for inference."""
    config = TrainingConfiguration()
    config.data.dataset_type = dataset_type
    config.model.model_type = model_type
    config.training.batch_size = batch_size

    # Set appropriate output features
    if model_type == "qspr":
        config.model.out_feature = 15  # Default for environmental impacts
    elif model_type == "GNN_M":
        config.model.out_feature = 15
    # GNN_C and GNN_E models have fixed architectures

    return config


def _load_model_for_inference(
    model_type: str, dataset_type: str, model_path: str, device: torch.device
) -> torch.nn.Module:
    """Load a model for inference."""

    # First, load the state dict to check output dimensions
    state_dict = torch.load(model_path, map_location=device)

    # For GNN_M and QSPR models, detect output size from the saved model
    if model_type in ["GNN_M", "qspr"]:
        # Check the final layer to determine output size
        if "fc13.weight" in state_dict:  # GNN_M final layer
            output_size = state_dict["fc13.weight"].shape[0]
        elif "linear_output.weight" in state_dict:  # QSPR final layer
            output_size = state_dict["linear_output.weight"].shape[0]
        else:
            # Default to 1 for single-task models
            output_size = 1

        logger.info(f"Detected output size for {model_type}: {output_size}")

        # Update model_type based on output size for better inference
        if output_size == 1 and model_type == "GNN_M":
            logger.info("Detected single-task GNN_M model (output_size=1)")
        elif output_size > 1 and model_type == "GNN_M":
            logger.info(f"Detected multi-task GNN_M model (output_size={output_size})")
    else:
        output_size = 15  # Default for other models

    # Create model instance based on type
    if model_type == "qspr":
        model = qspr(out_feature=output_size)
    elif model_type == "GNN_M":
        model = GNN_M(out_feature=output_size)
    elif model_type == "GNN_C_single":
        model = GNN_C_single()
    elif model_type == "GNN_C_multi":
        model = GNN_C_multi()
    elif model_type == "GNN_E_single":
        model = GNN_E_single()
    elif model_type == "GNN_E_multi":
        model = GNN_E_multi()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load trained weights
    try:
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise

    model = model.to(device)
    model.eval()

    return model


def _load_dataset_for_prediction(dataset_type: str, data_path: str) -> object:
    """Load dataset for prediction."""
    if data_path.endswith(".xlsx"):
        # If it's an Excel file, read it and create dataset
        dataexcel = pd.read_excel(data_path)

        # Determine root directory for dataset
        root_dir = str(Path(data_path).parent)

        if dataset_type == "QSPR":
            dataset = QSPR_dataset(root=root_dir, dataexcel=dataexcel)
        elif dataset_type == "GNN_M":
            dataset = GNN_M_dataset(root=root_dir, dataexcel=dataexcel)
        elif dataset_type == "GNN_C":
            dataset = GNN_C_dataset(root=root_dir, dataexcel=dataexcel)
        elif dataset_type == "GNN_E":
            dataset = GNN_E_dataset(root=root_dir, dataexcel=dataexcel)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    else:
        # Assume it's a directory with processed data
        if dataset_type == "QSPR":
            dataset = QSPR_dataset(root=data_path)
        elif dataset_type == "GNN_M":
            dataset = GNN_M_dataset(root=data_path)
        elif dataset_type == "GNN_C":
            dataset = GNN_C_dataset(root=data_path)
        elif dataset_type == "GNN_E":
            dataset = GNN_E_dataset(root=data_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return dataset


def _get_prediction_column_names(
    model_type: str, target_task: str, num_features: int
) -> list[str]:
    """Generate meaningful column names for predictions."""
    if num_features == 1:
        # Single output - use target task if available, otherwise generic name
        if target_task:
            return [target_task]
        else:
            # Try to infer from model type or use generic name
            return ["prediction"]
    elif num_features == 15 or "multi" in model_type:
        # Multi-task model - all environmental impact categories
        return [
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
        ][:num_features]
    else:
        # Generic names for other numbers of features
        return [f"prediction_{i}" for i in range(num_features)]
