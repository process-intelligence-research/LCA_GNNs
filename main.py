#!/usr/bin/env python3
"""
Main entry point for the LCA GNN project.

This script provides environmental impact prediction for molecules using trained GNN models.
The primary focus is single molecule inference from SMILES strings.

Primary Usage - Single Molecule Inference:
    python main.py --workflow inference --model_path path/to/model.pth --smiles "CCO"
    python main.py --workflow inference --model_path path/to/model.pth --smiles "CCO" --multitask
    python main.py --workflow inference --model_path path/to/model.pth --smiles "CCO" --dataset_type "GNN_E" --country_name "Germany"

Additional Functions (for research/development):
    python main.py --workflow batch --model_path path/to/model.pth --data_path test_data.xlsx
    python main.py --workflow config   # Setup configuration templates
    python main.py --workflow data     # Data preparation (requires proprietary dataset)
    python main.py --workflow train    # Model training (requires proprietary dataset)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src import scripts
from src.config.training_config import TrainingConfiguration
from src.engines.predict_engines import predict_single_molecule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lca_gnn.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_configuration(config_path: str = None) -> TrainingConfiguration:
    """
    Set up or load configuration for the LCA GNN project.

    Parameters
    ----------
    config_path : str, optional
        Path to existing configuration file. If None, creates default configuration.

    Returns
    -------
    TrainingConfiguration
        Configuration object for the project.
    """
    logger.info("Setting up configuration...")

    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from: {config_path}")
        config = TrainingConfiguration.from_yaml(config_path)
    else:
        logger.info("Creating default configuration...")
        config = TrainingConfiguration()

        # Set default paths
        config.data.raw_data_path = "data/raw"
        config.data.processed_data_path = "data/processed"
        config.training.model_save_path = "trained_models"
        config.training.results_path = "results"

        # Create directories if they don't exist
        os.makedirs(config.data.raw_data_path, exist_ok=True)
        os.makedirs(config.data.processed_data_path, exist_ok=True)
        os.makedirs(config.training.model_save_path, exist_ok=True)
        os.makedirs(config.training.results_path, exist_ok=True)

        # Save default configuration
        default_config_path = "configs/default_config.yaml"
        os.makedirs(os.path.dirname(default_config_path), exist_ok=True)
        config.save_yaml(default_config_path)
        logger.info(f"Default configuration saved to: {default_config_path}")

    logger.info("Configuration setup complete!")
    return config


def prepare_data_workflow(config: TrainingConfiguration, data_path: str = None):
    """
    Prepare data for training (requires proprietary dataset).

    Note: This function is for research/development use only.
    The training dataset is not publicly available.

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration object containing data processing parameters.
    data_path : str, optional
        Path to proprietary training data file.
    """
    logger.info("Starting data preparation workflow...")
    logger.warning(
        "This function requires proprietary training data which is not publicly available."
    )

    if data_path is None:
        data_path = os.path.join(config.data.raw_data_path, "data.xlsx")

    if not os.path.exists(data_path):
        logger.error(f"Training data file not found: {data_path}")
        logger.info(
            "This function requires access to the proprietary training dataset."
        )
        logger.info("For inference only, use: python main.py --workflow inference")
        return False

    try:
        scripts.prepare_data(
            data_path=data_path,
            dataset_type=config.data.dataset_type,
            output_dir=config.data.processed_data_path,
        )
        logger.info("Data preparation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return False


def train_model_workflow(config: TrainingConfiguration):
    """
    Train models using the configuration (requires proprietary dataset).

    Note: This function is for research/development use only.
    Model training requires the proprietary training dataset.

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration object containing training parameters.
    """
    logger.info("Starting model training workflow...")
    logger.warning(
        "Model training requires proprietary training data which is not publicly available."
    )

    try:
        scripts.train_model(
            config=config,
            dataset_path=config.data.processed_data_path,
            model_save_path=config.training.model_save_path,
        )
        logger.info("Model training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.info(
            "For inference with pre-trained models, use: python main.py --workflow inference"
        )
        return False


def final_model_training_workflow(config: TrainingConfiguration):
    """
    Train final models and perform comprehensive evaluation (requires proprietary dataset).

    Note: This function is for research/development use only.
    Final model training requires the proprietary training dataset.

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration object containing training parameters.
    """
    logger.info("Starting final model training and evaluation workflow...")
    logger.warning(
        "Final model training requires proprietary training data which is not publicly available."
    )

    try:
        # Train both single-task and multi-task models
        for model_type in ["single", "multi"]:
            logger.info(f"Training {model_type}-task models...")

            # Update config for current model type
            config.model.task_type = model_type

            scripts.train_model(
                config=config,
                dataset_path=config.data.processed_data_path,
                model_save_path=os.path.join(
                    config.training.model_save_path, model_type
                ),
                final_training=True,
            )

        logger.info("Final model training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Final model training failed: {e}")
        logger.info(
            "For inference with pre-trained models, use: python main.py --workflow inference"
        )
        return False


def single_molecule_inference_workflow(
    model_path: str,
    smiles: str,
    country_name: str = None,
    energy_mix: dict = None,
    dataset_type: str = "GNN_C",
    model_type: str = None,
    target_task: str = None,
    multitask: bool = False,
):
    """
    Perform single molecule inference using trained models.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    smiles : str
        SMILES string of the molecule to predict.
    country_name : str, optional
        Country name for GNN_C and GNN_E models. Required for GNN_C models.
        For GNN_E models, country name will be used to automatically retrieve
        the energy mix from data/raw/energy_mapping.json. Available countries
        are listed in data/raw/energy_mapping.json. Examples: "Germany",
        "United States", "Japan", "China", etc.
    energy_mix : dict, optional
        Custom energy mix dictionary for GNN_E models. If not provided and
        country_name is specified for GNN_E models, energy mix will be
        automatically retrieved. Only needed for custom energy scenarios.
    dataset_type : str, optional
        Type of dataset ("GNN_C", "GNN_E", "GNN_M").
    model_type : str, optional
        Type of model. If None, inferred from model_path.
    target_task : str, optional
        Target task for single-task models. Available tasks:
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
    multitask : bool, optional
        Whether to use multitask inference.
    """
    logger.info("Starting single molecule inference workflow...")
    logger.info(f"SMILES: {smiles}")
    logger.info(f"Model: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None

    try:
        results = predict_single_molecule(
            model_path=model_path,
            smiles=smiles,
            country_name=country_name,
            energy_mix=energy_mix,
            dataset_type=dataset_type,
            model_type=model_type,
            target_task=target_task,
            multitask=multitask,
        )

        logger.info("Single molecule inference completed successfully!")
        logger.info("Results:")

        if multitask or len(results["predictions"]) > 1:
            logger.info("Environmental impact predictions:")
            for category, value in results["predictions"].items():
                logger.info(f"  {category}: {value:.6e}")
        else:
            category = list(results["predictions"].keys())[0]
            value = list(results["predictions"].values())[0]
            logger.info(f"  {category}: {value:.6e}")

        # Save results
        results_file = (
            f"results/single_molecule_prediction_{smiles.replace('/', '_')}.json"
        )
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {results_file}")
        return results

    except Exception as e:
        logger.error(f"Single molecule inference failed: {e}")
        return None


def batch_inference_workflow(
    config: TrainingConfiguration, data_path: str, model_path: str
):
    """
    Perform batch inference on a dataset.

    Expected test file format (Excel):
    - Column 'SMILES': SMILES strings of molecules
    - Column 'country_name': Country names (for GNN_C/GNN_E models, optional)
    - Additional columns are ignored

    Example test_data.xlsx:
    | SMILES | country_name | compound_name |
    |--------|--------------|---------------|
    | CCO    | Germany      | Ethanol       |
    | CC(=O)O| Japan        | Acetic acid   |
    | c1ccccc1| China       | Benzene       |

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration object.
    data_path : str
        Path to Excel file with SMILES and optional country_name columns.
    model_path : str
        Path to trained model.
    """
    logger.info("Starting batch inference workflow...")
    logger.info(
        "Expected file format: Excel with 'SMILES' column (and optional 'country_name' column)"
    )

    try:
        results = scripts.predict_model(
            model_path=model_path,
            data_path=data_path,
            output_path=os.path.join(
                config.training.results_path, "batch_predictions.csv"
            ),
        )

        logger.info("Batch inference completed successfully!")
        logger.info(f"Results saved to: {results['predictions_saved_to']}")
        return results

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        return None


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="LCA GNN Project - Environmental Impact Prediction from SMILES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Primary Usage - Single Molecule Inference:
    # Single environmental impact prediction
    python main.py --workflow inference --model_path trained_models/GNN_C_Gwi.pth --smiles "CCO" --target_task "Gwi"
    
    # Multi-task prediction (all 15 environmental impacts)
    python main.py --workflow inference --model_path trained_models/GNN_C_multi.pth --smiles "CCO" --dataset_type "GNN_C" --multitask
    
    # GNN_E model with country-based energy mix
    python main.py --workflow inference --model_path trained_models/GNN_E_multi.pth --smiles "CCO" --dataset_type "GNN_E" --country_name "Germany" --multitask
    
    # GNN_M model (molecular structure only)
    python main.py --workflow inference --model_path trained_models/GNN_M_multi.pth --smiles "CCO" --dataset_type "GNN_M" --multitask

Batch Inference:
    # Batch prediction (requires Excel file with 'SMILES' column)
    python main.py --workflow batch --model_path trained_models/model.pth --data_path test_molecules.xlsx

Research/Development Functions (require proprietary dataset):
    python main.py --workflow config      # Setup configuration templates
    python main.py --workflow data        # Data preparation
    python main.py --workflow train       # Model training
    python main.py --workflow final_train # Final model training
        """,
    )

    parser.add_argument(
        "--workflow",
        choices=["inference", "batch", "config", "data", "train", "final_train"],
        required=True,
        help="Primary: 'inference' for single molecules, 'batch' for multiple molecules. Others require proprietary dataset.",
    )

    parser.add_argument("--config_path", help="Path to configuration file")
    parser.add_argument("--data_path", help="Path to data file")
    parser.add_argument("--model_path", help="Path to trained model file")
    parser.add_argument("--smiles", help="SMILES string for single molecule inference")
    parser.add_argument(
        "--country_name",
        default="Germany",
        help="Country name for GNN_C and GNN_E models (see data/raw/energy_mapping.json for available countries)",
    )
    parser.add_argument(
        "--dataset_type", default="GNN_C", help="Dataset type (GNN_C, GNN_E, GNN_M)"
    )
    parser.add_argument(
        "--model_type", help="Model type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--target_task",
        help="Target task for single-task models (Acid, Gwi, CTUe, ADP_f, Eutro_f, Eutro_m, Eutro_t, CTUh, Ionising, Soil, ADP_e, ODP, human_health, Photo, Water_use)",
    )
    parser.add_argument(
        "--multitask", action="store_true", help="Use multitask inference"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LCA GNN PROJECT - ENVIRONMENTAL IMPACT PREDICTION")
    logger.info("=" * 60)

    # Setup configuration
    config = setup_configuration(args.config_path)

    success = True

    if args.workflow == "inference":
        # Single molecule inference - PRIMARY USE CASE
        if not args.model_path or not args.smiles:
            logger.error("Model path and SMILES are required for inference")
            logger.info(
                "Example: python main.py --workflow inference --model_path model.pth --smiles 'CCO'"
            )
            success = False
        else:
            result = single_molecule_inference_workflow(
                model_path=args.model_path,
                smiles=args.smiles,
                country_name=args.country_name,
                dataset_type=args.dataset_type,
                model_type=args.model_type,
                target_task=args.target_task,
                multitask=args.multitask,
            )
            success = result is not None

    elif args.workflow == "batch":
        # Batch inference
        if not args.model_path or not args.data_path:
            logger.error("Model path and data path are required for batch inference")
            logger.info(
                "Example: python main.py --workflow batch --model_path model.pth --data_path test_data.xlsx"
            )
            success = False
        else:
            result = batch_inference_workflow(config, args.data_path, args.model_path)
            success = result is not None

    elif args.workflow == "config":
        # Configuration setup only
        logger.info("Configuration setup completed!")
        logger.info("This creates default configuration templates for development use.")

    elif args.workflow == "data":
        # Data preparation only - requires proprietary dataset
        logger.warning("Data preparation requires proprietary training dataset.")
        success = prepare_data_workflow(config, args.data_path)

    elif args.workflow == "train":
        # Model training only - requires proprietary dataset
        logger.warning("Model training requires proprietary training dataset.")
        success = train_model_workflow(config)

    elif args.workflow == "final_train":
        # Final model training only - requires proprietary dataset
        logger.warning("Final model training requires proprietary training dataset.")
        success = final_model_training_workflow(config)

    if success:
        logger.info("=" * 60)
        logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ WORKFLOW FAILED!")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
