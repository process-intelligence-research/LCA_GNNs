"""
Single molecule prediction test using the predict_engines interface.

This script demonstrates how to use the predict_single_molecule function
for single molecule inference with different model types.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.predict_engines import (
    predict_single_molecule,
    predict_single_molecule_multitask,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gnn_c_prediction():
    """Test GNN_C model prediction with country information."""
    logger.info("Testing GNN_C model prediction")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol
    country_name = "Germany"

    try:
        model_path = "./trained_models/GNN_C_single/GNN_C_Gwi_final_lr_5.00e-05.pth"

        logger.info(f"Predicting for molecule: {smiles}")
        logger.info(f"Country: {country_name}")
        logger.info(f"Using model: {model_path}")

        # Make prediction using the updated interface
        results = predict_single_molecule(
            model_path=model_path,
            smiles=smiles,
            country_name=country_name,
            dataset_type="GNN_C",
            target_task="Gwi",
        )

        logger.info("✓ GNN_C prediction completed!")
        logger.info(f"Molecule: {results['smiles']}")
        logger.info(f"Country: {results['country_name']}")
        logger.info(f"Prediction: {results['predictions']}")

        return results

    except Exception as e:
        logger.error(f"GNN_C prediction failed: {e}")
        return None


def test_gnn_e_prediction():
    """Test GNN_E model prediction with energy mix from country."""
    logger.info("Testing GNN_E model prediction")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol
    country_name = "Germany"  # Will use energy mix from JSON mapping

    try:
        model_path = "./trained_models/GNN_E_single/GNN_E_Gwi_final_lr_5.00e-04.pth"

        logger.info(f"Predicting for molecule: {smiles}")
        logger.info(f"Country (for energy mix): {country_name}")
        logger.info(f"Using model: {model_path}")

        # Make prediction using country-based energy mix lookup
        results = predict_single_molecule(
            model_path=model_path,
            smiles=smiles,
            country_name=country_name,  # This will lookup energy mix automatically
            dataset_type="GNN_E",
            target_task="Gwi",
        )

        logger.info("✓ GNN_E prediction completed!")
        logger.info(f"Molecule: {results['smiles']}")
        logger.info(f"Country: {results['country_name']}")
        logger.info(f"Prediction: {results['predictions']}")

        return results

    except Exception as e:
        logger.error(f"GNN_E prediction failed: {e}")
        return None


def test_gnn_e_prediction_custom_energy():
    """Test GNN_E model prediction with custom energy mix."""
    logger.info("Testing GNN_E model prediction with custom energy mix")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol
    custom_energy_mix = {
        "Coal, peat and oil shale": 0.3,
        "Crude, NGL and feedstocks": 0.0,
        "Oil products": 0.2,
        "Natural gas": 0.2,
        "Renewables and waste": 0.2,
        "Electricity": 0.1,
        "Heat": 0.0,
    }

    try:
        model_path = "./trained_models/GNN_E_single/GNN_E_Gwi_final_lr_5.00e-04.pth"

        logger.info(f"Predicting for molecule: {smiles}")
        logger.info(f"Custom energy mix: {custom_energy_mix}")
        logger.info(f"Using model: {model_path}")

        # Make prediction using custom energy mix
        results = predict_single_molecule(
            model_path=model_path,
            smiles=smiles,
            energy_mix=custom_energy_mix,
            dataset_type="GNN_E",
            target_task="Gwi",
        )

        logger.info("✓ GNN_E prediction with custom energy mix completed!")
        logger.info(f"Molecule: {results['smiles']}")
        logger.info("Energy mix: Custom")
        logger.info(f"Prediction: {results['predictions']}")

        return results

    except Exception as e:
        logger.error(f"GNN_E prediction with custom energy mix failed: {e}")
        return None


def test_multitask_prediction():
    """Test multitask model prediction for all environmental impact categories."""
    logger.info("Testing multitask model prediction")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol
    country_name = "Germany"

    try:
        # Test GNN_C multitask model
        model_path = "./trained_models/GNN_C_multi_best.pth"

        logger.info(f"Predicting for molecule: {smiles}")
        logger.info(f"Country: {country_name}")
        logger.info(f"Using multitask model: {model_path}")

        # Make prediction using multitask model
        results = predict_single_molecule_multitask(
            model_path=model_path,
            smiles=smiles,
            country_name=country_name,
            dataset_type="GNN_C",
        )

        logger.info("✓ Multitask prediction completed!")
        logger.info(f"Molecule: {results['smiles']}")
        logger.info(f"Country: {results['country_name']}")
        logger.info(f"Number of categories predicted: {results['num_categories']}")
        logger.info(f"Categories: {results['column_names']}")

        # Log first few predictions for brevity
        predictions = results["predictions"]
        logger.info("Sample predictions:")
        for i, (category, value) in enumerate(predictions.items()):
            if i < 5:  # Show first 5 predictions
                logger.info(f"  {category}: {value}")
            elif i == 5:
                logger.info(f"  ... and {len(predictions) - 5} more categories")
                break

        return results

    except Exception as e:
        logger.error(f"Multitask prediction failed: {e}")
        return None


def test_multitask_prediction_gnn_e():
    """Test GNN_E multitask model prediction for all environmental impact categories."""
    logger.info("Testing GNN_E multitask model prediction")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol
    country_name = "Germany"  # Will use energy mix from JSON mapping

    try:
        # Test GNN_E multitask model
        model_path = "./trained_models/GNN_E_multi_best.pth"

        logger.info(f"Predicting for molecule: {smiles}")
        logger.info(f"Country (for energy mix): {country_name}")
        logger.info(f"Using GNN_E multitask model: {model_path}")

        # Make prediction using GNN_E multitask model
        results = predict_single_molecule_multitask(
            model_path=model_path,
            smiles=smiles,
            country_name=country_name,
            dataset_type="GNN_E",
        )

        logger.info("✓ GNN_E multitask prediction completed!")
        logger.info(f"Molecule: {results['smiles']}")
        logger.info(f"Country: {results['country_name']}")
        logger.info(f"Number of categories predicted: {results['num_categories']}")
        logger.info(f"Categories: {results['column_names']}")

        # Log first few predictions for brevity
        predictions = results["predictions"]
        logger.info("Sample predictions:")
        for i, (category, value) in enumerate(predictions.items()):
            if i < 5:  # Show first 5 predictions
                logger.info(f"  {category}: {value}")
            elif i == 5:
                logger.info(f"  ... and {len(predictions) - 5} more categories")
                break

        return results

    except Exception as e:
        logger.error(f"GNN_E multitask prediction failed: {e}")
        return None


def test_qspr_error_handling():
    """Test that QSPR models raise appropriate error for single molecule inference."""
    logger.info("Testing QSPR model error handling")
    logger.info("=" * 50)

    # Example data
    smiles = "CCO"  # Ethanol

    try:
        # This should fail with a descriptive error message
        model_path = "./trained_models/QSPR/QSPR_model.pth"  # Hypothetical path

        logger.info(f"Attempting QSPR single molecule prediction for: {smiles}")
        logger.info("This should fail with an informative error message...")

        # This should raise a ValueError
        results = predict_single_molecule(
            model_path=model_path,
            smiles=smiles,
            dataset_type="QSPR",
        )

        # If we get here, the test failed
        logger.error("✗ QSPR prediction should have failed but didn't!")
        return None

    except ValueError as e:
        if "QSPR models are not supported" in str(e):
            logger.info("✓ QSPR prediction correctly failed with expected error!")
            logger.info(f"Error message: {e}")
            return True
        else:
            logger.error(f"✗ QSPR prediction failed with unexpected error: {e}")
            return None
    except Exception as e:
        logger.error(f"✗ QSPR prediction failed with unexpected exception: {e}")
        return None


def test_multiple_molecules():
    """Test prediction with multiple molecules."""
    logger.info("Testing prediction with multiple molecules")
    logger.info("=" * 50)

    # Test molecules
    molecules = [
        {"smiles": "CCO", "name": "Ethanol"},
        {"smiles": "CC(=O)O", "name": "Acetic acid"},
        {"smiles": "c1ccccc1", "name": "Benzene"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine"},
    ]

    country_name = "Germany"
    results = []

    try:
        model_path = "./trained_models/GNN_C_single/GNN_C_Eutro_f_final_lr_1.00e-03.pth"

        for molecule in molecules:
            logger.info(f"Predicting for {molecule['name']}: {molecule['smiles']}")

            result = predict_single_molecule(
                model_path=model_path,
                smiles=molecule["smiles"],
                country_name=country_name,
                dataset_type="GNN_C",
                target_task="Gwi",
            )

            result["name"] = molecule["name"]
            results.append(result)

            logger.info(f"  Prediction: {result['predictions']}")

        logger.info("✓ Multiple molecule predictions completed!")
        return results

    except Exception as e:
        logger.error(f"Multiple molecule prediction failed: {e}")
        return None


def run_all_tests():
    """Run all test functions."""
    logger.info("Starting single molecule prediction tests")
    logger.info("=" * 70)

    tests = [
        ("GNN_C Model Test", test_gnn_c_prediction),
        ("GNN_E Model Test (Country-based)", test_gnn_e_prediction),
        ("GNN_E Model Test (Custom Energy)", test_gnn_e_prediction_custom_energy),
        ("QSPR Error Handling Test", test_qspr_error_handling),
        ("GNN_C Multitask Test", test_multitask_prediction),
        ("GNN_E Multitask Test", test_multitask_prediction_gnn_e),
        ("Multiple Molecules Test", test_multiple_molecules),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'-' * 50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'-' * 50}")

        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} completed successfully")
            else:
                logger.warning(f"⚠ {test_name} completed with issues")
        except Exception as e:
            logger.error(f"✗ {test_name} failed: {e}")
            results[test_name] = None

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    run_all_tests()
