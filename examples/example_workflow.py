#!/usr/bin/env python3
"""
Example usage of the LCA GNN main workflow.

This script demonstrates how to use the main.py functions programmatically
instead of via command line interface.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import (
    setup_configuration,
    prepare_data_workflow,
    train_model_workflow,
    final_model_training_workflow,
    single_molecule_inference_workflow,
    batch_inference_workflow,
)


def example_complete_workflow():
    """Example of running the complete workflow programmatically."""
    print("=" * 60)
    print("LCA GNN COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)

    # Step 1: Setup configuration
    print("\n1. Setting up configuration...")
    config = setup_configuration()

    # Step 2: Prepare data (assuming data file exists)
    print("\n2. Preparing data...")
    data_path = "data/raw/example_data.xlsx"

    if os.path.exists(data_path):
        success = prepare_data_workflow(config, data_path)
        if not success:
            print("Data preparation failed!")
            return False
    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure your data file is available.")
        return False

    # Step 3: Train model
    print("\n3. Training models...")
    success = train_model_workflow(config)
    if not success:
        print("Model training failed!")
        return False

    # Step 4: Final model training
    print("\n4. Final model training...")
    success = final_model_training_workflow(config)
    if not success:
        print("Final model training failed!")
        return False

    print("\n✅ Complete workflow finished successfully!")
    return True


def example_single_molecule_inference():
    """Example of single molecule inference."""
    print("=" * 60)
    print("SINGLE MOLECULE INFERENCE EXAMPLE")
    print("=" * 60)

    # Example model path (adjust as needed)
    model_path = "trained_models/GNN_C_multi.pth"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train a model first or adjust the model path.")
        return None

    # Example molecules
    test_molecules = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCO",  # Ethanol (duplicate for testing)
    ]

    for smiles in test_molecules:
        print(f"\nPredicting for SMILES: {smiles}")

        # Single-task inference
        result = single_molecule_inference_workflow(
            model_path=model_path,
            smiles=smiles,
            country_name="Germany",
            dataset_type="GNN_C",
            multitask=True,  # Predict all categories
        )

        if result:
            print(f"✅ Prediction successful!")
            print(f"Number of categories: {result['num_categories']}")

            # Show top 3 impact categories
            predictions = result["predictions"]
            sorted_impacts = sorted(
                predictions.items(), key=lambda x: abs(x[1]), reverse=True
            )

            print("Top 3 environmental impacts:")
            for i, (category, value) in enumerate(sorted_impacts[:3]):
                print(f"  {i + 1}. {category}: {value:.6e}")
        else:
            print(f"❌ Prediction failed for {smiles}")

    return True


def example_batch_inference():
    """Example of batch inference."""
    print("=" * 60)
    print("BATCH INFERENCE EXAMPLE")
    print("=" * 60)

    # Setup configuration
    config = setup_configuration()

    # Example paths
    model_path = "trained_models/GNN_C_multi.pth"
    data_path = "data/test_molecules.xlsx"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please prepare test data first.")
        return None

    # Run batch inference
    result = batch_inference_workflow(config, data_path, model_path)

    if result:
        print(f"✅ Batch inference successful!")
        print(f"Predictions saved to: {result['predictions_saved_to']}")
        print(f"Number of samples: {result['num_samples']}")
        print(f"Number of features: {result['num_features']}")
    else:
        print("❌ Batch inference failed!")

    return result


def main():
    """Main function to run examples."""
    print("LCA GNN WORKFLOW EXAMPLES")
    print("Choose an example to run:")
    print("1. Complete workflow (config + data + train + final train)")
    print("2. Single molecule inference")
    print("3. Batch inference")
    print("4. All examples")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        example_complete_workflow()
    elif choice == "2":
        example_single_molecule_inference()
    elif choice == "3":
        example_batch_inference()
    elif choice == "4":
        # Run all examples
        print("\nRunning all examples...")
        example_complete_workflow()
        example_single_molecule_inference()
        example_batch_inference()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
