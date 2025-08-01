"""
Simple test example for the prepare_data function in scripts.py

This script demonstrates how to use the prepare_data function with existing data files.
It assumes you have actual dataset files available for testing.
"""

import datetime
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.scripts import prepare_data

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"test_prepare_data_{timestamp}.log"

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write to file
        logging.StreamHandler(),  # Also write to console
    ],
)
logger = logging.getLogger(__name__)

# Log the log file location
logger.info(f"Log file created: {log_file.absolute()}")


def test_prepare_data_with_existing_files(data_directory: str = "./data/raw"):
    """
    Test the prepare_data function with existing data files.

    Parameters
    ----------
    data_directory : str
        Path to the directory containing your actual data files.
        Default is "./data/raw" but you can change this to your data location.
    """
    logger.info("=" * 60)
    logger.info("Testing prepare_data function with existing data files")
    logger.info("=" * 60)

    # Define your actual data files here
    # Update these file names to match your actual data files
    test_cases = [
        {
            "dataset_type": "QSPR",
            "data_file": "2023_09_07_QSPR_mol_only_cc.xlsx",  # Update with your actual QSPR file name
            "description": "QSPR molecular descriptor data",
        },
        {
            "dataset_type": "GNN_M",
            "data_file": "2023_09_07_GNN_M_CC.xlsx",  # Update with your actual GNN_M file name
            "description": "GNN Molecular data",
        },
        {
            "dataset_type": "GNN_C",
            "data_file": "2023_09_14_finaldataset_country_combine.xlsx",  # Update with your actual GNN_C file name
            "description": "GNN Country-specific data",
        },
        {
            "dataset_type": "GNN_E",
            "data_file": "2023_09_18_finaldataset_energymix_combine.xlsx",  # Update with your actual GNN_E file name
            "description": "GNN Energy data",
        },
    ]

    # Output directory for processed data
    output_dir = "./data/processed"

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'-' * 50}")
        logger.info(
            f"Test {i}: {test_case['dataset_type']} - {test_case['description']}"
        )
        logger.info(f"{'-' * 50}")

        # Check if data file exists
        data_file_path = os.path.join(data_directory, test_case["data_file"])
        if not os.path.exists(data_file_path):
            logger.warning(f"⚠️  Data file not found: {data_file_path}")
            logger.warning(f"   Skipping {test_case['dataset_type']} test")
            logger.warning("   Please update the file name in the test_cases list")
            continue

        logger.info(f"Using data file: {data_file_path}")

        try:
            # Test prepare_data function
            logger.info(f"Processing {test_case['dataset_type']} data...")
            results = prepare_data(
                data_path=data_directory,
                dataset_type=test_case["dataset_type"],
                data_file=test_case["data_file"],
                output_dir=output_dir,
            )

            logger.info("✅ Success!")
            logger.info(f"Dataset size: {results['dataset_size']} samples")
            logger.info(f"Processed data saved to: {results['processed_data_path']}")

            # Verify the processed file
            if os.path.exists(results["processed_data_path"]):
                import torch

                # Load and inspect the processed dataset
                processed_dataset = torch.load(results["processed_data_path"])
                logger.info("✅ Processed dataset verified")
                logger.info(f"   Dataset length: {len(processed_dataset)}")

                # Check first sample structure
                if len(processed_dataset) > 0:
                    first_sample = processed_dataset[0]
                    logger.info(f"   Sample type: {type(first_sample)}")
                    if hasattr(first_sample, "x"):
                        logger.info(f"   Features shape: {first_sample.x.shape}")
                    if hasattr(first_sample, "y"):
                        logger.info(f"   Target shape: {first_sample.y.shape}")
                    if hasattr(first_sample, "edge_index"):
                        logger.info(
                            f"   Edge index shape: {first_sample.edge_index.shape}"
                        )
            else:
                logger.error(
                    f"❌ Processed file not created: {results['processed_data_path']}"
                )

        except FileNotFoundError as e:
            logger.error(f"❌ File not found: {str(e)}")
            logger.error("   Please check the data file path and name")

        except Exception as e:
            logger.error(f"❌ Error processing {test_case['dataset_type']}: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")

            # Show more detailed error for debugging
            import traceback

            logger.debug(f"   Full traceback: {traceback.format_exc()}")


def test_single_dataset(
    dataset_type: str, data_file: str, data_directory: str = "./data/raw"
):
    """
    Test prepare_data function for a single dataset.

    Parameters
    ----------
    dataset_type : str
        Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").
    data_file : str
        Name of the data file.
    data_directory : str
        Path to the directory containing the data file.
    """
    logger.info(f"Testing single dataset: {dataset_type}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Data directory: {data_directory}")

    try:
        results = prepare_data(
            data_path=data_directory,
            dataset_type=dataset_type,
            data_file=data_file,
            output_dir="./data/processed",
        )

        logger.info("✅ Single dataset test successful!")
        logger.info(f"Results: {results}")

        return results

    except Exception as e:
        logger.error(f"❌ Single dataset test failed: {str(e)}")
        return None


def main():
    """
    Main function to run the tests.

    You can modify this function to test specific datasets or configurations.
    """
    logger.info("Starting prepare_data function tests with existing data files...")

    # Update this path to point to your actual data directory
    data_directory = "./data/raw"

    # Check if data directory exists
    if not os.path.exists(data_directory):
        logger.warning(f"⚠️  Data directory not found: {data_directory}")
        logger.warning("Please update the data_directory path in the main() function")
        logger.warning("or create the directory and place your data files there.")
        return

    try:
        # Test all datasets
        test_prepare_data_with_existing_files(data_directory)

        # Uncomment and modify these lines to test individual datasets:
        # test_single_dataset("QSPR", "your_qspr_file.xlsx", data_directory)
        # test_single_dataset("GNN_M", "your_gnn_m_file.xlsx", data_directory)
        # test_single_dataset("GNN_C", "your_gnn_c_file.xlsx", data_directory)
        # test_single_dataset("GNN_E", "your_gnn_e_file.xlsx", data_directory)

        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ Fatal error during testing: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    # Example usage with command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Test prepare_data function")
    parser.add_argument(
        "--data-dir", default="./data/raw", help="Directory containing data files"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["QSPR", "GNN_M", "GNN_C", "GNN_E"],
        help="Test specific dataset type",
    )
    parser.add_argument("--data-file", help="Specific data file to test")

    args = parser.parse_args()

    if args.dataset_type and args.data_file:
        # Test single dataset
        test_single_dataset(args.dataset_type, args.data_file, args.data_dir)
    else:
        # Test all datasets
        main()
