"""
Simple test script for the train_model function.

This script provides basic tests for the four main training modes:
- QSPR training
- GNN_M training
- Single-task training
- Multi-task training
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.training_config import get_default_config
from src.scripts import train_model


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_train_model_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def test_qspr() -> bool:
    """Test QSPR model training."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing QSPR Model ===")

    try:
        config = get_default_config("QSPR", "qspr", "simple")
        config.data.path = "."
        config.training.epochs = 3
        config.training.k_fold = 10

        logger.info(
            f"Config: {config.data.dataset_type} dataset, {config.model.model_type} model"
        )

        results = train_model(
            config=config,
            data_path=".",
            save_results=True,
            results_dir="./test_results/qspr",
        )

        logger.info("QSPR training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"QSPR training failed: {str(e)}")
        return False


def test_gnn_m() -> bool:
    """Test GNN_M model training."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing GNN_M Model ===")

    try:
        config = get_default_config("GNN_M", "GNN_M", "simple")
        config.data.path = "."
        config.training.epochs = 3
        config.training.k_fold = 10

        logger.info(
            f"Config: {config.data.dataset_type} dataset, {config.model.model_type} model"
        )

        results = train_model(
            config=config,
            data_path=".",
            save_results=True,
            results_dir="./test_results/gnn_m",
        )

        logger.info("GNN_M training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"GNN_M training failed: {str(e)}")
        return False


def test_single_task_gnn_c() -> bool:
    """Test single-task training with category-specific learning rates."""
    logger = logging.getLogger(__name__)
    logger.info(
        "=== Testing Single-Task Training with Category-Specific Learning Rates ==="
    )

    try:
        # Test GNN_C single-task with category-specific learning rates
        config = get_default_config("GNN_C", "GNN_C_single", "single")
        config.data.path = "."
        config.training.epochs = 3
        config.training.k_fold = 2

        # Use only a subset of tasks for testing
        config.data.target_tasks = ["CTUe"]

        logger.info(
            f"Config: {config.data.dataset_type} dataset, {config.model.model_type} model"
        )
        logger.info(f"Target tasks: {config.data.target_tasks}")
        logger.info(
            f"Learning rates type: {type(config.hyperparameters.learning_rates)}"
        )

        # Show category-specific learning rates
        if isinstance(config.hyperparameters.learning_rates, dict):
            logger.info("Category-specific learning rates:")
            for task in config.data.target_tasks:
                if task in config.hyperparameters.learning_rates:
                    lr = config.hyperparameters.learning_rates[task][0]
                    logger.info(f"  {task}: {lr:.2e}")
                else:
                    logger.warning(f"  {task}: no learning rate specified")

        results = train_model(
            config=config,
            data_path=".",
            save_results=True,
            results_dir="./test_results/single",
        )

        logger.info("Single-task training completed successfully!")
        logger.info(
            f"Trained on {len(config.data.target_tasks)} environmental impact categories"
        )
        return True

    except Exception as e:
        logger.error(f"Single-task training failed: {str(e)}")
        return False


def test_single_task_gnn_e() -> bool:
    """Test GNN_E single-task training with category-specific learning rates."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing GNN_E Single-Task Training ===")

    try:
        # Test GNN_E single-task with category-specific learning rates
        config = get_default_config("GNN_E", "GNN_E_single", "single")
        config.data.path = "."
        config.training.epochs = 3
        config.training.k_fold = 2

        # Use only a subset of tasks for testing
        config.data.target_tasks = ["Acid", "Gwi", "human_health"]

        logger.info(
            f"Config: {config.data.dataset_type} dataset, {config.model.model_type} model"
        )
        logger.info(f"Target tasks: {config.data.target_tasks}")

        # Show category-specific learning rates for GNN_E
        if isinstance(config.hyperparameters.learning_rates, dict):
            logger.info("Category-specific learning rates for GNN_E:")
            for task in config.data.target_tasks:
                if task in config.hyperparameters.learning_rates:
                    lr = config.hyperparameters.learning_rates[task][0]
                    logger.info(f"  {task}: {lr:.2e}")

        results = train_model(
            config=config,
            data_path=".",
            save_results=True,
            results_dir="./test_results/single_gnn_e",
        )

        logger.info("GNN_E single-task training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"GNN_E single-task training failed: {str(e)}")
        return False


def test_multi() -> bool:
    """Test multi-task training."""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Multi-Task Training ===")

    try:
        config = get_default_config("GNN_C", "GNN_C_multi", "multi")
        config.data.path = "."
        config.hyperparameters.learning_rates = [1e-3]
        config.training.epochs = 1
        config.training.k_fold = 2

        logger.info(
            f"Config: {config.data.dataset_type} dataset, {config.model.model_type} model"
        )

        results = train_model(
            config=config,
            data_path=".",
            save_results=True,
            results_dir="./test_results/multi",
        )

        logger.info("Multi-task training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Multi-task training failed: {str(e)}")
        return False


def run_all_tests() -> dict[str, bool]:
    """Run all training tests."""
    logger = logging.getLogger(__name__)
    logger.info("=== Running All Tests ===")

    # Create test results directory
    test_results_dir = Path("./test_results")
    test_results_dir.mkdir(exist_ok=True)

    tests = [
        ("QSPR", test_qspr),
        ("GNN_M", test_gnn_m),
        ("Single-Task GNN_C", test_single_task_gnn_c),
        ("Single-Task GNN_E", test_single_task_gnn_e),
        ("Multi-Task", test_multi),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        results[test_name] = test_func()

    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")

    passed = sum(results.values())
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")

    return results


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()

    # Determine test mode
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    logger.info(f"Starting train_model testing in mode: {test_mode}")

    try:
        if test_mode == "all":
            results = run_all_tests()
            logger.info("All tests completed")

        elif test_mode == "qspr":
            success = test_qspr()
            logger.info(f"QSPR test result: {'SUCCESS' if success else 'FAILED'}")

        elif test_mode == "gnn_m":
            success = test_gnn_m()
            logger.info(f"GNN_M test result: {'SUCCESS' if success else 'FAILED'}")

        elif test_mode == "single_gnn_c":
            success = test_single_task_gnn_c()
            logger.info(
                f"Single-task GNN_C test result: {'SUCCESS' if success else 'FAILED'}"
            )

        elif test_mode == "single_gnn_e":
            success = test_single_task_gnn_e()
            logger.info(
                f"Single-task GNN_E test result: {'SUCCESS' if success else 'FAILED'}"
            )

        elif test_mode == "multi":
            success = test_multi()
            logger.info(f"Multi-task test result: {'SUCCESS' if success else 'FAILED'}")

        else:
            logger.error(f"Unknown test mode: {test_mode}")
            logger.info(
                "Available modes: all, qspr, gnn_m, single, single_gnn_e, multi"
            )

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")

    logger.info("Testing session completed")
