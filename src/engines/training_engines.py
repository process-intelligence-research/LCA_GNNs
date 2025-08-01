"""
Training Engine for GNN and QSPR Models.

This module provides a high-level training orchestration class that handles dataset loading,
data preprocessing, k-fold cross-validation, model creation, and training coordination.
"""

import logging
import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import wandb

# Import local modules
from src.config.training_config import get_default_config
from src.data_processing.make_dataset import (
    GNN_C_dataset,
    GNN_E_dataset,
    GNN_M_dataset,
    QSPR_dataset,
)
from src.engines.evaluation_engines import save_and_evaluate_results
from src.engines.predict_engines import prediction
from src.models.models import (
    GNN_M,
    GNN_C_multi,
    GNN_C_single,
    GNN_E_multi,
    GNN_E_single,
    qspr,
)
from src.trainer.trainer import Trainer
from src.utils.k_folder import calculate_index


class TrainingEngine:
    """
    High-level training orchestration class for GNN and QSPR models.

    This class handles the complete training pipeline including:
    - Dataset loading and preprocessing
    - K-fold cross-validation setup
    - Model instantiation
    - Training coordination with the Trainer class
    - Results collection and evaluation
    - Experiment tracking with Weights & Biases
    """

    def __init__(
        self,
        path: str,
        dataset_type: str,
        model_type: str,
        data_file: str,
        project_name: str = "GWP",
        entity: str = "qinghegao",
        experiment_prefix: str = "",
        k_fold: int = 10,
        batch_size: int = 20,
        val_length: float = 0.10,
        epochs: int = 500,
        enable_wandb: bool = True,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the training engine.

        Parameters
        ----------
        path : str
            Base path to the project directory.
        dataset_type : str
            Type of dataset ("QSPR", "GNN_M", "GNN_C", "GNN_E").
        model_type : str
            Type of model ("qspr", "GNN_M", "GNN_C_single", etc.).
        data_file : str
            Name of the Excel data file.
        project_name : str, optional
            Weights & Biases project name.
        entity : str, optional
            Weights & Biases entity name.
        experiment_prefix : str, optional
            Prefix for experiment names.
        k_fold : int, optional
            Number of cross-validation folds.
        batch_size : int, optional
            Training batch size.
        val_length : float, optional
            Validation set fraction.
        epochs : int, optional
            Maximum training epochs.
        enable_wandb : bool, optional
            Whether to enable Weights & Biases logging.
        device : str, optional
            Training device ("cuda" or "cpu").
        """
        self.path = path
        self.dataset_type = dataset_type
        self.model_type = model_type
        self.data_file = data_file
        self.k_fold = k_fold
        self.batch_size = batch_size
        self.val_length = val_length
        self.epochs = epochs
        self.project_name = project_name
        self.entity = entity
        self.experiment_prefix = experiment_prefix
        self.enable_wandb = enable_wandb

        # Set device with fallback to CPU if CUDA not available
        if device == "cuda" and not torch.cuda.is_available():
            logger = logging.getLogger(__name__)
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize dataset and indices
        self.dataset = None
        self.fold_indices = None
        self._load_dataset()
        self._setup_fold_indices()

    def _load_dataset(self) -> None:
        """Load the appropriate dataset based on dataset_type."""
        dataexcel = pd.read_excel(
            os.path.join(self.path, "data", "raw", self.data_file)
        )

        dataset_classes = {
            "QSPR": QSPR_dataset,
            "GNN_M": GNN_M_dataset,
            "GNN_C": GNN_C_dataset,
            "GNN_E": GNN_E_dataset,
        }

        if self.dataset_type not in dataset_classes:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        dataset_class = dataset_classes[self.dataset_type]
        self.dataset = dataset_class(os.path.join(self.path, "data"), dataexcel)
        print(f"Loaded {self.dataset_type} dataset with {len(self.dataset)} samples")

    def _setup_fold_indices(self) -> None:
        """Set up k-fold cross-validation indices."""
        number_of_mol = max(self.dataset.data.mol_id).item() + 1
        self.fold_indices = calculate_index(number_of_mol, self.k_fold)
        print(
            f"Set up {self.k_fold}-fold cross-validation for {number_of_mol} molecules"
        )

    def _create_model(self) -> torch.nn.Module:
        """Create and return the appropriate model."""
        model_classes = {
            "qspr": lambda: qspr(out_feature=1),
            "GNN_M": lambda: GNN_M(out_feature=1),
            "GNN_C_single": lambda: GNN_C_single(),
            "GNN_C_multi": lambda: GNN_C_multi(),
            "GNN_E_single": lambda: GNN_E_single(),
            "GNN_E_multi": lambda: GNN_E_multi(),
        }

        if self.model_type not in model_classes:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        model = model_classes[self.model_type]().to(self.device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Created {self.model_type} model with {num_params:,} parameters")
        return model

    def _prepare_fold_data(self, fold_num: int, task_order: int | None = None):
        """
        Prepare training and test data for a specific fold.

        Parameters
        ----------
        fold_num : int
            Current fold number.
        task_order : int, optional
            Task index for single-task learning on multi-output datasets.

        Returns
        -------
        tuple
            (train_loader, val_loader, test_loader, mean, std)
        """
        # Split data by molecule IDs
        test_mask = (self.dataset.data.mol_id >= self.fold_indices[fold_num]) & (
            self.dataset.data.mol_id <= self.fold_indices[fold_num + 1]
        )
        train_mask = ~test_mask

        testdataset = self.dataset[test_mask].copy()
        traindataset = self.dataset[train_mask].shuffle().copy()

        # Handle task selection for single-task learning on multi-output datasets
        if task_order is not None and self.dataset_type in ["GNN_C", "GNN_E"]:
            testdataset.data.y = testdataset.data.y[:, task_order]
            traindataset.data.y = traindataset.data.y[:, task_order]

        # Calculate normalization statistics
        if traindataset.data.y.dim() > 1 and traindataset.data.y.size(1) > 1:
            # Multi-task case
            mean = torch.as_tensor(traindataset.data.y, dtype=torch.float).mean(0)
            std = torch.as_tensor(traindataset.data.y, dtype=torch.float).std(0)
        else:
            # Single-task case
            mean = torch.as_tensor(traindataset.data.y, dtype=torch.float).mean()
            std = torch.as_tensor(traindataset.data.y, dtype=torch.float).std()

        # Normalize data
        traindataset.data.y = (traindataset.data.y - mean) / std
        testdataset.data.y = (testdataset.data.y - mean) / std

        # Create data loaders
        val_size = int(len(traindataset) * self.val_length)
        val_loader = DataLoader(
            traindataset[:val_size], batch_size=self.batch_size, shuffle=True
        )
        train_loader = DataLoader(
            traindataset[val_size:], batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(testdataset, batch_size=1)

        return train_loader, val_loader, test_loader, mean, std

    def _get_task_index(self, target_task: str) -> int:
        """
        Get the index of a target task in the multi-output dataset.

        Parameters
        ----------
        target_task : str
            Name of the environmental impact category

        Returns
        -------
        int
            Index of the task in the dataset's output tensor
        """
        # Standard task mapping for environmental impact categories
        # Order matches the columns in the dataset
        task_mapping = {
            "Acid": 0,  # Acidification
            "Gwi": 1,  # Global Warming Potential / Climate Change
            "CTUe": 2,  # Ecotoxicity
            "ADP_f": 3,  # Abiotic Depletion Potential - Fossil fuels (Mineral Resources)
            "Eutro_f": 4,  # Eutrophication freshwater
            "Eutro_m": 5,  # Eutrophication marine
            "Eutro_t": 6,  # Eutrophication terrestrial
            "CTUh": 7,  # Human Toxicity
            "Ionising": 8,  # Ionising Radiation
            "Soil": 9,  # Land Use/Soil
            "ADP_e": 10,  # Abiotic Depletion Potential - Elements (Energy Resources)
            "ODP": 11,  # Ozone Depletion Potential
            "human_health": 12,  # Particulate Matter Formation
            "Photo": 13,  # Photochemical Oxidation Formation
            "Water_use": 14,  # Water Use
        }

        if target_task not in task_mapping:
            logger = logging.getLogger(__name__)
            available_tasks = list(task_mapping.keys())
            logger.error(f"Unknown target task: {target_task}")
            logger.error(f"Available tasks: {available_tasks}")
            raise ValueError(
                f"Unknown target task: {target_task}. Available: {available_tasks}"
            )

        return task_mapping[target_task]

    def _create_wandb_config(self, learning_rate: float, **kwargs) -> object:
        """Create and return wandb configuration."""
        config_dict = {
            "epochs": self.epochs,
            "min_learning_rate": 1e-9,
            "batch_size": self.batch_size,
            "factor": 0.9,
            "cooldown": 2,
            "patience": 10,
            "early": 15,
            "val_length": self.val_length,
            "k_fold": self.k_fold,
            "learning_rate": learning_rate,
            "dataset_type": self.dataset_type,
            "model_type": self.model_type,
            **kwargs,
        }

        # Convert to wandb config object
        config = wandb.config
        for key, value in config_dict.items():
            setattr(config, key, value)

        return config

    def _create_simple_trainer(
        self, model: torch.nn.Module, learning_rate: float = 1e-4
    ) -> object:
        """Create a simple trainer with default parameters."""
        # Create a minimal config for the trainer
        config = get_default_config(self.dataset_type, self.model_type, "simple")
        config.optimizer.learning_rate = learning_rate

        return Trainer(model, config)

    def train_qspr(self) -> None:
        """Train QSPR model with fixed hyperparameters."""
        results = {}
        run_name = f"{self.experiment_prefix}fixed_params"

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            mode="online" if self.enable_wandb else "disabled",
        )

        for fold_num in range(self.k_fold):
            # Prepare data for this fold
            train_loader, val_loader, test_loader, mean, std = self._prepare_fold_data(
                fold_num
            )

            # Create model and trainer
            model = self._create_model()
            trainer = self._create_simple_trainer(
                model, learning_rate=5e-6
            )  # Fixed LR for QSPR

            # Train the model
            trained_model = trainer.train(
                train_loader, val_loader, mean, std, epochs=self.epochs
            )

            # Make predictions
            results = prediction(
                trained_model,
                test_loader,
                self.device,
                results,
                fold_num,
                mean,
                std,
                mode="single",
            )

        # Save results and evaluate
        save_and_evaluate_results(
            results,
            self.path,
            self.dataset_type,
            "fixed_params",
            self.experiment_prefix,
        )
        run.finish()

    def train_gnn_m(self) -> None:
        """Train GNN_M model with fixed hyperparameters."""
        results = {}
        run_name = f"{self.experiment_prefix}fixed_params"

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            mode="online" if self.enable_wandb else "disabled",
        )

        for fold_num in range(self.k_fold):
            # Prepare data for this fold
            train_loader, val_loader, test_loader, mean, std = self._prepare_fold_data(
                fold_num
            )

            # Create model and trainer
            model = self._create_model()
            trainer = self._create_simple_trainer(
                model, learning_rate=1e-4
            )  # Fixed LR for GNN_M

            # Train the model
            trained_model = trainer.train(
                train_loader, val_loader, mean, std, epochs=self.epochs
            )

            # Make predictions
            results = prediction(
                trained_model,
                test_loader,
                self.device,
                results,
                fold_num,
                mean,
                std,
                mode="single",
            )

        # Save results and evaluate
        save_and_evaluate_results(
            results,
            self.path,
            self.dataset_type,
            "fixed_params",
            self.experiment_prefix,
        )
        run.finish()

    def train_single_task(
        self,
        learning_rates: dict[str, list[float]] | list[float],
        target_tasks: list[str],
    ) -> None:
        """
        Train single-task models for GNN_C or GNN_E.

        Parameters
        ----------
        learning_rates : dict[str, list[float]] or list[float]
            Learning rates for each task (category-specific) or general rates
        target_tasks : list[str]
            List of target environmental impact categories to train
        """
        logger = logging.getLogger(__name__)

        # Handle both dictionary (category-specific) and list (general) learning rates
        if isinstance(learning_rates, dict):
            # Category-specific learning rates
            logger.info(
                f"Using category-specific learning rates for {len(target_tasks)} tasks"
            )
            for task_order, task in enumerate(target_tasks):
                if task not in learning_rates:
                    logger.warning(
                        f"No learning rate specified for task '{task}', skipping..."
                    )
                    continue

                task_learning_rates = learning_rates[task]
                logger.info(
                    f"Training task '{task}' with learning rates: {task_learning_rates}"
                )

                for lr in task_learning_rates:
                    self._train_single_task_with_lr(task, task_order, lr)
        else:
            # General learning rates - apply to all tasks
            logger.info(
                f"Using general learning rates {learning_rates} for {len(target_tasks)} tasks"
            )
            for task_order, task in enumerate(target_tasks):
                for lr in learning_rates:
                    self._train_single_task_with_lr(task, task_order, lr)

    def _train_single_task_with_lr(self, task: str, task_order: int, lr: float) -> None:
        """Train a single task with a specific learning rate."""
        logger = logging.getLogger(__name__)

        results = pd.DataFrame()
        run_name = f"{self.experiment_prefix}{self.dataset_type}_{task}_lr_{lr}"

        logger.info(
            f"Training {self.dataset_type} single-task model for '{task}' with LR={lr:.2e}"
        )

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            mode="online" if self.enable_wandb else "disabled",
        )

        for fold_num in range(self.k_fold):
            logger.info(f"Training fold {fold_num + 1}/{self.k_fold} for task '{task}'")

            # Prepare data for this fold with task selection
            train_loader, val_loader, test_loader, mean, std = self._prepare_fold_data(
                fold_num,
                task_order if self.dataset_type in ["GNN_C", "GNN_E"] else None,
            )

            # Create model and trainer
            model = self._create_model()
            trainer = self._create_simple_trainer(model, learning_rate=lr)

            # Train the model
            trained_model = trainer.train(
                train_loader, val_loader, mean, std, epochs=self.epochs
            )

            # Save the trained model with specific naming
            model_filename = (
                f"{self.dataset_type}_{task}_fold_{fold_num}_lr_{lr:.2e}.pth"
            )
            model_save_path = os.path.join(self.path, "trained_models", model_filename)
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(trained_model.state_dict(), model_save_path)
            logger.info(f"Saved model: {model_save_path}")

            # Make predictions
            results = prediction(
                trained_model,
                test_loader,
                self.device,
                results,
                fold_num,
                mean,
                std,
                mode="single",
            )

        # Save results and evaluate
        save_and_evaluate_results(
            results,
            self.path,
            self.dataset_type,
            f"{task}_lr_{lr}",
            self.experiment_prefix,
            task,
        )
        run.finish()
        logger.info(f"Completed training for task '{task}' with LR={lr:.2e}")

    def train_multi_task(self, learning_rates: list[float]) -> None:
        """Train multi-task models for GNN_C or GNN_E."""
        for lr in learning_rates:
            results = pd.DataFrame()
            run_name = f"{self.experiment_prefix}{self.dataset_type}_multitask_lr_{lr}"

            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                mode="online" if self.enable_wandb else "disabled",
            )

            for fold_num in range(self.k_fold):
                # Prepare data for this fold
                train_loader, val_loader, test_loader, mean, std = (
                    self._prepare_fold_data(fold_num)
                )

                # Create model and trainer
                model = self._create_model()

                # Create trainer with the specific learning rate
                trainer = self._create_simple_trainer(model, learning_rate=lr)

                # Train the model
                trained_model = trainer.train(
                    train_loader, val_loader, mean, std, epochs=self.epochs
                )

                # Make predictions
                results = prediction(
                    trained_model,
                    test_loader,
                    self.device,
                    results,
                    fold_num,
                    mean,
                    std,
                    mode="multi",
                )

            # Save results and evaluate
            save_and_evaluate_results(
                results,
                self.path,
                self.dataset_type,
                f"multitask_lr_{lr}",
                self.experiment_prefix,
            )
            run.finish()

    def train_final_qspr(self, learning_rate: float = 5e-6) -> torch.nn.Module:
        """Train final QSPR model on entire dataset for production use."""
        logger = logging.getLogger(__name__)

        logger.info("Training final QSPR model on entire dataset")

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=f"{self.experiment_prefix}final_qspr",
            mode="online" if self.enable_wandb else "disabled",
        )

        # Use entire dataset, split into train/val only (no test set)
        dataset = self.dataset.shuffle().copy()

        # Calculate normalization statistics from entire dataset
        mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
        std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()

        # Normalize data
        dataset.data.y = (dataset.data.y - mean) / std

        # Split into train/val for early stopping
        val_size = int(len(dataset) * self.val_length)
        val_dataset = dataset[:val_size]
        train_dataset = dataset[val_size:]

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Create model and trainer
        model = self._create_model()
        trainer = self._create_simple_trainer(model, learning_rate=learning_rate)

        # Train the model
        logger.info(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
        final_model = trainer.train(
            train_loader, val_loader, mean, std, epochs=self.epochs
        )

        run.finish()
        logger.info("Final QSPR model training completed")
        return final_model

    def train_final_gnn_m(self, learning_rate: float = 1e-4) -> torch.nn.Module:
        """Train final GNN_M model on entire dataset for production use."""
        logger = logging.getLogger(__name__)

        logger.info("Training final GNN_M model on entire dataset")

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=f"{self.experiment_prefix}final_gnn_m",
            mode="online" if self.enable_wandb else "disabled",
        )

        # Use entire dataset, split into train/val only (no test set)
        dataset = self.dataset.shuffle().copy()

        # Calculate normalization statistics
        mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
        std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()

        # Normalize data
        dataset.data.y = (dataset.data.y - mean) / std

        # Split into train/val for early stopping
        val_size = int(len(dataset) * self.val_length)
        val_dataset = dataset[:val_size]
        train_dataset = dataset[val_size:]

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Create model and trainer
        model = self._create_model()
        trainer = self._create_simple_trainer(model, learning_rate=learning_rate)

        # Train the model
        logger.info(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
        final_model = trainer.train(
            train_loader, val_loader, mean, std, epochs=self.epochs
        )

        run.finish()
        logger.info("Final GNN_M model training completed")
        return final_model

    def train_final_single_task(
        self, learning_rate: float, target_task: str
    ) -> torch.nn.Module:
        """Train final single-task model on entire dataset for production use."""
        logger = logging.getLogger(__name__)

        logger.info(
            f"Training final single-task model for {target_task} on entire dataset"
        )

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=f"{self.experiment_prefix}final_{target_task}",
            mode="online" if self.enable_wandb else "disabled",
        )

        # Use entire dataset
        dataset = self.dataset.shuffle().copy()

        # Handle task selection for single-task learning on multi-output datasets
        if self.dataset_type in ["GNN_C", "GNN_E"]:
            task_order = self._get_task_index(target_task)
            logger.info(
                f"Extracting task '{target_task}' (index {task_order}) from multi-output dataset"
            )
            dataset.data.y = dataset.data.y[:, task_order]

        # Calculate normalization statistics
        mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
        std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()

        # Normalize data
        dataset.data.y = (dataset.data.y - mean) / std

        # Split into train/val for early stopping
        val_size = int(len(dataset) * self.val_length)
        val_dataset = dataset[:val_size]
        train_dataset = dataset[val_size:]

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Create model and trainer
        model = self._create_model()
        trainer = self._create_simple_trainer(model, learning_rate=learning_rate)

        # Train the model
        logger.info(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
        final_model = trainer.train(
            train_loader, val_loader, mean, std, epochs=self.epochs
        )

        # Save the final trained model with specific naming
        model_filename = (
            f"{self.dataset_type}_{target_task}_final_lr_{learning_rate:.2e}.pth"
        )
        model_save_path = os.path.join(self.path, "trained_models", model_filename)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(final_model.state_dict(), model_save_path)
        logger.info(f"Final model saved: {model_save_path}")

        run.finish()
        logger.info(f"Final single-task model for {target_task} training completed")
        return final_model

    def train_final_multi_task(self, learning_rate: float) -> torch.nn.Module:
        """Train final multi-task model on entire dataset for production use."""
        logger = logging.getLogger(__name__)

        logger.info("Training final multi-task model on entire dataset")

        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=f"{self.experiment_prefix}final_multitask",
            mode="online" if self.enable_wandb else "disabled",
        )

        # Use entire dataset
        dataset = self.dataset.shuffle().copy()

        # Calculate normalization statistics
        if dataset.data.y.dim() > 1 and dataset.data.y.size(1) > 1:
            # Multi-task case
            mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean(0)
            std = torch.as_tensor(dataset.data.y, dtype=torch.float).std(0)
        else:
            # Single-task case
            mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
            std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()

        # Normalize data
        dataset.data.y = (dataset.data.y - mean) / std

        # Split into train/val for early stopping
        val_size = int(len(dataset) * self.val_length)
        val_dataset = dataset[:val_size]
        train_dataset = dataset[val_size:]

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Create model and trainer
        model = self._create_model()
        trainer = self._create_simple_trainer(model, learning_rate=learning_rate)

        # Train the model
        logger.info(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
        final_model = trainer.train(
            train_loader, val_loader, mean, std, epochs=self.epochs
        )

        run.finish()
        logger.info("Final multi-task model training completed")
        return final_model
