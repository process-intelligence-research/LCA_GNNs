"""
PyTorch training loop implementation for GNN and QSPR models.

This module contains the core training functionality with validation-based early stopping,
learning rate scheduling, and comprehensive metric logging.
"""

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.loader.dataloader
import wandb

from src.config.training_config import TrainingConfiguration


class Trainer:
    """
    Trainer class for GNN and QSPR models with advanced training features.

    This trainer implements a complete training pipeline including:
    - Forward and backward propagation
    - Validation monitoring with early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Comprehensive metric logging (loss, MAE, MRE)
    - Weights & Biases integration
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfiguration,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: torch.nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> None:
        """
        Initialize the trainer with model and training components.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to train.
        config : TrainingConfiguration
            Complete configuration for training.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for parameter updates. If None, creates from config.
        criterion : torch.nn.Module, optional
            Loss function for training. If None, creates MSELoss.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler. If None, creates from config.
        """
        self.model = model
        self.config = config
        self.device = config.get_device()

        # Create optimizer if not provided
        if optimizer is None:
            optimizer_class = getattr(torch.optim, config.optimizer.type)
            self.optimizer = optimizer_class(
                model.parameters(),
                lr=config.optimizer.learning_rate,
                weight_decay=config.optimizer.weight_decay,
                **(
                    {"betas": config.optimizer.betas, "eps": config.optimizer.eps}
                    if config.optimizer.type == "Adam"
                    else {}
                ),
            )
        else:
            self.optimizer = optimizer

        # Create criterion if not provided
        if criterion is None:
            self.criterion = nn.MSELoss().to(self.device)
        else:
            self.criterion = criterion.to(self.device)

        # Create scheduler if not provided
        if scheduler is None:
            if config.scheduler.type == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=config.scheduler.factor,
                    patience=config.scheduler.patience,
                    min_lr=config.scheduler.min_lr,
                    cooldown=config.scheduler.cooldown,
                    threshold=config.scheduler.threshold,
                )
            else:
                # Add other scheduler types as needed
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=config.scheduler.factor,
                    patience=config.scheduler.patience,
                    min_lr=config.scheduler.min_lr,
                )
        else:
            self.scheduler = scheduler

        # Training state
        self.min_val_loss = np.inf
        self.early_stopping_counter = 0
        self.current_epoch = 0

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize wandb if enabled
        if self.config.experiment.enable_wandb:
            wandb.watch(
                self.model, log="all", log_freq=self.config.experiment.log_frequency
            )

    def train_epoch(
        self,
        train_loader: torch_geometric.loader.DataLoader,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple:
        """
        Execute one training epoch.

        Parameters
        ----------
        train_loader : torch_geometric.loader.DataLoader
            DataLoader for training data.
        mean : torch.Tensor
            Mean value for normalization in error calculations.
        std : torch.Tensor
            Standard deviation for normalization in error calculations.

        Returns
        -------
        tuple
            (total_loss, mae, mre) - Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_mre = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch).squeeze()
            targets = batch.y

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Scale loss for gradient accumulation
            if self.config.training.accumulate_grad_batches > 1:
                loss = loss / self.config.training.accumulate_grad_batches

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.accumulate_grad_batches == 0:
                # Gradient clipping if specified
                if self.config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += (
                loss.item()
                * batch.num_graphs
                * self.config.training.accumulate_grad_batches
            )
            total_mae += (predictions - targets).abs().sum().item()
            total_mre += (
                ((predictions - targets) / (targets + mean / std)).abs().sum().item()
            )

        # Calculate average metrics
        num_samples = len(train_loader.dataset)
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        avg_mre = total_mre / num_samples

        return avg_loss, avg_mae, avg_mre

    def validate_epoch(
        self,
        val_loader: torch_geometric.loader.DataLoader,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple:
        """
        Execute one validation epoch.

        Parameters
        ----------
        val_loader : torch_geometric.loader.DataLoader
            DataLoader for validation data.
        mean : torch.Tensor
            Mean value for normalization in error calculations.
        std : torch.Tensor
            Standard deviation for normalization in error calculations.

        Returns
        -------
        tuple
            (total_loss, mae, mre) - Validation metrics for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mre = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                predictions = self.model(batch).squeeze()
                targets = batch.y

                # Compute loss
                loss = self.criterion(predictions, targets)

                # Accumulate metrics
                total_loss += loss.item() * batch.num_graphs
                total_mae += (predictions - targets).abs().sum().item()
                total_mre += (
                    ((predictions - targets) / (targets + mean / std))
                    .abs()
                    .sum()
                    .item()
                )

        # Calculate average metrics
        num_samples = len(val_loader.dataset)
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        avg_mre = total_mre / num_samples

        return avg_loss, avg_mae, avg_mre

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early based on validation loss.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if val_loss < self.min_val_loss:
            print(f"Saving model (new minimum validation loss: {val_loss:.6f})!")
            self.min_val_loss = val_loss

            # Save checkpoint if enabled
            if self.config.experiment.save_checkpoints:
                import os

                os.makedirs(self.config.experiment.checkpoint_dir, exist_ok=True)
                save_path = os.path.join(
                    self.config.experiment.checkpoint_dir,
                    f"{self.config.data.dataset_type}_{self.config.model.model_type}_best.pth",
                )
                torch.save(self.model.state_dict(), save_path)

            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if (
                self.early_stopping_counter
                >= self.config.training.early_stopping_patience
            ):
                print(
                    f"No improvement in validation loss for {self.config.training.early_stopping_patience} "
                    f"epochs. Early stopping..."
                )
                return True
            return False

    def log_metrics(
        self,
        train_loss: float,
        train_mae: float,
        train_mre: float,
        val_loss: float,
        val_mae: float,
        val_mre: float,
    ) -> None:
        """
        Log training metrics to console and Weights & Biases.

        Parameters
        ----------
        train_loss : float
            Training loss for current epoch.
        train_mae : float
            Training MAE for current epoch.
        train_mre : float
            Training MRE for current epoch.
        val_loss : float
            Validation loss for current epoch.
        val_mae : float
            Validation MAE for current epoch.
        val_mre : float
            Validation MRE for current epoch.
        """
        # Console logging
        print(
            f"Epoch {self.current_epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        # Weights & Biases logging
        if self.config.experiment.enable_wandb:
            wandb.log(
                {
                    "epoch": self.current_epoch,
                    "Training Loss": train_loss,
                    "Training MAE": train_mae,
                    "Training MRE": train_mre,
                    "Validation Loss": val_loss,
                    "Validation MAE": val_mae,
                    "Validation MRE": val_mre,
                    "Learning Rate": self.optimizer.param_groups[0]["lr"],
                }
            )

    def train(
        self,
        train_loader: torch_geometric.loader.DataLoader,
        val_loader: torch_geometric.loader.DataLoader,
        mean: torch.Tensor,
        std: torch.Tensor,
        epochs: int | None = None,
    ) -> torch.nn.Module:
        """
        Execute the complete training loop.

        This method implements the full training pipeline with validation monitoring,
        early stopping, learning rate scheduling, and metric logging.

        Parameters
        ----------
        train_loader : torch_geometric.loader.DataLoader
            DataLoader containing training data batches.
        val_loader : torch_geometric.loader.DataLoader
            DataLoader containing validation data batches.
        mean : torch.Tensor
            Mean value of target variable for normalization.
        std : torch.Tensor
            Standard deviation of target variable for normalization.
        epochs : int, optional
            Maximum number of training epochs. Uses config if None.

        Returns
        -------
        torch.nn.Module
            The trained model with the best validation performance.

        Notes
        -----
        - Model automatically loads the best checkpoint after training
        - Training stops early if validation loss doesn't improve
        - All metrics are logged to Weights & Biases if enabled
        """
        if epochs is None:
            epochs = self.config.training.epochs

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Ensure mean and std are on the correct device
        mean = mean.to(self.device) if hasattr(mean, "to") else mean
        std = std.to(self.device) if hasattr(std, "to") else std

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss, train_mae, train_mre = self.train_epoch(train_loader, mean, std)

            # Validation phase
            val_loss, val_mae, val_mre = self.validate_epoch(val_loader, mean, std)

            # Learning rate scheduling
            if hasattr(self.scheduler, "step"):
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log metrics
            self.log_metrics(
                train_loss, train_mae, train_mre, val_loss, val_mae, val_mre
            )

            # Early stopping check
            if self.check_early_stopping(val_loss):
                break

        # Load the best model if checkpointing is enabled
        if self.config.experiment.save_checkpoints:
            print("Loading best model checkpoint...")
            import os

            save_path = os.path.join(
                self.config.experiment.checkpoint_dir,
                f"{self.config.data.dataset_type}_{self.config.model.model_type}_best.pth",
            )
            if os.path.exists(save_path):
                self.model.load_state_dict(torch.load(save_path))

        print("Training completed!")

        return self.model


def create_trainer_from_config(
    model: torch.nn.Module,
    config: TrainingConfiguration,
    learning_rate: float | None = None,
) -> Trainer:
    """
    Create a trainer from a configuration object.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    config : TrainingConfiguration
        Complete training configuration.
    learning_rate : float, optional
        Override learning rate from config.

    Returns
    -------
    Trainer
        Configured trainer instance.
    """
    # Override learning rate if provided
    if learning_rate is not None:
        config.optimizer.learning_rate = learning_rate

    return Trainer(model=model, config=config)


def create_trainer(
    model: torch.nn.Module,
    learning_rate: float,
    device: torch.device,
    config,
    model_save_path: str = "./models/gnn_cc.pth",
) -> Trainer:
    """
    Create a trainer with standard configuration (legacy compatibility).

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    learning_rate : float
        Learning rate for the optimizer.
    device : torch.device
        Device for training.
    config : object
        Configuration object with training parameters.
    model_save_path : str, optional
        Path to save model checkpoints.

    Returns
    -------
    Trainer
        Configured trainer instance.
    """
    # Create a TrainingConfiguration from the legacy config
    training_config = TrainingConfiguration()

    # Map legacy config to new config structure
    training_config.optimizer.learning_rate = learning_rate
    training_config.scheduler.factor = getattr(config, "factor", 0.9)
    training_config.scheduler.patience = getattr(config, "patience", 10)
    training_config.scheduler.min_lr = getattr(config, "min_learning_rate", 1e-9)
    training_config.training.early_stopping_patience = getattr(config, "early", 15)
    training_config.training.epochs = getattr(config, "epochs", 500)
    training_config.training.batch_size = getattr(config, "batch_size", 20)
    training_config.training.val_length = getattr(config, "val_length", 0.10)
    training_config.training.k_fold = getattr(config, "k_fold", 10)
    training_config.system.device = str(device)

    # Update checkpoint path
    import os

    training_config.experiment.checkpoint_dir = os.path.dirname(model_save_path)

    return Trainer(model=model, config=training_config)
