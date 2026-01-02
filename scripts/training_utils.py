#!/usr/bin/env python
"""Common training utilities for HELM-BERT training scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

# Constants
SEPARATOR_LINE = "=" * 60
DEFAULT_MODEL = "Flansma/helm-bert"


def setup_training_env(seed: int) -> None:
    """Setup training environment with seed and matmul precision."""
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")


def setup_logging(output_dir: Path, timestamp: str, name: str) -> logging.Logger:
    """Setup logging configuration.

    Args:
        output_dir: Directory to save log files
        timestamp: Timestamp string for log filename
        name: Logger name

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / f"{name}_{timestamp}.log"),
        ],
    )
    return logging.getLogger(name)


def create_callbacks(
    checkpoint_dir: Path,
    early_stopping_patience: int,
) -> list[Callback]:
    """Create training callbacks.

    Args:
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping (0 to disable)

    Returns:
        List of Lightning callbacks
    """
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=10,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=2),
    ]

    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
                verbose=True,
            )
        )

    return callbacks


def create_output_dirs(base_dir: Path, run_name: str) -> tuple[Path, Path]:
    """Create output and checkpoint directories.

    Args:
        base_dir: Base output directory
        run_name: Name of this training run

    Returns:
        Tuple of (output_dir, checkpoint_dir)
    """
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return output_dir, checkpoint_dir


def load_best_checkpoint(trainer: L.Trainer, model_class, strict: bool = True):
    """Load best checkpoint from trainer.

    Args:
        trainer: Lightning trainer instance
        model_class: Model class to load checkpoint into
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Loaded model instance

    Raises:
        RuntimeError: If no best checkpoint found
    """
    if not trainer.checkpoint_callback or not trainer.checkpoint_callback.best_model_path:
        raise RuntimeError("No best checkpoint found - training may have failed")

    return model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        strict=strict
    )


def mark_completion(output_dir: Path):
    """Create completion marker file.

    Args:
        output_dir: Directory to create completion marker in
    """
    (output_dir / "COMPLETED").touch()


# =============================================================================
# Logging Helpers
# =============================================================================


def log_header(logger: logging.Logger, title: str) -> None:
    """Log a formatted header section.

    Args:
        logger: Logger instance
        title: Title to display
    """
    logger.info(SEPARATOR_LINE)
    logger.info(title)
    logger.info(SEPARATOR_LINE)


def log_config(logger: logging.Logger, config_path: Path) -> None:
    """Log configuration saved message.

    Args:
        logger: Logger instance
        config_path: Path where config was saved
    """
    logger.info(f"Configuration saved to {config_path}")


def log_training_config(
    logger: logging.Logger,
    max_epochs: int,
    batch_size: int,
    **kwargs,
) -> None:
    """Log training configuration details.

    Args:
        logger: Logger instance
        max_epochs: Maximum training epochs
        batch_size: Batch size
        **kwargs: Additional config items to log (key-value pairs)
    """
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {max_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    for key, value in kwargs.items():
        # Convert snake_case to readable format
        label = key.replace("_", " ").title()
        logger.info(f"  {label}: {value}")
    logger.info(SEPARATOR_LINE)


def log_training_start(logger: logging.Logger, task_name: str = "training") -> None:
    """Log training start message.

    Args:
        logger: Logger instance
        task_name: Name of the training task
    """
    logger.info(f"Starting {task_name}...")


def log_summary(
    logger: logging.Logger,
    duration_seconds: float,
    output_dir: Path,
    **kwargs,
) -> None:
    """Log training summary.

    Args:
        logger: Logger instance
        duration_seconds: Training duration in seconds
        output_dir: Output directory path
        **kwargs: Additional summary items to log
    """
    logger.info(SEPARATOR_LINE)
    logger.info("Training Summary")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Duration: {duration_seconds / 60:.2f} minutes")
    logger.info(f"Output directory: {output_dir}")
    for key, value in kwargs.items():
        label = key.replace("_", " ").title()
        logger.info(f"{label}: {value}")


def log_completion(logger: logging.Logger, task_name: str = "Training") -> None:
    """Log completion message.

    Args:
        logger: Logger instance
        task_name: Name of the completed task
    """
    logger.info(f"{task_name} completed successfully!")
