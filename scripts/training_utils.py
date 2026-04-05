#!/usr/bin/env python
"""Common training utilities for HELM-BERT training scripts."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from omegaconf import DictConfig, OmegaConf

# Config directory
CONFIG_DIR = Path(__file__).parent.parent / "configs"

# Constants
SEPARATOR_LINE = "=" * 60

# (config path, tag if True, tag if False)
_FREEZE_FIELDS: list[tuple[str, str, str]] = [
    ("model.drug_encoder.freeze", "drug-frozen", "drug-trainable"),
    ("model.target_encoder.freeze", "target-frozen", "target-trainable"),
    ("model.freeze_encoder", "frozen", "trainable"),
]


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.

    All fields are required - values come from YAML configuration.
    """

    save_top_k: int
    filename_pattern: str
    monitor: str
    mode: str
    save_last: bool


@dataclass
class DisplayConfig:
    """Configuration for display settings.

    All fields are required - values come from YAML configuration.
    """

    model_summary_max_depth: int



# =============================================================================
# Config Loading (OmegaConf + argparse)
# =============================================================================


def load_config(task: str, argv: List[str] = None) -> DictConfig:
    """Load configuration from YAML files with CLI overrides.

    Supports both formats:
        - Dotlist: training.batch_size=128
        - Argparse: --batch_size 128 (mapped to training.batch_size)

    Args:
        task: Task name (mlm, permeability, ppi)
        argv: Command line arguments (default: sys.argv[1:])

    Returns:
        Merged OmegaConf config
    """
    if argv is None:
        argv = sys.argv[1:]

    # Parse --config flag first
    config_file = None
    remaining_args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_file = argv[i + 1]
            i += 2
        else:
            remaining_args.append(argv[i])
            i += 1

    # Load base configs
    configs = []
    default_path = CONFIG_DIR / "default.yaml"
    if default_path.exists():
        configs.append(OmegaConf.load(default_path))

    # Load task config, then optional override config
    task_path = CONFIG_DIR / f"{task}.yaml"
    if task_path.exists():
        configs.append(OmegaConf.load(task_path))
    if config_file:
        configs.append(OmegaConf.load(config_file))

    # Merge base configs
    config = OmegaConf.merge(*configs) if configs else OmegaConf.create()

    # Parse CLI overrides (dotlist format: key=value)
    dotlist_overrides = [arg for arg in remaining_args if "=" in arg and not arg.startswith("-")]
    if dotlist_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist_overrides))

    # Parse argparse-style overrides (--key value)
    argparse_overrides = parse_argparse_overrides(remaining_args, task)
    if argparse_overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(argparse_overrides))

    return config


def parse_argparse_overrides(argv: List[str], task: str) -> List[str]:
    """Parse argparse-style arguments and convert to dotlist.

    Args:
        argv: Command line arguments
        task: Task name for task-specific mappings

    Returns:
        List of dotlist strings
    """
    # Common argument mappings
    mappings = {
        # Training
        "--batch_size": "training.batch_size",
        "--learning_rate": "training.learning_rate",
        "--lr": "training.learning_rate",
        "--max_epochs": "training.max_epochs",
        "--epochs": "training.max_epochs",
        "--encoder_lr": "training.encoder_lr",
        "--head_lr": "training.head_lr",
        "--seed": "training.seed",
        "--use_cached_embeddings": "training.use_cached_embeddings",
        # Model
        "--pretrained": "model.pretrained_path",
        "--from_scratch": "model.from_scratch",
        "--freeze_encoder": "model.freeze_encoder",
        "--freeze_drug_encoder": "model.drug_encoder.freeze",
        "--freeze_target_encoder": "model.target_encoder.freeze",
        # Data
        "--train_file": "data.train_file",
        "--test_file": "data.test_file",
        # Hardware
        "--devices": "hardware.devices",
        "--precision": "hardware.precision",
    }

    # Boolean flags (can be used without value: --from_scratch = --from_scratch true)
    boolean_flags = {
        "--from_scratch",
        "--freeze_encoder",
        "--freeze_drug_encoder",
        "--freeze_target_encoder",
        "--use_cached_embeddings",
    }

    overrides = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in mappings:
            # Check if next arg exists and is a value (not another flag)
            has_value = i + 1 < len(argv) and not argv[i + 1].startswith("-")
            if has_value:
                value = argv[i + 1]
                overrides.append(f"{mappings[arg]}={value}")
                i += 2
            elif arg in boolean_flags:
                # Boolean flag without value defaults to true
                overrides.append(f"{mappings[arg]}=true")
                i += 1
            else:
                i += 1
        else:
            i += 1

    return overrides


def get_model_tag(config: DictConfig) -> str:
    """Extract model tag from config for run naming.

    Handles both single-encoder (permeability) and dual-encoder (PPI) configs.
    """
    drug_path = OmegaConf.select(config, "model.drug_encoder.pretrained_path")
    path = drug_path or config.model.pretrained_path
    return Path(path).name


def to_dict(config: DictConfig) -> dict:
    """Convert OmegaConf config to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def build_tags(config: DictConfig, base_tags: list[str]) -> list[str]:
    """Build wandb tags from base tags, auto-generated config tags, and user tags.

    Auto-generated tags are derived from key hyperparameters so each run is
    self-describing in wandb without manual bookkeeping.
    """
    lambda_coeff = OmegaConf.select(config, "evidence.lambda_coeff")
    auto_tags = [f"lambda{lambda_coeff}"] if lambda_coeff is not None else []

    for path, frozen, trainable in _FREEZE_FIELDS:
        val = OmegaConf.select(config, path)
        if val is not None:
            auto_tags.append(frozen if val else trainable)

    return base_tags + auto_tags + list(config.logging.tags or [])


# =============================================================================
# Config Converters
# =============================================================================


def config_to_checkpoint_config(config: DictConfig) -> "CheckpointConfig":
    """Convert OmegaConf config to CheckpointConfig."""
    return CheckpointConfig(
        save_top_k=config.checkpoint.save_top_k,
        filename_pattern=config.checkpoint.filename_pattern,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_last=config.checkpoint.save_last,
    )


def config_to_display_config(config: DictConfig) -> "DisplayConfig":
    """Convert OmegaConf config to DisplayConfig."""
    return DisplayConfig(
        model_summary_max_depth=config.display.model_summary_max_depth,
    )


def setup_training_env(seed: int, matmul_precision: str, deterministic: bool) -> None:
    """Setup training environment with seed and matmul precision.

    Args:
        seed: Random seed for reproducibility
        matmul_precision: Float32 matmul precision ("highest", "high", "medium")
        deterministic: Whether deterministic CUDA algorithms are required
    """
    if deterministic:
        # Required for deterministic CuBLAS kernels on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)


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
    checkpoint_config: CheckpointConfig,
    display_config: DisplayConfig,
) -> list[Callback]:
    """Create training callbacks.

    Args:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_config: Checkpoint configuration from YAML
        display_config: Display configuration from YAML

    Returns:
        List of Lightning callbacks
    """
    return [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=checkpoint_config.filename_pattern,
            monitor=checkpoint_config.monitor,
            mode=checkpoint_config.mode,
            save_top_k=checkpoint_config.save_top_k,
            save_last=checkpoint_config.save_last,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=display_config.model_summary_max_depth),
    ]


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
        strict=strict,
        weights_only=False,
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
