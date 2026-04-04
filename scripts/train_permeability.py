#!/usr/bin/env python
"""Permeability regression training script.

Usage:
    python scripts/train_permeability.py
    python scripts/train_permeability.py --config configs/permeability.yaml
    python scripts/train_permeability.py model.freeze_encoder=true training.batch_size=64
    python scripts/train_permeability.py --batch_size 64 --freeze_encoder true
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.append(str(Path(__file__).parent.parent))

import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer

from scripts.training_utils import (
    SEPARATOR_LINE,
    config_to_checkpoint_config,
    config_to_display_config,
    create_callbacks,
    create_output_dirs,
    get_model_tag,
    load_best_checkpoint,
    load_config,
    log_completion,
    log_header,
    log_summary,
    log_training_start,
    mark_completion,
    setup_logging,
    setup_training_env,
    to_dict,
    build_tags,
)
from src.datamodules import PermeabilityDataModule, PermeabilityDataConfig
from src.models.permeability_lightning import (
    HELMBertPermeabilityLightning,
    PermeabilityTrainingConfig,
)


def main():
    """Main training function."""
    start_time = time.time()

    # Load configuration
    config = load_config(task="permeability")

    # Setup environment
    setup_training_env(
        config.training.seed,
        config.trainer.float32_matmul_precision,
        config.trainer.deterministic,
    )

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"permeability_{config.data.split_name}_{get_model_tag(config)}_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(config.paths.output_dir), run_name)

    # Setup logging
    logger = setup_logging(output_dir, timestamp, "train_permeability")
    log_header(logger, "Permeability Evidential Regression Training")

    # Training config will be created after datamodule setup (need total_steps)

    # Create data config
    data_config = PermeabilityDataConfig(
        train_file=config.data.train_file,
        test_file=config.data.test_file,
        helm_column=config.data.helm_column,
        target_column=config.data.target_column,
        val_ratio=config.data.val_ratio,
        batch_size=config.training.batch_size,
        max_seq_length=config.data.max_seq_length,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.training.seed,
    )

    # Save configurations
    config_path_out = output_dir / "config.json"
    config_dict = to_dict(config)
    with open(config_path_out, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path_out}")

    # Log configuration
    logger.info(f"\nPretrained model: {config.model.pretrained_path}")
    logger.info(f"Freeze encoder: {config.model.freeze_encoder}")
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {config.training.max_epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Encoder LR: {config.training.encoder_lr}")
    logger.info(f"  Head LR: {config.training.head_lr}")
    logger.info(SEPARATOR_LINE)

    # Create tokenizer and datamodule
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_path, trust_remote_code=config.model.trust_remote_code
    )
    datamodule = PermeabilityDataModule(config=data_config, tokenizer=tokenizer)

    # Calculate total steps for WSD scheduler
    datamodule.setup("fit")
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = steps_per_epoch * config.training.max_epochs
    logger.info(f"WSD scheduler: {total_steps} total steps ({steps_per_epoch} steps/epoch × {config.training.max_epochs} epochs)")

    # Create training config
    training_config = PermeabilityTrainingConfig(
        encoder_lr=config.training.encoder_lr,
        head_lr=config.training.head_lr,
        weight_decay=config.training.weight_decay,
        freeze_encoder=config.model.freeze_encoder,
        max_epochs=config.training.max_epochs,
        classifier_dropout=config.model.classifier.dropout,
        classifier_num_layers=config.model.classifier.num_layers,
        encoder_attribute_name=config.model.encoder_attribute_name,
        evidence_lambda_coeff=config.evidence.lambda_coeff,
        total_steps=total_steps,
        warmup_ratio=config.training.warmup_ratio,
        decay_ratio=config.training.decay_ratio,
    )

    # Create model
    model = HELMBertPermeabilityLightning(
        model_name_or_path=config.model.pretrained_path,
        training_config=training_config,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Create callbacks and logger
    checkpoint_config = config_to_checkpoint_config(config)
    display_config = config_to_display_config(config)
    callbacks = create_callbacks(
        checkpoint_dir,
        checkpoint_config,
        display_config,
    )
    wandb_logger = None if config.logging.disable_wandb else WandbLogger(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        save_dir=output_dir,
        config=config_dict,
        tags=build_tags(config, ["permeability", "downstream", "regression", "evidential", config.data.split_name]),
    )

    # Create trainer
    trainer = L.Trainer(
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=config.training.gradient_clip_val,
        deterministic=config.trainer.deterministic,
        default_root_dir=output_dir,
        log_every_n_steps=config.trainer.log_every_n_steps,
    )

    # Train
    log_training_start(logger, "permeability training")
    trainer.fit(model, datamodule)

    training_duration = time.time() - start_time

    # Load best checkpoint
    logger.info(f"Loading best model from: {trainer.checkpoint_callback.best_model_path}")
    model = load_best_checkpoint(trainer, HELMBertPermeabilityLightning, strict=False)

    # Get predictions and metrics on test set
    if datamodule.test_dataset is None:
        logger.warning("No test dataset found, skipping evaluation")
    else:
        # Prepare results directory
        results_dir = Path(config.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions with uncertainty (vectorized)
        predictions_output = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        predictions = np.concatenate([b["predictions"].cpu().numpy() for b in predictions_output]).flatten()
        targets = np.concatenate([b["targets"].cpu().numpy() for b in predictions_output]).flatten()
        aleatoric = np.concatenate([b["uncertainty"]["aleatoric"].cpu().numpy() for b in predictions_output]).flatten()
        epistemic = np.concatenate([b["uncertainty"]["epistemic"].cpu().numpy() for b in predictions_output]).flatten()

        # Save predictions with uncertainty
        pred_df = pd.DataFrame({
            "pred": predictions,
            "actual": targets,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
        })
        pred_file = results_dir / f"predictions_{run_name}.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Saved predictions to {pred_file}")

        # Test metrics
        metrics = trainer.test(model, datamodule)[0]
        logger.info("\nTest Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        # Save metrics
        metrics_file = results_dir / f"metrics_{run_name}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics to {metrics_file}")

    # Log summary and complete
    log_summary(logger, training_duration, output_dir)
    mark_completion(output_dir)
    log_completion(logger, "Permeability training")


if __name__ == "__main__":
    main()
