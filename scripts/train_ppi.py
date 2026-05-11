#!/usr/bin/env python
"""PPI (Peptide-Protein Interaction) training script.

Usage:
    python scripts/train_ppi.py
    python scripts/train_ppi.py --config configs/ppi.yaml
    python scripts/train_ppi.py training.use_cached_embeddings=true model.drug_encoder.freeze=false
    python scripts/train_ppi.py --batch_size 64
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
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.helpers.training import (
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
from src.datamodules import PPIDataModule, PPIDataConfig
from src.models.ppi_lightning import HELMGLaMLightning, PPITrainingConfig


def main():
    """Main training function."""
    start_time = time.time()

    # Load configuration
    if "--config" not in sys.argv:
        print("Error: --config is required for PPI training.")
        print("  python scripts/train_ppi.py --config configs/ppi_random.yaml")
        print("  python scripts/train_ppi.py --config configs/ppi_acsm.yaml")
        sys.exit(1)
    config = load_config(task="ppi")

    # Setup environment
    setup_training_env(
        config.training.seed,
        config.trainer.float32_matmul_precision,
        config.trainer.deterministic,
    )

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{config.logging.run_tag}" if config.logging.run_tag else ""
    run_name = f"ppi_{config.data.split_name}_{get_model_tag(config)}{tag}_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(config.paths.output_dir), run_name)

    # Setup logging
    logger = setup_logging(output_dir, timestamp, "train_ppi")
    log_header(logger, "PPI Evidential Classification Training")

    # Convert esm_hidden_sizes to dict
    esm_hidden_sizes = OmegaConf.to_container(config.esm_hidden_sizes, resolve=True)

    # Training config will be created after datamodule setup (need total_steps)

    # Create data config
    data_config = PPIDataConfig(
        train_file=config.data.train_file,
        test_file=config.data.test_file,
        drug_column=config.data.drug_column,
        target_column=config.data.target_column,
        label_column=config.data.label_column,
        target_encoder=config.model.target_encoder.pretrained_path,
        val_ratio=config.data.val_ratio,
        batch_size=config.training.batch_size,
        max_drug_length=config.data.max_drug_length,
        max_target_length=config.data.max_target_length,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.training.seed,
        use_cached_embeddings=config.training.use_cached_embeddings,
        cache_dir=config.paths.cache_dir,
        drug_encoder=config.model.drug_encoder.pretrained_path,
        trust_remote_code=config.model.trust_remote_code,
        cache_drug_encoder_name=config.cache.drug_encoder_name,
        cache_target_encoder_name=config.cache.target_encoder_name,
        cache_dataset_type=config.cache.dataset_type,
    )

    # Save configurations
    config_path_out = output_dir / "config.json"
    config_dict = to_dict(config)
    with open(config_path_out, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path_out}")

    # Log configuration
    logger.info(f"\nDrug encoder: {config.model.drug_encoder.pretrained_path}")
    logger.info(f"Target encoder: {config.model.target_encoder.pretrained_path}")
    logger.info(f"Freeze drug encoder: {config.model.drug_encoder.freeze}")
    logger.info(f"Freeze target encoder: {config.model.target_encoder.freeze}")
    logger.info(f"Use cached embeddings: {config.training.use_cached_embeddings}")
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {config.training.max_epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Encoder LR: {config.training.encoder_lr}")
    logger.info(f"  Head LR: {config.training.head_lr}")
    logger.info(SEPARATOR_LINE)

    # Create tokenizer and datamodule
    drug_tokenizer = AutoTokenizer.from_pretrained(
        config.model.drug_encoder.pretrained_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    datamodule = PPIDataModule(config=data_config, drug_tokenizer=drug_tokenizer)

    # Generate embedding cache before setup (setup loads cached embeddings)
    datamodule.prepare_data()

    # Calculate total steps for WSD scheduler
    datamodule.setup("fit")
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = steps_per_epoch * config.training.max_epochs
    logger.info(f"WSD scheduler: {total_steps} total steps ({steps_per_epoch} steps/epoch × {config.training.max_epochs} epochs)")

    # Create training config
    training_config = PPITrainingConfig(
        encoder_lr=config.training.encoder_lr,
        head_lr=config.training.head_lr,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        mlp_dropout=config.model.head.dropout,
        num_classes=config.model.head.num_classes,
        freeze_drug_encoder=config.model.drug_encoder.freeze,
        freeze_target_encoder=config.model.target_encoder.freeze,
        use_cached_embeddings=config.training.use_cached_embeddings,
        target_encoder=config.model.target_encoder.pretrained_path,
        esm_hidden_sizes=esm_hidden_sizes,
        prediction_threshold=config.classification.prediction_threshold,
        evidence_lambda_coeff=config.evidence.lambda_coeff,
        total_steps=total_steps,
        warmup_ratio=config.training.warmup_ratio,
        decay_ratio=config.training.decay_ratio,
    )

    # Create model
    model = HELMGLaMLightning(
        drug_model_path=config.model.drug_encoder.pretrained_path,
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
        tags=build_tags(config, ["ppi", "downstream", "classification", "evidential", config.data.split_name]),
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
    log_training_start(logger, "PPI training")
    trainer.fit(model, datamodule)

    training_duration = time.time() - start_time

    # Load best checkpoint
    logger.info(f"Loading best model from: {trainer.checkpoint_callback.best_model_path}")
    model = load_best_checkpoint(trainer, HELMGLaMLightning, strict=False)

    # Get predictions and metrics on test set
    if datamodule.test_dataset is None:
        logger.warning("No test dataset found, skipping evaluation")
    else:
        # Prepare results directory
        results_dir = Path(config.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions with uncertainty (vectorized)
        predictions_output = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        probs = np.concatenate([b["predictions"].cpu().numpy() for b in predictions_output]).flatten()
        targets = np.concatenate([b["targets"].cpu().numpy() for b in predictions_output]).flatten()
        uncertainty = np.concatenate([b["uncertainty"].cpu().numpy() for b in predictions_output]).flatten()

        # Dirichlet probabilities are already in [0, 1] — no sigmoid needed
        threshold = config.classification.prediction_threshold
        pred_labels = (probs >= threshold).astype(int)

        # Save predictions with uncertainty
        pred_df = pd.DataFrame({
            "pred_prob": probs,
            "pred_label": pred_labels,
            "actual": targets.astype(int),
            "uncertainty": uncertainty,
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
    log_completion(logger, "PPI training")


if __name__ == "__main__":
    main()
