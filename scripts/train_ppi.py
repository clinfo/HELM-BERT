#!/usr/bin/env python
"""PPI (Peptide-Protein Interaction) training script.

Example:
    python scripts/train_ppi.py
    python scripts/train_ppi.py --pretrained_path ./checkpoints/helmbert-base
    python scripts/train_ppi.py --finetune_drug_encoder  # unfreeze drug encoder
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Minimal environment setup before ML library imports
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.append(str(Path(__file__).parent.parent))

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer

from scripts.training_utils import (
    DEFAULT_MODEL,
    SEPARATOR_LINE,
    create_callbacks,
    create_output_dirs,
    load_best_checkpoint,
    log_completion,
    log_header,
    log_summary,
    log_training_start,
    mark_completion,
    setup_logging,
    setup_training_env,
)
from src.datamodules import PPIDataConfig, PPIDataModule
from src.models.ppi_lightning import HELMGLaMLightning, PPITrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PPI Classification Training")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/ppi",
        help="Output directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/ppi",
        help="Directory to save results",
    )

    # Model
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace Hub model ID or local path",
    )
    parser.add_argument(
        "--target_encoder",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="Target encoder model name (ESM-2)",
    )
    parser.add_argument("--freeze_drug_encoder", action="store_true",
                        help="Freeze drug encoder weights")
    parser.add_argument("--finetune_drug_encoder", dest="freeze_drug_encoder", action="store_false",
                        help="Finetune drug encoder weights")
    parser.set_defaults(freeze_drug_encoder=True)
    parser.add_argument("--freeze_target_encoder", action="store_true",
                        help="Freeze target encoder weights")
    parser.add_argument("--finetune_target_encoder", dest="freeze_target_encoder", action="store_false",
                        help="Finetune target encoder weights")
    parser.set_defaults(freeze_target_encoder=True)
    parser.add_argument("--head_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_lr", type=float, default=3e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # Data
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/downstream/propedia_ppi_train.csv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./data/downstream/propedia_ppi_test.csv",
    )
    parser.add_argument("--drug_column", type=str, default="Peptide_HELM")
    parser.add_argument("--target_column", type=str, default="Receptor_Sequence")
    parser.add_argument("--label_column", type=str, default="Label")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_drug_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)

    # Hardware
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="helmbert-ppi")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    return parser.parse_args()


def main():
    """Main training function."""
    start_time = time.time()
    args = parse_args()

    # Setup environment
    setup_training_env(args.seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppi_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(args.output_dir), run_name)

    # Setup logging
    logger = setup_logging(output_dir, timestamp, "train_ppi")
    log_header(logger, "PPI Classification Training")

    # Create configurations
    training_config = PPITrainingConfig(
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        freeze_drug_encoder=args.freeze_drug_encoder,
        freeze_target_encoder=args.freeze_target_encoder,
        target_encoder=args.target_encoder,
        mlp_dropout=args.head_dropout,
        use_cached_embeddings=False,
    )

    data_config = PPIDataConfig(
        train_file=args.train_file,
        test_file=args.test_file,
        drug_column=args.drug_column,
        target_column=args.target_column,
        label_column=args.label_column,
        target_encoder=args.target_encoder,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        max_drug_length=args.max_drug_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
    )

    # Save configurations
    config_path = output_dir / "config.json"
    config_dict = {
        "training": asdict(training_config),
        "data": asdict(data_config),
        "args": vars(args),
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Log configuration
    logger.info(f"\nDrug encoder: {args.pretrained_path}")
    logger.info(f"Target encoder: {args.target_encoder}")
    logger.info(f"Freeze drug encoder: {args.freeze_drug_encoder}")
    logger.info(f"Freeze target encoder: {args.freeze_target_encoder}")
    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Encoder LR: {args.encoder_lr}")
    logger.info(f"  Head LR: {args.head_lr}")
    logger.info(SEPARATOR_LINE)

    # Create tokenizer and datamodule
    drug_tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_path, trust_remote_code=True
    )
    datamodule = PPIDataModule(config=data_config, drug_tokenizer=drug_tokenizer)

    # Create model
    model = HELMGLaMLightning(
        drug_model_path=args.pretrained_path,
        training_config=training_config,
    )

    # Create callbacks and logger
    callbacks = create_callbacks(checkpoint_dir, args.early_stopping_patience)
    wandb_logger = None if args.disable_wandb else WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        save_dir=output_dir,
        config=config_dict,
    )

    # Create trainer
    trainer = L.Trainer(
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        default_root_dir=output_dir,
        log_every_n_steps=10,
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
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions (vectorized)
        predictions_output = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        predictions = torch.cat([b["predictions"].cpu() for b in predictions_output]).numpy()
        labels = torch.cat([b["labels"].cpu() for b in predictions_output]).numpy()

        # Compute probabilities and labels (vectorized)
        probs = 1 / (1 + np.exp(-predictions))
        pred_labels = (probs >= 0.5).astype(int)

        # Save predictions
        pred_df = pd.DataFrame({
            "pred_prob": probs.flatten(),
            "pred_label": pred_labels.flatten(),
            "actual": labels.astype(int),
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
