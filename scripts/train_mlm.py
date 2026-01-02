#!/usr/bin/env python
"""MLM training script for HELM-BERT.

Supports two modes:
1. Continue pre-training (default): Load from existing checkpoint and continue MLM training
2. From scratch: Initialize with random weights (use --from_scratch flag)

Example (continue pre-training):
    python scripts/train_mlm.py
    python scripts/train_mlm.py --pretrained_path Flansma/helm-bert
    python scripts/train_mlm.py --pretrained_path ./my-checkpoint --max_epochs 100

Example (from scratch):
    python scripts/train_mlm.py --from_scratch
    python scripts/train_mlm.py --from_scratch --num_hidden_layers 12
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
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoConfig, AutoTokenizer

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
from src.datamodules import MLMDataConfig, MLMDataModule
from src.models.mlm_lightning import HELMBertMLMLightning, MLMTrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HELM-BERT MLM Training (continue pre-training or from scratch)"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mlm",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save final checkpoints",
    )

    # Model
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace Hub model ID or local path for continue pre-training",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch instead of continue pre-training",
    )

    # Architecture (only used with --from_scratch)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_position_embeddings", type=int, default=512)

    # Training
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/deduplicated",
        help="Directory containing training data",
    )
    parser.add_argument("--num_workers", type=int, default=8)

    # Hardware
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="helmbert-mlm")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    return parser.parse_args()


def main():
    """Main MLM training function."""
    start_time = time.time()
    args = parse_args()

    # Setup environment
    setup_training_env(args.seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "scratch" if args.from_scratch else "continue"
    run_name = f"helmbert_mlm_{mode_str}_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(args.output_dir), run_name)

    # Setup logging
    logger = setup_logging(output_dir, timestamp, "train_mlm")
    title = "HELM-BERT MLM Pretraining (from scratch)" if args.from_scratch else "HELM-BERT MLM Continue Pre-training"
    log_header(logger, title)

    # Create training config
    training_config = MLMTrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    data_config = MLMDataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_position_embeddings,
        mlm_probability=args.mlm_probability,
        num_workers=args.num_workers,
    )

    # Prepare model config (only used for from_scratch mode)
    model_config = None
    if args.from_scratch:
        model_config = AutoConfig.from_pretrained(
            args.pretrained_path, trust_remote_code=True
        )
        model_config.hidden_size = args.hidden_size
        model_config.num_hidden_layers = args.num_hidden_layers
        model_config.num_attention_heads = args.num_attention_heads
        model_config.intermediate_size = args.intermediate_size
        model_config.max_position_embeddings = args.max_position_embeddings

    # Save configurations
    config_path = output_dir / "config.json"
    config_dict = {
        "mode": "from_scratch" if args.from_scratch else "continue_pretraining",
        "pretrained_path": args.pretrained_path,
        "training": asdict(training_config),
        "data": asdict(data_config),
        "args": vars(args),
    }
    if args.from_scratch:
        config_dict["model"] = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "intermediate_size": args.intermediate_size,
            "max_position_embeddings": args.max_position_embeddings,
        }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")

    # Log configuration
    if args.from_scratch:
        logger.info("\nModel Architecture (from scratch):")
        logger.info(f"  Hidden size: {args.hidden_size}")
        logger.info(f"  Num layers: {args.num_hidden_layers}")
        logger.info(f"  Num heads: {args.num_attention_heads}")
        logger.info(f"  Intermediate size: {args.intermediate_size}")
    else:
        logger.info(f"\nContinue pre-training from: {args.pretrained_path}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  MLM probability: {args.mlm_probability}")
    logger.info(SEPARATOR_LINE)

    # Create tokenizer and datamodule
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_path, trust_remote_code=True
    )
    datamodule = MLMDataModule(config=data_config, tokenizer=tokenizer)

    # Create model
    model = HELMBertMLMLightning(
        model_name_or_path=args.pretrained_path,
        from_scratch=args.from_scratch,
        model_config=model_config,
        training_config=training_config,
        max_epochs=args.max_epochs,
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
    mode_msg = f"MLM training ({'from scratch' if args.from_scratch else f'from {args.pretrained_path}'})"
    log_training_start(logger, mode_msg)
    trainer.fit(model, datamodule)

    training_duration = time.time() - start_time

    # Load best checkpoint
    logger.info(f"Loading best model from: {trainer.checkpoint_callback.best_model_path}")
    model = load_best_checkpoint(trainer, HELMBertMLMLightning, strict=True)

    # Save model in HuggingFace format
    hf_checkpoint_dir = Path(args.checkpoint_dir) / "helmbert-base"
    hf_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(hf_checkpoint_dir))
    tokenizer.save_pretrained(str(hf_checkpoint_dir))
    logger.info(f"Model saved in HuggingFace format to {hf_checkpoint_dir}")

    # Log summary and complete
    log_summary(logger, training_duration, output_dir, huggingface_checkpoint=hf_checkpoint_dir)
    mark_completion(output_dir)
    log_completion(logger, "MLM training")


if __name__ == "__main__":
    main()
