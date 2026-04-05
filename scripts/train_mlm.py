#!/usr/bin/env python
"""MLM training script for HELM-BERT.

Usage:
    python scripts/train_mlm.py
    python scripts/train_mlm.py --config configs/mlm.yaml
    python scripts/train_mlm.py training.batch_size=128 model.from_scratch=true
    python scripts/train_mlm.py --batch_size 128 --from_scratch true
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
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoConfig, AutoTokenizer

from scripts.training_utils import (
    SEPARATOR_LINE,
    config_to_checkpoint_config,
    config_to_display_config,
    create_callbacks,
    create_output_dirs,
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
)
from src.datamodules import MLMDataModule, MLMDataConfig
from src.datamodules.mlm_datamodule import DatasetInfo
from src.models.mlm_lightning import HELMBertMLMLightning, MLMTrainingConfig


def main():
    """Main MLM training function."""
    start_time = time.time()

    # Load configuration
    config = load_config(task="mlm")

    # Setup environment
    setup_training_env(
        config.training.seed,
        config.trainer.float32_matmul_precision,
        config.trainer.deterministic,
    )

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "scratch" if config.model.from_scratch else "continue"
    run_name = f"mlm_{mode_str}_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(config.paths.output_dir), run_name)

    # Setup logging
    logger = setup_logging(output_dir, timestamp, "train_mlm")
    title = "HELM-BERT MLM Pretraining (from scratch)" if config.model.from_scratch else "HELM-BERT MLM Continue Pre-training"
    log_header(logger, title)

    # Create data config
    datasets = [
        DatasetInfo(name=ds.name, file=ds.file, helm_column=ds.helm_column)
        for ds in config.data.datasets
    ]
    data_config = MLMDataConfig(
        data_dir=config.paths.data_dir,
        datasets=datasets,
        train_ratio=config.data.train_ratio,
        batch_size=config.training.batch_size,
        max_seq_length=config.data.max_seq_length,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.training.seed,
        mlm_probability=config.data.masking.mlm_probability,
        mask_ratio=config.data.masking.mask_ratio,
        random_ratio=config.data.masking.random_ratio,
        keep_ratio=config.data.masking.keep_ratio,
        min_span_length=config.data.masking.min_span_length,
        max_span_length=config.data.masking.max_span_length,
        geometric_p=config.data.masking.geometric_p,
        ignore_index=config.mlm.ignore_index,
    )

    # Prepare model config (only used for from_scratch mode)
    model_config = None
    if config.model.from_scratch:
        arch = config.model.architecture
        model_config = AutoConfig.from_pretrained(
            config.model.pretrained_path, trust_remote_code=config.model.trust_remote_code
        )
        model_config.hidden_size = arch.hidden_size
        model_config.num_hidden_layers = arch.num_hidden_layers
        model_config.num_attention_heads = arch.num_attention_heads
        model_config.intermediate_size = arch.intermediate_size
        model_config.max_position_embeddings = arch.max_position_embeddings

    # Save configurations
    config_path_out = output_dir / "config.json"
    config_dict = to_dict(config)
    with open(config_path_out, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path_out}")

    # Log configuration
    if config.model.from_scratch:
        arch = config.model.architecture
        logger.info("\nModel Architecture (from scratch):")
        logger.info(f"  Hidden size: {arch.hidden_size}")
        logger.info(f"  Num layers: {arch.num_hidden_layers}")
        logger.info(f"  Num heads: {arch.num_attention_heads}")
        logger.info(f"  Intermediate size: {arch.intermediate_size}")
    else:
        logger.info(f"\nContinue pre-training from: {config.model.pretrained_path}")

    logger.info("\nTraining Configuration:")
    logger.info(f"  Max epochs: {config.training.max_epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  MLM probability: {config.data.masking.mlm_probability}")
    logger.info(SEPARATOR_LINE)

    # Create tokenizer and datamodule
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_path, trust_remote_code=config.model.trust_remote_code
    )
    datamodule = MLMDataModule(config=data_config, tokenizer=tokenizer)

    # Calculate total steps for WSD scheduler
    datamodule.setup("fit")
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = steps_per_epoch * config.training.max_epochs
    logger.info(f"WSD scheduler: {total_steps} total steps ({steps_per_epoch} steps/epoch × {config.training.max_epochs} epochs)")

    # Create training config
    training_config = MLMTrainingConfig(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        ignore_index=config.mlm.ignore_index,
        total_steps=total_steps,
        warmup_ratio=config.training.warmup_ratio,
        decay_ratio=config.training.decay_ratio,
    )

    # Create model
    model = HELMBertMLMLightning(
        model_name_or_path=config.model.pretrained_path,
        training_config=training_config,
        from_scratch=config.model.from_scratch,
        model_config=model_config,
        trust_remote_code=config.model.trust_remote_code,
        use_emd=config.model.use_emd,
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
        tags=["mlm", "pretrain", mode_str] + (config.logging.tags or []),
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
    mode_msg = f"MLM training ({'from scratch' if config.model.from_scratch else f'from {config.model.pretrained_path}'})"
    log_training_start(logger, mode_msg)
    trainer.fit(model, datamodule)

    training_duration = time.time() - start_time

    # Use final model (WSD: stable phase explores, decay phase settles)
    logger.info("Using final model after WSD decay phase")

    # Save model in HuggingFace format
    hf_checkpoint_dir = Path(config.paths.checkpoint_dir) / config.paths.hf_checkpoint_name
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
