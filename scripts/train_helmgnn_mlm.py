#!/usr/bin/env python
"""HELM-GNN MLM training script.

Usage:
    python scripts/train_helmgnn_mlm.py
    python scripts/train_helmgnn_mlm.py --config configs/mlm_gnn.yaml
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

from scripts.training_utils import (
    SEPARATOR_LINE,
    config_to_checkpoint_config,
    config_to_display_config,
    create_callbacks,
    create_output_dirs,
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
from src.datamodules.helmgnn_mlm_datamodule import (
    HELMGNNMLMDataConfig,
    HELMGNNMLMDataModule,
    HELMGNNDatasetInfo,
)
from src.models.configuration_helmgnn import HELMGNNConfig
from src.models.helmgnn_mlm_lightning import HELMGNNMLMLightning, HELMGNNMLMTrainingConfig
from src.models.tokenization_helmgnn import HELMGNNTokenizer


def main():
    start_time = time.time()

    config = load_config(task="mlm_gnn")

    setup_training_env(
        config.training.seed,
        config.trainer.float32_matmul_precision,
        config.trainer.deterministic,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"helmgnn_mlm_{timestamp}"
    output_dir, checkpoint_dir = create_output_dirs(Path(config.paths.output_dir), run_name)

    logger = setup_logging(output_dir, timestamp, "train_helmgnn_mlm")
    log_header(logger, "HELM-GNN MLM Pre-training (End-to-End GPS + Transformer)")

    # Build tokenizer from monomer library
    monomer_library = config.model.monomer_library_path
    tokenizer = HELMGNNTokenizer(monomer_library_path=monomer_library)
    logger.info(f"Monomer tokenizer: vocab_size={tokenizer.vocab_size}")

    # Build id_to_symbol mapping for GPS
    id_to_symbol = {v: k for k, v in tokenizer.vocab.items()}

    # Data config
    datasets = [
        HELMGNNDatasetInfo(name=ds.name, file=ds.file, helm_column=ds.helm_column)
        for ds in config.data.datasets
    ]
    data_config = HELMGNNMLMDataConfig(
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
        max_graph_distance=config.model.max_graph_distance,
    )

    datamodule = HELMGNNMLMDataModule(config=data_config, tokenizer=tokenizer)

    # Calculate total steps
    datamodule.setup("fit")
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = steps_per_epoch * config.training.max_epochs
    logger.info(f"WSD scheduler: {total_steps} total steps ({steps_per_epoch}/epoch × {config.training.max_epochs} epochs)")

    # Model config
    model_config = HELMGNNConfig(
        vocab_size=tokenizer.vocab_size,
        gps_hidden_dim=config.model.architecture.gps_hidden_dim,
        gps_num_layers=config.model.architecture.gps_num_layers,
        gps_num_heads=config.model.architecture.gps_num_heads,
        gps_dropout=config.model.architecture.gps_dropout,
        hidden_size=config.model.architecture.hidden_size,
        num_hidden_layers=config.model.architecture.num_hidden_layers,
        num_attention_heads=config.model.architecture.num_attention_heads,
        intermediate_size=config.model.architecture.intermediate_size,
        max_position_embeddings=config.model.architecture.max_position_embeddings,
        hidden_dropout_prob=config.model.architecture.hidden_dropout_prob,
        attention_probs_dropout_prob=config.model.architecture.attention_probs_dropout_prob,
        max_graph_distance=config.model.max_graph_distance,
        num_graph_distance_buckets=config.model.max_graph_distance + 1,
        monomer_library_path=monomer_library,
    )

    training_config = HELMGNNMLMTrainingConfig(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        ignore_index=config.mlm.ignore_index,
        total_steps=total_steps,
        warmup_ratio=config.training.warmup_ratio,
        decay_ratio=config.training.decay_ratio,
    )

    model = HELMGNNMLMLightning(
        config=model_config,
        training_config=training_config,
        id_to_symbol=id_to_symbol,
        use_emd=config.model.use_emd,
    )

    # Save config
    config_path_out = output_dir / "config.json"
    config_dict = to_dict(config)
    with open(config_path_out, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("\nModel Architecture:")
    logger.info(f"  GPS: {model_config.gps_num_layers} layers, hidden={model_config.gps_hidden_dim}, heads={model_config.gps_num_heads}")
    logger.info(f"  Transformer: {model_config.num_hidden_layers} layers, hidden={model_config.hidden_size}, heads={model_config.num_attention_heads}")
    logger.info(f"  Vocab: {model_config.vocab_size} monomers")
    logger.info(f"  Graph distance: max={model_config.max_graph_distance}")
    logger.info(f"\nTraining: epochs={config.training.max_epochs}, batch={config.training.batch_size}, lr={config.training.learning_rate}")
    logger.info(SEPARATOR_LINE)

    # Callbacks and logger
    checkpoint_config = config_to_checkpoint_config(config)
    display_config = config_to_display_config(config)
    callbacks = create_callbacks(checkpoint_dir, checkpoint_config, display_config)
    wandb_logger = None if config.logging.disable_wandb else WandbLogger(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        save_dir=output_dir,
        config=config_dict,
        tags=["mlm", "helmgnn", "gps", "end-to-end"] + (config.logging.tags or []),
    )

    # Train
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

    log_training_start(logger, "HELM-GNN MLM (end-to-end GPS + Transformer)")
    trainer.fit(model, datamodule)

    training_duration = time.time() - start_time

    # Export last checkpoint
    last_ckpt = Path(checkpoint_dir) / "last.ckpt"
    if not last_ckpt.exists():
        raise RuntimeError("No last checkpoint found after training")
    logger.info(f"Using last checkpoint: {last_ckpt}")
    export_model = HELMGNNMLMLightning.load_from_checkpoint(
        str(last_ckpt), strict=True, id_to_symbol=id_to_symbol
    )

    hf_checkpoint_dir = Path(config.paths.checkpoint_dir) / config.paths.hf_checkpoint_name
    hf_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_model.save_pretrained(str(hf_checkpoint_dir))
    tokenizer.save_pretrained(str(hf_checkpoint_dir))
    logger.info(f"Model saved to {hf_checkpoint_dir}")

    log_summary(logger, training_duration, output_dir, huggingface_checkpoint=hf_checkpoint_dir)
    mark_completion(output_dir)
    log_completion(logger, "HELM-GNN MLM training")


if __name__ == "__main__":
    main()
