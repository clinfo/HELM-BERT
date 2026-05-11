from __future__ import annotations

from pathlib import Path

from src.models.mlm_lightning import HELMBertMLMLightning


def resolve_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    """Resolve a named MLM checkpoint within a run directory."""
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_name == "last":
        return checkpoint_dir / "last.ckpt"
    if checkpoint_name == "best":
        candidates = sorted(
            checkpoint_dir.glob("epoch=*-val_loss=*.ckpt"),
            key=lambda path: float(path.stem.split("val_loss=")[-1]),
        )
        if not candidates:
            raise RuntimeError(f"No best checkpoint candidates found in {checkpoint_dir}")
        return candidates[0]
    return checkpoint_dir / checkpoint_name


def export_checkpoint(
    checkpoint_path: Path,
    hf_checkpoint_dir: Path,
    tokenizer,
    logger,
) -> None:
    """Export a Lightning MLM checkpoint to Hugging Face format."""
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Using checkpoint: {checkpoint_path}")
    export_model = HELMBertMLMLightning.load_from_checkpoint(
        str(checkpoint_path),
        strict=True,
        weights_only=False,
    )
    hf_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_model.save_pretrained(str(hf_checkpoint_dir))
    tokenizer.save_pretrained(str(hf_checkpoint_dir))
    logger.info(f"Model saved in HuggingFace format to {hf_checkpoint_dir}")
