#!/usr/bin/env python
"""Export an MLM Lightning checkpoint to Hugging Face format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from scripts.helpers.training import setup_logging
from scripts.helpers.mlm_checkpoint import export_checkpoint, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLM checkpoint to Hugging Face format")
    parser.add_argument("--run-dir", required=True, help="MLM output run directory")
    parser.add_argument("--checkpoint", default="best", help="'best', 'last', or checkpoint filename")
    parser.add_argument("--tokenizer-path", default="Flansma/helm-bert", help="Tokenizer source path")
    parser.add_argument("--output-name", required=True, help="Checkpoint directory name under ./checkpoints")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path("./outputs") / f"mlm_export_{run_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, run_dir.name, "export_mlm")

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    export_dir = Path("./checkpoints") / args.output_name
    export_checkpoint(checkpoint_path, export_dir, tokenizer, logger)


if __name__ == "__main__":
    main()
