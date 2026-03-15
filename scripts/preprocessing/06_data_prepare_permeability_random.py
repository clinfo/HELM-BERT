#!/usr/bin/env python
"""Prepare permeability data for regression task.

Creates train/test split. Train is further split into train/val by datamodule.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging

import pandas as pd
import lightning as L
from sklearn.model_selection import train_test_split

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/mlm/cycpeptmpdb_deduplicated.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"

SEED = 42
TEST_RATIO = 0.1  # 8:1:1 split (10% test)
INVALID_THRESHOLD = -10  # Filter permeability <= -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
TARGET_COL = "Permeability"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare permeability data")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    L.seed_everything(args.seed)

    source_file = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Permeability Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} samples")

    # Sort for deterministic split regardless of input order
    df = df.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Filter invalid samples
    df_filtered = df[df[TARGET_COL] > INVALID_THRESHOLD].copy()
    n_filtered = len(df) - len(df_filtered)
    logger.info(f"Filtered {n_filtered} samples with {TARGET_COL} <= {INVALID_THRESHOLD}")
    logger.info(f"Remaining: {len(df_filtered)} samples")

    # Check required columns
    required_cols = [SMILES_COL, HELM_COL, TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Split train/test
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True
    )

    logger.info(f"\nSplit: {len(train_df)} train, {len(test_df)} test")

    # Select output columns
    output_cols = [SMILES_COL, HELM_COL, TARGET_COL]

    # Save
    train_file = output_dir / "cycpeptmpdb_permeability_train.csv"
    test_file = output_dir / "cycpeptmpdb_permeability_test.csv"

    train_df[output_cols].to_csv(train_file, index=False)
    test_df[output_cols].to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    # Statistics
    logger.info(f"\n{TARGET_COL} statistics:")
    for name, data in [("Train", train_df), ("Test", test_df)]:
        logger.info(f"  {name}: mean={data[TARGET_COL].mean():.3f}, std={data[TARGET_COL].std():.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
