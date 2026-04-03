#!/usr/bin/env python
"""Prepare multi-assay permeability data (PAMPA + Caco2).

Keeps PAMPA and Caco2 as separate target columns.
Rows must have at least one valid assay value.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import Tuple

import pandas as pd
import lightning as L
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/mlm/cycpeptmpdb_deduplicated.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/preprocessing"

SEED = 42
TEST_RATIO = 0.1
INVALID_THRESHOLD = -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
ASSAY_COLS = ["PAMPA", "Caco2"]

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"multi_assay_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_multi_assay.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file.absolute()}")

    return logger, log_dir


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare multi-assay permeability data")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logger, log_dir = setup_logging()

    L.seed_everything(args.seed)

    source_file = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-Assay Permeability Data Preparation (PAMPA + Caco2)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    df = pd.read_csv(source_file, low_memory=False)
    logger.info(f"Loaded {len(df)} samples")

    # Check required columns
    required_cols = [SMILES_COL, HELM_COL] + ASSAY_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter: keep rows with at least one valid assay value
    has_any_assay = df[ASSAY_COLS].notna().any(axis=1)
    df_filtered = df[has_any_assay].copy()
    logger.info(f"Rows with at least one assay value: {len(df_filtered)}")

    # Filter invalid values per assay
    for col in ASSAY_COLS:
        invalid_mask = df_filtered[col].notna() & (df_filtered[col] <= INVALID_THRESHOLD)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df_filtered.loc[invalid_mask, col] = float("nan")
            logger.info(f"Set {n_invalid} invalid {col} values (<= {INVALID_THRESHOLD}) to NaN")

    # Drop rows that lost all assay values after filtering
    has_any_assay = df_filtered[ASSAY_COLS].notna().any(axis=1)
    df_filtered = df_filtered[has_any_assay].copy()
    logger.info(f"After invalid filtering: {len(df_filtered)} samples")

    # Log per-assay statistics
    for col in ASSAY_COLS:
        valid = df_filtered[col].notna()
        logger.info(f"  {col}: {valid.sum()} valid values")

    both_valid = df_filtered[ASSAY_COLS].notna().all(axis=1)
    logger.info(f"  Both PAMPA+Caco2: {both_valid.sum()} rows")

    # Sort for deterministic split
    df_filtered = df_filtered.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Build stratification key: preserve assay-type ratios across split
    has_pampa = df_filtered["PAMPA"].notna()
    has_caco2 = df_filtered["Caco2"].notna()
    strat_key = has_pampa.astype(int).astype(str) + "_" + has_caco2.astype(int).astype(str)
    logger.info(f"Stratification groups: {strat_key.value_counts().to_dict()}")

    # Split train/test (stratified by assay availability)
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=strat_key,
    )

    logger.info(f"\nSplit: {len(train_df)} train, {len(test_df)} test")

    # Select output columns
    output_cols = [SMILES_COL, HELM_COL] + ASSAY_COLS

    # Save
    train_file = output_dir / "cycpeptmpdb_multi_assay_random_train.csv"
    test_file = output_dir / "cycpeptmpdb_multi_assay_random_test.csv"

    train_df[output_cols].to_csv(train_file, index=False)
    test_df[output_cols].to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    # Per-assay statistics
    for col in ASSAY_COLS:
        logger.info(f"\n{col} statistics:")
        for name, data in [("Train", train_df), ("Test", test_df)]:
            valid = data[col].dropna()
            logger.info(f"  {name}: n={len(valid)}, mean={valid.mean():.3f}, std={valid.std():.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
