#!/usr/bin/env python
"""Prepare permeability data with random train/test split.

Keeps Permeability, PAMPA, and Caco2 columns.
Stratifies by PAMPA/Caco2 availability.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import Tuple

import pandas as pd
import lightning as L
from sklearn.model_selection import train_test_split

from scripts.preprocessing.helpers.downstream import (
    aggregate_median_by_canonical_smiles,
    log_mlm_coverage,
)
from scripts.preprocessing.helpers.paths import (
    INTERMEDIATE_PRODUCT_DIR,
    PREPROCESSING_OUTPUT_DIR,
    REPO_ROOT,
)

DEFAULT_SOURCE = INTERMEDIATE_PRODUCT_DIR / "cycpeptmpdb_helm_normalized.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = PREPROCESSING_OUTPUT_DIR

SEED = 42
TEST_RATIO = 0.1
INVALID_THRESHOLD = -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
ASSAY_COLS = ["PAMPA", "Caco2"]
TARGET_COLS = ["Permeability"] + ASSAY_COLS

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def _build_strat_key(df: pd.DataFrame) -> pd.Series:
    """Build a non-target stratification key from assay availability."""
    has_pampa = df["PAMPA"].notna()
    has_caco2 = df["Caco2"].notna()
    return has_pampa.astype(int).astype(str) + "_" + has_caco2.astype(int).astype(str)


def _save_split_files(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    """Save one train/test pair and log where it was written."""
    train_file = output_dir / f"{stem}_train.csv"
    test_file = output_dir / f"{stem}_test.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")


def _log_task_statistics(task_name: str, target_col: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log basic target stats for one task-specific split."""
    logger.info(f"\n{task_name} statistics:")
    for split_name, df in [("Train", train_df), ("Test", test_df)]:
        values = df[target_col].dropna()
        logger.info(
            f"  {split_name}: n={len(values)}, mean={values.mean():.3f}, std={values.std():.3f}"
        )


def _run_task_random_split(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    stem: str,
    output_dir: Path,
    test_ratio: float,
    seed: int,
) -> None:
    """Prepare one task-specific random split without deriving it from another task."""
    task_df = df.loc[df[target_col].notna()].copy().reset_index(drop=True)
    logger.info(f"\nPreparing {task_name} random split from {len(task_df)} rows")

    strat_key = _build_strat_key(task_df)
    logger.info(f"  Stratification groups: {strat_key.value_counts().to_dict()}")
    train_df, test_df = train_test_split(
        task_df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strat_key,
    )

    log_mlm_coverage(train_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"{stem}/train")
    log_mlm_coverage(test_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"{stem}/test")
    _save_split_files(train_df, test_df, output_dir, stem)
    _log_task_statistics(task_name, target_col, train_df, test_df)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"permeability_random_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_permeability_random.log"

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

    parser = argparse.ArgumentParser(description="Prepare permeability data (random split)")
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
    logger.info("Permeability Data Preparation (Random Split)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    df = pd.read_csv(source_file, low_memory=False)
    logger.info(f"Loaded {len(df)} samples (HELM already normalized in 02_* stage)")

    # Check required columns
    required_cols = [SMILES_COL, HELM_COL] + ASSAY_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter: keep rows with valid Permeability value
    has_perm = df["Permeability"].notna()
    df_filtered = df[has_perm].copy()
    logger.info(f"Rows with valid Permeability: {len(df_filtered)}")

    # Filter invalid values
    for col in ["Permeability"] + ASSAY_COLS:
        invalid_mask = df_filtered[col].notna() & (df_filtered[col] <= INVALID_THRESHOLD)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df_filtered.loc[invalid_mask, col] = float("nan")
            logger.info(f"Set {n_invalid} invalid {col} values (<= {INVALID_THRESHOLD}) to NaN")

    # Drop rows that lost Permeability after filtering
    has_perm = df_filtered["Permeability"].notna()
    df_filtered = df_filtered[has_perm].copy()
    logger.info(f"After invalid filtering: {len(df_filtered)} samples")

    # Aggregate duplicate measurements per molecule using median assay values.
    df_filtered = df_filtered.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)
    df_filtered = aggregate_median_by_canonical_smiles(
        df_filtered, SMILES_COL, TARGET_COLS, logger
    )
    logger.info(f"After molecule aggregation: {len(df_filtered)} unique molecules")

    # Log per-assay statistics
    for col in ASSAY_COLS:
        valid = df_filtered[col].notna()
        logger.info(f"  {col}: {valid.sum()} valid values")

    both_valid = df_filtered[ASSAY_COLS].notna().all(axis=1)
    logger.info(f"  Both PAMPA+Caco2: {both_valid.sum()} rows")

    # Sort for deterministic split
    df_filtered = df_filtered.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    logger.info(f"\nSaved:")
    _run_task_random_split(
        df_filtered,
        "Permeability",
        "Permeability",
        "cycpeptmpdb_permeability_random",
        output_dir,
        args.test_ratio,
        args.seed,
    )
    _run_task_random_split(
        df_filtered,
        "PAMPA",
        "PAMPA",
        "cycpeptmpdb_permeability_pampa_random",
        output_dir,
        args.test_ratio,
        args.seed,
    )
    _run_task_random_split(
        df_filtered,
        "Caco2",
        "Caco2",
        "cycpeptmpdb_permeability_caco2_random",
        output_dir,
        args.test_ratio,
        args.seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
