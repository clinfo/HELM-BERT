#!/usr/bin/env python
"""Prepare permeability data with scaffold-based train/test split.

Uses Murcko scaffolds to ensure molecules sharing the same scaffold
never appear in both train and test sets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import List, Tuple

import lightning as L
import pandas as pd
from src.utils import build_scaffold_groups, flatten_groups, greedy_scaffold_partition

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/mlm/cycpeptmpdb_deduplicated.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/preprocessing"

SEED = 42
TEST_RATIO = 0.1
INVALID_THRESHOLD = -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
TARGET_COL = "Permeability"

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"permeability_scaffold_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_permeability_scaffold.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []

    for handler_cls, args in [
        (logging.StreamHandler, (sys.stdout,)),
        (logging.FileHandler, (log_file,)),
    ]:
        h = handler_cls(*args)
        h.setLevel(LOG_LEVEL)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(h)

    logger.info(f"Log file: {log_file.absolute()}")
    return logger, log_dir


def _size_only_key(test_size: int, _: Tuple[()], target_test_size: int) -> Tuple[float]:
    """Key for single-task scaffold assignment: size only, no label information."""
    return (abs(test_size - target_test_size),)


def scaffold_split(
    df: pd.DataFrame, smiles_col: str, test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by scaffold using direct size-balanced assignment."""
    groups = build_scaffold_groups(df[smiles_col].tolist())
    logger.info(f"Found {len(groups)} unique scaffolds from {len(df)} molecules")
    target_test_size = round(len(df) * test_ratio)
    empty_state: Tuple[()] = tuple()

    test_groups, train_groups, test_size, _ = greedy_scaffold_partition(
        groups=groups,
        group_states=[empty_state] * len(groups),
        target_test_size=target_test_size,
        empty_state=empty_state,
        combine_states=lambda current, _: current,
        key_fn=_size_only_key,
    )

    test_indices = flatten_groups(test_groups)
    train_indices = flatten_groups(train_groups)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    logger.info(
        f"Selected test set: {len(test_groups)} scaffold groups, "
        f"{test_size} samples ({test_size/len(df):.3f} of dataset)"
    )
    logger.info(
        f"Scaffold split: {len(train_df)} train, {len(test_df)} test "
        f"(actual test ratio: {len(test_df)/len(df):.3f})"
    )
    return train_df, test_df


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare permeability data (scaffold split)")
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
    logger.info("Permeability Data Preparation (Scaffold Split)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} samples")

    # Sort for deterministic processing
    df = df.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Filter invalid samples
    df_filtered = df[df[TARGET_COL] > INVALID_THRESHOLD].copy()
    n_filtered = len(df) - len(df_filtered)
    logger.info(f"Filtered {n_filtered} samples with {TARGET_COL} <= {INVALID_THRESHOLD}")
    logger.info(f"Remaining: {len(df_filtered)} samples")

    required_cols = [SMILES_COL, HELM_COL, TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Scaffold-based split
    train_df, test_df = scaffold_split(df_filtered, SMILES_COL, args.test_ratio)

    output_cols = [SMILES_COL, HELM_COL, TARGET_COL]

    train_file = output_dir / "cycpeptmpdb_permeability_scaffold_train.csv"
    test_file = output_dir / "cycpeptmpdb_permeability_scaffold_test.csv"

    train_df[output_cols].to_csv(train_file, index=False)
    test_df[output_cols].to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    logger.info(f"\n{TARGET_COL} statistics:")
    for name, data in [("Train", train_df), ("Test", test_df)]:
        logger.info(f"  {name}: mean={data[TARGET_COL].mean():.3f}, std={data[TARGET_COL].std():.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
