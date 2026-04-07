#!/usr/bin/env python
"""Prepare SST2 binding data with scaffold train/test split.

Source: local_data/raw/sst2_dataset.csv (306 rows, 297 unique molecules)
Steps:
  1. Deduplicate SMILES by activity type priority (Ki > Kd > IC50)
  2. Scaffold split (10% test, Murcko scaffold-based)

Output: data/downstream/sst2_scaffold_{train,test}.csv
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
from src.utils import build_scaffold_groups

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "local_data/raw/sst2_dataset.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/preprocessing"

SEED = 42
TEST_RATIO = 0.1

SMILES_COL = "canonical_smiles"
HELM_COL = "helm_notation"
TARGET_COL = "pchembl_value"

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)

ACTIVITY_PRIORITY = {"Ki": 0, "Kd": 1, "IC50": 2}


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"sst2_scaffold_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_sst2_scaffold.log"
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(logging.INFO)
    logger_.handlers = []

    for handler_cls, args in [
        (logging.StreamHandler, (sys.stdout,)),
        (logging.FileHandler, (log_file,)),
    ]:
        h = handler_cls(*args)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        logger_.addHandler(h)

    logger_.info(f"Log file: {log_file.absolute()}")
    return logger_, log_dir


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """For duplicate SMILES, keep the row with highest-priority activity type.

    Priority: Ki > Kd > IC50  (Ki is the most direct binding affinity measure).
    """
    df = df.copy()
    df["_priority"] = df["activity_type"].map(ACTIVITY_PRIORITY).fillna(len(ACTIVITY_PRIORITY))
    df = df.sort_values(["_priority", TARGET_COL], ascending=[True, False])
    dedup = df.drop_duplicates(subset=SMILES_COL, keep="first").drop(columns=["_priority"])
    return dedup


def scaffold_split(
    df: pd.DataFrame, test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by Murcko scaffold, packing small multi-molecule groups into test.

    Strategy:
      1. Oversized groups (> target_test // 4) go to train.
      2. Singletons go to train.
      3. Remaining multi-molecule groups fill test, smallest first.
    """
    groups = build_scaffold_groups(df[SMILES_COL].tolist())
    logger.info(f"Found {len(groups)} unique scaffolds from {len(df)} molecules")

    target_test_size = round(len(df) * test_ratio)
    max_group_size = target_test_size // 2

    # Sort eligible multi-molecule groups largest-first for packing
    indexed = [(i, len(g)) for i, g in enumerate(groups)]
    eligible = sorted([(i, s) for i, s in indexed if 2 <= s <= max_group_size], key=lambda x: (x[1], x[0]), reverse=True)
    singletons = [i for i, s in indexed if s == 1]
    oversized = [i for i, s in indexed if s > max_group_size]

    test_indices: list[int] = []
    train_indices: list[int] = []
    test_size = 0

    for i, size in eligible:
        if test_size + size <= target_test_size:
            test_indices.extend(groups[i])
            test_size += size
        else:
            train_indices.extend(groups[i])

    for i in oversized + singletons:
        train_indices.extend(groups[i])

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    test_group_sizes = sorted(
        [len(groups[i]) for i, _ in eligible if all(idx in test_indices for idx in groups[i][:1])],
        reverse=True,
    )
    logger.info(
        f"Scaffold split: {len(train_df)} train, {len(test_df)} test "
        f"(target={target_test_size}, actual ratio={len(test_df)/len(df):.3f})"
    )
    return train_df, test_df


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare SST2 binding data (scaffold split)")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logger, _ = setup_logging()
    L.seed_everything(args.seed)

    source_file = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SST2 Data Preparation (Scaffold Split)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} rows")

    for col in [SMILES_COL, HELM_COL, TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    n_before = len(df)
    df = deduplicate(df)
    logger.info(f"After deduplication: {len(df)} molecules ({n_before - len(df)} duplicates removed)")

    df = df.sort_values([HELM_COL, SMILES_COL]).reset_index(drop=True)

    logger.info(f"{TARGET_COL}: mean={df[TARGET_COL].mean():.3f}, std={df[TARGET_COL].std():.3f}, "
                f"range=[{df[TARGET_COL].min():.3f}, {df[TARGET_COL].max():.3f}]")

    train_df, test_df = scaffold_split(df, args.test_ratio)

    logger.info(f"Scaffold split: {len(train_df)} train, {len(test_df)} test")

    for name, split_df in [("Train", train_df), ("Test", test_df)]:
        logger.info(f"  {name}: n={len(split_df)}, mean={split_df[TARGET_COL].mean():.3f}, "
                     f"std={split_df[TARGET_COL].std():.3f}")

    train_df.to_csv(output_dir / "sst2_scaffold_train.csv", index=False)
    test_df.to_csv(output_dir / "sst2_scaffold_test.csv", index=False)
    logger.info(f"Saved: sst2_scaffold_{{train,test}}.csv")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
