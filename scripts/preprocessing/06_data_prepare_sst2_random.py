#!/usr/bin/env python
"""Prepare SST2 binding data with random train/test split.

Source: local_data/raw/sst2_dataset.csv (306 rows, 297 unique molecules)
Steps:
  1. Deduplicate SMILES by activity type priority (Ki > Kd > IC50)
  2. Random split (10% test)

Output: data/downstream/sst2_random_{train,test}.csv
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
    log_dir = log_base / f"sst2_random_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_sst2_random.log"
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


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare SST2 binding data (random split)")
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
    logger.info("SST2 Data Preparation (Random Split)")
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

    train_df, test_df = train_test_split(
        df, test_size=args.test_ratio, random_state=args.seed, shuffle=True,
    )

    logger.info(f"Random split: {len(train_df)} train, {len(test_df)} test")

    for name, split_df in [("Train", train_df), ("Test", test_df)]:
        logger.info(f"  {name}: n={len(split_df)}, mean={split_df[TARGET_COL].mean():.3f}, "
                     f"std={split_df[TARGET_COL].std():.3f}")

    train_df.to_csv(output_dir / "sst2_random_train.csv", index=False)
    test_df.to_csv(output_dir / "sst2_random_test.csv", index=False)
    logger.info(f"Saved: sst2_random_{{train,test}}.csv")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
