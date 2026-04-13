#!/usr/bin/env python3
"""Deduplicate datasets for MLM pre-training.

Priority order: CycPeptMPDB > ChEMBL > Propedia > CREMP
- HELM normalization is done upstream in 02_* scripts
- This script performs canonical SMILES deduplication within and across datasets
"""

import sys
from pathlib import Path

import argparse
import logging
from datetime import datetime
from typing import Tuple

import pandas as pd
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = {
    'cycpeptmpdb': REPO_ROOT / 'local_data/intermediate_product/cycpeptmpdb_helm_normalized.csv',
    'chembl': REPO_ROOT / 'local_data/intermediate_product/chembl_helm_normalized.csv',
    'propedia': REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_with_HELM_SMILES.csv',
    'cremp': REPO_ROOT / 'local_data/intermediate_product/CREMP_v1.1_helm.csv',
}

SMILES_COLUMNS = {
    'cycpeptmpdb': 'SMILES',
    'chembl': 'canonical_smiles',
    'propedia': 'Peptide_SMILES',
    'cremp': 'smiles',
}

HELM_COLUMNS = {
    'cycpeptmpdb': 'HELM',
    'chembl': 'helm_notation',
    'propedia': 'Peptide_HELM',
    'cremp': 'helm',
}

PRIORITY_ORDER = ['cycpeptmpdb', 'chembl', 'propedia', 'cremp']

DEFAULT_OUTPUT_DIR = REPO_ROOT / 'data/mlm'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"mlm_deduplication_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "deduplication.log"

    logger_ = logging.getLogger(__name__)
    logger_.setLevel(logging.INFO)
    logger_.handlers = []

    for handler_cls, handler_args in [
        (logging.StreamHandler, (sys.stdout,)),
        (logging.FileHandler, (log_file,)),
    ]:
        h = handler_cls(*handler_args)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        logger_.addHandler(h)

    logger_.info(f"Log file: {log_file.absolute()}")
    return logger_, log_dir


def smiles_to_canonical(smiles: str) -> str:
    """Convert SMILES to canonical form. Returns '' on failure."""
    if not smiles or pd.isna(smiles) or smiles == '':
        return ''
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol else ''
    except Exception:
        return ''


def deduplicate_by_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Remove duplicates within a dataset using canonical SMILES."""
    if df.empty or smiles_col not in df.columns:
        return pd.DataFrame()

    before = len(df)
    df = df[df[smiles_col].notna() & (df[smiles_col] != '')].copy()

    canonical = df[smiles_col].apply(smiles_to_canonical)
    valid = canonical != ''
    df = df[valid.values].copy()
    df['_canonical'] = canonical[valid].values
    df = df.drop_duplicates(subset=['_canonical']).drop(columns=['_canonical'])

    logger.info(f"  SMILES dedup: {before} → {len(df)}")
    return df


def main():
    global logger

    parser = argparse.ArgumentParser(description="MLM dataset deduplication")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    logger, _ = setup_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MLM Deduplication (HELM already normalized in 02_* stage)")
    logger.info("=" * 60)
    logger.info(f"Priority: {' > '.join(PRIORITY_ORDER)}")
    logger.info(f"Output: {output_dir}")

    logger.info(f"\n{'='*60}")
    logger.info("Step 1: Load → SMILES self-dedup")
    logger.info(f"{'='*60}")

    datasets = {}

    for name in PRIORITY_ORDER:
        input_file = INPUT_FILES[name]
        if not input_file.exists():
            logger.warning(f"{name}: file not found ({input_file}), skipping")
            continue

        logger.info(f"\n--- {name} ---")
        df = pd.read_csv(input_file, low_memory=False)
        logger.info(f"  Loaded: {len(df):,} rows")

        helm_col = HELM_COLUMNS[name]
        smiles_col = SMILES_COLUMNS[name]

        if helm_col not in df.columns:
            logger.warning(f"  HELM column '{helm_col}' not found, skipping")
            continue

        before_null = len(df)
        df = df[df[helm_col].notna() & (df[helm_col].astype(str).str.strip() != '')].copy()
        null_removed = before_null - len(df)
        if null_removed:
            logger.info(f"  Null HELM removed: {null_removed}")

        df = deduplicate_by_smiles(df, smiles_col)
        datasets[name] = df

    logger.info(f"\n{'='*60}")
    logger.info("Step 2: Cross-dataset deduplication")
    logger.info(f"{'='*60}")

    seen_smiles: set = set()

    for i, name in enumerate(PRIORITY_ORDER):
        if name not in datasets:
            continue

        df = datasets[name]
        smiles_col = SMILES_COLUMNS[name]
        before = len(df)

        if i > 0:
            canonical = df[smiles_col].apply(smiles_to_canonical)
            keep = [s not in seen_smiles and s != '' for s in canonical]
            df = df[keep].copy()
            datasets[name] = df
            removed = before - len(df)
            if removed > 0:
                logger.info(f"  {name}: removed {removed:,} cross-duplicates")

        current = df[smiles_col].apply(smiles_to_canonical)
        seen_smiles.update(s for s in current if s != '')
        logger.info(f"  {name}: {len(df):,} rows")

    logger.info(f"\n{'='*60}")
    logger.info("Step 3: Save")
    logger.info(f"{'='*60}")

    for name, df in datasets.items():
        out_file = output_dir / f"{name}_deduplicated.csv"
        df.to_csv(out_file, index=False)
        logger.info(f"  {name}: {len(df):,} rows → {out_file}")

    total = sum(len(df) for df in datasets.values())
    logger.info(f"\nTotal: {total:,} rows across {len(datasets)} datasets")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
