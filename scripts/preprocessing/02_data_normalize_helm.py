#!/usr/bin/env python3
"""Normalize HELM notation in raw datasets.

Applies shared HELM normalization and monomer-library validation to raw HELM
datasets and writes normalized copies to `intermediate_product/`.

Usage:
    python scripts/preprocessing/02_data_normalize_helm.py
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import Any, Tuple, cast

import pandas as pd

from scripts.preprocessing.preprocessing_utils.helm_utils import (
    apply_helm_normalization,
    load_monomer_library,
)
from scripts.preprocessing.preprocessing_utils.paths import (
    MONOMER_LIBRARY_PATH,
    PREPROCESSING_OUTPUT_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    INTERMEDIATE_PRODUCT_DIR,
)

DATASETS = {
    'cycpeptmpdb': {
        'input': RAW_DATA_DIR / 'CycPeptMPDB_Peptide_All_V1.2.csv',
        'output': INTERMEDIATE_PRODUCT_DIR / 'cycpeptmpdb_helm_normalized.csv',
        'helm_col': 'HELM',
    },
    'chembl': {
        'input': RAW_DATA_DIR / 'chembl36_helm_compounds.csv',
        'output': INTERMEDIATE_PRODUCT_DIR / 'chembl_helm_normalized.csv',
        'helm_col': 'bt_helm_notation',
        'usecols': ['md_chembl_id', 'md_molecule_type', 'bt_helm_notation', 'cs_canonical_smiles'],
        'filter_in': {'md_molecule_type': ['Protein', 'Small molecule']},
        'drop_cols': ['md_molecule_type'],
        'rename': {'bt_helm_notation': 'helm_notation', 'cs_canonical_smiles': 'canonical_smiles'},
    },
    'chembl_ppi': {
        'input': RAW_DATA_DIR / 'helm_ppi_dataset.csv',
        'output': INTERMEDIATE_PRODUCT_DIR / 'chembl_ppi_helm_normalized.csv',
        'helm_col': 'helm_notation',
    },
}

DEFAULT_MONOMER_LIBRARY = MONOMER_LIBRARY_PATH
DEFAULT_LOG_DIR = PREPROCESSING_OUTPUT_DIR

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"normalize_helm_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "normalize.log"

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


def main():
    global logger

    parser = argparse.ArgumentParser(description="Normalize HELM in raw datasets")
    parser.add_argument("--monomer-library", type=str, default=str(DEFAULT_MONOMER_LIBRARY))
    args = parser.parse_args()

    logger, _ = setup_logging()

    logger.info("=" * 60)
    logger.info("Normalize HELM Datasets")
    logger.info("=" * 60)

    valid_symbols, alt_to_canonical = load_monomer_library(args.monomer_library, logger)

    for name, cfg in DATASETS.items():
        input_file = cfg['input']
        output_file = cfg['output']
        helm_col = cfg['helm_col']

        logger.info(f"\n--- {name} ---")

        if not input_file.exists():
            logger.warning(f"  File not found: {input_file}, skipping")
            continue

        usecols = cfg.get('usecols')
        df = cast(
            pd.DataFrame,
            pd.read_csv(input_file, low_memory=False, usecols=cast(Any, usecols)),
        )
        logger.info(f"  Loaded: {len(df):,} rows")

        filter_in = cfg.get('filter_in')
        if filter_in:
            for col, vals in filter_in.items():
                before = len(df)
                col_series = cast(pd.Series, df[col])
                df = cast(pd.DataFrame, df[col_series.isin(vals)].copy())
                logger.info(f"  Filter {col} in {vals}: {before:,} → {len(df):,}")

        drop_cols = cfg.get('drop_cols')
        if drop_cols:
            df = cast(pd.DataFrame, df.drop(columns=[c for c in drop_cols if c in df.columns]))

        if helm_col not in df.columns:
            logger.warning(f"  HELM column '{helm_col}' not found, skipping")
            continue

        df, _ = apply_helm_normalization(df, helm_col, valid_symbols, alt_to_canonical, logger)

        rename_map = cfg.get('rename')
        if rename_map:
            df = cast(pd.DataFrame, df.rename(columns=rename_map))
            logger.info(f"  Renamed columns: {rename_map}")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"  Saved: {len(df):,} rows → {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
