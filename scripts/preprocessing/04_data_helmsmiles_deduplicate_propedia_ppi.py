#!/usr/bin/env python3
"""Deduplicate Propedia v2 peptide-receptor pairs.

Removes duplicate (Peptide_Sequence, Receptor_Sequence) pairs that appear
across multiple PDB structures, keeping the first occurrence.

Input: Propedia_v2_with_HELM_SMILES.csv (from 02)
Output: Propedia_v2_unique_ppi_HELM_SMILES.csv (used by 06_ppi_*)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import argparse

# Configuration (repo-relative by default; CLI-overridable)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_with_HELM_SMILES.csv'
DEFAULT_OUTPUT = REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'

# Column names
PEPTIDE_COL = 'Peptide_Sequence'
RECEPTOR_COL = 'Receptor_Sequence'
PDB_COL = 'PDB'
PEPTIDE_CHAIN_COL = 'Peptide_Chain'
RECEPTOR_CHAIN_COL = 'Receptor_Chain'


def setup_logging():
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(DEFAULT_LOG_DIR) / f"propedia_v2_unique_helm_smiles_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "deduplication.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_dir


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """Analyze duplicate patterns in the dataset."""
    # Count occurrences of each peptide-receptor pair
    pair_counts = df.groupby([PEPTIDE_COL, RECEPTOR_COL]).size()
    
    # Find duplicated pairs
    duplicated_pairs = pair_counts[pair_counts > 1].sort_values(ascending=False)
    
    # Analyze duplication patterns
    stats = {
        'total_rows': len(df),
        'unique_pairs': len(pair_counts),
        'duplicate_rows': len(df) - len(pair_counts),
        'pairs_appearing_once': len(pair_counts[pair_counts == 1]),
        'pairs_appearing_multiple': len(pair_counts[pair_counts > 1]),
        'max_appearances': int(pair_counts.max()),
        'top_duplicates': []
    }
    
    # Get top 10 most duplicated pairs with their PDB info
    for i, ((pep, rec), count) in enumerate(duplicated_pairs.head(10).items()):
        pair_data = df[(df[PEPTIDE_COL] == pep) & (df[RECEPTOR_COL] == rec)]
        unique_pdbs = pair_data[PDB_COL].nunique()
        pdb_list = pair_data[PDB_COL].unique()[:5].tolist()
        
        stats['top_duplicates'].append({
            'peptide_preview': pep[:50] + '...' if len(pep) > 50 else pep,
            'receptor_preview': rec[:50] + '...' if len(rec) > 50 else rec,
            'occurrences': int(count),
            'unique_pdbs': int(unique_pdbs),
            'pdb_examples': pdb_list
        })
    
    # Analyze duplication sources
    dup_from_diff_pdb = 0
    dup_from_same_pdb = 0
    
    for (pep, rec), count in duplicated_pairs.items():
        if count > 1:
            pair_data = df[(df[PEPTIDE_COL] == pep) & (df[RECEPTOR_COL] == rec)]
            if pair_data[PDB_COL].nunique() > 1:
                dup_from_diff_pdb += 1
            else:
                dup_from_same_pdb += 1
    
    stats['duplicates_from_different_pdbs'] = dup_from_diff_pdb
    stats['duplicates_from_same_pdb'] = dup_from_same_pdb
    
    return stats


def deduplicate_peptide_receptor_pairs(input_file: str, output_file: str) -> pd.DataFrame:
    """Remove duplicate peptide-receptor pairs, keeping first occurrence."""
    logger, log_dir = setup_logging()
    
    logger.info("=" * 70)
    logger.info("Propedia v2 Peptide-Receptor Pair Deduplication")
    logger.info("=" * 70)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log directory: {log_dir}")
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(input_file)
    
    # Validate required columns
    required_cols = [PEPTIDE_COL, RECEPTOR_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Analyze duplicates before deduplication
    logger.info("\n" + "=" * 50)
    logger.info("Analyzing duplicates...")
    stats_before = analyze_duplicates(df)
    
    logger.info(f"\nDuplicate Analysis:")
    logger.info(f"  Total rows: {stats_before['total_rows']:,}")
    logger.info(f"  Unique peptide-receptor pairs: {stats_before['unique_pairs']:,}")
    logger.info(f"  Duplicate rows to remove: {stats_before['duplicate_rows']:,}")
    logger.info(f"  Duplication rate: {stats_before['duplicate_rows']/stats_before['total_rows']*100:.1f}%")
    
    logger.info(f"\nDuplication patterns:")
    logger.info(f"  Pairs appearing only once: {stats_before['pairs_appearing_once']:,}")
    logger.info(f"  Pairs appearing multiple times: {stats_before['pairs_appearing_multiple']:,}")
    logger.info(f"  Maximum appearances of a single pair: {stats_before['max_appearances']}")
    logger.info(f"  Duplicates from different PDB structures: {stats_before['duplicates_from_different_pdbs']:,}")
    logger.info(f"  Duplicates from same PDB (different chains): {stats_before['duplicates_from_same_pdb']:,}")
    
    logger.info("\nTop 10 most duplicated pairs:")
    for i, dup in enumerate(stats_before['top_duplicates'], 1):
        logger.info(f"\n  {i}. Appears {dup['occurrences']} times in {dup['unique_pdbs']} PDB structures")
        logger.info(f"     Peptide: {dup['peptide_preview']}")
        logger.info(f"     Receptor: {dup['receptor_preview']}")
        logger.info(f"     PDB examples: {', '.join(dup['pdb_examples'])}")
    
    # Perform deduplication
    logger.info("\n" + "=" * 50)
    logger.info("Performing deduplication...")
    
    # Sort for deterministic dedup regardless of input order
    df = df.sort_values(by=[PEPTIDE_COL, RECEPTOR_COL, PDB_COL]).reset_index(drop=True)

    # Keep first occurrence of each peptide-receptor pair
    df_dedup = df.drop_duplicates(subset=[PEPTIDE_COL, RECEPTOR_COL], keep='first').copy()
    
    # Calculate statistics
    rows_removed = len(df) - len(df_dedup)
    logger.info(f"\nDeduplication complete:")
    logger.info(f"  Original rows: {len(df):,}")
    logger.info(f"  Deduplicated rows: {len(df_dedup):,}")
    logger.info(f"  Rows removed: {rows_removed:,} ({rows_removed/len(df)*100:.1f}%)")
    
    # Additional statistics
    logger.info(f"\nUnique counts after deduplication:")
    logger.info(f"  Unique peptides: {df_dedup[PEPTIDE_COL].nunique():,}")
    logger.info(f"  Unique receptors: {df_dedup[RECEPTOR_COL].nunique():,}")
    logger.info(f"  Unique PDB structures retained: {df_dedup[PDB_COL].nunique():,}")
    
    # Save deduplicated data
    logger.info(f"\nSaving deduplicated data to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(output_path, index=False)
    
    # Verify saved file
    df_verify = pd.read_csv(output_path)
    logger.info(f"Verification: Saved file contains {len(df_verify):,} rows")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("DEDUPLICATION SUCCESSFUL")
    logger.info("=" * 70)
    logger.info(f"✓ Removed {rows_removed:,} duplicate peptide-receptor pairs")
    logger.info(f"✓ Retained {len(df_dedup):,} unique molecular interactions")
    logger.info(f"✓ Preserved first occurrence with original PDB information")
    logger.info(f"✓ Output saved to: {output_file}")
    logger.info("=" * 70)
    
    return df_dedup


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Deduplicate Propedia v2 peptide-receptor pairs")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to Propedia_v2_with_HELM_SMILES.csv")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path for unique pairs")
    args = parser.parse_args()

    try:
        deduplicate_peptide_receptor_pairs(args.input, args.output)
    except Exception as e:
        logging.error(f"Deduplication failed: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
