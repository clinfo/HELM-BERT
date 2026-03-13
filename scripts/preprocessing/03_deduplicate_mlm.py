#!/usr/bin/env python3
"""Simple dataset deduplication for MLM pretraining using canonical SMILES."""

import pandas as pd
import logging
import sys
from pathlib import Path
from rdkit import Chem

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
INPUT_FILES = {
    'cycpeptmpdb': ['data/CycPeptMPDB_Peptide_All_V1.2.csv'],
    'propedia': ['local_data/intermediate/propedia_helm.csv'],
    'chembl': ['data/chembl_v35_with_helm_notation.csv']
}

# SMILES column names for each dataset
SMILES_COLUMNS = {
    'cycpeptmpdb': 'SMILES',
    'propedia': 'Peptide_SMILES',
    'chembl': 'canonical_smiles'
}

# Output directory
OUTPUT_DIR = 'local_data/deduplicated'

# Priority order (leftmost preserved in cross-dataset deduplication)
PRIORITY_ORDER = ['cycpeptmpdb', 'propedia', 'chembl']

# =============================================================================

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime

# Create output directory for logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("outputs") / "preprocessing" / f"deduplication_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging with file handler
log_file = log_dir / "deduplication.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")


def smiles_to_canonical_smiles(smiles: str) -> str:
    """Convert SMILES to canonical SMILES."""
    if not smiles or pd.isna(smiles) or smiles == '':
        return ''
    
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            logger.warning(f"Invalid SMILES string: {smiles}")
            return ''
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.warning(f"Error converting SMILES '{smiles}' to canonical form: {e}")
        return ''


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load dataset from available file paths."""
    for file_path in INPUT_FILES[dataset_name]:
        if Path(file_path).exists():
            logger.info(f"Loading {dataset_name} from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df):,} rows")
            return df
    
    logger.warning(f"No file found for {dataset_name}")
    return pd.DataFrame()


def deduplicate_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Remove duplicates within dataset using canonical SMILES."""
    if df.empty:
        return df
    
    smiles_col = SMILES_COLUMNS[dataset_name]
    logger.info(f"Deduplicating {dataset_name}...")
    
    # Check if SMILES column exists
    if smiles_col not in df.columns:
        logger.warning(f"SMILES column '{smiles_col}' not found in {dataset_name}")
        return pd.DataFrame()
    
    # Log initial dataset size
    logger.info(f"  Original rows: {len(df):,}")
    
    # Filter rows with valid SMILES
    logger.info(f"  Filtering valid SMILES...")
    valid_mask = df[smiles_col].notna() & (df[smiles_col] != '')
    df_valid = df[valid_mask].copy()
    logger.info(f"  After valid SMILES filter: {len(df_valid):,}")
    
    # Convert to canonical SMILES
    logger.info(f"  Converting to canonical SMILES...")
    canonical_smiles = df_valid[smiles_col].apply(smiles_to_canonical_smiles)

    failed_conversion = (canonical_smiles == '').sum()
    success_rate = ((len(canonical_smiles) - failed_conversion) / len(canonical_smiles) * 100) if len(canonical_smiles) > 0 else 0
    logger.info(f"  SMILES conversion failures: {failed_conversion:,} ({100-success_rate:.1f}%)")
    
    df_valid = df_valid[canonical_smiles != ''].copy()
    df_valid['_temp_canonical'] = canonical_smiles[canonical_smiles != '']
    logger.info(f"  After conversion filter: {len(df_valid):,}")
    
    # Sort before deduplication for deterministic results
    sort_cols = [col for col in ['PDB', 'Peptide_Chain', 'Receptor_Chain', 'chembl_id', 'ID'] if col in df_valid.columns]
    if sort_cols:
        df_valid = df_valid.sort_values(sort_cols)

    # Remove duplicates
    logger.info(f"  Removing duplicates...")
    df_dedup = df_valid.drop_duplicates(subset=['_temp_canonical']).drop(columns=['_temp_canonical'])
    duplicates = len(df_valid) - len(df_dedup)
    duplicate_rate = (duplicates / len(df_valid) * 100) if len(df_valid) > 0 else 0
    logger.info(f"  Duplicates removed: {duplicates:,} ({duplicate_rate:.1f}%)")
    logger.info(f"  Final deduplicated rows: {len(df_dedup):,}")
    
    return df_dedup


def main():
    """Main deduplication process for MLM pretraining."""
    logger.info("Starting deduplication with full cross-dataset deduplication...")
    logger.info(f"Log directory: {log_dir}")
    
    # Step 1: Load and deduplicate each dataset individually
    logger.info(f"\n{'='*50}")
    logger.info("Step 1: Loading and deduplicating individual datasets...")
    logger.info(f"{'='*50}")
    
    datasets = {}
    dedup_stats = {}
    
    # Process all datasets in priority order
    for dataset_name in PRIORITY_ORDER:
        logger.info(f"\nProcessing {dataset_name}...")
        df = load_dataset(dataset_name)
        
        if df.empty:
            logger.warning(f"Failed to load {dataset_name}, skipping...")
            continue
            
        original_size = len(df)
        df_dedup = deduplicate_dataset(df, dataset_name)
        datasets[dataset_name] = df_dedup
        
        reduction = ((original_size - len(df_dedup)) / original_size * 100) if original_size > 0 else 0
        dedup_stats[dataset_name] = {
            'original': original_size,
            'after_self_dedup': len(df_dedup),
            'self_reduction': reduction
        }
        logger.info(f"{dataset_name}: {original_size:,} → {len(df_dedup):,} (-{reduction:.1f}%)")
    
    # Step 2: Cross-dataset deduplication
    logger.info(f"\n{'='*50}")
    logger.info("Step 2: Cross-dataset deduplication...")
    logger.info(f"Priority order: {' > '.join(PRIORITY_ORDER)}")
    logger.info(f"{'='*50}")
    
    seen_smiles = set()
    
    # Process each dataset according to priority
    for i, dataset_name in enumerate(PRIORITY_ORDER):
        if dataset_name not in datasets:
            continue
            
        df = datasets[dataset_name]
        smiles_col = SMILES_COLUMNS[dataset_name]
        before_cross = len(df)
        
        logger.info(f"\nProcessing {dataset_name} for cross-dataset duplicates...")
        
        # Remove duplicates from higher priority datasets
        if i > 0:  # Not the first dataset
            smiles_list = df[smiles_col].apply(smiles_to_canonical_smiles)
            keep_mask = [smiles not in seen_smiles and smiles != '' for smiles in smiles_list]
            df = df[keep_mask].copy()
            datasets[dataset_name] = df
            
            cross_removed = before_cross - len(df)
            cross_rate = (cross_removed / before_cross * 100) if before_cross > 0 else 0
            logger.info(f"  Removed {cross_removed:,} duplicates from higher priority datasets (-{cross_rate:.1f}%)")
            
            dedup_stats[dataset_name]['after_cross_dedup'] = len(df)
            dedup_stats[dataset_name]['cross_removed'] = cross_removed
        else:
            dedup_stats[dataset_name]['after_cross_dedup'] = len(df)
            dedup_stats[dataset_name]['cross_removed'] = 0
            logger.info(f"  No cross-dataset deduplication needed (highest priority)")
        
        # Add current dataset's SMILES to seen set
        current_smiles = df[smiles_col].apply(smiles_to_canonical_smiles)
        valid_smiles = current_smiles[current_smiles != '']
        seen_smiles.update(valid_smiles)
        logger.info(f"  Added {len(valid_smiles):,} unique SMILES to seen set")
        logger.info(f"  {dataset_name}: {before_cross:,} → {len(df):,} rows")
    
    # Step 3: Save all deduplicated datasets
    logger.info(f"\n{'='*50}")
    logger.info("Step 3: Saving all deduplicated datasets...")
    logger.info(f"{'='*50}")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, df in datasets.items():
        output_file = output_path / f"{dataset_name}_deduplicated.csv"
        
        logger.info(f"Saving {dataset_name}...")
        df.to_csv(output_file, index=False)
        logger.info(f"  Saved {len(df):,} rows to {output_file}")
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("DEDUPLICATION COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info("SUMMARY:")
    
    total_original = sum(stats['original'] for stats in dedup_stats.values())
    total_after_self = sum(stats['after_self_dedup'] for stats in dedup_stats.values())
    total_after_cross = sum(stats['after_cross_dedup'] for stats in dedup_stats.values())
    
    logger.info(f"\nDataset-wise statistics:")
    for dataset_name in PRIORITY_ORDER:
        if dataset_name in dedup_stats:
            stats = dedup_stats[dataset_name]
            logger.info(f"\n{dataset_name.upper()}:")
            logger.info(f"  Original: {stats['original']:,} rows")
            logger.info(f"  After self-dedup: {stats['after_self_dedup']:,} rows (-{stats['self_reduction']:.1f}%)")
            logger.info(f"  After cross-dedup: {stats['after_cross_dedup']:,} rows (-{stats['cross_removed']:,} from higher priority)")
            total_reduction = ((stats['original'] - stats['after_cross_dedup']) / stats['original'] * 100) if stats['original'] > 0 else 0
            logger.info(f"  Total reduction: {stats['original'] - stats['after_cross_dedup']:,} rows ({total_reduction:.1f}%)")
    
    logger.info(f"\nTOTAL:")
    logger.info(f"  Original: {total_original:,} rows")
    logger.info(f"  After self-dedup: {total_after_self:,} rows")
    logger.info(f"  After cross-dedup: {total_after_cross:,} rows")
    logger.info(f"  Total reduction: {total_original - total_after_cross:,} rows ({(total_original - total_after_cross) / total_original * 100:.1f}%)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()