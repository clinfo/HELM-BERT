#!/usr/bin/env python3
"""Generate aCSM-ALL signatures for Propedia v2 complexes.

This script generates aCSM-ALL molecular signatures for entire peptide-protein complexes
from Propedia v2. These signatures capture the structural properties of the full complex,
including interaction interfaces, and are used for clustering-based k-fold splits.

Requirements:
- Signa library (included in src/utils/signa/)
- Complex PDB files in local_data/propedia_v2/complexes/
- Propedia v2 deduplicated data file

Output:
- Complex signatures: local_data/propedia_v2/signatures_acsm_all/complex_signatures_acsm_all.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import contextlib
import io
import argparse

# Safe tqdm import
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Resolve repo root and add Signa path
REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNA_PATH_DEFAULT = REPO_ROOT / 'src/utils/signa/Signa'
sys.path.append(str(SIGNA_PATH_DEFAULT))
import signa

# Configuration (repo-relative defaults; CLI-overridable)
DEFAULT_SOURCE_FILE = REPO_ROOT / 'local_data/intermediate_product/Propedia_v2_with_HELM_SMILES.csv'
DEFAULT_COMPLEX_DIR = REPO_ROOT / 'local_data/raw/propedia_v2/complex2_3/complex'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'local_data/intermediate_product/signatures_acsm_all'
DEFAULT_LOG_DIR = REPO_ROOT / 'outputs/preprocessing'

# Signa configuration
SIGNA_TYPE = 'acsm-all'
CUTOFF_LIMIT = 20.0
CUTOFF_STEP = 0.2
SIGNATURE_DIM = int(CUTOFF_LIMIT / CUTOFF_STEP) * 36  # 3600 features

# Processing configuration
BATCH_SIZE = 100  # Number of complexes to process in a batch

# Column names
PDB_COL = 'PDB'
PEPTIDE_CHAIN_COL = 'Peptide_Chain'
PROTEIN_CHAIN_COL = 'Receptor_Chain'
COMPLEX_FILE_COL = 'Complex_File'

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging() -> Tuple[logging.Logger, Path]:
    """Set up logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(DEFAULT_LOG_DIR)
    log_dir = base / f"acsm_signature_generation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "signature_generation.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    print(f"Log file: {log_file.absolute()}")
    
    return logger, log_dir


logger, log_dir = (logging.getLogger(__name__), Path('.'))


def check_signa_installation() -> bool:
    """Check if Signa is properly imported and accessible."""
    try:
        # Test if signa module is loaded
        hasattr(signa, 'read')
        return True
    except:
        return False


def generate_complex_signature(pdb_file: Path) -> Optional[np.ndarray]:
    """Generate aCSM-ALL signature for entire complex using Signa.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Signature array or None if failed
    """
    try:
        # Suppress signa's print statements
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # Generate signature using Signa for entire complex (no chain specification)
            signature = signa.read(
                str(pdb_file),
                signa_type=SIGNA_TYPE,
                cutoff_limit=CUTOFF_LIMIT,
                cutoff_step=CUTOFF_STEP,
                output_csv=False,
                chain='ALL',  # Process entire complex
                verbose=False
            )
        
        if signature is None:
            logger.error(f"No signature generated for {pdb_file.name}")
            return None
        
        # Convert to numpy array
        if isinstance(signature, list):
            signature_array = np.array(signature)
        else:
            signature_array = signature
        
        # Ensure correct length
        if len(signature_array) != SIGNATURE_DIM:
            logger.warning(f"Unexpected signature length for {pdb_file.name}: "
                         f"{len(signature_array)} (expected {SIGNATURE_DIM})")
            # Pad or truncate as needed
            if len(signature_array) < SIGNATURE_DIM:
                padded_array = np.zeros(SIGNATURE_DIM, dtype=signature_array.dtype)
                padded_array[:len(signature_array)] = signature_array
                signature_array = padded_array
            else:
                signature_array = signature_array[:SIGNATURE_DIM]
        
        return signature_array
        
    except Exception as e:
        logger.error(f"Error generating signature for {pdb_file.name}: {type(e).__name__}: {str(e)}")
        return None


def process_complex(row: pd.Series, complex_dir: Path) -> Optional[Dict]:
    """Process a single complex and extract signature for the entire complex.
    
    Args:
        row: DataFrame row with complex information
        complex_dir: Directory containing complex PDB files
        
    Returns:
        Dictionary with complex information and signature, or None if failed
    """
    pdb_id = row[PDB_COL]
    complex_file = row[COMPLEX_FILE_COL]
    
    # Construct PDB file path
    pdb_file = complex_dir / complex_file
    
    if not pdb_file.exists():
        logger.warning(f"Complex file not found: {pdb_file}")
        return None
    
    # Generate signature for entire complex
    complex_sig = generate_complex_signature(pdb_file)
    
    if complex_sig is not None:
        return {
            'pdb_id': pdb_id,
            'peptide_chain': row[PEPTIDE_CHAIN_COL],
            'protein_chain': row[PROTEIN_CHAIN_COL],
            'complex_file': complex_file,
            'signature': complex_sig
        }
    
    return None


def process_batch(batch_df: pd.DataFrame, complex_dir: Path) -> Tuple[List[Dict], List[str]]:
    """Process a batch of complexes.
    
    Args:
        batch_df: DataFrame with batch of complexes
        complex_dir: Directory containing complex PDB files
        
    Returns:
        Tuple of (list of complex signature results, list of failed PDB IDs)
    """
    results = []
    failed_pdbs = []
    
    for idx, row in batch_df.iterrows():
        result = process_complex(row, complex_dir)
        
        if result is not None:
            results.append(result)
        else:
            failed_pdbs.append(row[PDB_COL])
    
    return results, failed_pdbs


def save_signatures(signatures: List[Dict], output_file: Path):
    """Save signatures to CSV file.
    
    Args:
        signatures: List of signature dictionaries
        output_file: Output CSV file path
    """
    if not signatures:
        logger.warning(f"No signatures to save for {output_file}")
        return
    
    # Convert to DataFrame
    rows = []
    for sig_dict in signatures:
        row = {
            'pdb_id': sig_dict['pdb_id'],
            'peptide_chain': sig_dict['peptide_chain'],
            'protein_chain': sig_dict['protein_chain'],
            'complex_file': sig_dict['complex_file']
        }
        # Add signature values
        for i, val in enumerate(sig_dict['signature']):
            row[f'sig_{i}'] = val
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} signatures to {output_file}")


def main():
    """Main function to generate aCSM-ALL signatures."""
    global DEFAULT_LOG_DIR
    parser = argparse.ArgumentParser(description='Generate aCSM-ALL signatures for complexes')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument('--complex-dir', type=str, default=str(DEFAULT_COMPLEX_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--signa-path', type=str, default=str(SIGNA_PATH_DEFAULT))
    args = parser.parse_args()

    # Adjust signa path if overridden
    if args.signa_path and args.signa_path not in sys.path:
        sys.path.insert(0, args.signa_path)
    
    # Configure logging
    DEFAULT_LOG_DIR = Path(args.log_dir)
    global logger, log_dir
    logger, log_dir = setup_logging()
    logger.info("=" * 60)
    logger.info("aCSM-ALL Signature Generation for Propedia v2")
    logger.info("=" * 60)
    
    # Check Signa installation
    if not check_signa_installation():
        logger.error("Signa not found! Please check the path to Signa library")
        logger.error(f"Expected path: {SIGNA_PATH_DEFAULT}")
        return
    
    logger.info("Signa library loaded successfully")
    
    # Load data
    logger.info(f"Loading data from: {args.source}")
    df = pd.read_csv(args.source)
    logger.info(f"Loaded {len(df)} entries")
    
    # Get unique complexes (sorted for deterministic processing order)
    unique_complexes = (df[[PDB_COL, PEPTIDE_CHAIN_COL, PROTEIN_CHAIN_COL,
                           COMPLEX_FILE_COL]]
                        .sort_values(by=[PDB_COL, PEPTIDE_CHAIN_COL, PROTEIN_CHAIN_COL])
                        .drop_duplicates()
                        .reset_index(drop=True))
    logger.info(f"Found {len(unique_complexes)} unique complexes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process complexes in batches
    all_complex_sigs = []
    all_failed_pdbs = []
    
    complex_dir = Path(args.complex_dir)
    
    # Process with progress bar
    n_batches = (len(unique_complexes) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=len(unique_complexes), desc="Processing complexes") as pbar:
        for i in range(0, len(unique_complexes), BATCH_SIZE):
            batch_df = unique_complexes.iloc[i:i+BATCH_SIZE]
            
            # Process batch
            results, failed_pdbs = process_batch(batch_df, complex_dir)
            
            # Collect results
            all_complex_sigs.extend(results)
            all_failed_pdbs.extend(failed_pdbs)
            
            # Log failures immediately
            if failed_pdbs:
                logger.warning(f"Failed to generate signatures for {len(failed_pdbs)} complexes in this batch: {failed_pdbs[:5]}..." if len(failed_pdbs) > 5 else f"Failed to generate signatures for {len(failed_pdbs)} complexes: {failed_pdbs}")
            
            pbar.update(len(batch_df))
    
    # Save results
    logger.info("Saving signatures...")
    
    complex_output = output_dir / 'complex_signatures_acsm_all.csv'
    
    save_signatures(all_complex_sigs, complex_output)
    
    # Save list of failed PDBs
    if all_failed_pdbs:
        failed_output = output_dir / 'failed_signatures.txt'
        with open(failed_output, 'w') as f:
            f.write("\n".join(sorted(set(all_failed_pdbs))))
        logger.info(f"Saved list of {len(all_failed_pdbs)} failed PDBs to {failed_output}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"Processed complexes: {len(unique_complexes)}")
    logger.info(f"Complex signatures generated: {len(all_complex_sigs)}")
    logger.info(f"Failed signatures: {len(all_failed_pdbs)}")
    if all_failed_pdbs:
        logger.info(f"Failed PDB IDs (first 10): {sorted(set(all_failed_pdbs))[:10]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)
