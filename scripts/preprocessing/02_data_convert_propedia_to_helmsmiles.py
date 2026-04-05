#!/usr/bin/env python3
"""Convert Propedia v2 peptide sequences to HELM notation using RDKit."""

import pandas as pd
import logging
from pathlib import Path
from rdkit import Chem
import re
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime

# Create output directory for logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("outputs") / "preprocessing" / f"propedia_v2_helm_conversion_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging with file handler
log_file = log_dir / "conversion.log"
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


def peptide_to_helm(peptide_sequence: str) -> str:
    """Convert peptide sequence to HELM notation using RDKit.
    
    Args:
        peptide_sequence: Single-letter amino acid sequence
        
    Returns:
        HELM notation string
    """
    if not peptide_sequence or peptide_sequence == '-' or 'X' in peptide_sequence:
        return ''
    
    # Clean sequence - remove non-standard characters
    clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', peptide_sequence.upper())
    
    if not clean_sequence:
        return ''
    
    try:
        # Create molecule from peptide sequence using RDKit
        mol = Chem.MolFromSequence(clean_sequence)
        
        if mol is None:
            logger.warning(f"RDKit could not process sequence: {peptide_sequence}")
            return ''
        
        # Convert to HELM notation
        # RDKit standard amino acid codes map directly to HELM
        aa_codes = list(clean_sequence)
        
        # HELM notation format: PEPTIDE1{AA1.AA2.AA3...}$$$$
        helm_notation = f"PEPTIDE1{{{'.'.join(aa_codes)}}}$$$$"
        return helm_notation
        
    except Exception as e:
        logger.error(f"Error converting peptide '{peptide_sequence}' to HELM: {e}")
        return ''


def peptide_to_smiles(peptide_sequence: str) -> str:
    """Convert peptide sequence to SMILES notation using RDKit.
    
    Args:
        peptide_sequence: Single-letter amino acid sequence
        
    Returns:
        SMILES string
    """
    if not peptide_sequence or peptide_sequence == '-' or 'X' in peptide_sequence:
        return ''
    
    # Clean sequence - remove non-standard characters
    clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', peptide_sequence.upper())
    
    if not clean_sequence:
        return ''
    
    try:
        # Create molecule from peptide sequence using RDKit
        mol = Chem.MolFromSequence(clean_sequence)
        
        if mol is None:
            logger.warning(f"RDKit could not process sequence: {peptide_sequence}")
            return ''
        
        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol)
        return smiles
        
    except Exception as e:
        logger.error(f"Error converting peptide '{peptide_sequence}' to SMILES: {e}")
        return ''


def process_propedia_v2_dataset(input_file: str, output_file: str):
    """Process Propedia v2 dataset and add HELM notation column.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with HELM column
    """
    logger.info(f"Reading dataset from: {input_file}")
    
    # Read the dataset
    df = pd.read_csv(input_file)
    logger.info(f"Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")
    
    # Check if Peptide_Sequence column exists
    if 'Peptide_Sequence' not in df.columns:
        raise ValueError("'Peptide_Sequence' column not found in dataset")
    
    logger.info("Filtering out rows with invalid peptide sequences...")
    
    # Filter out rows with X or non-standard characters
    initial_count = len(df)
    
    # Remove rows where peptide sequence contains X or non-standard amino acids
    logger.info("  Checking for valid amino acid sequences...")
    valid_mask = df['Peptide_Sequence'].apply(lambda x: 
        isinstance(x, str) and 
        x != '-' and 
        'X' not in x.upper() and 
        bool(re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', x.upper()))
    )
    
    df_filtered = df[valid_mask].copy()
    removed_count = initial_count - len(df_filtered)
    removal_rate = (removed_count / initial_count * 100) if initial_count > 0 else 0
    
    logger.info(f"  Removed {removed_count:,} rows with invalid peptide sequences ({removal_rate:.1f}%)")
    logger.info(f"  Remaining {len(df_filtered):,} rows with valid sequences")
    
    if len(df_filtered) == 0:
        logger.warning("No valid peptide sequences found!")
        return
    
    logger.info("Converting peptide sequences to HELM notation and SMILES...")
    
    # Convert peptide sequences to HELM notation and SMILES
    helm_sequences = []
    smiles_sequences = []
    failed_helm = 0
    failed_smiles = 0
    
    total_seqs = len(df_filtered)
    logger.info(f"  Processing {total_seqs:,} peptide sequences...")
    
    for idx, (_, row) in enumerate(df_filtered.iterrows(), 1):
        peptide_seq = str(row['Peptide_Sequence'])
        helm_notation = peptide_to_helm(peptide_seq)
        smiles_notation = peptide_to_smiles(peptide_seq)
        
        if not helm_notation:
            failed_helm += 1
        if not smiles_notation:
            failed_smiles += 1
        
        helm_sequences.append(helm_notation)
        smiles_sequences.append(smiles_notation)
        
        # Log progress every 1000 sequences
        if idx % 1000 == 0 or idx == total_seqs:
            logger.info(f"  Progress: {idx:,}/{total_seqs:,} sequences processed ({idx/total_seqs*100:.1f}%)")
    
    # Report conversion success rates
    total_converted = len(df_filtered)
    helm_success = ((total_converted - failed_helm) / total_converted * 100) if total_converted > 0 else 0
    smiles_success = ((total_converted - failed_smiles) / total_converted * 100) if total_converted > 0 else 0
    
    logger.info(f"  HELM conversion: {total_converted - failed_helm:,}/{total_converted:,} successful ({helm_success:.1f}%)")
    logger.info(f"  SMILES conversion: {total_converted - failed_smiles:,}/{total_converted:,} successful ({smiles_success:.1f}%)")
    
    if failed_helm > 0:
        logger.warning(f"  {failed_helm:,} HELM conversions failed")
    if failed_smiles > 0:
        logger.warning(f"  {failed_smiles:,} SMILES conversions failed")
    
    # Add HELM and SMILES columns next to Peptide_Sequence
    peptide_seq_idx = df_filtered.columns.get_loc('Peptide_Sequence')
    
    # Ensure we have an integer index
    if not isinstance(peptide_seq_idx, int):
        peptide_seq_idx = 0  # Fallback to beginning if not found
    
    # Insert the new columns right after Peptide_Sequence
    logger.info("Adding new columns to dataset...")
    df_filtered.insert(peptide_seq_idx + 1, 'Peptide_HELM', helm_sequences)
    df_filtered.insert(peptide_seq_idx + 2, 'Peptide_SMILES', smiles_sequences)
    
    logger.info(f"Successfully processed {len(df_filtered):,} peptide sequences")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed dataset
    logger.info(f"Saving processed dataset to: {output_file}")
    df_filtered.to_csv(output_file, index=False)
    
    logger.info(f"Final dataset saved with {len(df_filtered):,} rows and {len(df_filtered.columns)} columns")
    logger.info("Added columns: 'Peptide_HELM' and 'Peptide_SMILES'")
    logger.info("Rows with X or non-standard amino acids have been removed")


def main():
    """Main function."""
    # Define file paths
    input_file = "local_data/intermediate_product/Propedia_v2.csv"
    output_file = "local_data/intermediate_product/Propedia_v2_with_HELM_SMILES.csv"
    
    logger.info("Starting Propedia v2 peptide-to-HELM and SMILES conversion...")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"{'='*60}")
    logger.info(f"Input:  {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"{'='*60}")
    
    try:
        process_propedia_v2_dataset(input_file, output_file)
        logger.info(f"{'='*60}")
        logger.info("Conversion completed successfully!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()