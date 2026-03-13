#!/usr/bin/env python3
"""
Process Propedia data to extract ALL positive PPI samples to CSV format.

This script:
1. Reads PDB complex files to identify positive protein-peptide interactions
2. Matches complex files with their corresponding sequence FASTA files
3. Extracts all sequences without any filtering
4. Creates a consolidated CSV file with all positive samples
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from glob import glob
import re
from dataclasses import dataclass
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime

# Create output directory for logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("outputs") / "preprocessing" / f"propedia_processing_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging with file handler
log_file = log_dir / "processing.log"
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


@dataclass
class PDBComplexInfo:
    """Store information about a PDB complex file."""
    pdb_id: str
    peptide_chain: str
    receptor_chain: str
    filename: str
    
    @classmethod
    def from_filename(cls, filename: str) -> Optional['PDBComplexInfo']:
        """
        Parse PDB filename to extract complex information.
        
        Expected format: {PDB_ID}_{PEPTIDE_CHAIN}_{RECEPTOR_CHAIN}.pdb
        Example: 1a07_C_A.pdb -> PDB: 1a07, Peptide: C, Receptor: A
        """
        if not filename.endswith('.pdb'):
            return None
            
        # Remove .pdb extension
        base = filename[:-4]
        
        # Split by underscore
        parts = base.split('_')
        
        if len(parts) == 3:
            return cls(
                pdb_id=parts[0].lower(),
                peptide_chain=parts[1],
                receptor_chain=parts[2],
                filename=filename
            )
        else:
            logger.warning(f"Unexpected PDB filename format: {filename}")
            return None


class FastaReader:
    """Handle FASTA file reading operations."""
    
    @staticmethod
    def read_sequence(filepath: Path) -> Optional[str]:
        """
        Read a single sequence from a FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            Sequence string or None if file not found or empty
        """
        if not filepath.exists():
            return None
            
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Skip empty files
            if len(lines) < 2:
                return None
                
            # Skip header line(s), join sequence lines
            sequence_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('>'):
                    sequence_lines.append(line)
                    
            sequence = ''.join(sequence_lines)
            return sequence if sequence else None
            
        except Exception as e:
            logger.error(f"Error reading FASTA file {filepath}: {e}")
            return None




class PropediaProcessor:
    """Main processor for Propedia data."""

    def __init__(self, base_path: str):
        """
        Initialize processor with base path.

        Args:
            base_path: Base path to propedia directory
        """
        self.base_path = Path(base_path)
        self.complex_path = self.base_path / "complex2_3" / "complex"
        self.peptide_path = self.base_path / "sequences2_3" / "peptide"
        self.receptor_path = self.base_path / "sequences2_3" / "receptor"
        
        # Validate paths
        self._validate_paths()
        
        # Initialize components
        self.fasta_reader = FastaReader()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _validate_paths(self):
        """Validate that all required directories exist."""
        paths_to_check = [
            (self.complex_path, "Complex"),
            (self.peptide_path, "Peptide"),
            (self.receptor_path, "Receptor")
        ]
        
        for path, name in paths_to_check:
            if not path.exists():
                raise ValueError(f"{name} path not found: {path}")
            logger.info(f"{name} path found: {path}")
    
    def get_complex_files(self) -> List[PDBComplexInfo]:
        """
        Get all PDB complex files and parse their information.
        
        Returns:
            List of PDBComplexInfo objects
        """
        pdb_files = sorted(self.complex_path.glob("*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files in complex directory")
        
        complex_infos = []
        for pdb_file in pdb_files:
            info = PDBComplexInfo.from_filename(pdb_file.name)
            if info:
                complex_infos.append(info)
            else:
                self.stats['invalid_filenames'] += 1
                
        logger.info(f"Successfully parsed {len(complex_infos)} PDB filenames")
        return complex_infos
    
    def process_complex(self, complex_info: PDBComplexInfo) -> Optional[Dict]:
        """
        Process a single PDB complex to extract sequences.
        
        Args:
            complex_info: PDBComplexInfo object
            
        Returns:
            Dictionary with sample data or None if processing fails
        """
        # Construct FASTA file paths
        peptide_fasta = self.peptide_path / f"{complex_info.pdb_id}_{complex_info.peptide_chain}.fasta"
        receptor_fasta = self.receptor_path / f"{complex_info.pdb_id}_{complex_info.receptor_chain}.fasta"
        
        # Read sequences
        peptide_seq = self.fasta_reader.read_sequence(peptide_fasta)
        receptor_seq = self.fasta_reader.read_sequence(receptor_fasta)
        
        # Check if both sequences exist
        if not peptide_seq:
            self.stats['missing_peptide_seq'] += 1
            return None
            
        if not receptor_seq:
            self.stats['missing_receptor_seq'] += 1
            return None
        
        # Create sample dictionary (NO FILTERING, NO CONVERSION)
        sample = {
            'PDB': complex_info.pdb_id.upper(),
            'Peptide_Chain': complex_info.peptide_chain,
            'Receptor_Chain': complex_info.receptor_chain,
            'Peptide_Sequence': peptide_seq,
            'Receptor_Sequence': receptor_seq,
            'Peptide_Length': len(peptide_seq),
            'Receptor_Length': len(receptor_seq),
            'Complex_File': complex_info.filename,
            'Label': 1  # Positive sample
        }
        
        self.stats['valid_samples'] += 1
        return sample
    
    def process_all(self) -> pd.DataFrame:
        """
        Process all PDB complexes and create DataFrame.
        
        Returns:
            DataFrame with all positive samples
        """
        # Get all complex files
        complex_infos = self.get_complex_files()
        
        # Process each complex
        positive_samples = []
        
        logger.info("Processing PDB complexes...")
        for idx, complex_info in enumerate(complex_infos):
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Progress: {idx}/{len(complex_infos)} complexes processed")
                
            sample = self.process_complex(complex_info)
            if sample:
                positive_samples.append(sample)
        
        logger.info(f"Completed processing {len(complex_infos)} complexes")
        logger.info(f"Generated {len(positive_samples)} valid positive samples")
        
        # Create DataFrame
        df = pd.DataFrame(positive_samples)
        
        # Sort by PDB ID, Peptide_Chain, Receptor_Chain for consistency
        if not df.empty:
            df = df.sort_values(['PDB', 'Peptide_Chain', 'Receptor_Chain'])
        
        return df
    
    def print_statistics(self, df: pd.DataFrame):
        """Print processing statistics."""
        logger.info("="*60)
        logger.info("Processing Statistics:")
        logger.info("="*60)
        logger.info(f"Total PDB files found: {len(df) + self.stats['invalid_filenames'] + self.stats['missing_peptide_seq'] + self.stats['missing_receptor_seq']}")
        logger.info(f"Invalid filenames: {self.stats['invalid_filenames']}")
        logger.info(f"Missing peptide sequences: {self.stats['missing_peptide_seq']}")
        logger.info(f"Missing receptor sequences: {self.stats['missing_receptor_seq']}")
        logger.info(f"Total positive samples: {len(df)}")
        
        if not df.empty:
            logger.info("\nDataset Statistics:")
            logger.info(f"Unique PDB IDs: {df['PDB'].nunique()}")
            logger.info(f"Average peptide length: {df['Peptide_Length'].mean():.1f} ± {df['Peptide_Length'].std():.1f}")
            logger.info(f"Peptide length range: {df['Peptide_Length'].min()}-{df['Peptide_Length'].max()}")
            logger.info(f"Average receptor length: {df['Receptor_Length'].mean():.1f} ± {df['Receptor_Length'].std():.1f}")
            logger.info(f"Receptor length range: {df['Receptor_Length'].min()}-{df['Receptor_Length'].max()}")
            
            # Count sequences with special characters
            peptides_with_x = df['Peptide_Sequence'].str.contains('X', case=False).sum()
            logger.info(f"Peptides containing 'X': {peptides_with_x}")
        
        logger.info("="*60)


def main():
    """Main function to run the processing pipeline."""
    # Configuration (repo-relative defaults)
    repo_root = Path(__file__).resolve().parents[2]
    base_path = str(repo_root / "local_data/propedia")
    output_file = str(repo_root / "local_data/intermediate/propedia.csv")

    logger.info("Starting Propedia positive sample extraction (NO FILTERING)...")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"{'='*60}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"{'='*60}")
    
    try:
        # Initialize processor
        processor = PropediaProcessor(base_path)
        
        # Process all data
        df = processor.process_all()
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {output_file}")
        df.to_csv(output_file, index=False)
        
        # Print statistics
        processor.print_statistics(df)
        
        # Show sample data
        if not df.empty:
            logger.info("\nFirst 5 entries:")
            logger.info(df[['PDB', 'Peptide_Chain', 'Peptide_Length', 'Peptide_Sequence']].head().to_string())
        
        logger.info(f"{'='*60}")
        logger.info("Processing completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
