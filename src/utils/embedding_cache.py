"""Simple embedding cache for frozen encoders with drug/target role separation."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple cache for pre-computed embeddings with role-based organization.

    Cache structure:
        embeddings/{encoder}/{dataset}/{role}/

    Example:
        embeddings/helmbert/propedia_ppi_random/drugs/abc123.pt
        embeddings/esm2/propedia_ppi_random/targets/def456.pt
    """

    def __init__(self, cache_dir: Path):
        """Initialize embedding cache.

        Args:
            cache_dir: Base directory for cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_sequence_hash(sequence: str) -> str:
        """Get MD5 hash of sequence for caching."""
        return hashlib.md5(sequence.encode()).hexdigest()

    def get_cache_path(
        self,
        encoder_name: str,
        dataset_type: str,
        model_version: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Path:
        """Get path for encoder/dataset/role combination.

        Args:
            encoder_name: Encoder type (e.g., 'helmbert', 'esm2')
            dataset_type: Dataset name (e.g., 'propedia_ppi_random')
            model_version: Optional model version
            role: Sequence role ('drugs' or 'targets')

        Returns:
            Path to cache directory

        Examples:
            >>> cache.get_cache_path('helmbert', 'propedia_ppi_random', role='drugs')
            Path('embeddings/helmbert/propedia_ppi_random/drugs')

            >>> cache.get_cache_path('esm2', 'propedia_ppi_random', role='targets')
            Path('embeddings/esm2/propedia_ppi_random/targets')
        """
        path = self.cache_dir / encoder_name

        if model_version:
            path = path / model_version

        path = path / dataset_type

        if role:
            path = path / role

        return path

    def save_embeddings(
        self,
        encoder_name: str,
        dataset_type: str,
        sequences: List[str],
        embeddings: List[torch.Tensor],
        model_version: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        """Save embeddings to cache.

        Args:
            encoder_name: Encoder type (e.g., 'helmbert', 'esm2')
            dataset_type: Dataset name (e.g., 'propedia_ppi_random')
            sequences: List of sequences
            embeddings: List of embedding tensors
            model_version: Optional model version
            role: Sequence role ('drugs' or 'targets')
        """
        cache_path = self.get_cache_path(
            encoder_name, dataset_type, model_version, role
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save each embedding
        sequence_to_file = {}
        for seq, emb in zip(sequences, embeddings):
            seq_hash = self.get_sequence_hash(seq)
            emb_file = cache_path / f"{seq_hash}.pt"
            torch.save(emb, emb_file)
            sequence_to_file[seq] = str(emb_file)

        # Update index
        index_file = cache_path / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                existing = json.load(f)
            existing.update(sequence_to_file)
            sequence_to_file = existing

        with open(index_file, "w") as f:
            json.dump(sequence_to_file, f, indent=2)

    def load_embeddings(
        self,
        encoder_name: str,
        dataset_type: str,
        sequences: List[str],
        model_version: Optional[str] = None,
        role: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load cached embeddings.

        Args:
            encoder_name: Encoder type (e.g., 'helmbert', 'esm2')
            dataset_type: Dataset name (e.g., 'propedia_ppi_random')
            sequences: List of sequences to load (should be unique)
            model_version: Optional model version
            role: Sequence role ('drugs' or 'targets')

        Returns:
            Dictionary mapping sequences to embeddings
        """
        cache_path = self.get_cache_path(
            encoder_name, dataset_type, model_version, role
        )
        index_file = cache_path / "index.json"

        if not index_file.exists():
            logger.warning(f"Index file not found: {index_file}")
            return {}

        logger.info(f"Loading index from {index_file}...")
        with open(index_file, "r") as f:
            sequence_to_file = json.load(f)

        # Filter sequences that exist in cache
        sequences_to_load = [seq for seq in sequences if seq in sequence_to_file]

        if len(sequences_to_load) < len(sequences):
            missing = len(sequences) - len(sequences_to_load)
            logger.warning(f"Missing {missing}/{len(sequences)} sequences from cache")

        # Load embeddings with progress bar
        embeddings = {}
        logger.info(f"Loading {len(sequences_to_load)} embeddings from disk...")
        for seq in tqdm(
            sequences_to_load,
            desc=f"Loading {role}",
            disable=len(sequences_to_load) < 100,
        ):
            emb_file = Path(sequence_to_file[seq])
            if emb_file.exists():
                embeddings[seq] = torch.load(emb_file)
            else:
                logger.warning(f"Embedding file not found: {emb_file}")

        return embeddings

    def check_cached_count(
        self,
        encoder_name: str,
        dataset_type: str,
        sequences: List[str],
        model_version: Optional[str] = None,
        role: Optional[str] = None,
    ) -> int:
        """Count how many sequences are already cached (without loading embeddings).

        Args:
            encoder_name: Encoder type (e.g., 'helmbert', 'esm2')
            dataset_type: Dataset name (e.g., 'propedia_ppi_random')
            sequences: List of sequences to check
            model_version: Optional model version
            role: Sequence role ('drugs' or 'targets')

        Returns:
            Number of sequences found in cache (does NOT load actual embeddings)
        """
        cache_path = self.get_cache_path(
            encoder_name, dataset_type, model_version, role
        )
        index_file = cache_path / "index.json"

        if not index_file.exists():
            return 0

        with open(index_file, "r") as f:
            sequence_to_file = json.load(f)

        # Count sequences present in index (no disk I/O for .pt files)
        return sum(1 for seq in sequences if seq in sequence_to_file)

    def generate_missing_embeddings(
        self,
        encoder_name: str,
        dataset_type: str,
        sequences: List[str],
        embed_fn: Callable[[List[str]], List[torch.Tensor]],
        batch_size: int = 32,
        model_version: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        """Generate and cache embeddings for sequences not already cached.

        This function is optimized for the generation use case and does NOT return embeddings.
        Use load_embeddings() if you need to retrieve cached embeddings.

        Args:
            encoder_name: Encoder type (e.g., 'helmbert', 'esm2')
            dataset_type: Dataset name (e.g., 'propedia_ppi_random')
            sequences: Unique sequences to embed (caller should deduplicate)
            embed_fn: Function to generate embeddings for a batch
            batch_size: Batch size for generation
            model_version: Optional model version
            role: Sequence role ('drugs' or 'targets')
        """
        logger.info(f"    Checking cache for {len(sequences)} unique sequences...")

        # Assume sequences are already unique (caller's responsibility)
        unique_sequences = sequences

        # Fast check: which sequences are missing from cache?
        cache_path = self.get_cache_path(
            encoder_name, dataset_type, model_version, role
        )
        index_file = cache_path / "index.json"

        logger.info(f"    Checking cache at {index_file}...")
        if index_file.exists():
            logger.info("    Loading index.json...")
            with open(index_file, "r") as f:
                sequence_to_file = json.load(f)
            logger.info("    Index loaded, checking missing sequences...")
            missing = [seq for seq in unique_sequences if seq not in sequence_to_file]
            logger.info(f"    Found {len(missing)} missing sequences")
        else:
            logger.info("    No index.json found, all sequences need generation")
            missing = unique_sequences

        if not missing:
            logger.info(f"    ✓ All {len(unique_sequences)} embeddings found in cache")
            return

        num_cached = len(unique_sequences) - len(missing)
        logger.info(
            f"    Generating {len(missing)} embeddings ({num_cached} cached)..."
        )

        # Generate missing in batches
        new_embeddings = []
        for i in tqdm(
            range(0, len(missing), batch_size), desc=f"    Generating {role}"
        ):
            batch = missing[i : i + batch_size]
            batch_embeddings = embed_fn(batch)
            new_embeddings.extend(batch_embeddings)

        # Save new embeddings
        logger.info(f"    Saving {len(new_embeddings)} new embeddings...")
        self.save_embeddings(
            encoder_name, dataset_type, missing, new_embeddings, model_version, role
        )
        logger.info(f"    ✓ Saved {len(new_embeddings)} embeddings")
