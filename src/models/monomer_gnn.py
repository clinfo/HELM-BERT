"""GPS-based monomer encoder for HELM-GNN.

Converts monomer SMILES to atom graphs via RDKit, then encodes each monomer
into a fixed-size embedding using GPSConv layers with global mean pooling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GPSConv, GINEConv, global_mean_pool

logger = logging.getLogger(__name__)

# Atom feature dimensions
ATOM_FEATURES = {
    "atomic_num": 119,     # 0-118
    "degree": 7,           # 0-6
    "formal_charge": 11,   # -5 to +5
    "num_hs": 9,           # 0-8
    "hybridization": 7,    # SP, SP2, SP3, SP3D, SP3D2, other, unspecified
    "is_aromatic": 2,
    "is_in_ring": 2,
}
ATOM_FEAT_DIM = sum(ATOM_FEATURES.values())

# Bond feature dimensions
BOND_FEATURES = {
    "bond_type": 5,        # SINGLE, DOUBLE, TRIPLE, AROMATIC, other
    "is_conjugated": 2,
    "is_in_ring": 2,
}
BOND_FEAT_DIM = sum(BOND_FEATURES.values())

_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.OTHER: 5,
}

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def _one_hot(value: int, num_classes: int) -> list[float]:
    vec = [0.0] * num_classes
    if 0 <= value < num_classes:
        vec[value] = 1.0
    return vec


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string to a PyG Data object.

    Returns None if SMILES is invalid or has no atoms.
    """
    # Strip extended SMILES annotations
    smiles = smiles.split("|")[0].strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = (
            _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
            + _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
            + _one_hot(atom.GetFormalCharge() + 5, ATOM_FEATURES["formal_charge"])
            + _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
            + _one_hot(
                _HYBRIDIZATION_MAP.get(atom.GetHybridization(), 6),
                ATOM_FEATURES["hybridization"],
            )
            + _one_hot(int(atom.GetIsAromatic()), ATOM_FEATURES["is_aromatic"])
            + _one_hot(int(atom.IsInRing()), ATOM_FEATURES["is_in_ring"])
        )
        atom_feats.append(feat)

    x = torch.tensor(atom_feats, dtype=torch.float)

    # Bond features + edge index
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = (
                _one_hot(
                    _BOND_TYPE_MAP.get(bond.GetBondType(), 4),
                    BOND_FEATURES["bond_type"],
                )
                + _one_hot(int(bond.GetIsConjugated()), BOND_FEATURES["is_conjugated"])
                + _one_hot(int(bond.IsInRing()), BOND_FEATURES["is_in_ring"])
            )
            # Undirected: add both directions
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([feat, feat])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MonomerGPSEncoder(nn.Module):
    """GPS-based encoder that maps monomer SMILES to fixed-dim embeddings.

    Architecture: atom features -> Linear proj -> GPS layers -> global mean pool -> output proj

    Args:
        hidden_dim: Hidden dimension of GPS layers.
        output_dim: Output embedding dimension (should match Transformer hidden_size).
        num_layers: Number of GPS layers.
        num_heads: Number of attention heads in GPS.
        dropout: Dropout rate.
        monomer_library_path: Path to monomer library CSV for SMILES lookup.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        monomer_library_path: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projections
        self.atom_proj = nn.Linear(ATOM_FEAT_DIM, hidden_dim)
        self.bond_proj = nn.Linear(BOND_FEAT_DIM, hidden_dim)

        # GPS layers
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            gin_conv = GINEConv(gin_nn, edge_dim=hidden_dim)
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=gin_conv,
                heads=num_heads,
                dropout=dropout,
                attn_kwargs={"dropout": dropout},
            )
            self.gps_layers.append(gps_layer)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # SMILES lookup table (monomer symbol -> SMILES)
        self._smiles_lookup: Dict[str, str] = {}
        if monomer_library_path:
            self._load_smiles_lookup(monomer_library_path)

        # Cache: SMILES -> PyG Data
        self._graph_cache: Dict[str, Data] = {}

    def _load_smiles_lookup(self, path: str | Path) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            symbol = str(row["symbol"]).strip()
            smiles = str(row["smiles"]).strip()
            if symbol and smiles and smiles != "nan":
                self._smiles_lookup[symbol] = smiles
        logger.info(f"Loaded SMILES for {len(self._smiles_lookup)} monomers")

    def get_monomer_graph(self, symbol: str) -> Optional[Data]:
        """Get atom graph for a monomer, using cache."""
        smiles = self._smiles_lookup.get(symbol)
        if smiles is None:
            return None
        if smiles not in self._graph_cache:
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self._graph_cache[smiles] = graph
            else:
                return None
        return self._graph_cache[smiles].clone()

    def forward(
        self,
        monomer_symbols: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a list of unique monomer symbols into embeddings.

        Args:
            monomer_symbols: List of unique monomer symbols to encode.
            device: Target device for tensors.

        Returns:
            (num_monomers, output_dim) tensor of monomer embeddings.
        """
        if not monomer_symbols:
            return torch.zeros(0, self.output_dim, device=device)

        graphs = []
        valid_indices = []
        for i, symbol in enumerate(monomer_symbols):
            graph = self.get_monomer_graph(symbol)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)

        # Initialize output with zeros for unknown monomers
        output = torch.zeros(len(monomer_symbols), self.output_dim, device=device)

        if not graphs:
            return output

        batch = Batch.from_data_list(graphs).to(device)

        # Project atom and bond features
        x = self.atom_proj(batch.x)
        edge_attr = self.bond_proj(batch.edge_attr)

        # GPS layers
        for gps_layer in self.gps_layers:
            x = gps_layer(x, batch.edge_index, batch.batch, edge_attr=edge_attr)

        # Global mean pooling per graph
        pooled = global_mean_pool(x, batch.batch)  # (num_valid, hidden_dim)

        # Output projection
        embeddings = self.output_proj(pooled)  # (num_valid, output_dim)

        # Place valid embeddings into output tensor
        for idx, emb in zip(valid_indices, embeddings):
            output[idx] = emb

        return output
