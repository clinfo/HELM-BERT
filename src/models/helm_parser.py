"""HELM notation parser for monomer-level tokenization.

Parses HELM strings into monomer sequences and connectivity information
for use with the HELM-GNN model.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


def parse_helm_to_monomers(helm: str) -> list[str] | None:
    """Extract monomer symbols from all PEPTIDE chains in HELM notation.

    Args:
        helm: HELM notation string, e.g.
              'PEPTIDE1{A.[meL].G}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$'

    Returns:
        List of monomer symbols (without brackets), or None if unparseable.
    """
    if not helm:
        return None
    helm = str(helm).strip()
    matches = re.findall(r"PEPTIDE\d*\{([^}]+)\}", helm)
    if not matches:
        return None
    monomers: list[str] = []
    for seq in matches:
        for token in seq.split("."):
            token = token.strip()
            if not token:
                continue
            if token.startswith("[") and token.endswith("]"):
                token = token[1:-1]
            monomers.append(token)
    return monomers


def parse_helm_connections(helm: str) -> list[tuple[int, int]]:
    """Parse HELM connection section to get monomer-level edges.

    Connections like 'PEPTIDE1,PEPTIDE1,1:R1-11:R2' are parsed into
    zero-indexed (source, target) tuples.

    Args:
        helm: Full HELM notation string.

    Returns:
        List of (src_idx, tgt_idx) zero-indexed monomer pairs.
    """
    if not helm:
        return []
    parts = str(helm).split("$")
    if len(parts) < 2 or not parts[1].strip():
        return []

    connections: list[tuple[int, int]] = []
    for conn_str in parts[1].split("|"):
        conn_str = conn_str.strip()
        if not conn_str:
            continue
        # Format: PEPTIDE1,PEPTIDE1,1:R1-11:R2
        match = re.match(
            r"PEPTIDE\d+,PEPTIDE\d+,(\d+):R\d+-(\d+):R\d+", conn_str
        )
        if match:
            src = int(match.group(1)) - 1  # 1-indexed -> 0-indexed
            tgt = int(match.group(2)) - 1
            connections.append((src, tgt))
    return connections


def build_graph_distance_matrix(
    num_monomers: int,
    extra_connections: list[tuple[int, int]],
    max_distance: int = 32,
) -> np.ndarray:
    """Build pairwise shortest-path distance matrix between monomers.

    Edges: sequential peptide bonds (i, i+1) + explicit HELM connections.
    Uses BFS for shortest paths.

    Args:
        num_monomers: Number of monomers in the sequence.
        extra_connections: Additional connections from HELM (cyclic bonds, etc.).
        max_distance: Distances beyond this are clipped.

    Returns:
        (num_monomers, num_monomers) int32 distance matrix.
    """
    if num_monomers == 0:
        return np.zeros((0, 0), dtype=np.int32)

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(num_monomers)]
    for i in range(num_monomers - 1):
        adj[i].append(i + 1)
        adj[i + 1].append(i)
    for src, tgt in extra_connections:
        if 0 <= src < num_monomers and 0 <= tgt < num_monomers:
            adj[src].append(tgt)
            adj[tgt].append(src)

    # BFS from each node
    dist = np.full((num_monomers, num_monomers), max_distance, dtype=np.int32)
    for start in range(num_monomers):
        dist[start, start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in adj[node]:
                if dist[start, neighbor] > dist[start, node] + 1:
                    dist[start, neighbor] = dist[start, node] + 1
                    queue.append(neighbor)

    np.clip(dist, 0, max_distance, out=dist)
    return dist


def parse_helm_full(
    helm: str, max_distance: int = 32
) -> Optional[tuple[list[str], np.ndarray]]:
    """Parse HELM string into monomers and graph distance matrix.

    Convenience function combining parse + distance computation.

    Returns:
        (monomer_symbols, distance_matrix) or None if unparseable.
    """
    monomers = parse_helm_to_monomers(helm)
    if monomers is None:
        return None
    connections = parse_helm_connections(helm)
    dist = build_graph_distance_matrix(len(monomers), connections, max_distance)
    return monomers, dist
