"""Utilities for scaffold-based split construction."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str) -> str:
    """Generate a Murcko scaffold string.

    Returns an empty string when scaffold generation fails.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles,
            includeChirality=False,
        )
    except Exception:
        return ""


def build_scaffold_groups(smiles_iter: Iterable[str]) -> List[List[int]]:
    """Group row indices by scaffold."""
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, smiles in enumerate(smiles_iter):
        scaffold = generate_scaffold(smiles) or "__no_scaffold__"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    return sorted(
        scaffold_to_indices.values(),
        key=lambda group: (len(group), group[0]),
        reverse=True,
    )


def flatten_groups(groups: List[List[int]]) -> List[int]:
    """Flatten scaffold groups into a list of row indices."""
    return [idx for group in groups for idx in group]


StateT = TypeVar("StateT")


def greedy_scaffold_partition(
    groups: List[List[int]],
    group_states: List[StateT],
    target_test_size: int,
    empty_state: StateT,
    combine_states: Callable[[StateT, StateT], StateT],
    key_fn: Callable[[int, StateT, int], Tuple[float, ...]],
) -> Tuple[List[List[int]], List[List[int]], int, StateT]:
    """Greedily assign scaffold groups into test/train.

    Groups are assumed to be sorted largest-first. The same direct assignment
    procedure is shared across permeability splits; only the state
    representation and comparison key differ.
    """
    test_groups: List[List[int]] = []
    train_groups: List[List[int]] = []
    train_states: List[StateT] = []

    remaining_size = sum(len(group) for group in groups)
    test_size = 0
    test_state = empty_state

    for group, group_state in zip(groups, group_states):
        group_size = len(group)
        remaining_size -= group_size

        current_key = key_fn(test_size, test_state, target_test_size)
        candidate_state = combine_states(test_state, group_state)
        candidate_key = key_fn(test_size + group_size, candidate_state, target_test_size)

        must_add = test_size < target_test_size and (test_size + remaining_size) < target_test_size
        improves = candidate_key < current_key
        overshoots = (test_size + group_size) > target_test_size

        if must_add or (not overshoots and improves):
            test_groups.append(group)
            test_size += group_size
            test_state = candidate_state
        else:
            train_groups.append(group)
            train_states.append(group_state)

    while test_size < target_test_size and train_groups:
        best_idx = min(
            range(len(train_groups)),
            key=lambda idx: key_fn(
                test_size + len(train_groups[idx]),
                combine_states(test_state, train_states[idx]),
                target_test_size,
            ),
        )
        group = train_groups.pop(best_idx)
        group_state = train_states.pop(best_idx)
        test_groups.append(group)
        test_size += len(group)
        test_state = combine_states(test_state, group_state)

    return test_groups, train_groups, test_size, test_state
