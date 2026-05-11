#!/usr/bin/env python3
"""Per-source deduplication on (helm_col, normalized_smiles, *dedup_extra_keys).

Reads ``processed/04_smiles_normalized/{key}.csv``, collapses rows that
share the dataset-specific dedup key, and writes
``processed/05_final/{key}.csv``.

The grouping key uses ``normalized_smiles`` (RDKit canonical SMILES) as
the "same molecule" identity — strict byte-equality of the canonical
form. Two rows must have identical HELM AND identical canonical SMILES
to collapse. This treats salts, protonation states, isotope labels,
and tautomer drawings as DISTINCT, preserving each as registered.

Determinism contract:
    * Within each duplicate group, the surviving row is the one with the
      lex-smallest tuple over ``cfg.id_cols`` — independent of input row
      order. Re-running the pipeline yields bit-identical output.
    * The collapsed group's full set of original IDs is preserved in a
      new ``Source_IDs`` column (semicolon-joined per id_col, then
      pipe-joined across id_cols when there are multiple).

Diagnostics emitted to ``05_final/{key}.csv``:
    * ``helm_smiles_consistent`` — False if a row's HELM appears with
      more than one normalized_smiles across the dataset (i.e. HELM
      cannot uniquely identify the molecule, usually a ChEMBL
      X-monomer curation artifact, or salts / protonation differences
      that the source registered against the same HELM). True otherwise.

Sanity checks:
    * After dedup, ``df.groupby(dedup_key).size().max() == 1``
      (no remaining duplicates on the canonical key).
    * Group count + drop count + final row count consistent.

Usage:
    python 08_dedup.py
    python 08_dedup.py --datasets cycpept_permeability_compounds chembl_ppi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from helpers.datasets import DATASETS, DatasetConfig
from helpers.logging_utils import setup_logger
from helpers.paths import ensure_dirs


DEDUP_BASE_MOL_KEY = "normalized_smiles"  # the "same molecule" column from stage 08


def _build_dedup_key(cfg: DatasetConfig) -> list[str]:
    """The full grouping key for this dataset.

    Always: ``[helm_col, normalized_smiles, *dedup_extra_keys]``. Canonical
    SMILES is the molecular-identity column (strict byte-equality);
    ``dedup_extra_keys`` carry measurement / pair context that should NOT
    collapse (chembl_ppi target+type+value, propedia_ppi receptor
    sequence, …).
    """
    base = [cfg.helm_col, DEDUP_BASE_MOL_KEY]
    return base + list(cfg.dedup_extra_keys)


def _collapse_ids(group: pd.DataFrame, id_cols: tuple[str, ...]) -> str:
    """Format a group's original IDs into a single string for Source_IDs.

    With one id_col: ``"id1;id2;id3"``.
    With multiple: ``"col1=id1a;id1b|col2=id2a;id2b"`` (per-column unique).
    Sorted alphabetically for determinism.
    """
    parts: list[str] = []
    for col in id_cols:
        if col not in group.columns:
            continue
        vals = (
            group[col]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        vals.sort()
        if not vals:
            continue
        if len(id_cols) == 1:
            parts.append(";".join(vals))
        else:
            parts.append(f"{col}=" + ";".join(vals))
    return "|".join(parts)


def _representative_index(group: pd.DataFrame, id_cols: tuple[str, ...]) -> int:
    """Index of the lex-smallest id-tuple row in the group.

    Falls back to the first row when no id_cols are available on the frame
    (defensive — a misconfigured dataset would otherwise raise).
    """
    available = [c for c in id_cols if c in group.columns]
    if not available:
        return cast(int, group.index[0])
    sorted_group = group.sort_values(by=available, kind="mergesort", na_position="last")
    return cast(int, sorted_group.index[0])


def dedup_dataset(cfg: DatasetConfig, log) -> dict[str, int]:
    """Run dedup for one dataset; emit final CSV."""
    in_path = cfg.stage_path("smiles_normalized")
    out_path = cfg.stage_path("final")
    if not in_path.exists():
        raise FileNotFoundError(
            f"[{cfg.key}] missing input {in_path}; run 07_normalize_smiles.py first"
        )

    log.info("=" * 70)
    log.info("[%s] %s", cfg.key, in_path)
    log.info("=" * 70)

    df = cast(pd.DataFrame, pd.read_csv(in_path, low_memory=False))
    before = len(df)
    log.info("[%s] loaded %d rows", cfg.key, before)

    key_cols = _build_dedup_key(cfg)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{cfg.key}] dedup key columns missing: {missing}. "
            f"Available: {list(df.columns)[:20]}"
        )

    # Make the dedup contract explicit in the log so a reviewer can audit
    # exactly what was collapsed and what was preserved without reading code.
    log.info("[%s] dedup key (group by): %s", cfg.key, key_cols)
    log.info("[%s] id_cols (keep + Source_IDs): %s", cfg.key, list(cfg.id_cols))
    log.info(
        "[%s] keep policy: row with lex-smallest tuple over id_cols; "
        "collapsed IDs joined into Source_IDs column",
        cfg.key,
    )

    # Drop rows whose dedup key is incomplete — they cannot be safely
    # collapsed and would create a phantom NaN group.
    key_complete = df[key_cols].notna().all(axis=1)
    dropped_incomplete = int((~key_complete).sum())
    df = cast(pd.DataFrame, df.loc[key_complete].copy())
    if dropped_incomplete:
        log.warning(
            "[%s] dropped %d rows with incomplete dedup key %s",
            cfg.key,
            dropped_incomplete,
            key_cols,
        )

    # Group + collapse
    keep_indices: list[int] = []
    source_ids: list[str] = []
    group_sizes: list[int] = []
    grouped = df.groupby(key_cols, sort=False, dropna=False)
    n_groups = grouped.ngroups
    for _, group in grouped:
        rep_idx = _representative_index(group, cfg.id_cols)
        keep_indices.append(rep_idx)
        source_ids.append(_collapse_ids(group, cfg.id_cols))
        group_sizes.append(len(group))

    out = cast(pd.DataFrame, df.loc[keep_indices].copy())
    out["Source_IDs"] = source_ids

    # helm_smiles_consistent: False when one HELM maps to multiple
    # canonical SMILES — i.e. HELM cannot uniquely identify the molecule.
    # Causes include: ChEMBL X-monomer that needs splitting (E/Z, etc.),
    # source registering the same HELM with salt-form variants, or
    # tautomer-only drawing differences in the source. Per-row flag lets
    # downstream filter or audit without re-grouping.
    helm_to_keys = out.groupby(cfg.helm_col)[DEDUP_BASE_MOL_KEY].nunique()
    ambiguous_helms = set(helm_to_keys[helm_to_keys > 1].index)
    out["helm_smiles_consistent"] = ~out[cfg.helm_col].isin(ambiguous_helms)

    out = cast(pd.DataFrame, out.sort_values(by=list(cfg.id_cols), kind="mergesort").reset_index(drop=True))

    after = len(out)
    collapsed = before - dropped_incomplete - after
    n_inconsistent = int((~out["helm_smiles_consistent"]).sum())
    if n_inconsistent:
        log.warning(
            "[%s] %d rows flagged helm_smiles_consistent=False "
            "(HELM maps to multiple canonical SMILES — see %d HELM groups)",
            cfg.key,
            n_inconsistent,
            len(ambiguous_helms),
        )
    log.info(
        "[%s] %d -> %d (collapsed %d duplicates across %d groups, "
        "dropped %d incomplete)",
        cfg.key,
        before,
        after,
        collapsed,
        n_groups,
        dropped_incomplete,
    )

    # Group-size distribution: lets a reviewer eyeball whether dedup is
    # behaving sensibly. A lone outlier with size 100+ is usually a sign
    # the dedup key is too loose.
    sizes_series = pd.Series(group_sizes)
    multi_groups = int((sizes_series > 1).sum())
    if multi_groups:
        log.info(
            "[%s] group-size distribution: max=%d, mean(size>1)=%.2f, "
            "groups_with_dups=%d (%.1f%%)",
            cfg.key,
            int(sizes_series.max()),
            float(sizes_series[sizes_series > 1].mean()),
            multi_groups,
            100.0 * multi_groups / n_groups,
        )
    else:
        log.info("[%s] no duplicate groups (all rows already unique on key)", cfg.key)

    # Sanity check: zero duplicates remain on the dedup key.
    leftover = int(out.groupby(key_cols, dropna=False).size().max() if len(out) else 0)
    if leftover > 1:
        raise AssertionError(
            f"[{cfg.key}] {leftover} duplicates remain after dedup — bug in grouping logic"
        )
    log.info("[%s] sanity check: 0 duplicate keys remain", cfg.key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("[%s] wrote %d rows -> %s", cfg.key, after, out_path)

    return {
        "before": before,
        "dropped_incomplete": dropped_incomplete,
        "groups": n_groups,
        "collapsed": collapsed,
        "after": after,
        "max_group_size": int(sizes_series.max()) if len(sizes_series) else 0,
        "groups_with_dups": multi_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedup datasets on canonical (helm, smiles)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        help="subset of dataset keys; default = all",
    )
    args = parser.parse_args()

    ensure_dirs()
    logger, _ = setup_logger("dedup")

    targets = args.datasets or list(DATASETS.keys())
    summary: dict[str, dict[str, int]] = {}
    for key in targets:
        cfg = DATASETS[key]
        try:
            summary[key] = dedup_dataset(cfg, logger)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", key, e)
        except Exception:
            logger.exception("[%s] failed", key)
            return 2

    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(
        "%-18s %8s %8s %10s %8s %12s %14s",
        "dataset", "before", "after", "collapsed", "groups", "max_grp_sz", "groups_w_dups",
    )
    for key, s in summary.items():
        logger.info(
            "%-18s %8d %8d %10d %8d %12d %14d",
            key,
            s["before"],
            s["after"],
            s["collapsed"],
            s["groups"],
            s["max_group_size"],
            s["groups_with_dups"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
