from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

MLM_FILES: dict[str, tuple[str, str, str]] = {
    "propedia": ("data/mlm/propedia_deduplicated.csv", "Peptide_HELM", "Peptide_SMILES"),
    "cremp": ("data/mlm/cremp_deduplicated.csv", "helm", "smiles"),
    "cycpeptmpdb": ("data/mlm/cycpeptmpdb_deduplicated.csv", "HELM", "SMILES"),
    "chembl": ("data/mlm/chembl_deduplicated.csv", "helm_notation", "canonical_smiles"),
}


def smiles_to_canonical(smiles: str) -> str:
    if not smiles or pd.isna(smiles):
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _load_mlm_reference(repo_root_str: str) -> tuple[set[str], set[str]]:
    repo_root = Path(repo_root_str)
    mlm_helms: set[str] = set()
    mlm_smiles: set[str] = set()

    for rel_path, helm_col, smiles_col in MLM_FILES.values():
        path = repo_root / rel_path
        if not path.exists():
            continue
        df = cast(
            pd.DataFrame,
            pd.read_csv(path, usecols=cast(Any, [helm_col, smiles_col]), low_memory=False),
        )
        helm_series = cast(pd.Series, df[helm_col])
        smiles_series = cast(pd.Series, df[smiles_col])
        mlm_helms.update(helm_series.dropna().astype(str).str.strip())
        mlm_smiles.update(
            s for s in (smiles_to_canonical(v) for v in smiles_series.dropna().astype(str)) if s
        )

    return mlm_helms, mlm_smiles


def log_mlm_coverage(
    df: pd.DataFrame,
    helm_col: str,
    smiles_col: str | None,
    repo_root: Path,
    log: logging.Logger | None = None,
    label: str = "dataset",
) -> None:
    _log = log or logger
    mlm_helms, mlm_smiles = _load_mlm_reference(str(repo_root.resolve()))

    helm_series = cast(pd.Series, df[helm_col])
    helms = set(helm_series.dropna().astype(str).str.strip())
    missing_helms = sorted(helms - mlm_helms)
    _log.info(
        f"MLM coverage [{label}] exact HELM: {len(helms) - len(missing_helms)}/{len(helms)} "
        f"(missing: {len(missing_helms)})"
    )

    if missing_helms:
        _log.info(f"  Sample exact-HELM mismatches: {missing_helms[:3]}")

    if smiles_col is None:
        return

    unique = cast(pd.DataFrame, df[[helm_col, smiles_col]].drop_duplicates().copy())
    unique_helm_series = cast(pd.Series, unique[helm_col])
    missing = cast(
        pd.DataFrame,
        unique[~unique_helm_series.astype(str).str.strip().isin(list(mlm_helms))].copy(),
    )
    missing_smiles_series = cast(pd.Series, missing[smiles_col])
    missing["_canonical_smiles"] = missing_smiles_series.map(smiles_to_canonical)
    canonical_smiles_series = cast(pd.Series, missing["_canonical_smiles"])
    smiles_covered = int(canonical_smiles_series.isin(list(mlm_smiles)).sum())
    _log.info(
        f"MLM coverage [{label}] canonical SMILES: {smiles_covered}/{len(missing)} "
        f"of exact-HELM mismatches covered"
    )


def aggregate_median_by_canonical_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    numeric_cols: list[str],
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    _log = log or logger
    if df.empty:
        return df.copy()

    work = df.copy()
    smiles_series = cast(pd.Series, work[smiles_col])
    work["_canonical_smiles"] = smiles_series.map(smiles_to_canonical)
    canonical_smiles_series = cast(pd.Series, work["_canonical_smiles"])
    invalid = int((canonical_smiles_series == "").sum())
    if invalid:
        _log.warning(
            f"Found {invalid} rows with invalid SMILES during aggregation; keeping original SMILES key"
        )
        fallback = cast(pd.Series, canonical_smiles_series == "")
        work.loc[fallback, "_canonical_smiles"] = cast(
            pd.Series, work.loc[fallback, smiles_col]
        ).astype(str)

    before = len(work)
    canonical_smiles_series = cast(pd.Series, work["_canonical_smiles"])
    if canonical_smiles_series.nunique() == before:
        work[smiles_col] = canonical_smiles_series
        return cast(pd.DataFrame, work.drop(columns=["_canonical_smiles"]))

    aggregated_rows = []
    for canonical_smiles, group in work.groupby("_canonical_smiles", sort=False, dropna=False):
        rep = group.iloc[0].copy()
        rep[smiles_col] = canonical_smiles
        for col in numeric_cols:
            if col in group.columns:
                values = cast(pd.Series, pd.to_numeric(cast(pd.Series, group[col]), errors="coerce"))
                rep[col] = values.median() if values.notna().any() else float("nan")
        aggregated_rows.append(rep)

    result = cast(pd.DataFrame, pd.DataFrame(aggregated_rows).drop(columns=["_canonical_smiles"]))
    _log.info(
        f"Canonical-SMILES median aggregation: {before} rows -> {len(result)} molecules "
        f"(collapsed {before - len(result)} duplicates)"
    )
    return result.reset_index(drop=True)
