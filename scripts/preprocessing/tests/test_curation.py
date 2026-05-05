"""Tests for hand-curated layer (manual_monomer_additions.csv + helm_corrections.csv).

These tests guard the warehouse from a class of silent data-corruption bugs:
declaring an (E)/(Z) suffix that does NOT match what RDKit's CIP labeller
sees in the molecule. A wrong label here would tag a different molecule
in HELM than what the SMILES actually represents — irreversible chaos
once the warehouse is consumed downstream.

Run from scripts/:
    python -m unittest tests.test_curation
"""
from __future__ import annotations

import csv
import io
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from helpers.paths import (
    FINAL_DIR,
    HELM_CORRECTIONS_PATH,
    MANUAL_MONOMER_ADDITIONS_PATH,
)


def _read_skip_comments(path: Path) -> pd.DataFrame:
    """Read a CSV ignoring leading comment lines (lines starting with '#')."""
    with open(path) as f:
        body = "".join(l for l in f if not l.lstrip().startswith("#"))
    return pd.read_csv(io.StringIO(body))


def _cip_ez_codes(smi: str) -> list[str]:
    """Return a list of CIP E/Z codes for every stereo double bond.

    Uses ``rdCIPLabeler.AssignCIPLabels`` (the modern, IUPAC-compliant
    CIP implementation). Parses the bare SMILES portion only —
    CXSMILES extensions like ``|w:1.0,$_R1;...$|`` are stripped before
    parsing so RDKit doesn't choke on them.
    """
    bare = smi.split(" |")[0] if " |" in smi else smi
    mol = Chem.MolFromSmiles(bare)
    if mol is None:
        return []
    rdCIPLabeler.AssignCIPLabels(mol)
    out: list[str] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if bond.HasProp("_CIPCode"):
            out.append(bond.GetProp("_CIPCode"))
    return out


SUFFIX_RE = re.compile(r"\((E|Z)\)$")
TOKEN_SUFFIX_RE = re.compile(r"^\[.+\((E|Z)\)\]$")


@unittest.skipUnless(
    MANUAL_MONOMER_ADDITIONS_PATH.exists(),
    "manual_monomer_additions.csv absent — nothing to verify",
)
class TestManualAdditionsLabels(unittest.TestCase):
    """For every monomer with an (E) or (Z) suffix, RDKit CIP must agree.

    The suffix is the human-readable part of the label; it has to match
    what a chemistry tool computes from the SMILES, otherwise downstream
    consumers see one geometry but the symbol claims another.
    """

    def test_every_suffix_matches_cip(self):
        df = _read_skip_comments(MANUAL_MONOMER_ADDITIONS_PATH)
        suffix_rows = []
        for _, r in df.iterrows():
            sym = str(r["symbol"])
            m = SUFFIX_RE.search(sym)
            if not m:
                continue
            suffix_rows.append((sym, m.group(1), str(r["smiles"])))

        self.assertGreater(
            len(suffix_rows),
            0,
            "expected at least one (E)/(Z) entry in manual_monomer_additions.csv",
        )

        for sym, declared, smi in suffix_rows:
            with self.subTest(symbol=sym):
                # Replace [*] R-group markers with [H] so RDKit can fully parse
                # and compute CIP. R-group context can shift CIP priorities,
                # but for the monomer's own double bond the local geometry
                # is preserved.
                test_smi = smi.replace("[*]", "[H]")
                codes = _cip_ez_codes(test_smi)
                self.assertIn(
                    declared,
                    codes,
                    f"{sym!r} declares ({declared}) but RDKit CIP sees {codes}",
                )


@unittest.skipUnless(
    HELM_CORRECTIONS_PATH.exists(),
    "helm_corrections.csv absent — nothing to verify",
)
class TestHelmCorrectionLabels(unittest.TestCase):
    """For every compound rewrite, the corrected_token's (E)/(Z) suffix
    must match what RDKit CIP computes from the compound's normalized
    SMILES. CIP can flip between the isolated monomer fragment and the
    full peptide context (R-group neighbors change priorities) — what
    matters for the warehouse is the FULL-COMPOUND classification, since
    that is what dedup, the model, and downstream joins all observe.
    """

    def test_every_correction_matches_compound_cip(self):
        corr = _read_skip_comments(HELM_CORRECTIONS_PATH)
        compounds_path = FINAL_DIR / "chembl_compounds.csv"
        if not compounds_path.exists():
            self.skipTest(f"missing prerequisite: {compounds_path}")
        df = pd.read_csv(
            compounds_path,
            low_memory=False,
            usecols=["compound_chembl_id", "normalized_smiles"],
        )

        for _, row in corr.iterrows():
            cid = str(row["compound_chembl_id"])
            new_token = str(row["corrected_token"])
            m = TOKEN_SUFFIX_RE.match(new_token)
            if not m:
                continue
            declared = m.group(1)
            with self.subTest(compound=cid, token=new_token):
                sub = df[df["compound_chembl_id"] == cid]
                self.assertEqual(
                    len(sub),
                    1,
                    f"{cid} not found in chembl_compounds (corrections file is stale)",
                )
                smi = sub.iloc[0]["normalized_smiles"]
                codes = _cip_ez_codes(smi)
                self.assertIn(
                    declared,
                    codes,
                    f"{cid} corrects to {new_token} ({declared}) but full-compound "
                    f"CIP gives {codes}",
                )


@unittest.skipUnless(
    HELM_CORRECTIONS_PATH.exists() and MANUAL_MONOMER_ADDITIONS_PATH.exists(),
    "curation files not both present",
)
class TestCurationCrossReference(unittest.TestCase):
    """The two curation files must agree: every corrected_token used in
    helm_corrections.csv must be backed by a matching symbol in
    manual_monomer_additions.csv (or be an existing library symbol — but we
    don't have access to the live library here, so we only check the
    stronger condition for symbols ending in (E)/(Z))."""

    def test_corrected_tokens_have_matching_monomer_entries(self):
        manual_syms = set(_read_skip_comments(MANUAL_MONOMER_ADDITIONS_PATH)["symbol"])
        corr = _read_skip_comments(HELM_CORRECTIONS_PATH)
        missing: list[tuple[str, str]] = []
        for _, row in corr.iterrows():
            new_token = str(row["corrected_token"])
            m = TOKEN_SUFFIX_RE.match(new_token)
            if not m:
                continue
            symbol = new_token.strip("[]")
            if symbol not in manual_syms:
                missing.append((str(row["compound_chembl_id"]), new_token))
        self.assertEqual(
            missing,
            [],
            f"corrected tokens reference symbols absent from manual_monomer_additions.csv: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
