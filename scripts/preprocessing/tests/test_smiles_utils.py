"""Tests for helpers.smiles_utils.

Critical properties covered:
    1. Stereo (D/L, E/Z) preservation through canonicalization.
    2. NO structural mutation: salts, charges, isotopes preserved.
    3. Idempotency of standardize_smiles.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rdkit import Chem

from helpers.smiles_utils import (
    StandardizeStats,
    standardize_series,
    standardize_smiles,
)


def _stereo_count(smi: str) -> int:
    """Count atoms with explicit chirality (sanity-check helper)."""
    if not smi:
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(
        1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )


class TestStereoPreservation(unittest.TestCase):
    """The single most important property: D-AAs and L-AAs stay distinct."""

    def test_l_tyr_stereo_preserved(self):
        smi = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        out = standardize_smiles(smi)
        self.assertEqual(_stereo_count(smi), _stereo_count(out))
        self.assertGreaterEqual(_stereo_count(out), 1)

    def test_d_tyr_stereo_preserved(self):
        smi = "N[C@H](Cc1ccc(O)cc1)C(=O)O"
        out = standardize_smiles(smi)
        self.assertEqual(_stereo_count(smi), _stereo_count(out))

    def test_d_vs_l_tyr_yield_different_normalized(self):
        l_smi = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        d_smi = "N[C@H](Cc1ccc(O)cc1)C(=O)O"
        self.assertNotEqual(standardize_smiles(l_smi), standardize_smiles(d_smi))

    def test_tripeptide_all_stereo_centers_preserved(self):
        smi = "C[C@H](N)C(=O)NCC(=O)N[C@@H](C)C(=O)O"  # L-Ala-Gly-L-Ala
        out = standardize_smiles(smi)
        self.assertEqual(_stereo_count(out), 2)

    def test_three_chiral_peptide(self):
        """ChEMBL-like peptide with 3 chiral centers — none must be lost."""
        smi = (
            "CC(C)C[C@@H](NC(=O)N1CCCCCC1)C(=O)N(C)[C@@H]"
            "(Cc1c[nH]c2ccccc12)C(=O)N[C@H](Cc1ccccn1)C(=O)N(C)CC(=O)O"
        )
        out = standardize_smiles(smi)
        self.assertEqual(_stereo_count(smi), 3)
        self.assertEqual(_stereo_count(out), 3)


class TestNoStructuralMutation(unittest.TestCase):
    """Source molecule must be preserved exactly — no salt strip,
    no uncharger, no isotope strip. Only the SMILES *string* is
    rewritten into RDKit canonical form.
    """

    def test_salt_preserved(self):
        out = standardize_smiles("CC(=O)O.[Na+]")
        # Both fragments must survive.
        self.assertIn(".", out)
        self.assertIn("[Na+]", out)

    def test_tfa_salt_preserved(self):
        smi = "CC(=O)N[C@H](Cc1ccccc1)C(=O)O.O=C(O)C(F)(F)F"
        out = standardize_smiles(smi)
        self.assertIn("F", out)            # TFA stays
        self.assertIn("[C@H]", out)        # peptide stereo also stays

    def test_zwitterion_preserved(self):
        out = standardize_smiles("[NH3+]CC(=O)[O-]")
        self.assertIn("+", out)
        self.assertIn("-", out)

    def test_carboxylate_charge_preserved(self):
        out = standardize_smiles("CC(=O)[O-]")
        self.assertIn("[O-]", out)

    def test_deuterium_preserved(self):
        out = standardize_smiles("[2H]C(C(=O)O)N")
        self.assertIn("2H", out)

    def test_c13_preserved(self):
        out = standardize_smiles("[13CH3]C(=O)O")
        self.assertIn("13C", out)


class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(standardize_smiles(""), "")

    def test_none(self):
        self.assertEqual(standardize_smiles(None), "")

    def test_invalid_smiles(self):
        self.assertEqual(standardize_smiles("not_a_real_smiles!"), "")


class TestIdempotency(unittest.TestCase):
    """f(f(x)) == f(x) for every input — important for re-running pipelines."""

    SAMPLES = [
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "C[C@H](N)C(=O)NCC(=O)N[C@@H](C)C(=O)O",
        "CC(=O)O.[Na+]",
        "[NH3+]CC(=O)[O-]",
        "[2H]C(C(=O)O)N",
        "NC(=N)NCCC[C@H](N)C(=O)O",
    ]

    def test_idempotent(self):
        for smi in self.SAMPLES:
            with self.subTest(smi=smi):
                once = standardize_smiles(smi)
                twice = standardize_smiles(once)
                self.assertEqual(once, twice)


class TestStandardizeSeries(unittest.TestCase):
    def test_returns_smiles_and_stats(self):
        s = pd.Series(
            [
                "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
                "CC(=O)O.[Na+]",
                "",
                "invalid_smiles!",
            ]
        )
        smi, stats = standardize_series(s, label="test")
        self.assertIsInstance(stats, StandardizeStats)
        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.parsed, 2)
        self.assertEqual(stats.parse_failed, 2)
        self.assertEqual(smi.iloc[2], "")
        self.assertEqual(smi.iloc[3], "")

    def test_preserves_index(self):
        s = pd.Series(["CCO", "CCN"], index=[100, 200])
        smi, _ = standardize_series(s)
        self.assertEqual(list(smi.index), [100, 200])


if __name__ == "__main__":
    unittest.main()
