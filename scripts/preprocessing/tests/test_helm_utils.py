"""Tests for helpers.helm_utils.

Run from the scripts/ directory:
    python -m pytest tests/ -v
or:
    python -m unittest discover tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running via `python -m unittest discover tests` from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from helpers.helm_utils import (
    apply_helm_normalization,
    canonicalize_connections,
    get_peptide_chain_ids,
    normalize_single_chain,
    parse_helm_monomers,
    remap_alt_symbols,
    validate_monomers,
)


class TestParseHelmMonomers(unittest.TestCase):
    def test_simple_chain(self):
        self.assertEqual(parse_helm_monomers("PEPTIDE1{A.G.L}$$$$"), ["A", "G", "L"])

    def test_bracketed_monomers(self):
        self.assertEqual(
            parse_helm_monomers("PEPTIDE1{[meL].[dP].A}$$$$"), ["meL", "dP", "A"]
        )

    def test_multi_chain(self):
        self.assertEqual(
            parse_helm_monomers("PEPTIDE1{A}|PEPTIDE2{B}$$$$"), ["A", "B"]
        )

    def test_empty(self):
        self.assertIsNone(parse_helm_monomers(""))
        self.assertIsNone(parse_helm_monomers(None))

    def test_no_peptide_block(self):
        self.assertIsNone(parse_helm_monomers("CHEM1{xyz}$$$$"))


class TestGetChainIds(unittest.TestCase):
    def test_single(self):
        self.assertEqual(
            get_peptide_chain_ids("PEPTIDE1{A.B.C}$$$$"), {"PEPTIDE1"}
        )

    def test_multi(self):
        ids = get_peptide_chain_ids(
            "PEPTIDE1{A}|PEPTIDE2{B}$PEPTIDE1,PEPTIDE2,1:R1-1:R1$$$"
        )
        self.assertEqual(ids, {"PEPTIDE1", "PEPTIDE2"})

    def test_empty(self):
        self.assertEqual(get_peptide_chain_ids(""), set())


class TestNormalizeSingleChain(unittest.TestCase):
    def test_renames_peptide2_to_peptide1(self):
        out = normalize_single_chain("PEPTIDE2{A.B.C}$$$$")
        self.assertEqual(out, "PEPTIDE1{A.B.C}$$$$")

    def test_no_op_when_already_peptide1(self):
        s = "PEPTIDE1{A.B.C}$$$$"
        self.assertEqual(normalize_single_chain(s), s)

    def test_renames_in_connection_token_too(self):
        out = normalize_single_chain(
            "PEPTIDE5{A.B.C}$PEPTIDE5,PEPTIDE5,1:R1-3:R2$$$"
        )
        self.assertEqual(out, "PEPTIDE1{A.B.C}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$")

    def test_multi_chain_not_renamed(self):
        """Multi-chain (dimer) molecules must NOT be collapsed."""
        s = "PEPTIDE1{A}|PEPTIDE2{B}$PEPTIDE1,PEPTIDE2,1:R1-1:R1$$$"
        self.assertEqual(normalize_single_chain(s), s)

    def test_multi_chain_starting_at_peptide2_not_renamed(self):
        """Even if the lowest chain id is PEPTIDE2, stay multi-chain."""
        s = "PEPTIDE2{A}|PEPTIDE3{B}$PEPTIDE2,PEPTIDE3,1:R1-1:R1$$$"
        self.assertEqual(normalize_single_chain(s), s)

    def test_double_digit_chain_no_partial_match(self):
        """PEPTIDE2 must not collide with PEPTIDE20 substring."""
        # Synthetic case: only PEPTIDE20 chain present; rename should produce
        # PEPTIDE1 not PEPTIDE10 or partial mangling.
        s = "PEPTIDE20{A.B}$PEPTIDE20,PEPTIDE20,1:R1-2:R2$$$"
        self.assertEqual(
            normalize_single_chain(s),
            "PEPTIDE1{A.B}$PEPTIDE1,PEPTIDE1,1:R1-2:R2$$$",
        )

    def test_idempotent(self):
        s = "PEPTIDE7{A.B.C}$$$$"
        once = normalize_single_chain(s)
        self.assertEqual(once, normalize_single_chain(once))

    def test_empty(self):
        self.assertEqual(normalize_single_chain(""), "")


class TestRemapAltSymbols(unittest.TestCase):
    def setUp(self):
        self.alt = {"meL": "MeLeu", "dP": "DPro", "alpha-Aib": "Aib"}

    def test_simple_remap(self):
        out = remap_alt_symbols("PEPTIDE1{[meL].A.[dP]}$$$$", self.alt)
        self.assertEqual(out, "PEPTIDE1{[MeLeu].A.[DPro]}$$$$")

    def test_unknown_passes_through(self):
        out = remap_alt_symbols("PEPTIDE1{[xyz123].A}$$$$", self.alt)
        self.assertEqual(out, "PEPTIDE1{[xyz123].A}$$$$")

    def test_empty_mapping_no_op(self):
        s = "PEPTIDE1{[meL].A}$$$$"
        self.assertEqual(remap_alt_symbols(s, {}), s)

    def test_preserves_outside_block(self):
        out = remap_alt_symbols(
            "PEPTIDE1{[meL]}$PEPTIDE1,PEPTIDE1,1:R1-1:R2$$$", self.alt
        )
        # Connection block stays intact; only block contents rewritten.
        self.assertEqual(out, "PEPTIDE1{[MeLeu]}$PEPTIDE1,PEPTIDE1,1:R1-1:R2$$$")

    def test_idempotent(self):
        s = "PEPTIDE1{[meL].[dP]}$$$$"
        once = remap_alt_symbols(s, self.alt)
        self.assertEqual(once, remap_alt_symbols(once, self.alt))


class TestCanonicalizeConnections(unittest.TestCase):
    def test_swaps_endpoints_single_chain(self):
        out = canonicalize_connections(
            "PEPTIDE1{A.B.C.D.E.F}$PEPTIDE1,PEPTIDE1,6:R2-1:R1$$$"
        )
        self.assertEqual(
            out, "PEPTIDE1{A.B.C.D.E.F}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$"
        )

    def test_already_canonical(self):
        s = "PEPTIDE1{A.B.C}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$"
        self.assertEqual(canonicalize_connections(s), s)

    def test_swaps_chain_ids_dimer(self):
        """Multi-chain: PEPTIDE2,PEPTIDE1 -> PEPTIDE1,PEPTIDE2 with positions adjusted."""
        out = canonicalize_connections(
            "PEPTIDE1{C.A.B}|PEPTIDE2{D.C.E}$PEPTIDE2,PEPTIDE1,2:R3-1:R3$$$"
        )
        self.assertEqual(
            out,
            "PEPTIDE1{C.A.B}|PEPTIDE2{D.C.E}$PEPTIDE1,PEPTIDE2,1:R3-2:R3$$$",
        )

    def test_chain_blocks_untouched_for_dimer(self):
        """The {...} blocks must remain bit-identical after canonicalization."""
        helm = "PEPTIDE1{X.Y}|PEPTIDE2{Z.W}$PEPTIDE2,PEPTIDE1,1:R1-2:R2$$$"
        out = canonicalize_connections(helm)
        self.assertIn("PEPTIDE1{X.Y}", out)
        self.assertIn("PEPTIDE2{Z.W}", out)

    def test_multiple_connections(self):
        out = canonicalize_connections(
            "PEPTIDE1{A.B.C.D}$PEPTIDE1,PEPTIDE1,4:R2-1:R1|PEPTIDE1,PEPTIDE1,3:R3-2:R3$$$"
        )
        # Both swapped where needed
        self.assertIn("1:R1-4:R2", out)
        self.assertIn("2:R3-3:R3", out)

    def test_wildcard_token_passes_through(self):
        """Wildcard tokens don't match the strict regex; left untouched."""
        s = "PEPTIDE1{A.B.C}$PEPTIDE1,PEPTIDE1,?:R1-3:R2$$$"
        self.assertEqual(canonicalize_connections(s), s)

    def test_no_connection_section(self):
        s = "PEPTIDE1{A.B.C}$$$$"
        self.assertEqual(canonicalize_connections(s), s)

    def test_idempotent(self):
        s = "PEPTIDE1{A.B.C.D.E.F}$PEPTIDE1,PEPTIDE1,6:R2-1:R1$$$"
        once = canonicalize_connections(s)
        self.assertEqual(once, canonicalize_connections(once))


class TestValidateMonomers(unittest.TestCase):
    def test_all_valid(self):
        self.assertTrue(
            validate_monomers("PEPTIDE1{A.G.L}$$$$", {"A", "G", "L"})
        )

    def test_one_invalid(self):
        self.assertFalse(
            validate_monomers("PEPTIDE1{A.X.L}$$$$", {"A", "G", "L"})
        )

    def test_no_peptide_block_returns_false(self):
        self.assertFalse(validate_monomers("CHEM1{xyz}$$$$", {"A"}))


class TestApplyHelmNormalization(unittest.TestCase):
    def setUp(self):
        self.valid = {"A", "G", "L", "MeLeu", "DPro"}
        self.alt = {"meL": "MeLeu", "dP": "DPro"}

    def test_full_pipeline(self):
        df = pd.DataFrame({
            "helm": [
                "PEPTIDE2{A.[meL].[dP]}$PEPTIDE2,PEPTIDE2,3:R2-1:R1$$$",
                "PEPTIDE1{A.G.L}$$$$",
            ]
        })
        out, stats = apply_helm_normalization(df, "helm", self.valid, self.alt)
        self.assertEqual(stats["original"], 2)
        self.assertEqual(stats["final"], 2)
        self.assertEqual(stats["renumbered"], 1)
        self.assertEqual(stats["remapped"], 1)
        self.assertEqual(stats["connection_swapped"], 1)
        self.assertEqual(
            out["helm"].iloc[0],
            "PEPTIDE1{A.[MeLeu].[DPro]}$PEPTIDE1,PEPTIDE1,1:R1-3:R2$$$",
        )

    def test_invalid_monomer_dropped(self):
        df = pd.DataFrame({
            "helm": [
                "PEPTIDE1{A.G.L}$$$$",
                "PEPTIDE1{A.UNKNOWN.L}$$$$",
            ]
        })
        out, stats = apply_helm_normalization(df, "helm", self.valid, self.alt)
        self.assertEqual(stats["invalid_removed"], 1)
        self.assertEqual(len(out), 1)

    def test_null_helm_dropped(self):
        df = pd.DataFrame({"helm": ["PEPTIDE1{A.G.L}$$$$", None, ""]})
        out, stats = apply_helm_normalization(df, "helm", self.valid, self.alt)
        self.assertEqual(stats["null_removed"], 2)
        self.assertEqual(len(out), 1)

    def test_idempotent(self):
        df = pd.DataFrame({
            "helm": ["PEPTIDE2{A.[meL].[dP]}$PEPTIDE2,PEPTIDE2,3:R2-1:R1$$$"]
        })
        once, _ = apply_helm_normalization(df.copy(), "helm", self.valid, self.alt)
        twice, _ = apply_helm_normalization(once.copy(), "helm", self.valid, self.alt)
        self.assertTrue((once["helm"].values == twice["helm"].values).all())


class TestHelmCorrections(unittest.TestCase):
    """Per-compound HELM token rewrites (manual curation overlay)."""

    def setUp(self):
        from helpers.helm_utils import apply_helm_corrections, load_helm_corrections
        self.apply = apply_helm_corrections
        self.load = load_helm_corrections

    def test_load_skips_missing_file(self):
        from pathlib import Path
        result = self.load(Path("/nonexistent/path/corrections.csv"))
        self.assertEqual(result, {})

    def test_load_parses_csv_skipping_comments(self, tmp_path=None):
        import tempfile, csv as _csv
        from pathlib import Path
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("# this is a comment\n")
            f.write("# another\n")
            f.write("compound_chembl_id,original_token,corrected_token,added_by,reason\n")
            f.write("CHEMBL1,[X1],[X1(E)],test,test\n")
            f.write("CHEMBL2,[X2],[X2(Z)],test,test\n")
            path = Path(f.name)
        try:
            result = self.load(path)
            self.assertEqual(result, {
                "CHEMBL1": [("[X1]", "[X1(E)]")],
                "CHEMBL2": [("[X2]", "[X2(Z)]")],
            })
        finally:
            path.unlink()

    def test_apply_rewrites_target_compound_only(self):
        df = pd.DataFrame({
            "compound_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "helm": [
                "PEPTIDE1{[X1].A.G}$$$$",
                "PEPTIDE1{[X1].A.G}$$$$",   # same HELM but different compound — should NOT be rewritten
                "PEPTIDE1{A.[X1].G}$$$$",
            ],
        })
        corr = {"CHEMBL1": [("[X1]", "[X1(E)]")]}
        out, n = self.apply(df, "helm", corr, id_cols=("compound_chembl_id",))
        self.assertEqual(n, 1)
        self.assertEqual(out["helm"].iloc[0], "PEPTIDE1{[X1(E)].A.G}$$$$")
        self.assertEqual(out["helm"].iloc[1], "PEPTIDE1{[X1].A.G}$$$$")  # untouched
        self.assertEqual(out["helm"].iloc[2], "PEPTIDE1{A.[X1].G}$$$$")  # untouched

    def test_apply_skips_dataset_without_id_column(self):
        df = pd.DataFrame({"sequence": ["ABC"], "helm": ["PEPTIDE1{A.B.C}$$$$"]})
        corr = {"CHEMBL1": [("[X1]", "[X1(E)]")]}
        out, n = self.apply(df, "helm", corr, id_cols=("compound_chembl_id",))
        self.assertEqual(n, 0)
        self.assertEqual(out["helm"].iloc[0], "PEPTIDE1{A.B.C}$$$$")

    def test_apply_empty_corrections_is_noop(self):
        df = pd.DataFrame({"compound_chembl_id": ["CHEMBL1"], "helm": ["X"]})
        out, n = self.apply(df, "helm", {}, id_cols=("compound_chembl_id",))
        self.assertEqual(n, 0)
        self.assertTrue((out == df).all().all())


if __name__ == "__main__":
    unittest.main()
