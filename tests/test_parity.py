"""Regression evidence against actual DEXSeq/DESeq2 execution, not Python answers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exonara import estimate_size_factors, test_deu

FIXTURES = Path(__file__).parents[1] / "reference" / "fixtures"


def test_reference_provenance_and_integrity():
    root = FIXTURES.parent
    report = json.loads((root / "comparison-report.json").read_text())
    for name, expected in report["fixture_sha256"].items():
        assert "\\" not in name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256((root / "generate.R").read_bytes()).hexdigest()
        == report["generator_sha256"]
    )
    metadata = json.loads((FIXTURES / "metadata.json").read_text())
    assert metadata["observed_results"] is True
    assert metadata["DEXSeq"] == "1.58.0"
    assert metadata["DESeq2"] == "1.52.0"


@pytest.fixture
def inputs():
    counts = pd.read_csv(FIXTURES / "counts.csv").set_index("feature_id")
    genes = counts.pop("gene_id")
    condition = pd.read_csv(FIXTURES / "samples.csv").set_index("sample_id").condition
    return counts, genes, condition


def test_size_factors_match_r(inputs):
    expected = pd.read_csv(FIXTURES / "size_factors.csv").set_index("sample_id").size_factor
    np.testing.assert_allclose(estimate_size_factors(inputs[0]), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["fixed", "estimated"])
def test_full_reduced_nb_lrt_against_dexseq(inputs, mode):
    result = test_deu(*inputs, dispersions=0.2 if mode == "fixed" else None).table
    expected = pd.read_csv(FIXTURES / f"{mode}_dispersion.csv").set_index("feature_id")
    np.testing.assert_array_equal(result.pvalue.isna(), expected.pvalue.isna())
    valid = expected.pvalue.notna()
    assert valid.sum() == 72
    assert (expected.loc[valid, "padj"] < 0.05).any()
    assert (expected.loc[valid, "padj"] >= 0.05).any()
    if mode == "fixed":
        np.testing.assert_allclose(
            result.loc[valid, "stat"], expected.loc[valid, "stat"], atol=1e-6, rtol=1e-5
        )
        np.testing.assert_allclose(
            result.loc[valid, "pvalue"], expected.loc[valid, "pvalue"], atol=2e-6, rtol=1e-5
        )
    else:
        components = pd.read_csv(FIXTURES / "dispersion_components.csv").set_index("feature_id")
        assert result.loc[valid, "dispersion_outlier"].any()
        np.testing.assert_array_equal(
            result.loc[valid, "dispersion_outlier"], components.loc[valid, "outlier"].astype(bool)
        )
        np.testing.assert_allclose(
            result.loc[valid, "dispersion"],
            expected.loc[valid, "dispersion"],
            atol=1e-6,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            result.loc[valid, "stat"], expected.loc[valid, "stat"], atol=1e-4, rtol=1e-4
        )
        np.testing.assert_allclose(
            result.loc[valid, "pvalue"], expected.loc[valid, "pvalue"], atol=3e-6, rtol=1e-4
        )
    np.testing.assert_allclose(
        result.loc[valid, "padj"], expected.loc[valid, "padj"], atol=1e-5, rtol=1e-4
    )
    np.testing.assert_array_equal(
        result.loc[valid, "padj"] < 0.05, expected.loc[valid, "padj"] < 0.05
    )


def test_identifier_alignment_and_condition_reversal(inputs):
    counts, genes, condition = inputs
    original = test_deu(counts, genes, condition, dispersions=0.2).table
    reversed_conditions = 1 - condition.iloc[::-1]
    reversed_result = test_deu(
        counts, genes.iloc[::-1], reversed_conditions, dispersions=0.2
    ).table
    np.testing.assert_allclose(original.stat, reversed_result.stat, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(
        original.log2_usage_odds_ratio,
        -reversed_result.log2_usage_odds_ratio,
        atol=1e-6,
        equal_nan=True,
    )
