import numpy as np
import pandas as pd
import pytest

from exonara import estimate_size_factors, model, test_deu


@pytest.fixture
def small():
    samples = list("abcdef")
    counts = pd.DataFrame(
        [[10, 20, 30, 40, 60, 80], [20, 40, 60, 80, 120, 160]], index=["e1", "e2"], columns=samples
    )
    genes = pd.Series(["g", "g"], index=counts.index)
    condition = pd.Series([0, 0, 0, 1, 1, 1], index=samples)
    return counts, genes, condition


def test_gene_expression_change_is_not_exon_usage_change(small):
    result = test_deu(*small, dispersions=0.2).table
    assert (result.status == "ok").all()
    np.testing.assert_allclose(result.stat, 0, atol=1e-9)
    np.testing.assert_allclose(result.log2_usage_odds_ratio, 0, atol=1e-7)


def test_two_bin_complement_has_identical_likelihood_statistic(small):
    counts, genes, condition = small
    counts.iloc[0, 3:] *= 4
    result = test_deu(counts, genes, condition, dispersions=0.2).table
    assert result.stat.iloc[0] > 0
    np.testing.assert_allclose(result.stat.iloc[0], result.stat.iloc[1], atol=1e-8)


@pytest.mark.parametrize("bad", [-1, 1.5, np.nan, np.inf, 2**32])
def test_invalid_counts_rejected(small, bad):
    counts, genes, condition = small
    counts = counts.astype(float)
    counts.iloc[0, 0] = bad
    with pytest.raises(ValueError, match="counts"):
        test_deu(counts, genes, condition, dispersions=0.2)


def test_missing_sample_metadata_rejected(small):
    counts, genes, condition = small
    with pytest.raises(ValueError, match="identifiers"):
        test_deu(counts, genes, condition.iloc[:-1], dispersions=0.2)


def test_unreplicated_condition_rejected(small):
    counts, genes, condition = small
    condition[:] = [0, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="two samples"):
        test_deu(counts, genes, condition, dispersions=0.2)


def test_low_residual_df_requires_explicit_prior(small):
    counts, genes, condition = small
    subset = ["a", "b", "d", "e"]
    with pytest.raises(ValueError, match="prior_variance"):
        test_deu(counts[subset], genes, condition[subset])


def test_sparse_ratio_normalization_requires_explicit_factors(small):
    counts, _, _ = small
    counts.iloc[:, 0] = 0
    with pytest.raises(ValueError, match="no exon is positive"):
        estimate_size_factors(counts)


def test_zero_exon_and_single_bin_are_not_tested(small):
    counts, genes, condition = small
    counts.loc["absent"] = 0
    genes.loc["absent"] = "g"
    counts.loc["single"] = 10
    genes.loc["single"] = "singleton"
    result = test_deu(counts, genes, condition, dispersions=0.2).table
    assert result.loc[["absent", "single"], "pvalue"].isna().all()
    assert (result.loc[["absent", "single"], "status"] == "zero_exon_or_others").all()


def test_complete_separation_is_reported(small):
    counts, genes, condition = small
    counts.iloc[0, :3] = 0
    result = test_deu(counts, genes, condition, dispersions=0.2).table
    assert (result.status == "condition_boundary").all()
    assert result.pvalue.isna().all()


def test_all_zero_counts_with_explicit_factors_return_missing_results(small):
    counts, genes, condition = small
    counts.iloc[:, :] = 0
    factors = pd.Series(1.0, index=counts.columns)
    result = test_deu(counts, genes, condition, size_factors=factors, dispersions=0.2).table
    assert result.pvalue.isna().all()
    assert (result.status == "zero_gene_sample").all()


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan, np.nextafter(12.0, np.inf), 1e308])
def test_invalid_dispersions_rejected(small, bad):
    with pytest.raises(ValueError, match="dispersions"):
        test_deu(*small, dispersions=bad)


def test_supplied_dispersion_upper_bound_is_inclusive(small):
    result = test_deu(*small, dispersions=12.0).table
    assert (result.status == "ok").all()
    assert np.isfinite(result.pvalue).all()


@pytest.mark.parametrize("invalid_likelihood", [np.nan, np.inf, -np.inf])
def test_numerical_likelihood_failure_never_yields_valid_pvalues(
    small, monkeypatch, invalid_likelihood
):
    monkeypatch.setattr(model, "_log_likelihood", lambda *args: invalid_likelihood)
    result = test_deu(*small, dispersions=0.2).table
    assert (result.status == "glm_fit_failed").all()
    assert result[["stat", "pvalue", "padj"]].isna().all().all()


def test_overflowing_likelihood_ratio_never_becomes_null_result(small, monkeypatch):
    def finite_fit_with_overflowing_difference(y, x, offset, alpha):
        # Both model likelihoods are finite; their subtraction is not.
        ll = 1e308 if x.shape[1] == 8 else -1e308
        return model._Fit(np.zeros(x.shape[1]), np.ones(len(y)), ll, True)

    monkeypatch.setattr(model, "_fit_glm", finite_fit_with_overflowing_difference)
    result = test_deu(*small, dispersions=0.2).table
    assert (result.status == "nonfinite_likelihood").all()
    assert result[["stat", "pvalue", "padj"]].isna().all().all()
