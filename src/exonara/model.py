"""Two-condition DEXSeq-style exon-versus-other-exons negative-binomial models.

Algorithmic adaptation informed by DEXSeq and DESeq2 (see NOTICE/PROVENANCE).
Original Python implementation, GPL-3.0-or-later.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import digamma, polygamma
from scipy.stats import chi2, nbinom, trim_mean

FloatArray = NDArray[np.float64]
MIN_DISPERSION = 1e-8


@dataclass
class DEUResult:
    """Per-exon results and the diagnostics needed to interpret the analysis."""

    table: pd.DataFrame
    size_factors: pd.Series
    diagnostics: dict


@dataclass
class _Fit:
    beta: FloatArray
    mu: FloatArray
    log_likelihood: float
    converged: bool


def _counts(values: pd.DataFrame) -> FloatArray:
    if not isinstance(values, pd.DataFrame) or values.empty:
        raise ValueError("counts must be a nonempty exon-by-sample DataFrame")
    if not values.index.is_unique or not values.columns.is_unique:
        raise ValueError("exon and sample identifiers must be unique")
    if values.index.hasnans or values.columns.hasnans:
        raise ValueError("exon and sample identifiers cannot be missing")
    data = values.to_numpy(dtype=float)
    if not np.isfinite(data).all() or (data < 0).any() or (data != np.floor(data)).any():
        raise ValueError("counts must contain finite, nonnegative integers")
    if (data > 2**31 - 1).any():
        raise ValueError("counts exceed the supported signed 32-bit integer range")
    return data


def estimate_size_factors(counts: pd.DataFrame) -> pd.Series:
    """DEXSeq/DESeq2 median of log ratios, using exons positive in every sample.

    No additional geometric-mean rescaling is applied, matching upstream's
    default ratio estimator with internally computed geometric means.
    """
    data = _counts(counts)
    positive = np.all(data > 0, axis=1)
    if not positive.any():
        raise ValueError("no exon is positive in every sample; supply explicit size_factors")
    log_counts = np.log(data[positive])
    factors = np.exp(np.median(log_counts - log_counts.mean(axis=1, keepdims=True), axis=0))
    return pd.Series(factors, index=counts.columns, name="size_factor")


def _aligned(series: pd.Series, index: pd.Index, name: str) -> pd.Series:
    if not isinstance(series, pd.Series) or not series.index.is_unique:
        raise ValueError(f"{name} must be a Series with unique identifiers")
    if len(series) != len(index) or set(series.index) != set(index):
        raise ValueError(f"{name} identifiers must exactly match the corresponding count axis")
    result = series.reindex(index)
    if result.isna().any():
        raise ValueError(f"{name} cannot contain missing values")
    return result


def _design(condition: FloatArray) -> tuple[FloatArray, FloatArray]:
    n = len(condition)
    sample = np.vstack((np.eye(n), np.eye(n)))
    exon = np.r_[np.ones(n), np.zeros(n)]
    reduced = np.column_stack((sample, exon))
    full = np.column_stack((reduced, exon * np.tile(condition, 2)))
    return full, reduced


def _log_likelihood(y: FloatArray, mu: FloatArray, alpha: float) -> float:
    return float(nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu)).sum())


def _fit_glm(y: FloatArray, x: FloatArray, offset: FloatArray, alpha: float) -> _Fit:
    """Unpenalized NB log-link MLE using damped Newton steps and observed Hessian."""
    beta = np.linalg.lstsq(x, np.log(y + 0.1) - offset, rcond=None)[0]
    for _ in range(150):
        mu = np.exp(np.clip(x @ beta + offset, -40, 40))
        ll = _log_likelihood(y, mu, alpha)
        if not np.isfinite(ll):
            return _Fit(beta, mu, ll, False)
        score = x.T @ ((y - mu) / (1 + alpha * mu))
        if not np.isfinite(score).all():
            return _Fit(beta, mu, ll, False)
        if np.max(np.abs(score)) < 1e-7:
            return _Fit(beta, mu, ll, True)
        weights = mu * (1 + alpha * y) / (1 + alpha * mu) ** 2
        try:
            step = np.linalg.solve(x.T @ (weights[:, None] * x), score)
        except np.linalg.LinAlgError:
            return _Fit(beta, mu, ll, False)
        accepted = False
        for power in range(30):
            candidate = beta + step * 0.5**power
            eta = x @ candidate + offset
            if np.any(np.abs(eta) > 40):
                continue
            candidate_mu = np.exp(eta)
            candidate_ll = _log_likelihood(y, candidate_mu, alpha)
            if np.isfinite(candidate_ll) and candidate_ll >= ll - 1e-10:
                beta = candidate
                accepted = True
                if np.max(np.abs(step * 0.5**power)) < 1e-7:
                    return _Fit(beta, candidate_mu, candidate_ll, True)
                break
        if not accepted:
            return _Fit(beta, mu, ll, np.max(np.abs(score)) < 1e-5)
    mu = np.exp(np.clip(x @ beta + offset, -40, 40))
    return _Fit(beta, mu, _log_likelihood(y, mu, alpha), False)


def _cr_objective(
    log_alpha: float,
    y: FloatArray,
    x: FloatArray,
    mu: FloatArray,
    prior_mean: float | None = None,
    prior_variance: float | None = None,
) -> float:
    alpha = float(np.exp(log_alpha))
    weights = mu / (1 + alpha * mu)
    sign, logdet = np.linalg.slogdet(x.T @ (weights[:, None] * x))
    if sign <= 0:
        return np.inf
    value = -_log_likelihood(y, mu, alpha) + 0.5 * logdet
    if prior_mean is not None:
        value += (log_alpha - np.log(prior_mean)) ** 2 / (2 * prior_variance)
    return float(value)


def _dispersion_score(
    log_alpha: float,
    y: FloatArray,
    x: FloatArray,
    mu: FloatArray,
    prior_mean: float | None,
    prior_variance: float | None,
) -> float:
    alpha = np.exp(log_alpha)
    weights = mu / (1 + alpha * mu)
    information = x.T @ (weights[:, None] * x)
    derivative = x.T @ ((-(weights**2))[:, None] * x)
    cr_score = -0.5 * np.trace(np.linalg.solve(information, derivative))
    ll_score = (
        np.sum(
            digamma(1 / alpha)
            + np.log1p(mu * alpha)
            - mu * alpha / (1 + mu * alpha)
            - digamma(y + 1 / alpha)
            + y / (mu + 1 / alpha)
        )
        / alpha**2
    )
    score = (ll_score + cr_score) * alpha
    if prior_mean is not None:
        score -= (log_alpha - np.log(prior_mean)) / prior_variance
    return float(score)


def _dispersion_line_search(
    initial: float,
    y: FloatArray,
    x: FloatArray,
    mu: FloatArray,
    prior_mean: float | None = None,
    prior_variance: float | None = None,
) -> tuple[float, int]:
    """Adapt DESeq2's Armijo search and stopping rules; see NOTICE."""
    args = (y, x, mu, prior_mean, prior_variance)
    a = float(np.log(initial))
    lp = -_cr_objective(a, *args)
    score = _dispersion_score(a, *args)
    kappa, accepted = 1.0, 0
    for _iteration in range(1, 101):
        if a + kappa * score < -30:
            kappa = (-30 - a) / score
        if a + kappa * score > 10:
            kappa = (10 - a) / score
        candidate = a + kappa * score
        lp_new = -_cr_objective(candidate, *args)
        if lp_new >= lp + kappa * 1e-4 * score**2:
            accepted += 1
            a = candidate
            if lp_new - lp < 1e-6 or a < np.log(MIN_DISPERSION / 10):
                break
            lp = lp_new
            score = _dispersion_score(a, *args)
            kappa = min(kappa * 1.1, 1.0)
            if accepted % 5 == 0:
                kappa /= 2
        else:
            kappa /= 2
    return float(np.exp(a)), _iteration


def _dispersion_grid(
    y: FloatArray,
    x: FloatArray,
    mu: FloatArray,
    maximum: float,
    prior_mean: float | None = None,
    prior_variance: float | None = None,
) -> float:
    args = (y, x, mu, prior_mean, prior_variance)
    grid = np.linspace(np.log(MIN_DISPERSION), np.log(maximum), 20)
    best = min(grid, key=lambda a: _cr_objective(a, *args))
    delta = grid[1] - grid[0]
    fine = np.linspace(best - delta, best + delta, 20)
    return float(np.exp(min(fine, key=lambda a: _cr_objective(a, *args))))


def _raw_dispersion(y: FloatArray, x: FloatArray, offset: FloatArray) -> tuple[float, _Fit]:
    normalized = y / np.exp(offset)
    linear_mu = np.maximum(x @ np.linalg.lstsq(x, normalized, rcond=None)[0], 1)
    rough = max(
        0,
        float(
            np.sum(((normalized - linear_mu) ** 2 - linear_mu) / linear_mu**2)
            / (len(y) - x.shape[1])
        ),
    )
    mean = normalized.mean()
    moments = (normalized.var(ddof=1) - np.mean(np.exp(-offset)) * mean) / mean**2
    maximum = max(10, len(y))
    alpha = float(np.clip(min(rough, moments), MIN_DISPERSION, maximum))
    for _ in range(10):
        fit = _fit_glm(y, x, offset, alpha)
        if not fit.converged:
            return np.nan, fit
        mu = np.maximum(fit.mu, 0.5)
        new_alpha, iterations = _dispersion_line_search(alpha, y, x, mu)
        new_alpha = min(new_alpha, maximum)
        changed = abs(np.log(new_alpha) - np.log(alpha))
        alpha = new_alpha
        if changed <= 0.05:
            break
    # Upstream MAP conditions on this final alternating fit, rather than
    # refitting means at each candidate dispersion.
    if (iterations == 1 or iterations == 100) and alpha > MIN_DISPERSION * 10:
        alpha = _dispersion_grid(y, x, mu, maximum)
    fit.mu = mu
    return float(np.clip(alpha, MIN_DISPERSION, maximum)), fit


def _bh(pvalues: FloatArray) -> FloatArray:
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid], kind="stable")]
    if len(order):
        ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def test_deu(
    counts: pd.DataFrame,
    gene_ids: pd.Series,
    condition: pd.Series,
    *,
    size_factors: pd.Series | None = None,
    dispersions: pd.Series | float | None = None,
    prior_variance: float | None = None,
) -> DEUResult:
    """Test condition-dependent exon usage with paired full/reduced NB models.

    ``condition`` must contain numeric 0 and 1, each with at least two samples.
    Automatic dispersion estimation requires six samples total (residual df>=4),
    unless ``prior_variance`` is supplied. Dispersion trend is the upstream
    ``fitType='mean'`` option. Supplied dispersions must lie in the inclusive
    range [1e-8, max(10, 2 * number_of_samples)]. BH adjustment covers finite exon
    p-values; neither independent filtering nor Cook's-distance filtering is performed.
    """
    data = _counts(counts)
    genes = _aligned(gene_ids, counts.index, "gene_ids")
    conditions = _aligned(condition, counts.columns, "condition").to_numpy(dtype=float)
    if set(conditions) != {0.0, 1.0} or min(np.bincount(conditions.astype(int))) < 2:
        raise ValueError("condition must be numeric 0/1 with at least two samples per condition")
    factors = (
        estimate_size_factors(counts)
        if size_factors is None
        else _aligned(size_factors, counts.columns, "size_factors").astype(float)
    )
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("size_factors must be finite and positive")
    if prior_variance is not None and (not np.isfinite(prior_variance) or prior_variance <= 0):
        raise ValueError("prior_variance must be finite and positive")
    n_exons, n_samples = data.shape
    full, reduced = _design(conditions)
    residual_df = len(full) - full.shape[1]
    if dispersions is None and residual_df <= 3 and prior_variance is None:
        raise ValueError("automatic prior estimation requires >=6 samples; supply prior_variance")
    others = np.empty_like(data)
    statuses = np.full(n_exons, "ok", dtype=object)
    for gene in pd.unique(genes):
        idx = np.flatnonzero(genes.to_numpy() == gene)
        others[idx] = data[idx].sum(axis=0) - data[idx]
        if len(idx) == 1:
            statuses[idx] = "single_exon_gene"
    statuses[(data.sum(axis=1) == 0) | (others.sum(axis=1) == 0)] = "zero_exon_or_others"
    statuses[np.any(data + others == 0, axis=1)] = "zero_gene_sample"
    # Complete separation requires boundary inference not implemented in this alpha.
    boundary = np.zeros(n_exons, dtype=bool)
    for group in (0, 1):
        boundary |= data[:, conditions == group].sum(axis=1) == 0
        boundary |= others[:, conditions == group].sum(axis=1) == 0
    statuses[(statuses == "ok") & boundary] = "condition_boundary"
    paired = np.hstack((data, others))
    offset = np.log(np.tile(factors.to_numpy(), 2))
    raw = np.full(n_exons, np.nan)
    final = np.full(n_exons, np.nan)
    trend = np.full(n_exons, np.nan)
    outlier = np.zeros(n_exons, dtype=bool)
    fitted: dict[int, _Fit] = {}
    eligible = np.flatnonzero(statuses == "ok")
    diagnostics = {
        "method": "exon-versus-others NB LRT",
        "residual_df": residual_df,
        "dispersion_fit": "supplied" if dispersions is not None else "mean",
        "independent_filtering": False,
        "cooks_filtering": False,
        "prior_variance": prior_variance,
    }
    if dispersions is not None:
        if np.isscalar(dispersions):
            final[:] = float(dispersions)
        else:
            final[:] = _aligned(dispersions, counts.index, "dispersions").to_numpy(dtype=float)
        maximum = max(10, 2 * n_samples)
        if (
            not np.isfinite(final).all()
            or (final < MIN_DISPERSION).any()
            or (final > maximum).any()
        ):
            raise ValueError(f"dispersions must be finite and within [1e-8, {maximum}]")
    elif len(eligible):
        for i in eligible:
            raw[i], fitted[i] = _raw_dispersion(paired[i], full, offset)
            if not np.isfinite(raw[i]):
                statuses[i] = "dispersion_fit_failed"
        valid = np.flatnonzero((statuses == "ok") & (raw > 100 * MIN_DISPERSION))
        if len(valid) < 2:
            raise ValueError("insufficient non-boundary dispersion estimates; supply dispersions")
        mean_values = raw[(statuses == "ok") & (raw > 10 * MIN_DISPERSION)]
        mean_dispersion = float(trim_mean(mean_values, 0.001))
        trend[eligible] = mean_dispersion
        residuals = np.log(raw[valid]) - np.log(mean_dispersion)
        variance = float((1.4826 * np.median(np.abs(residuals - np.median(residuals)))) ** 2)
        if prior_variance is None:
            prior_variance = max(variance - float(polygamma(1, residual_df / 2)), 0.25)
        diagnostics.update(
            prior_variance=prior_variance,
            dispersion_trend=mean_dispersion,
            variance_log_dispersions=variance,
        )
        for i in eligible:
            if statuses[i] != "ok":
                continue
            outlier[i] = np.log(raw[i]) > np.log(mean_dispersion) + 2 * np.sqrt(variance)
            initial = raw[i] if raw[i] > 0.1 * mean_dispersion else mean_dispersion
            map_alpha, iterations = _dispersion_line_search(
                initial, paired[i], full, fitted[i].mu, mean_dispersion, prior_variance
            )
            if iterations == 100:
                map_alpha = _dispersion_grid(
                    paired[i],
                    full,
                    fitted[i].mu,
                    max(10, 2 * n_samples),
                    mean_dispersion,
                    prior_variance,
                )
            final[i] = (
                raw[i]
                if outlier[i]
                else np.clip(map_alpha, MIN_DISPERSION, max(10, 2 * n_samples))
            )
            final[i] = min(final[i], 2 * n_samples)
    statistics = np.full(n_exons, np.nan)
    pvalues = np.full(n_exons, np.nan)
    effects = np.full(n_exons, np.nan)
    for i in np.flatnonzero(statuses == "ok"):
        fit_full = _fit_glm(paired[i], full, offset, final[i])
        fit_reduced = _fit_glm(paired[i], reduced, offset, final[i])
        if not fit_full.converged or not fit_reduced.converged:
            statuses[i] = "glm_fit_failed"
            continue
        if not np.isfinite([fit_full.log_likelihood, fit_reduced.log_likelihood]).all():
            statuses[i] = "nonfinite_likelihood"
            continue
        if min(fit_full.mu.min(), fit_reduced.mu.min()) < 0.5:
            statuses[i] = "fitted_mean_below_supported_floor"
            continue
        stat = 2 * (fit_full.log_likelihood - fit_reduced.log_likelihood)
        if not np.isfinite(stat):
            statuses[i] = "nonfinite_likelihood"
            continue
        if stat < -1e-6:
            statuses[i] = "likelihood_order_failure"
            continue
        statistics[i] = max(0, stat)
        pvalues[i] = chi2.sf(statistics[i], 1)
        effects[i] = fit_full.beta[-1] / np.log(2)
    table = pd.DataFrame(
        {
            "gene_id": genes,
            "exon_base_mean": (data / factors.to_numpy()).mean(axis=1),
            "dispersion_raw": raw,
            "dispersion_trend": trend,
            "dispersion": final,
            "dispersion_outlier": outlier,
            "log2_usage_odds_ratio": effects,
            "stat": statistics,
            "pvalue": pvalues,
            "padj": _bh(pvalues),
            "status": statuses,
        },
        index=counts.index,
    )
    diagnostics["status_counts"] = table.status.value_counts().to_dict()
    return DEUResult(table, factors.rename("size_factor"), diagnostics)


# Prevent pytest from collecting the public API when imported into test modules.
test_deu.__test__ = False
