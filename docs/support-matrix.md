# Supported behavior

| Area | Initial support and limits |
| --- | --- |
| Inputs | Prepared exon/bin × sample integer DataFrame; unique identifiers; finite counts ≤2³¹−1 |
| Gene map | One gene per counting bin; all rows required; no ambiguity resolution |
| Design | Numeric condition 0/1; independent biological replicates; ≥2 per group |
| Model | Full sample + exon + condition:exon versus reduced sample + exon; 1-df NB LRT |
| Normalization | Upstream default ratio estimator over exons positive in every sample; or user-supplied positive factors |
| Raw dispersion | Cox–Reid adjustment; alternating fitted means/dispersion, up to 10 iterations; upstream Armijo search and grid fallback |
| Trend | DEXSeq `fitType="mean"`, a 0.1% trimmed mean; no parametric/local trend |
| Shrinkage | Log-normal MAP prior; MAD/trigamma variance estimator for residual df ≥4; minimum variance 0.25; high-dispersion outliers retained |
| Smaller designs | Explicit prior variance or dispersions required; upstream low-df Monte Carlo prior estimator is not implemented |
| Testing | Supplied or estimated dispersions; same final dispersion in full/reduced GLMs; χ² tail with one degree of freedom |
| Supplied dispersion range | Finite values from 1e-8 through max(10, 2 × number of samples), inclusive; larger values are rejected before fitting |
| Adjustment | BH over finite exon p-values; no independent filtering, Cook's filter, gene-level aggregation, or stage-wise FDR |
| Effect | Change in this/other exon log₂ odds; not DEXSeq's alternative fold-change estimator |
| Zero exon / zero others | Missing test result with `zero_exon_or_others` status |
| Zero gene count in a sample | `zero_gene_sample`; excluded rather than silently regularized |
| Complete separation | `condition_boundary`; one condition has no this or no others counts; excluded |
| Very small fitted means | Fits below 0.5 are excluded; boundary behavior is not claimed equivalent to upstream |
| Failed fit | Explicit failure status and missing statistic/p-value; nonfinite likelihoods or LRT statistics never produce valid p-values |
| Other designs | No covariates, paired-subject effects, >2 conditions, continuous predictors, or observation weights |
| Input preparation | No GTF flattening, BAM/CRAM counting, transcript quantification, or annotation download |

Library-size effects are represented as offsets, while sample coefficients
absorb gene-level changes. Counts of other exons are summed within each gene,
excluding the tested exon. The same gene can therefore generate correlated tests;
the reported adjusted values follow the stated BH procedure, not a separate
gene-level error-control guarantee.

The GLM fitter uses an unpenalized, damped Newton optimizer with an observed
Hessian. DESeq2 uses a small coefficient ridge and Fisher scoring with fallback
optimization. This implementation difference produces small numerical differences
on the documented fixtures; it is not hidden by an exact-parity claim.
