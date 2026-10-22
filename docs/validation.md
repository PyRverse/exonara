# Validation evidence

`reference/generate.R` runs actual DEXSeq and DESeq2. The generated counts, sample
design, size factors, dispersion components, fixed-dispersion results,
estimated-dispersion results, version metadata, and session information are
committed under `reference/fixtures/`. Inputs are synthetic, generated with R seed
20261022, with an explicit strong usage change and a high-dispersion outlier.
Both rejected and non-rejected BH decisions and retention of the dispersion
outlier are checked. No human or restricted research data are included.

Run from the repository root, with DEXSeq, DESeq2, and jsonlite installed:

```sh
Rscript reference/generate.R
python reference/compare.py
python -m pytest
```

Use `R_LIBS_USER` to point R at a separately managed library when needed. Package
versions and platform are recorded in the fixture metadata; pin those versions
to reproduce this reference. The source audit also records official release
revisions and SHA-256 hashes. No R installation is needed for routine Python CI,
which compares against committed outputs.

Two comparisons separate the layers of inference:

1. Known dispersion 0.2 isolates size-factor estimation and the paired full/reduced
   negative-binomial likelihood-ratio test.
2. Estimated dispersions exercise count-to-result behavior with DEXSeq's mean
   trend, empirical prior variance, MAP shrinkage, and LRT.

The report records maximum absolute error and RMSE for dispersion, statistics,
p-values, and BH-adjusted values, along with missing-value and 0.05 BH-decision
agreement. Fixture hashes prevent ambiguous provenance. Python-generated
comparison results are not used as the reference test oracle.

The fixed-dispersion tests allow absolute statistical error 1e-6 and p-value error
2e-6. End-to-end tests allow absolute statistical error 1e-4 and p-value error
3e-6, with relative tolerance 1e-4. Small absolute tolerances matter for near-zero
statistics and near-one p-values; relative checks are retained for effectful
tests. These are regression thresholds for this fixture and its known optimizer
differences, not universal scientific tolerances.

Additional tests verify sample-label alignment, condition reversal, identical
tests for complementary bins, a gene-expression-only change with unchanged exon
usage, missing/invalid inputs, insufficient replication, zero bins, and complete
separation. Excluded sparse cases do not acquire fabricated p-values.

Before wider research use, validate against real datasets, more library-size and
dispersion regimes, known null simulations, rejection behavior, optimizer
failures, and external reproducibility on supported platforms. The current
synthetic fixture is deliberately limited and does not establish all-model parity.
