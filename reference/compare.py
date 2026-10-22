"""Write transparent numerical differences against committed R results."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from exonara import test_deu

root = Path(__file__).parent
fixtures = root / "fixtures"
for fixture in fixtures.iterdir():
    if fixture.suffix in {".csv", ".json", ".txt"}:
        fixture.write_bytes(fixture.read_bytes().replace(b"\r\n", b"\n"))
generator = root / "generate.R"
generator.write_bytes(generator.read_bytes().replace(b"\r\n", b"\n"))
counts = pd.read_csv(fixtures / "counts.csv").set_index("feature_id")
genes = counts.pop("gene_id")
condition = pd.read_csv(fixtures / "samples.csv").set_index("sample_id").condition
report = {
    "fixture": "synthetic seed 20261022, 72 tested exons and 2 excluded bins",
    "scope": "six samples, two conditions, DEXSeq mean dispersion trend",
    "comparisons": {},
    "python_runtime": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    },
}
for mode in ("fixed", "estimated"):
    actual = test_deu(counts, genes, condition, dispersions=0.2 if mode == "fixed" else None)
    expected = pd.read_csv(fixtures / f"{mode}_dispersion.csv").set_index("feature_id")
    comparison = {"diagnostics": actual.diagnostics, "columns": {}}
    for col in ("dispersion", "stat", "pvalue", "padj"):
        mask = expected[col].notna() & actual.table[col].notna()
        delta = np.abs(actual.table.loc[mask, col] - expected.loc[mask, col])
        comparison["columns"][col] = {
            "max_absolute_error": float(delta.max()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "compared": int(mask.sum()),
            "max_relative_error_nonzero_reference": float(
                (
                    delta[expected.loc[mask, col] != 0]
                    / np.abs(expected.loc[mask, col][expected.loc[mask, col] != 0])
                ).max()
            ),
        }
    comparison["same_na_pvalues"] = bool(
        np.array_equal(actual.table.pvalue.isna(), expected.pvalue.isna())
    )
    comparison["same_bh_decisions_at_0_05"] = bool(
        np.array_equal(actual.table.padj < 0.05, expected.padj < 0.05)
    )
    report["comparisons"][mode] = comparison
report["fixture_sha256"] = {
    file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
    for file in sorted(fixtures.iterdir())
    if file.is_file() and not file.name.startswith("python_")
}
report["generator_sha256"] = hashlib.sha256(generator.read_bytes()).hexdigest()
(root / "comparison-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["comparisons"], indent=2))
