"""Run from the repository root after installing Exonara."""

from pathlib import Path

import pandas as pd

from exonara import test_deu

root = Path(__file__).parents[1] / "reference" / "fixtures"
counts = pd.read_csv(root / "counts.csv").set_index("feature_id")
gene_ids = counts.pop("gene_id")
condition = pd.read_csv(root / "samples.csv").set_index("sample_id").condition
result = test_deu(counts, gene_ids, condition)
print(result.table.sort_values("pvalue").head(10).to_string())
print(result.diagnostics)
