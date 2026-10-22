"""Differential exon usage from prepared count tables; alpha research software."""

from exonara.model import DEUResult, estimate_size_factors, test_deu

__version__ = "0.1.0a1"
__all__ = ["DEUResult", "estimate_size_factors", "test_deu"]
