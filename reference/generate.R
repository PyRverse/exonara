# SPDX-License-Identifier: GPL-3.0-or-later
# Run from repository root with Rscript reference/generate.R.
# Set R_LIBS_USER externally when using a non-default Bioconductor library.
suppressPackageStartupMessages(library(DEXSeq))
suppressPackageStartupMessages(library(jsonlite))
dir.create("reference/fixtures", recursive=TRUE, showWarnings=FALSE)
set.seed(20261022)
n_genes <- 24L
n_samples <- 6L
gene_ids <- rep(sprintf("g%02d", seq_len(n_genes)), each=3)
feature_ids <- sprintf("e%03d", seq_along(gene_ids))
sample_ids <- paste0("s", seq_len(n_samples))
sample_data <- data.frame(condition=factor(rep(c("control", "treated"), each=3)),
                          row.names=sample_ids)
library_sizes <- c(0.7, 1.0, 1.3, 0.8, 1.1, 1.4)
means <- runif(length(gene_ids), 80, 500)
alphas <- runif(length(gene_ids), 0.08, 0.35)
counts <- matrix(0L, nrow=length(gene_ids), ncol=n_samples,
                 dimnames=list(feature_ids, sample_ids))
for (i in seq_len(nrow(counts))) {
    mu <- means[i] * library_sizes
    if (i %in% c(1, 7, 19)) mu[4:6] <- mu[4:6] * 4
    counts[i,] <- rnbinom(n_samples, mu=mu, size=1/alphas[i])
}
# Include a strong usage change so FDR-decision agreement is not all-negative.
counts[1, 4:6] <- counts[1, 4:6] * 8L
# Include an unusually variable exon to exercise retention of dispersion outliers.
counts[12, ] <- c(5L, 20000L, 5L, 5L, 20000L, 5L)
# An absent exon and a single-bin gene exercise upstream NA behavior.
counts <- rbind(counts, e_zero=rep(0L, n_samples), e_single=c(40L,50L,60L,70L,80L,90L))
gene_ids <- c(gene_ids, "g24", "single")
feature_ids <- rownames(counts)
write.csv(data.frame(feature_id=feature_ids, gene_id=gene_ids, counts,
                     check.names=FALSE), "reference/fixtures/counts.csv", row.names=FALSE)
write.csv(data.frame(sample_id=sample_ids, condition=rep(0:1, each=3)),
          "reference/fixtures/samples.csv", row.names=FALSE)
make_object <- function() {
    dxd <- DEXSeqDataSet(counts, sample_data, design=~sample+exon+condition:exon,
                         featureID=feature_ids, groupID=gene_ids)
    estimateSizeFactors(dxd)
}
dxd <- make_object()
write.csv(data.frame(sample_id=sample_ids, size_factor=sizeFactors(dxd)[seq_len(n_samples)]),
          "reference/fixtures/size_factors.csv", row.names=FALSE)
save_result <- function(object, filename) {
    result <- results(object, independentFiltering=FALSE, cooksCutoff=FALSE)
    write.csv(data.frame(feature_id=mcols(object)$featureID,
                         dispersion=dispersions(object), stat=result$stat,
                         pvalue=result$pvalue, padj=result$padj), filename,
              row.names=FALSE, na="NA")
}
# Explicitly mark non-testable exon/others pairs, as estimateDispersions does.
dxd <- DESeq2:::getBaseMeansAndVariances(dxd)
mcols(dxd)$allZero <- rowSums(featureCounts(dxd)) == 0 |
                     rowSums(counts(dxd)[,colData(dxd)$exon == "others"]) == 0
dispersions(dxd) <- rep(0.2, nrow(dxd))
dxd <- testForDEU(dxd)
save_result(dxd, "reference/fixtures/fixed_dispersion.csv")
estimated <- estimateDispersions(make_object(), fitType="mean", quiet=TRUE)
write.csv(data.frame(feature_id=mcols(estimated)$featureID,
                     raw=mcols(estimated)$dispGeneEst,
                     trend=mcols(estimated)$dispFit,
                     outlier=mcols(estimated)$dispOutlier),
          "reference/fixtures/dispersion_components.csv", row.names=FALSE, na="NA")
estimated <- testForDEU(estimated)
save_result(estimated, "reference/fixtures/estimated_dispersion.csv")
writeLines(capture.output(sessionInfo()), "reference/fixtures/session-info.txt")
metadata <- list(generator="reference/generate.R", seed=20261022,
                 R=as.character(getRversion()), DEXSeq=as.character(packageVersion("DEXSeq")),
                 DESeq2=as.character(packageVersion("DESeq2")),
                 fit_type="mean", independent_filtering=FALSE, cooks_cutoff=FALSE,
                 fixed_dispersion=0.2,
                 estimated_prior_variance=DESeq2:::estimateDispersionsPriorVar(
                     estimated, modelMatrix=DEXSeq:::rmDepCols(model.matrix(
                         design(estimated), as.data.frame(colData(estimated))))),
                 synthetic_data=TRUE, observed_results=TRUE)
write_json(metadata, "reference/fixtures/metadata.json", pretty=TRUE, auto_unbox=TRUE)
