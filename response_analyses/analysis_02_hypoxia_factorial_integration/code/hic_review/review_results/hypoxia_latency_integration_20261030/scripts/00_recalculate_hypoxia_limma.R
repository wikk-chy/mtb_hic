#!/usr/bin/env Rscript

# Recalculate the eight-sample genotype x hypoxia RNA-seq model using
# TMM normalization, limma-voom and empirical-Bayes moderated t tests.
# The biological contrasts and validated gene mapping are unchanged from
# the earlier integration; only RNA normalization and inference change.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_path <- normalizePath(file_arg[1])
OUT <- dirname(dirname(script_path))
ROOT <- normalizePath(file.path(OUT, "..", "..", ".."))
LOCAL_LIB <- file.path(ROOT, "tmp", "R_libs")
.libPaths(c(LOCAL_LIB, .libPaths()))

suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
})

RNA_DIR <- file.path(ROOT, "hic_review", "review_data", "rnaseq")
RA_DIR <- file.path(ROOT, "hic_review", "review_results", "R1_mechanism_reanalysis")
MAP_DIR <- file.path(ROOT, "hic_review", "review_results", "R1_mechanism", "R1_mechanism_zhenghe", "tables")
TAB_DIR <- file.path(OUT, "tables")
SRC_DIR <- file.path(OUT, "source_data")
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(SRC_DIR, recursive = TRUE, showWarnings = FALSE)

counts_raw <- read.csv(file.path(RNA_DIR, "AKR_RNA_seq_merged.csv"),
                       check.names = FALSE, stringsAsFactors = FALSE)
mapping <- read.csv(file.path(MAP_DIR, "ebg_to_rv_mapping.csv"),
                    stringsAsFactors = FALSE)
old <- read.csv(file.path(RA_DIR, "04_rna_integration", "gene_level_rna_v2.csv"),
                stringsAsFactors = FALSE)

stopifnot(!anyDuplicated(counts_raw$gene), !anyDuplicated(mapping$gene_id))
mapping <- mapping[tolower(as.character(mapping$validated)) == "true", , drop = FALSE]
idx <- match(mapping$merged_ebg_id, counts_raw$gene)
stopifnot(!anyNA(idx), nrow(mapping) == 3814L)
counts_mapped <- counts_raw[idx, , drop = FALSE]
stopifnot(identical(mapping$merged_ebg_id, counts_mapped$gene))
old_tested <- old$gene_id[is.finite(old$hyp_response_ko)]
stopifnot(setequal(mapping$gene_id, old_tested))

sample_cols <- c(
  "WT-rep1", "WT-rep2",
  "NapM-KO-rep1", "NapM-KO-rep2",
  "Hypoxia-rep1", "Hypoxia-rep2",
  "NapM_Ko-Hypoxia-rep1", "NapM_Ko-Hypoxia-rep2"
)
counts <- as.matrix(counts_mapped[, sample_cols])
rownames(counts) <- mapping$gene_id
storage.mode(counts) <- "double"
stopifnot(all(is.finite(counts)), all(counts >= 0))

sample_info <- data.frame(
  sample = sample_cols,
  genotype = rep(c("WT", "KO", "WT", "KO"), each = 2),
  condition = rep(c("normoxia", "hypoxia"), each = 4),
  group = rep(c("WT", "KO", "WT_hypoxia", "KO_hypoxia"), each = 2),
  stringsAsFactors = FALSE
)
sample_info$group <- factor(sample_info$group,
                            levels = c("WT", "KO", "WT_hypoxia", "KO_hypoxia"))
design <- model.matrix(~0 + sample_info$group)
colnames(design) <- levels(sample_info$group)
rownames(design) <- sample_cols
contrast_matrix <- makeContrasts(
  baseline_KO_WT = KO - WT,
  WT_hypoxia = WT_hypoxia - WT,
  KO_hypoxia = KO_hypoxia - KO,
  hypoxia_interaction = (KO_hypoxia - KO) - (WT_hypoxia - WT),
  levels = design
)

dge <- DGEList(counts = counts)
dge <- calcNormFactors(dge, method = "TMM")
v <- voom(dge, design = design, normalize.method = "none", plot = FALSE)
fit <- lmFit(v, design)
fit <- contrasts.fit(fit, contrast_matrix)
fit <- eBayes(fit, robust = FALSE)

contrast_names <- colnames(contrast_matrix)
result_list <- lapply(contrast_names, function(name) {
  tt <- topTable(fit, coef = name, number = Inf, sort.by = "none",
                 adjust.method = "BH")
  data.frame(
    gene_id = rownames(tt),
    merged_ebg_id = mapping$merged_ebg_id[match(rownames(tt), mapping$gene_id)],
    contrast = name,
    log2FC = tt$logFC,
    AveExpr = tt$AveExpr,
    moderated_t = tt$t,
    P_Value = tt$P.Value,
    adj_P_Val = tt$adj.P.Val,
    B = tt$B,
    df_residual = fit$df.residual[match(rownames(tt), rownames(fit))],
    sigma = fit$sigma[match(rownames(tt), rownames(fit))],
    stringsAsFactors = FALSE
  )
})
results <- do.call(rbind, result_list)
results$DEG <- with(results, adj_P_Val < 0.05 & abs(log2FC) >= 1)
results$direction <- ifelse(!results$DEG, "not_DEG",
                            ifelse(results$log2FC > 0, "up", "down"))
write.csv(results, file.path(TAB_DIR, "Hypoxia_limma_voom_gene_results_long.csv"),
          row.names = FALSE)

# Wide table with the original integration field names for a controlled update.
wide <- data.frame(gene_id = mapping$gene_id, stringsAsFactors = FALSE)
field_map <- list(
  baseline_KO_WT = c("ko_effect_base", "p_ko", "padj_ko"),
  WT_hypoxia = c("hyp_response_wt", "p_hyp", "padj_hyp"),
  KO_hypoxia = c("hyp_response_ko", "p_hyp_ko", "padj_hyp_ko"),
  hypoxia_interaction = c("interaction", "p_int", "padj_int")
)
for (name in names(field_map)) {
  z <- result_list[[match(name, contrast_names)]]
  m <- match(wide$gene_id, z$gene_id)
  wide[[field_map[[name]][1]]] <- z$log2FC[m]
  wide[[field_map[[name]][2]]] <- z$P_Value[m]
  wide[[field_map[[name]][3]]] <- z$adj_P_Val[m]
}
write.csv(wide, file.path(TAB_DIR, "Hypoxia_limma_voom_for_integration.csv"),
          row.names = FALSE)

count_rows <- lapply(result_list, function(z) {
  deg <- z$adj_P_Val < 0.05 & abs(z$log2FC) >= 1
  data.frame(
    contrast = z$contrast[1],
    n_tested = nrow(z),
    n_FDR_0_05 = sum(z$adj_P_Val < 0.05),
    n_DEG = sum(deg),
    n_up = sum(deg & z$log2FC > 0),
    n_down = sum(deg & z$log2FC < 0),
    median_log2FC = median(z$log2FC),
    median_abs_log2FC = median(abs(z$log2FC))
  )
})
counts_summary <- do.call(rbind, count_rows)
write.csv(counts_summary, file.path(TAB_DIR, "Hypoxia_limma_voom_counts.csv"),
          row.names = FALSE)

# Audit the exact change against the superseded unmoderated OLS result.
ko_new <- result_list[[match("KO_hypoxia", contrast_names)]]
ko_new$DEG <- with(ko_new, adj_P_Val < 0.05 & abs(log2FC) >= 1)
ko_new$direction <- ifelse(!ko_new$DEG, "not_DEG", ifelse(ko_new$log2FC > 0, "up", "down"))
ko_old <- old[, c("gene_id", "hyp_response_ko", "p_hyp_ko", "padj_hyp_ko")]
names(ko_old) <- c("gene_id", "OLS_log2FC", "OLS_P_Value", "OLS_adj_P_Val")
audit <- merge(ko_old, ko_new[, c("gene_id", "log2FC", "P_Value", "adj_P_Val", "DEG", "direction")],
               by = "gene_id", all = TRUE, sort = FALSE)
names(audit)[names(audit) == "log2FC"] <- "limma_log2FC"
names(audit)[names(audit) == "P_Value"] <- "limma_P_Value"
names(audit)[names(audit) == "adj_P_Val"] <- "limma_adj_P_Val"
audit$OLS_DEG <- with(audit, OLS_adj_P_Val < 0.05 & abs(OLS_log2FC) >= 1)
audit$status_change <- ifelse(audit$OLS_DEG & audit$DEG, "DEG_in_both",
                              ifelse(!audit$OLS_DEG & audit$DEG, "gained_with_limma",
                                     ifelse(audit$OLS_DEG & !audit$DEG, "lost_with_limma", "not_DEG_in_both")))
write.csv(audit, file.path(TAB_DIR, "KO_hypoxia_OLS_to_limma_audit.csv"),
          row.names = FALSE)

logcpm <- cpm(dge, log = TRUE, prior.count = 0.5, normalized.lib.sizes = TRUE)
logcpm_out <- data.frame(gene_id = rownames(logcpm), logcpm, check.names = FALSE)
write.csv(logcpm_out, file.path(SRC_DIR, "Hypoxia_limma_voom_normalized_logCPM.csv"),
          row.names = FALSE)
write.csv(sample_info, file.path(SRC_DIR, "Hypoxia_limma_sample_design.csv"), row.names = FALSE)
write.csv(data.frame(sample = rownames(design), design, check.names = FALSE),
          file.path(SRC_DIR, "Hypoxia_limma_design_matrix.csv"), row.names = FALSE)
write.csv(data.frame(contrast = rownames(contrast_matrix), contrast_matrix, check.names = FALSE),
          file.path(SRC_DIR, "Hypoxia_limma_contrast_matrix.csv"), row.names = FALSE)
write.csv(data.frame(sample = sample_cols, library_size = dge$samples$lib.size,
                     TMM_norm_factor = dge$samples$norm.factors),
          file.path(TAB_DIR, "Hypoxia_limma_library_normalization.csv"), row.names = FALSE)

metadata <- c(
  "Primary RNA model: eight-sample genotype x hypoxia factorial design",
  "Input: AKR_RNA_seq_merged.csv; 3814 KO-count-validated gene mappings",
  "Normalization: edgeR TMM",
  "Transformation and weights: limma voom",
  "Inference: limma contrasts.fit + eBayes (robust=FALSE)",
  "KO hypoxia response: (KO_hypoxia - KO)",
  "Interaction: (KO_hypoxia - KO) - (WT_hypoxia - WT)",
  "DEG: BH FDR < 0.05 and absolute log2FC >= 1",
  paste0("R: ", R.version.string),
  paste0("limma: ", packageVersion("limma")),
  paste0("edgeR: ", packageVersion("edgeR"))
)
writeLines(metadata, file.path(TAB_DIR, "Hypoxia_limma_voom_model_metadata.txt"))
writeLines(capture.output(sessionInfo()), file.path(TAB_DIR, "Hypoxia_limma_sessionInfo.txt"))

cat("limma-voom analysis complete\n")
cat("Validated genes:", nrow(counts), "\n")
cat("Residual degrees of freedom:", unique(fit$df.residual), "\n")
print(counts_summary)
cat("KO hypoxia status changes:\n")
print(table(audit$status_change, useNA = "ifany"))
