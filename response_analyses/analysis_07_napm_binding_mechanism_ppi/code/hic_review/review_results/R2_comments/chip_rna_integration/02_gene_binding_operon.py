#!/usr/bin/env python3
"""R2 Comment 6 - chip_rna_integration, part 2.

Gene-level binding status of NapM peaks, three tiers:
  direct_promoter : peak within 300 bp upstream of strand-aware TSS
  direct_body     : peak overlapping gene body (and not promoter tier)
  operon_extended : not directly bound, but in the same operon
                    (operon_metadata/operon_list_updated_with_bin_coverage.csv)
                    as a directly bound gene -- i.e. a peak in the upstream
                    part of the operon counts downstream genes as
                    "binding-associated" (approximation, stated as such)
  unbound

Direct tiers are taken from the R1 reanalysis gene-level table
(coordinate-derived, GTF ASM19595v2); the operon extension is added here.

Outputs: tables/gene_binding_status.csv
"""
import warnings
warnings.filterwarnings("ignore")
import ast
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
GENE = BASE + "hic_review/review_results/R1_mechanism_reanalysis/04_rna_integration/gene_level_rna_v2.csv"
OPERON = BASE + "hic_review/review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv"
OUT_T = BASE + "hic_review/review_results/R2_comments/chip_rna_integration/tables/"

g = pd.read_csv(GENE)
g["direct_tier"] = np.select(
    [g.binding_class == "promoter_bound", g.binding_class == "gene_body_bound"],
    ["direct_promoter", "direct_body"], default="unbound")

# ---- operon map --------------------------------------------------------------
op = pd.read_csv(OPERON)
gene2op = {}
for _, r in op.iterrows():
    try:
        genes = ast.literal_eval(r.gene_list)
    except Exception:
        genes = [x.strip() for x in str(r.gene_names).split(",")]
    for gid in genes:
        gene2op[gid] = r.operon_id
g["operon_id"] = g.gene_id.map(gene2op)
n_mapped = g.operon_id.notna().sum()

# operons containing at least one directly bound gene
bound_ops = set(g.loc[g.direct_tier != "unbound", "operon_id"].dropna())
g["binding_tier"] = g.direct_tier
mask = (g.direct_tier == "unbound") & (g.operon_id.isin(bound_ops))
g.loc[mask, "binding_tier"] = "operon_extended"

g[["gene_id", "gene_name", "start", "end", "strand", "operon_id",
   "direct_tier", "binding_tier",
   "gene_peak_score_max", "gene_peak_vpm_max",
   "log2FC_KOvsWT", "p_KOvsWT", "padj_KOvsWT", "class_KOvsWT"]].to_csv(
    OUT_T + "gene_binding_status.csv", index=False)

print("genes:", len(g), "| operon-mapped:", n_mapped)
print(g.binding_tier.value_counts().to_dict())
print("operons total:", op.operon_id.nunique(),
      "| operons with >=1 directly bound gene:", len(bound_ops))
print("OK")
