#!/usr/bin/env python3
"""R2 Comment 6 - chip_rna_integration, part 1.

NapM (Rv0047c) expression across conditions from the merged RNA-seq count
table (library-size CPM normalization; note: merged table carries n = 2
replicates per condition) and its genome-wide percentile rank among all
genes, plus limma values from the four AKR DEG tables.

Auxiliary information only: NO protein copy-number inference is made
(no absolute quantification data exist).

Recomputes and extends the existing R2_comments/R_napm_expression_dependence
result (WT CPM 421, hypoxia 584) and adds percentile ranks.

Outputs: tables/napm_expression.csv,
         figures/fig1_napm_expression.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt
from scipy.stats import rankdata

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
RNA = BASE + "hic_review/review_data/rnaseq/AKR_RNA_seq_merged.csv"
DEG_DIR = BASE + "hic_review/review_data/rnaseq/"
OUT_T = BASE + "hic_review/review_results/R2_comments/chip_rna_integration/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/chip_rna_integration/figures/"

rna = pd.read_csv(RNA).set_index("gene")
conds = {
    "WT": ["WT-rep1", "WT-rep2"],
    "napM-KO": ["NapM-KO-rep1", "NapM-KO-rep2"],
    "hypoxia": ["Hypoxia-rep1", "Hypoxia-rep2"],
    "latent": ["Latent-rep1", "Latent-rep2"],
    "KO-hypoxia": ["NapM_Ko-Hypoxia-rep1", "NapM_Ko-Hypoxia-rep2"],
}
libsize = rna.sum(axis=0)
rows = []
for cond, reps in conds.items():
    cpm_rep = [1e6 * rna[r] / libsize[r] for r in reps]
    napm_cpm = float(np.mean([c.loc["Rv0047c"] for c in cpm_rep]))
    mean_cpm_all = np.mean([c.values for c in cpm_rep], axis=0)
    pct = float(100 * (rankdata(mean_cpm_all)[list(rna.index).index("Rv0047c")] - 1)
                / (len(mean_cpm_all) - 1))
    rows.append(dict(condition=cond, napM_CPM=round(napm_cpm, 1),
                     genome_percentile=round(pct, 1),
                     n_reps_in_merged_table=len(reps)))
napm = pd.DataFrame(rows)

# limma values for Rv0047c from the four AKR DEG tables
deg_rows = []
for f, name in [("KOvsWT_supo_DEG_limma.csv", "KO vs WT"),
                ("hypoxiaWTvsWT_supo_DEG_limma.csv", "hypoxia vs WT"),
                ("LatentvsWT_supo_DEG_limma.csv", "latent vs WT"),
                ("hypoxia_KOvsWT_supo_DEG_limma.csv", "KO-hypoxia vs WT")]:
    d = pd.read_csv(DEG_DIR + f, index_col=0)
    if "Rv0047c" in d.index:
        r = d.loc["Rv0047c"]
        deg_rows.append(dict(contrast=name, log2FC=round(r.avg_log2FC, 3),
                             p_val=r.p_val, regulate=r.regulate))
deg = pd.DataFrame(deg_rows)
out = napm.to_csv(OUT_T + "napm_expression.csv", index=False)
deg.to_csv(OUT_T + "napm_limma_deg_rows.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
ax = axes[0]
colors = ["#3182bd" if c == "WT" else "#bdbdbd" for c in napm.condition]
ax.bar(napm.condition, napm.napM_CPM, color=colors, width=0.6)
for k, r in napm.iterrows():
    ax.text(k, r.napM_CPM + 10, f"{r.napM_CPM:.0f}", ha="center", fontsize=7)
ax.set_ylabel("NapM mRNA (CPM)")
ax.set_title("A  NapM expression (CPM)", loc="left", fontweight="bold")
ax.tick_params(axis="x", rotation=30)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")

ax = axes[1]
ax.bar(napm.condition, napm.genome_percentile, color="#9ecae1", width=0.6)
for k, r in napm.iterrows():
    ax.text(k, r.genome_percentile + 1, f"{r.genome_percentile:.0f}%",
            ha="center", fontsize=7)
ax.set_ylabel("expression percentile\namong all genes")
ax.set_ylim(0, 105)
ax.set_title("B  genome-wide percentile", loc="left", fontweight="bold")
ax.tick_params(axis="x", rotation=30)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig1_napm_expression.png", dpi=300)
fig.savefig(OUT_F + "fig1_napm_expression.pdf")
print(napm.to_string(index=False))
print(deg.to_string(index=False))
print("note: merged table has only n=2 replicates per condition;",
      "limma tables were computed on the full design")
print("OK")
