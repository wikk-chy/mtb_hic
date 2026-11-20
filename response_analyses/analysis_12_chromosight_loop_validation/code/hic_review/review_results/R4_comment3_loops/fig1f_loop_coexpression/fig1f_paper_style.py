#!/usr/bin/env python
"""Fig 1f (paper style): loops vs random anchors boxplot, p-value annotated.

Two groups only, no scatter spots, paper palette (loops #AE3019, random #4D779B).
Metric: (max/min)-1 of anchor genes' RPKM (gene length-normalised);
lower = more coordinated.

Run: python fig1f_paper_style.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/fig1f_loop_coexpression")
RNA = os.path.join(BASE, "review_data/rnaseq/AKR_RNA_seq_merged.csv")
GTF = os.path.join(BASE, "annotation.gtf")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")

C_LOOP = "#8C2F39"   # deep burgundy
C_RAND = "#3D5A80"   # deep slate blue
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42

# ---------------- data ----------------
rna = pd.read_csv(RNA)
rna["WT"] = (rna["WT-rep1"] + rna["WT-rep2"]) / 2
gtf = pd.read_csv(GTF, sep="\t", comment="#", header=None,
                  names=["chr", "src", "feat", "start", "end", "score", "strand", "frame", "attr"])
gtf = gtf[gtf.feat == "gene"].copy()
gtf["gene_id"] = gtf.attr.str.extract(r'gene_id "?([^";]+)"?')
m = gtf.merge(rna[["gene", "WT"]], left_on="gene_id", right_on="gene").sort_values("start").reset_index(drop=True)
m["expr_norm"] = m.WT / (m.end - m.start)
m = m[m.expr_norm > 0].reset_index(drop=True)

def pos_to_gene(pos):
    inside = m[(m.start <= pos) & (m.end >= pos)]
    if len(inside):
        return inside.index[0]
    nxt = m[m.start > pos]
    return nxt.index[0] if len(nxt) else m.index[-1]

cs = pd.read_csv(CS_LOOPS, sep="\t")
cs["m1"] = (cs.start1 + cs.end1) // 2
cs["m2"] = (cs.start2 + cs.end2) // 2

def divergence(gi, gj):
    a, b = m.loc[gi, "expr_norm"], m.loc[gj, "expr_norm"]
    return max(a, b) / min(a, b) - 1.0

loop_gene_pairs = [(pos_to_gene(r.m1), pos_to_gene(r.m2)) for _, r in cs.iterrows()]
rab = np.array([divergence(gi, gj) for gi, gj in loop_gene_pairs])

# paper's random anchor method: anchor1 kept, random second gene, 10 rounds pooled
rng = np.random.default_rng(42)
r1 = []
for gi, _ in loop_gene_pairs:
    for _ in range(10):
        h = int(rng.integers(0, len(m)))
        if h != gi:
            r1.append(divergence(gi, h))
r1 = np.array(r1)

# p-values: Welch t-test (paper) + Mann-Whitney
t_stat, p_t = stats.ttest_ind(rab, r1, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(rab, r1, alternative="less")

print(f"loops (n={len(rab)}): median={np.median(rab):.3f}")
print(f"random anchors (n={len(r1)}): median={np.median(r1):.3f}")
print(f"Welch t-test p={p_t:.3e}, Mann-Whitney p={p_u:.3e}")

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(2.8, 4.6))

bp = ax.boxplot([rab, r1], labels=["loops", "random anchors"],
                patch_artist=True, showfliers=False, widths=0.52,
                medianprops=dict(color="black", lw=1.4),
                whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
for patch, c in zip(bp["boxes"], [C_LOOP, C_RAND]):
    patch.set_facecolor(c)
    patch.set_alpha(0.9)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)

ax.set_ylabel("Expression divergence\n(max / min − 1)", fontsize=10)
ax.set_yscale("log")
vals_all = np.concatenate([rab, r1])
ax.set_ylim(max(vals_all[vals_all > 0].min() * 0.7, 1e-2), vals_all.max() * 2.6)
from matplotlib.ticker import ScalarFormatter, NullFormatter
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100])
ax.set_yticklabels(["0.01", "0.1", "0.5", "1", "5", "20", "100"])

# p-value bar
y = vals_all.max() * 1.3
star = "***" if p_u < 0.001 else "**" if p_u < 0.01 else "*" if p_u < 0.05 else "ns"
ax.plot([1, 1, 2, 2], [y, y * 1.07, y * 1.07, y], lw=0.9, color="black")
ax.text(1.5, y * 1.12, star, ha="center", fontsize=12)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig1f_paper_style.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_paper_style.png"), dpi=300, bbox_inches="tight")

json.dump(dict(loops_median=float(np.median(rab)),
               random_median=float(np.median(r1)),
               welch_p=float(p_t), mannwhitney_p=float(p_u)),
          open(os.path.join(OUT, "tables/fig1f_paper_style_stats.json"), "w"), indent=2)
print("saved fig1f_paper_style")
