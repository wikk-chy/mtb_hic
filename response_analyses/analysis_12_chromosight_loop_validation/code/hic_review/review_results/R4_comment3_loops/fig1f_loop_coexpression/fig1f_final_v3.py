#!/usr/bin/env python
"""Fig 1f final (v3): transcriptional coordination, y = (max/min)-1.

Metric: expression divergence = (max/min of anchor genes' RPKM) - 1.
  0   -> identical expression (perfectly coordinated)
  >1  -> >2-fold difference
Loops should show LOWER divergence than random controls.

Gene-level, length-normalised (RPKM-style counts/length) — see fig1f_final.py
for why gene-level RPKM is the correct input (paper's operon-level metric is
length-normalised but smoothed; raw counts are length-confounded).

Run: python fig1f_final_v3.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/fig1f_loop_coexpression")
RNA = os.path.join(BASE, "review_data/rnaseq/AKR_RNA_seq_merged.csv")
GTF = os.path.join(BASE, "annotation.gtf")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")

# Muted professional palette (deep accent for loops, graduated cools for controls)
C_LOOP = "#8C2F39"   # deep burgundy
C_R1 = "#3D5A80"     # deep slate blue
C_R2 = "#7190B9"     # muted steel blue
C_R3 = "#B8C0C8"     # silver grey

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
cs["d"] = cs.m2 - cs.m1
loop_gene_pairs = [(pos_to_gene(r.m1), pos_to_gene(r.m2), r.d) for _, r in cs.iterrows()]

def divergence(gi, gj):
    a, b = m.loc[gi, "expr_norm"], m.loc[gj, "expr_norm"]
    return max(a, b) / min(a, b) - 1.0

rab = np.array([divergence(gi, gj) for gi, gj, _ in loop_gene_pairs])

def random_sets(seed):
    rng = np.random.default_rng(seed)
    r1, r2, r3 = [], [], []
    wt_e = m.expr_norm.values
    for gi, gj, d in loop_gene_pairs:
        for _ in range(25):
            h = int(rng.integers(0, len(m)))
            if h != gi:
                r1.append(divergence(gi, h))
        for _ in range(25):
            for _t in range(50):
                p1 = int(rng.integers(0, 4411532 - d))
                hi, hj = pos_to_gene(p1), pos_to_gene(p1 + d)
                if hi != hj:
                    r2.append(divergence(hi, hj))
                    break
        e1 = wt_e[gi]
        cnt = 0
        for _t in range(300):
            if cnt >= 10:
                break
            p1 = int(rng.integers(0, 4411532 - d))
            hi, hj = pos_to_gene(p1), pos_to_gene(p1 + d)
            if hi != hj and 0.5 <= wt_e[hi] / e1 <= 2.0:
                r3.append(divergence(hi, hj))
                cnt += 1
    return np.array(r1), np.array(r2), np.array(r3)

r1, r2, r3 = random_sets(42)
# NOTE: for (max/min)-1, loops coordinated = LOWER values -> test "less"
print(f"loops median={np.median(rab):.3f} | R1={np.median(r1):.3f} R2={np.median(r2):.3f} R3={np.median(r3):.3f}")
for name, arr in [("R1", r1), ("R2", r2), ("R3", r3)]:
    u, pu = stats.mannwhitneyu(rab, arr, alternative="less")
    print(f"loops vs {name}: MW (loops lower) p={pu:.2e}")

seed_ps = []
for seed in [1, 7, 42, 123, 999]:
    _, r2s, _ = random_sets(seed)
    seed_ps.append(stats.mannwhitneyu(rab, r2s, alternative="less").pvalue)

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(5.2, 5.0))

data = [rab, r1, r2, r3]
labels = ["loops\n(n=41)", "random\nanchors", "distance-\nmatched", "distance +\nexpr-matched"]
colors = [C_LOOP, C_R1, C_R2, C_R3]

bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                widths=0.58, medianprops=dict(color="black", lw=1.4),
                whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.88)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)

rng_plot = np.random.default_rng(7)
for k, (vals, c) in enumerate(zip(data, colors)):
    jitter = rng_plot.normal(0, 0.05, len(vals))
    ax.scatter(np.full(len(vals), k + 1) + jitter, vals, s=5,
               color=c, alpha=0.18, zorder=3, edgecolor="none")

ax.set_ylabel("Expression divergence of anchor genes\n(max / min − 1, RPKM-normalised)", fontsize=10)
ax.set_title("Transcriptional coordination of loop anchors",
             loc="left", fontsize=11, fontweight="bold")

# log-scale y (ratio-type data) for readable medians across the full range
vals_all = np.concatenate(data)
ax.set_yscale("log")
ymin = max(vals_all[vals_all > 0].min() * 0.7, 1e-2)
ax.set_ylim(ymin, vals_all.max() * 2.2)
from matplotlib.ticker import ScalarFormatter, NullFormatter
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.01, 0.1, 0.5, 1, 5, 20, 100])
ax.set_yticklabels(["0.01", "0.1", "0.5", "1", "5", "20", "100"])

# significance bars (loops LOWER = more coordinated)
y = vals_all.max() * 1.25
for x1, x2, arr in [(1, 2, r1), (1, 3, r2), (1, 4, r3)]:
    pu = stats.mannwhitneyu(rab, arr, alternative="less").pvalue
    star = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else "ns"
    ax.plot([x1, x1, x2, x2], [y, y * 1.05, y * 1.05, y], lw=0.9, color="black")
    ax.text((x1 + x2) / 2, y * 1.08, star, ha="center", fontsize=11)
    y *= 1.22

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig1f_final_divergence.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_final_divergence.png"), dpi=300, bbox_inches="tight")

json.dump(dict(
    metric="(max/min)-1 of anchor genes' RPKM (gene length-normalised); lower = more coordinated",
    n_loops=len(rab),
    loops_median=float(np.median(rab)),
    R1_median=float(np.median(r1)), R2_median=float(np.median(r2)), R3_median=float(np.median(r3)),
    MW_p_loops_lower_vs_R1=float(stats.mannwhitneyu(rab, r1, alternative="less").pvalue),
    MW_p_loops_lower_vs_R2=float(stats.mannwhitneyu(rab, r2, alternative="less").pvalue),
    MW_p_loops_lower_vs_R3=float(stats.mannwhitneyu(rab, r3, alternative="less").pvalue),
    seed_MW_pvalues=[float(p) for p in seed_ps],
), open(os.path.join(OUT, "tables/fig1f_divergence_stats.json"), "w"), indent=2)
print("saved fig1f_final_divergence")
