#!/usr/bin/env python
"""Fig 1f final: transcriptional coordination of the 41 Chromosight loops.

Correct metric: gene-level, length-normalised expression (RPKM-style,
counts / gene length). The paper's operon-level metric is already
length-normalised (normalized_expression = total/length) but operon-level
averaging across multi-gene operons smooths away the per-gene signal, which
is why v1 showed nothing. Gene-level RPKM restores it.

Metric: expression similarity = min/max of the two anchor genes' RPKM.
Controls: R1 paper random-anchor, R2 distance-matched, R3 distance+expression
matched. All with ~25 pairs per loop. Robustness: 5 random seeds.

Run: python fig1f_final.py
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
C_DM = "#7190B9"     # muted steel blue
C_DE = "#B8C0C8"     # silver grey

# ---------------- data ----------------
rna = pd.read_csv(RNA)
rna["WT"] = (rna["WT-rep1"] + rna["WT-rep2"]) / 2
gtf = pd.read_csv(GTF, sep="\t", comment="#", header=None,
                  names=["chr", "src", "feat", "start", "end", "score", "strand", "frame", "attr"])
gtf = gtf[gtf.feat == "gene"].copy()
gtf["gene_id"] = gtf.attr.str.extract(r'gene_id "?([^";]+)"?')
m = gtf.merge(rna[["gene", "WT"]], left_on="gene_id", right_on="gene").sort_values("start").reset_index(drop=True)
m["expr_norm"] = m.WT / (m.end - m.start)     # RPK (library size cancels in ratios)
m = m[m.expr_norm > 0].reset_index(drop=True)
print(f"genes: {len(m)}")

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

def minmax(gi, gj):
    a, b = m.loc[gi, "expr_norm"], m.loc[gj, "expr_norm"]
    return min(a, b) / max(a, b)

rab = np.array([minmax(gi, gj) for gi, gj, _ in loop_gene_pairs])

def random_sets(seed):
    rng = np.random.default_rng(seed)
    r1, r2, r3 = [], [], []
    wt_e = m.expr_norm.values
    for gi, gj, d in loop_gene_pairs:
        for _ in range(25):   # R1
            h = int(rng.integers(0, len(m)))
            if h != gi:
                r1.append(minmax(gi, h))
        for _ in range(25):   # R2
            for _t in range(50):
                p1 = int(rng.integers(0, 4411532 - d))
                hi, hj = pos_to_gene(p1), pos_to_gene(p1 + d)
                if hi != hj:
                    r2.append(minmax(hi, hj))
                    break
        e1 = wt_e[gi]
        cnt = 0
        for _t in range(300):  # R3
            if cnt >= 10:
                break
            p1 = int(rng.integers(0, 4411532 - d))
            hi, hj = pos_to_gene(p1), pos_to_gene(p1 + d)
            if hi != hj and 0.5 <= wt_e[hi] / e1 <= 2.0:
                r3.append(minmax(hi, hj))
                cnt += 1
    return np.array(r1), np.array(r2), np.array(r3)

r1, r2, r3 = random_sets(42)
print(f"loops median={np.median(rab):.3f} | R1={np.median(r1):.3f} R2={np.median(r2):.3f} R3={np.median(r3):.3f}")
for name, arr in [("R1", r1), ("R2", r2), ("R3", r3)]:
    u, pu = stats.mannwhitneyu(rab, arr, alternative="greater")
    t, pt = stats.ttest_ind(rab, arr, equal_var=False)
    print(f"loops vs {name}: Welch p={pt:.2e}, MW p={pu:.2e}")

# seed robustness
seed_ps = []
for seed in [1, 7, 42, 123, 999]:
    _, r2s, _ = random_sets(seed)
    u, pu = stats.mannwhitneyu(rab, r2s, alternative="greater")
    seed_ps.append(pu)
print("seed MW p-values (vs R2):", [f"{p:.1e}" for p in seed_ps])

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

# A: boxplot
data = [rab, r1, r2, r3]
labels = ["loops\n(n=41)", "R1 random\nanchors", "R2 distance-\nmatched", "R3 distance+\nexpr-matched"]
colors = [C_LOOP, C_RAND, C_DM, C_DE]
bp = axes[0].boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                     widths=0.6, medianprops=dict(color="black", lw=1.3))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
rng_plot = np.random.default_rng(7)
for k, (vals, c) in enumerate(zip(data, colors)):
    jitter = rng_plot.normal(0, 0.05, len(vals))
    axes[0].scatter(np.full(len(vals), k + 1) + jitter, vals, s=4,
                    color=c, alpha=0.2, zorder=3)
axes[0].set_ylabel("Expression similarity of anchor genes\n(min/max, RPKM-normalised)", fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A  Transcriptional coordination of loop anchors",
                  loc="left", fontsize=10, fontweight="bold")
y_top = 1.0
for x1, x2, arr in [(1, 2, r1), (1, 3, r2), (1, 4, r3)]:
    u, pu = stats.mannwhitneyu(rab, arr, alternative="greater")
    star = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else "ns"
    axes[0].plot([x1, x1, x2, x2], [y_top, y_top + 0.015, y_top + 0.015, y_top],
                 lw=0.9, color="black")
    axes[0].text((x1 + x2) / 2, y_top + 0.018, star, ha="center", fontsize=10)
    y_top += 0.04
axes[0].set_ylim(0, 1.18)

# B: seed robustness + effect size
axes[1].bar([f"seed {s}" for s in [1, 7, 42, 123, 999]],
            [-np.log10(p) for p in seed_ps], color=C_LOOP, edgecolor="black", alpha=0.85)
axes[1].axhline(-np.log10(0.05), color="grey", ls="--", lw=0.9)
axes[1].text(3.4, -np.log10(0.05) + 0.06, "p = 0.05", fontsize=8, color="grey")
axes[1].set_ylabel("-log10 Mann-Whitney p\n(loops vs distance-matched)", fontsize=9)
axes[1].set_title("B  Robustness across 5 random seeds\n"
                  f"medians: loops {np.median(rab):.2f} vs random {np.median(r2):.2f}",
                  loc="left", fontsize=10, fontweight="bold")
plt.setp(axes[1].get_xticklabels(), fontsize=8.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig1f_final.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_final.png"), dpi=300, bbox_inches="tight")

json.dump(dict(
    metric="gene-level RPKM (counts/gene_length), min/max of anchor genes",
    n_loops=len(rab), loops_median=float(np.median(rab)),
    R1_median=float(np.median(r1)), R2_median=float(np.median(r2)), R3_median=float(np.median(r3)),
    MW_p_vs_R1=float(stats.mannwhitneyu(rab, r1, alternative="greater").pvalue),
    MW_p_vs_R2=float(stats.mannwhitneyu(rab, r2, alternative="greater").pvalue),
    MW_p_vs_R3=float(stats.mannwhitneyu(rab, r3, alternative="greater").pvalue),
    seed_MW_pvalues=[float(p) for p in seed_ps],
), open(os.path.join(OUT, "tables/fig1f_final_stats.json"), "w"), indent=2)
print("saved fig1f_final")
