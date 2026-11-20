#!/usr/bin/env python
"""Fig 1f (v4): loops vs controls, linear broken y-axis.

- linear, evenly-divided y-axis, minimum shown = -1
- broken axis: main panel shows -1 .. (just above whisker cluster); the top
  panel shows the extreme whisker range, with the empty middle omitted
- whiskers drawn at full range (0-100th percentile), no scatter spots

Metric: (max/min)-1 of anchor genes' RPKM; lower = more coordinated.

Run: python fig1f_final_v4.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/fig1f_loop_coexpression")
RNA = os.path.join(BASE, "review_data/rnaseq/AKR_RNA_seq_merged.csv")
GTF = os.path.join(BASE, "annotation.gtf")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")

C_LOOP = "#8C2F39"
C_R1 = "#3D5A80"
C_R2 = "#7190B9"
C_R3 = "#B8C0C8"
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
data = [rab, r1, r2, r3]
labels = ["loops\n(n=41)", "random\nanchors", "distance-\nmatched", "distance +\nexpr-matched"]
colors = [C_LOOP, C_R1, C_R2, C_R3]

# choose break range: bottom panel shows up to just above the whisker cluster;
# top panel covers the main extreme range; single extreme values beyond it are
# clipped and annotated with an arrow
vals = np.concatenate(data)
bottom_hi = 16.0
top_lo, top_hi = 40.0, 300.0
group_max = [d.max() for d in data]
print(f"break: bottom -1..{bottom_hi}, top {top_lo}..{top_hi}, group maxima={[round(v,1) for v in group_max]}")

# ---------------- figure with broken axis ----------------
# bottom: clean 1.5-IQR boxplots, linear, min=-1
# top: each group's maximum shown as a labelled marker line (the "large
# values" after the omitted segment); no scatter spots anywhere
fig = plt.figure(figsize=(5.4, 5.4))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3.4], hspace=0.08,
                       left=0.16, right=0.96, top=0.90, bottom=0.11)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

bp = ax_bot.boxplot(data, labels=labels,
                    patch_artist=True, showfliers=False, widths=0.58,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="#444444", lw=0.9),
                    capprops=dict(color="#444444", lw=0.9))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.9)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)

# top panel: max value of each group as a labelled marker line; values beyond
# the panel are drawn at the top edge with an up-arrow and the true value
for xi, (gmax, c) in enumerate(zip(group_max, colors), start=1):
    if gmax <= top_hi:
        ax_top.hlines(gmax, xi - 0.22, xi + 0.22, color=c, lw=3.0)
        ax_top.text(xi, gmax + (top_hi - top_lo) * 0.03, f"{gmax:.0f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333333")
    else:
        ax_top.annotate(f"{gmax:.0f}", xy=(xi, top_hi), xytext=(xi, top_hi * 0.86),
                        ha="center", fontsize=8.5, fontweight="bold", color=c,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))
ax_top.text(0.55, top_lo + (top_hi - top_lo) * 0.90, "group maxima",
            fontsize=8, color="#666666", style="italic")

ax_top.set_ylim(top_lo, top_hi)
ax_bot.set_ylim(-1, 18)

# ticks: even divisions
ax_bot.set_yticks(np.arange(-1, 19, 3))
ax_top.set_yticks(np.arange(50, int(top_hi) + 1, 100))

ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
for spine in ["top", "right"]:
    ax_top.spines[spine].set_visible(False)
    ax_bot.spines[spine].set_visible(False)

# break marks (diagonal slashes)
d = 0.012
kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.0)
ax_top.plot((-d, +d), (-d * 2.2, +d * 2.2), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d * 2.2, +d * 2.2), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax_bot.set_ylabel("Expression divergence of anchor genes\n(max / min − 1, RPKM-normalised)", fontsize=10)
ax_bot.yaxis.set_label_coords(-0.13, 0.62, transform=ax_bot.transAxes)
fig.suptitle("Transcriptional coordination of loop anchors",
             fontsize=11.5, fontweight="bold", x=0.55, y=0.965)

# significance stars: loops vs each control, above the whiskers in bottom panel
pu_r1 = stats.mannwhitneyu(rab, r1, alternative="less").pvalue
pu_r2 = stats.mannwhitneyu(rab, r2, alternative="less").pvalue
pu_r3 = stats.mannwhitneyu(rab, r3, alternative="less").pvalue
for x2, pu in [(2, pu_r1), (3, pu_r2), (4, pu_r3)]:
    star = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else "ns"
    y = 16.2
    ax_bot.plot([1, 1, x2, x2], [y, y + 0.4, y + 0.4, y], lw=0.8, color="black")
    ax_bot.text((1 + x2) / 2, y + 0.5, star, ha="center", fontsize=10)

fig.savefig(os.path.join(OUT, "figures/fig1f_final_broken_axis.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_final_broken_axis.png"), dpi=300, bbox_inches="tight")

json.dump(dict(
    y_axis="linear broken axis: bottom -1..16, top shows extreme whisker range",
    whiskers="full range (0-100th percentile), no scatter spots",
    loops_median=float(np.median(rab)),
    R1_median=float(np.median(r1)), R2_median=float(np.median(r2)), R3_median=float(np.median(r3)),
    MW_p_vs_R1=float(pu_r1), MW_p_vs_R2=float(pu_r2), MW_p_vs_R3=float(pu_r3),
), open(os.path.join(OUT, "tables/fig1f_broken_axis_stats.json"), "w"), indent=2)
print("saved fig1f_final_broken_axis")
