#!/usr/bin/env python
"""Fig 1f redo v2: transcriptional coordination with proper co-expression metric.

Key change vs v1: the paper's min/max ratio measures similarity of absolute
expression in ONE condition — not coordination. True co-expression = correlation
of expression ACROSS samples/conditions (5 conditions x 2 replicates).

Metrics:
  M1: min/max WT expression (paper's metric, for continuity)
  M2: Spearman correlation across the 10 RNA-seq samples (real co-expression)

Random controls:
  R1: paper's random anchor (anchor1 kept, random operon C)
  R2: distance-matched random pairs
  R3: distance- AND expression-matched random pairs (strictest)

Positive control: operon pairs inside the same CID (known co-regulated unit).

Run: python fig1f_coexpression_v2.py
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
OPERON = os.path.join(BASE, "review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")
RNA = os.path.join(BASE, "review_data/rnaseq/AKR_RNA_seq_merged.csv")
GTF = os.path.join(BASE, "annotation.gtf")

SEED = 42
rng = np.random.default_rng(SEED)
N_ROUNDS = 100  # random pair sets for distribution
C_LOOP, C_RAND, C_DM, C_POS = "#AE3019", "#4D779B", "#999999", "#009E73"

print("=" * 60)
print("Fig 1f v2: co-expression across conditions")
print("=" * 60)

# ---------------- load data ----------------
op = pd.read_csv(OPERON)[["operon_id", "operon_start", "operon_end", "normalized_expression"]]
op = op.sort_values("operon_start").reset_index(drop=True)

rna = pd.read_csv(RNA)
sample_cols = [c for c in rna.columns if c != "gene"]
print(f"RNA-seq samples: {sample_cols}")

gtf = pd.read_csv(GTF, sep="\t", comment="#", header=None,
                  names=["chr", "src", "feat", "start", "end", "score", "strand", "frame", "attr"])
gtf = gtf[gtf.feat == "gene"].copy()
gtf["gene_id"] = gtf.attr.str.extract(r'gene_id "?([^";]+)"?')
m = gtf.merge(rna, left_on="gene_id", right_on="gene").sort_values("start").reset_index(drop=True)
print(f"genes with position + expression: {len(m)}")

# log2 expression matrix (genes x 10 samples)
expr_mat = np.log2(m[sample_cols].values + 1)

def pos_to_gene(pos):
    inside = m[(m.start <= pos) & (m.end >= pos)]
    if len(inside):
        return inside.index[0]
    nxt = m[m.start > pos]
    return nxt.index[0] if len(nxt) else m.index[-1]

def pos_to_operon(pos):
    inside = op[(op.operon_start <= pos) & (op.operon_end >= pos)]
    if len(inside):
        return inside.index[0]
    nxt = op[op.operon_start > pos]
    return nxt.index[0] if len(nxt) else op.index[-1]

def minmax(a, b):
    return min(a, b) / max(a, b) if a > 0 and b > 0 else np.nan

def spearman_gene(i, j):
    x, y = expr_mat[i], expr_mat[j]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return stats.spearmanr(x, y).statistic

# ---------------- loops ----------------
cs = pd.read_csv(CS_LOOPS, sep="\t")
cs["m1"] = (cs.start1 + cs.end1) // 2
cs["m2"] = (cs.start2 + cs.end2) // 2
cs["dist"] = cs.m2 - cs.m1

loop_m1, loop_m2 = [], []
loop_pairs = []
for _, r in cs.iterrows():
    gi, gj = pos_to_gene(r.m1), pos_to_gene(r.m2)
    oi, oj = pos_to_operon(r.m1), pos_to_operon(r.m2)
    loop_pairs.append((gi, gj, oi, oj, r.dist))
    loop_m1.append(minmax(op.loc[oi, "normalized_expression"], op.loc[oj, "normalized_expression"]))
    loop_m2.append(spearman_gene(gi, gj))

loop_m1 = np.array([v for v in loop_m1 if np.isfinite(v)])
loop_m2 = np.array([v for v in loop_m2 if np.isfinite(v)])
print(f"loops: M1 median={np.median(loop_m1):.3f}, M2 (Spearman) median={np.median(loop_m2):.3f}")

# ---------------- random controls ----------------
def rand_pairs_distance_matched(n_per_loop=1):
    """R2: distance-matched random gene pairs."""
    out = []
    for (_, _, _, _, d) in loop_pairs:
        for _ in range(n_per_loop):
            for _try in range(50):
                p1 = int(rng.integers(0, 4411532 - d))
                p2 = p1 + d
                gi, gj = pos_to_gene(p1), pos_to_gene(p2)
                if gi != gj:
                    out.append((gi, gj))
                    break
    return out

def rand_pairs_paper(n_per_loop=25):
    """R1: anchor1 kept, random second gene."""
    out = []
    for (gi, _, _, _, _) in loop_pairs:
        for _ in range(n_per_loop):
            gj = int(rng.integers(0, len(m)))
            if gj != gi:
                out.append((gi, gj))
    return out

def rand_pairs_matched(n_per_loop=10):
    """R3: distance- and expression-matched."""
    out = []
    wt_expr = m["WT-rep1"].values + m["WT-rep2"].values
    for (gi, _, _, _, d) in loop_pairs:
        e1 = wt_expr[gi]
        cnt = 0
        for _try in range(200):
            if cnt >= n_per_loop:
                break
            p1 = int(rng.integers(0, 4411532 - d))
            p2 = p1 + d
            hi, hj = pos_to_gene(p1), pos_to_gene(p2)
            if hi == hj:
                continue
            # expression match: anchor1 expression within 2x
            if e1 > 0 and 0.5 <= wt_expr[hi] / e1 <= 2.0:
                out.append((hi, hj))
                cnt += 1
    return out

r1_pairs = rand_pairs_paper()
r2_pairs = rand_pairs_distance_matched(n_per_loop=25)   # ~1000 pairs for stable null
r3_pairs = rand_pairs_matched()
print(f"random pairs: R1={len(r1_pairs)}, R2={len(r2_pairs)}, R3={len(r3_pairs)}")

def eval_pairs(pairs, metric):
    vals = []
    for gi, gj in pairs:
        if metric == "M2":
            v = spearman_gene(gi, gj)
        else:
            oi, oj = pos_to_operon(int(m.loc[gi, "start"])), pos_to_operon(int(m.loc[gj, "start"]))
            v = minmax(op.loc[oi, "normalized_expression"], op.loc[oj, "normalized_expression"])
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)

r1_m2 = eval_pairs(r1_pairs, "M2")
r2_m2 = eval_pairs(r2_pairs, "M2")
r3_m2 = eval_pairs(r3_pairs, "M2")

# positive control: operon pairs inside the same CID (known co-regulated unit)
cid_dir = os.path.join(BASE, "review_data/cid_boundary")
cid = pd.read_csv(os.path.join(cid_dir, "wt-boundary.bed"), sep="\t", header=None)
bounds = sorted(cid[1].tolist())
def cid_of(pos):
    for k in range(len(bounds) - 1):
        if bounds[k] <= pos < bounds[k + 1]:
            return k
    return len(bounds) - 1

pos_pairs = []
tries = 0
while len(pos_pairs) < 500 and tries < 20000:
    tries += 1
    p1 = int(rng.integers(0, 4411532))
    p2 = int(rng.integers(0, 4411532))
    if abs(p2 - p1) < 5000:
        continue
    if cid_of(p1) == cid_of(p2):
        gi, gj = pos_to_gene(p1), pos_to_gene(p2)
        if gi != gj:
            pos_pairs.append((gi, gj))
pos_ctrl = eval_pairs(pos_pairs, "M2")
print(f"positive control (same CID): n={len(pos_ctrl)}, M2 median={np.median(pos_ctrl):.3f}")

# ---------------- statistics ----------------
def cmp(a, b, name):
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="greater")
    print(f"loops vs {name}: medians {np.median(a):.3f} vs {np.median(b):.3f}, "
          f"Welch p={pt:.3e}, MW p={pu:.3e}")
    return pt, pu

print("\n--- M2 (Spearman co-expression across 10 samples) ---")
p_r1 = cmp(loop_m2, r1_m2, "R1 paper-random")
p_r2 = cmp(loop_m2, r2_m2, "R2 distance-matched")
p_r3 = cmp(loop_m2, r3_m2, "R3 dist+expr-matched")
p_pos = cmp(pos_ctrl, r2_m2, "same-CID positive control vs R2")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# A: distributions
data = [loop_m2, r1_m2, r2_m2, r3_m2, pos_ctrl]
labels = ["loops\n(n=41)", f"R1 random\nanchors\n(n={len(r1_m2)})", f"R2 distance-\nmatched\n(n={len(r2_m2)})",
          f"R3 distance+\nexpr-matched\n(n={len(r3_m2)})", f"same-CID\npairs\n(n={len(pos_ctrl)})\n(pos. control)"]
colors = [C_LOOP, C_RAND, C_DM, C_DM, C_POS]
bp = axes[0].boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                     widths=0.6, medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
for k, (vals, c) in enumerate(zip(data, colors)):
    jitter = rng.normal(0, 0.05, len(vals))
    axes[0].scatter(np.full(len(vals), k + 1) + jitter, vals, s=4,
                    color=c, alpha=0.25, zorder=3)
axes[0].axhline(0, color="grey", ls=":", lw=0.8)
axes[0].set_ylabel("Spearman co-expression\nacross 10 RNA-seq samples", fontsize=9)
axes[0].set_title("A  Co-expression of loop anchors vs random controls",
                  loc="left", fontsize=10, fontweight="bold")
# sig bars vs R2 and R3
y_top = max(np.nanmax(d) for d in data) * 1.05
for x1, x2, (_, pu) in [(1, 3, p_r2), (1, 4, p_r3)]:
    axes[0].plot([x1, x1, x2, x2], [y_top, y_top + 0.03, y_top + 0.03, y_top],
                 lw=1, color="black")
    star = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else "ns"
    axes[0].text((x1 + x2) / 2, y_top + 0.035, star, ha="center", fontsize=11)
    y_top += 0.09

# B: distance-dependence of co-expression, loops vs random
axes[1].scatter([p[4] / 1000 for p in loop_pairs], loop_m2, s=30, color=C_LOOP,
                alpha=0.85, edgecolor="black", lw=0.3, label="loops", zorder=3)
r2_dist = [(abs(int(m.loc[gj, "start"]) - int(m.loc[gi, "start"])) / 1000) for gi, gj in r2_pairs]
axes[1].scatter(r2_dist, r2_m2, s=8, color=C_RAND, alpha=0.15, label="distance-matched random", zorder=2)
axes[1].axhline(np.median(loop_m2), color=C_LOOP, ls="--", lw=0.9)
axes[1].axhline(np.median(r2_m2), color=C_RAND, ls="--", lw=0.9)
axes[1].axhline(0, color="grey", ls=":", lw=0.8)
axes[1].set_xlabel("Anchor distance (kb)", fontsize=9)
axes[1].set_ylabel("Spearman co-expression", fontsize=9)
axes[1].legend(fontsize=8, frameon=False)
axes[1].set_title("B  Co-expression vs anchor distance\n(dashed = group medians)", loc="left", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig1f_coexpression_v2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_coexpression_v2.png"), dpi=300, bbox_inches="tight")

result = dict(
    metric_M2="Spearman correlation of anchor genes across 10 RNA-seq samples",
    loops_median_M2=float(np.median(loop_m2)),
    R1_random_anchor_median=float(np.median(r1_m2)),
    R2_distance_matched_median=float(np.median(r2_m2)),
    R3_dist_expr_matched_median=float(np.median(r3_m2)),
    positive_control_neighbours_median=float(np.median(pos_ctrl)),
    loops_vs_R1_MW_p=float(p_r1[1]),
    loops_vs_R2_MW_p=float(p_r2[1]),
    loops_vs_R3_MW_p=float(p_r3[1]),
)
with open(os.path.join(OUT, "tables/fig1f_v2_stats.json"), "w") as f:
    json.dump(result, f, indent=2)
print("\n" + json.dumps(result, indent=2))
print("done.")
