#!/usr/bin/env python
"""Redo Fig. 1f: transcriptional coordination of loop anchors.

41 Chromosight loops vs random anchor pairs, following the paper's approach
(random_pairs.R + hiccompared_randomcidsloops.ipynb):

- expression_similarity = min(expr_A, expr_B) / max(expr_A, expr_B)
  on operon-level normalized_expression (operon_list_updated_with_bin_coverage.csv)
- anchor midpoint -> containing operon (paper's mapping: position inside
  operon_start..operon_end; if in an intergenic gap, the next operon)
- ratio_AB = actual loop anchor pair
- ratio_AC = anchor 1 kept, anchor 2 replaced by a random operon (10 rounds,
  pooled) — the paper's "random_anchor" control
- robustness: distance-matched random operon pairs (both anchors random,
  same anchor distance), 10 rounds pooled
- statistics: Welch t-test (as in the paper) + Mann-Whitney

Run: python fig1f_loop_coexpression.py
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
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

OPERON = os.path.join(BASE, "review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")

SEED = 42
rng = np.random.default_rng(SEED)
N_RANDOM_ROUNDS = 10

C_LOOP, C_RAND = "#AE3019", "#4D779B"   # paper's palette

print("=" * 60)
print("Fig 1f redo: transcriptional coordination of loop anchors")
print("=" * 60)

# ---------------- load operons ----------------
op = pd.read_csv(OPERON)
print(f"operons: {len(op)}")
# columns: operon_id, ..., operon_start(11), operon_end(12), ..., normalized_expression(16)
op = op[["operon_id", "operon_start", "operon_end", "normalized_expression"]].copy()
op = op.sort_values("operon_start").reset_index(drop=True)

def pos_to_operon(pos):
    """Paper's mapping: operon containing pos; if intergenic, the next operon."""
    inside = op[(op.operon_start <= pos) & (op.operon_end >= pos)]
    if len(inside):
        return inside.index[0]
    nxt = op[op.operon_start > pos]
    if len(nxt):
        return nxt.index[0]
    return op.index[-1]

def expr_sim(i, j):
    a, b = op.loc[i, "normalized_expression"], op.loc[j, "normalized_expression"]
    if a <= 0 or b <= 0:
        return np.nan
    return min(a, b) / max(a, b)

# ---------------- map loop anchors to operons ----------------
cs = pd.read_csv(CS_LOOPS, sep="\t")
cs["mid1"] = ((cs.start1 + cs.end1) // 2).astype(int)
cs["mid2"] = ((cs.start2 + cs.end2) // 2).astype(int)

loop_rows = []
for _, r in cs.iterrows():
    i = pos_to_operon(r.mid1)
    j = pos_to_operon(r.mid2)
    loop_rows.append(dict(
        loop=f"{r.mid1//1000}-{r.mid2//1000}kb",
        mid1=r.mid1, mid2=r.mid2,
        anchor1_kb=r.mid1 // 1000, anchor2_kb=r.mid2 // 1000,
        dist_kb=(r.mid2 - r.mid1) // 1000,
        operon_A=op.loc[i, "operon_id"], operon_B=op.loc[j, "operon_id"],
        expr_A=op.loc[i, "normalized_expression"],
        expr_B=op.loc[j, "normalized_expression"],
        ratio_AB=expr_sim(i, j),
        same_operon=(i == j),
    ))
loops_df = pd.DataFrame(loop_rows)
print(f"loops: {len(loops_df)}, anchors in same operon: {loops_df.same_operon.sum()}")
loops_df.to_csv(os.path.join(OUT, "tables/loop_anchor_operons.csv"), index=False)

ratio_ab = loops_df.ratio_AB.dropna().values
print(f"ratio_AB: n={len(ratio_ab)}, median={np.median(ratio_ab):.3f}, "
      f"mean={np.mean(ratio_ab):.3f}")

# ---------------- paper's random control: anchor1 kept, random C ----------------
ratio_ac_all = []
for it in range(N_RANDOM_ROUNDS):
    for _, r in loops_df.iterrows():
        i = pos_to_operon(r.mid1)
        j = int(rng.integers(0, len(op)))
        v = expr_sim(i, j)
        if np.isfinite(v):
            ratio_ac_all.append(v)
ratio_ac_all = np.array(ratio_ac_all)
print(f"ratio_AC (paper's random anchor): n={len(ratio_ac_all)}, "
      f"median={np.median(ratio_ac_all):.3f}, mean={np.mean(ratio_ac_all):.3f}")

# ---------------- robustness: distance-matched random pairs ----------------
ratio_dm_all = []
for it in range(N_RANDOM_ROUNDS):
    for _, r in loops_df.iterrows():
        d = r.mid2 - r.mid1
        tries = 0
        while tries < 50:
            p1 = int(rng.integers(0, 4411532 - d))
            p2 = p1 + d
            i, j = pos_to_operon(p1), pos_to_operon(p2)
            v = expr_sim(i, j)
            if np.isfinite(v) and i != j:
                ratio_dm_all.append(v)
                break
            tries += 1
ratio_dm_all = np.array(ratio_dm_all)
print(f"ratio_distance-matched: n={len(ratio_dm_all)}, "
      f"median={np.median(ratio_dm_all):.3f}, mean={np.mean(ratio_dm_all):.3f}")

# ---------------- statistics ----------------
t_stat, p_t = stats.ttest_ind(ratio_ab, ratio_ac_all, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(ratio_ab, ratio_ac_all, alternative="greater")
t_stat2, p_t2 = stats.ttest_ind(ratio_ab, ratio_dm_all, equal_var=False)
u_stat2, p_u2 = stats.mannwhitneyu(ratio_ab, ratio_dm_all, alternative="greater")

print(f"\nloops vs paper-random:  Welch t p={p_t:.3e}, Mann-Whitney p={p_u:.3e}")
print(f"loops vs dist-matched:  Welch t p={p_t2:.3e}, Mann-Whitney p={p_u2:.3e}")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))

# A: boxplot, paper style
data = [ratio_ab, ratio_ac_all, ratio_dm_all]
labels = ["loops\n(n=41)", "random anchors\n(paper method,\n10 rounds)",
          "distance-matched\nrandom pairs\n(10 rounds)"]
bp = axes[0].boxplot(data, labels=labels, patch_artist=True,
                     showfliers=False, widths=0.55,
                     medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], [C_LOOP, C_RAND, "#999999"]):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
# strip points
for k, (vals, c) in enumerate(zip(data, [C_LOOP, C_RAND, "#666666"])):
    jitter = rng.normal(0, 0.04, len(vals))
    axes[0].scatter(np.full(len(vals), k + 1) + jitter, vals,
                    s=5, color=c, alpha=0.35, zorder=3)
axes[0].set_ylabel("Expression similarity\n(min/max of anchor operons)", fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("A  Transcriptional coordination of loop anchors",
                  loc="left", fontsize=10, fontweight="bold")
# significance bars
y_top = 0.98
for x1, x2, p in [(1, 2, p_t), (1, 3, p_t2)]:
    axes[0].plot([x1, x1, x2, x2], [y_top, y_top + 0.02, y_top + 0.02, y_top],
                 lw=1, color="black")
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    axes[0].text((x1 + x2) / 2, y_top + 0.025, star, ha="center", fontsize=11)
    y_top += 0.055

# B: per-round comparison (each round's random vs loops)
axes[1].axhline(np.median(ratio_ab), color=C_LOOP, lw=1.6,
                label=f"loops median = {np.median(ratio_ab):.2f}")
rand_medians_ac, rand_medians_dm = [], []
for it in range(N_RANDOM_ROUNDS):
    ra = ratio_ac_all[it * len(loops_df):(it + 1) * len(loops_df)]
    rd = ratio_dm_all[it * len(loops_df):(it + 1) * len(loops_df)]
    rand_medians_ac.append(np.median(ra))
    rand_medians_dm.append(np.median(rd))
xp = np.arange(N_RANDOM_ROUNDS)
axes[1].bar(xp - 0.2, rand_medians_ac, 0.4, color=C_RAND, alpha=0.85,
            edgecolor="black", label="random anchors (per round)")
axes[1].bar(xp + 0.2, rand_medians_dm, 0.4, color="#999999", alpha=0.85,
            edgecolor="black", label="distance-matched (per round)")
axes[1].set_xticks(xp, [f"r{i+1}" for i in xp], fontsize=8)
axes[1].set_ylabel("Median expression similarity", fontsize=9)
axes[1].set_ylim(0, 1.0)
axes[1].legend(fontsize=8, frameon=False)
axes[1].set_title("B  Loops vs 10 independent random rounds",
                  loc="left", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig1f_loop_coexpression.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1f_loop_coexpression.png"), dpi=300, bbox_inches="tight")

# ---------------- save ----------------
result = dict(
    n_loops=len(ratio_ab),
    loops_median=float(np.median(ratio_ab)),
    loops_mean=float(np.mean(ratio_ab)),
    random_anchor_median=float(np.median(ratio_ac_all)),
    dist_matched_median=float(np.median(ratio_dm_all)),
    welch_p_vs_random_anchor=float(p_t),
    mannwhitney_p_vs_random_anchor=float(p_u),
    welch_p_vs_dist_matched=float(p_t2),
    mannwhitney_p_vs_dist_matched=float(p_u2),
)
with open(os.path.join(OUT, "tables/fig1f_stats.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\n" + json.dumps(result, indent=2))
print("outputs in", OUT)
