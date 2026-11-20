#!/usr/bin/env python
"""CID boundary validation by pileup / aggregate analyses (R4 comment 2).

Four independent lines of evidence that the re-called (confirmed) CID
boundaries are correct:

  P1. Boundary-centred contact pileup ("insulation cross"): mean O/E matrix
      of +/-150 kb windows centred on boundaries shows a depletion cross
      through the centre; random positions do not.
  P2. Insulation score metaplot: sliding-window insulation score dips at
      boundaries (aggregate over all confirmed boundaries vs random).
  P3. Within-CID vs cross-boundary contact enrichment with a permutation
      test: pixel pairs inside the same CID are more enriched (O/E) than
      same-distance pairs spanning a boundary, compared to 1000 random
      partitions with the same CID-size distribution.
  P4. DI metaplot: average DI profile aligned on boundaries shows a clean
      negative-to-positive transition; random positions show none.

Uses the 13 CONFIRMED boundaries (transition + >=3/4 replicate reproducibility)
from cid_boundary_validation.py.

Run: ../../.venv_cooltools/bin/python cid_pileup_validation.py
"""

import os
import json
import numpy as np
import pandas as pd
import cooler
import cooltools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment2_CID")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
DI_WT = os.path.join(BASE, "review_data/DI_Score_results/wt.csv")
BD_CSV = os.path.join(OUT, "tables/cid_boundaries_recalled.csv")

BINSIZE = 5000
SEED = 42
rng = np.random.default_rng(SEED)

C_WT, C_RAND = "#0072B2", "#999999"
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list("oe", ["#2166ac", "#f7f7f7", "#b2182b"])

print("=" * 60)
print("CID boundary pileup validation")
print("=" * 60)

# ---------------- load ----------------
clr = cooler.Cooler(WT_COOL)
mat = np.nan_to_num(clr.matrix(sparse=False).fetch("Chromosome"), nan=0.0)
n = mat.shape[0]
di = pd.read_csv(DI_WT, header=None).values.flatten()

bd = pd.read_csv(BD_CSV)
bd_conf = bd[bd.confirmed].reset_index(drop=True)
conf_bins = bd_conf["bin"].astype(int).tolist()
print(f"confirmed boundaries: {len(conf_bins)}")

# expected (distance-decay) for O/E
exp = cooltools.expected_cis(clr, nproc=1)
exp_diag = exp.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()
oe = np.full_like(mat, np.nan, dtype=float)
for d in range(n):
    e = exp_diag.get(d, np.nan)
    if e and e > 0:
        rows = np.arange(n - d)
        cols = rows + d
        vals = np.diag(mat, k=d) / e
        oe[rows, cols] = vals
        oe[cols, rows] = vals

# ---------------- P1: boundary-centred contact pileup ----------------
print("\n[P1] boundary-centred contact pileup (insulation cross) ...")
FL = 30  # +-150 kb

def pileup_windows(centers, matrix, diag_mask=3):
    stack = []
    for c in centers:
        if c - FL < 0 or c + FL + 1 > matrix.shape[0]:
            continue
        w = matrix[c - FL:c + FL + 1, c - FL:c + FL + 1].copy()
        # mask diagonal band so the domain structure (not the diagonal) shows
        nd = w.shape[0]
        for d in range(-diag_mask, diag_mask + 1):
            idx = np.arange(max(0, -d), nd - max(0, d))
            w[idx, idx + d] = np.nan
        stack.append(w)
    return np.nanmean(np.array(stack), axis=0) if stack else None

rand_centers = []
while len(rand_centers) < len(conf_bins):
    c = int(rng.integers(FL + 5, n - FL - 5))
    if all(abs(c - b) > 10 for b in conf_bins):
        rand_centers.append(c)

pile_bd = pileup_windows(conf_bins, oe)
pile_rd = pileup_windows(rand_centers, oe)

def cross_depletion(p, arm=5, corner=10):
    """insulation cross: mean O/E along the boundary row/column (arms, diagonal
    already masked) vs mean O/E in the four corner blocks (within-domain).
    <1 means cross-boundary contacts are depleted (a real boundary)."""
    c = p.shape[0] // 2
    arms = np.concatenate([
        p[c - arm:c + arm + 1, :c - corner].ravel(),   # left arm
        p[c - arm:c + arm + 1, c + corner + 1:].ravel(),  # right arm
        p[:c - corner, c - arm:c + arm + 1].ravel(),   # top arm
        p[c + corner + 1:, c - arm:c + arm + 1].ravel(),  # bottom arm
    ])
    corners = np.concatenate([
        p[:c - corner, :c - corner].ravel(),
        p[:c - corner, c + corner + 1:].ravel(),
        p[c + corner + 1:, :c - corner].ravel(),
        p[c + corner + 1:, c + corner + 1:].ravel(),
    ])
    arms = arms[np.isfinite(arms)]
    corners = corners[np.isfinite(corners)]
    return float(np.nanmedian(arms)), float(np.nanmedian(corners)), \
           float(np.nanmedian(arms) / np.nanmedian(corners))

# ---------------- P2: insulation score metaplot ----------------
print("[P2] insulation score metaplot ...")

def insulation_score(m, w=20):
    """diamond insulation on the O/E matrix: mean O/E in a w x w square
    immediately downstream of each bin. Smoothed with a moving average."""
    n = m.shape[0]
    ins = np.full(n, np.nan)
    for i in range(n - 2 * w):
        v = m[i:i + w, i + w:i + 2 * w]
        v = v[np.isfinite(v)]
        if len(v) > w * w * 0.5:
            ins[i] = np.nanmedian(v)
    # light smoothing
    k = 5
    sm = np.copy(ins)
    for i in range(n):
        lo, hi = max(0, i - k // 2), min(n, i + k // 2 + 1)
        sm[i] = np.nanmedian(ins[lo:hi])
    return sm

ins = insulation_score(oe, w=20)

def meta_profile(centers, track, half=40):
    prof = []
    for c in centers:
        if c - half < 0 or c + half + 1 > len(track):
            continue
        prof.append(track[c - half:c + half + 1])
    return np.nanmean(np.array(prof), axis=0)

ins_bd = meta_profile(conf_bins, ins)
ins_rd = meta_profile(rand_centers, ins)
di_bd = meta_profile(conf_bins, di)
di_rd = meta_profile(rand_centers, di)

# ---------------- P3: within-CID vs cross-boundary enrichment ----------------
print("[P3] within-CID vs cross-boundary enrichment + permutation ...")

def within_vs_cross(boundaries_bins, oe, d_lo=6, d_hi=30):
    """MEDIAN O/E of pixel pairs within same CID vs spanning a boundary.
    Short-to-mid distances (30-150 kb) where domain structure is strongest."""
    bounds = sorted(boundaries_bins)
    within, cross = [], []
    for i in range(n):
        # precompute how many bounds after i
        for d in range(d_lo, d_hi):
            j = i + d
            if j >= n:
                continue
            v = oe[i, j]
            if not np.isfinite(v):
                continue
            nb = sum(1 for b in bounds if i < b <= j)
            (cross if nb > 0 else within).append(v)
    return np.array(within), np.array(cross)

w_in, w_cross = within_vs_cross(conf_bins, oe)
obs_ratio = np.nanmedian(w_in) / np.nanmedian(w_cross)
obs_stat, obs_p = stats.mannwhitneyu(w_in, w_cross, alternative="greater")
print(f"  within-CID median O/E={np.nanmedian(w_in):.3f} (n={len(w_in)}), "
      f"cross-boundary median O/E={np.nanmedian(w_cross):.3f} (n={len(w_cross)}), "
      f"ratio={obs_ratio:.3f}, MW p={obs_p:.2e}")

# permutation: random partitions with same CID sizes
sizes = np.diff([0] + sorted(conf_bins) + [n - 1])
null_ratios = []
for it in range(1000):
    shuffled = np.concatenate([[0], np.cumsum(rng.permutation(sizes))[:-1]])
    rb = [int(x) for x in shuffled[1:] if 0 < x < n - 1]
    wi, wc = within_vs_cross(rb, oe)
    if len(wc) > 100:
        null_ratios.append(np.nanmedian(wi) / np.nanmedian(wc))
null_ratios = np.array(null_ratios)
perm_p = (np.sum(null_ratios >= obs_ratio) + 1) / (len(null_ratios) + 1)
print(f"  permutation null ratio: {null_ratios.mean():.3f}±{null_ratios.std():.3f}, "
      f"empirical p={perm_p:.4f}")

# ---------------- figures ----------------
print("rendering figures ...")
fig = plt.figure(figsize=(15, 9))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                       left=0.06, right=0.97, top=0.90, bottom=0.09)

# P1a: pileup boundaries
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(pile_bd, cmap=cmap_oe, vmin=0.7, vmax=1.3, interpolation="nearest")
cb_ = pile_bd.shape[0] // 2
ax.axhline(cb_, color="black", ls="--", lw=0.8)
ax.axvline(cb_, color="black", ls="--", lw=0.8)
arm_b, cor_b, depl_b = cross_depletion(pile_bd)
ax.set_title(f"A  Contact pileup at boundaries (n={len(conf_bins)})\n"
             f"cross-arms/corners = {depl_b:.2f} (<1 = insulation)",
             loc="left", fontsize=10, fontweight="bold")
ax.set_xticks([0, cb_, 2 * cb_], ["-150 kb", "boundary", "+150 kb"], fontsize=8)
ax.set_yticks([0, cb_, 2 * cb_], ["-150 kb", "boundary", "+150 kb"], fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.7, label="agg. O/E")

# P1b: pileup random
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(pile_rd, cmap=cmap_oe, vmin=0.7, vmax=1.3, interpolation="nearest")
ax.axhline(cb_, color="black", ls="--", lw=0.8)
ax.axvline(cb_, color="black", ls="--", lw=0.8)
arm_r, cor_r, depl_r = cross_depletion(pile_rd)
ax.set_title(f"B  Pileup at random positions (n={len(rand_centers)})\n"
             f"cross-arms/corners = {depl_r:.2f}",
             loc="left", fontsize=10, fontweight="bold")
ax.set_xticks([0, cb_, 2 * cb_], ["-150 kb", "centre", "+150 kb"], fontsize=8)
ax.set_yticks([0, cb_, 2 * cb_], ["-150 kb", "centre", "+150 kb"], fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.7, label="agg. O/E")

# P3: permutation null for within/cross ratio
ax = fig.add_subplot(gs[0, 2])
ax.hist(null_ratios, bins=40, color=C_RAND, edgecolor="black", alpha=0.8)
ax.axvline(obs_ratio, color=C_WT, lw=2)
ax.text(obs_ratio, ax.get_ylim()[1] * 0.85, f"observed={obs_ratio:.2f}\np={perm_p:.3f}",
        color=C_WT, fontsize=9, ha="center", fontweight="bold")
ax.set_xlabel("within-partition / cross-boundary median O/E ratio", fontsize=9)
ax.set_ylabel("Random partitions", fontsize=9)
ax.set_title("C  Within-CID contact enrichment\n(permutation test, 1000 random partitions)",
             loc="left", fontsize=10, fontweight="bold")

# P2: insulation metaplot
ax = fig.add_subplot(gs[1, 0])
xmeta = (np.arange(-40, 41) * BINSIZE / 1000)
ax.plot(xmeta, ins_bd, color=C_WT, lw=1.6, label=f"boundaries (n={len(conf_bins)})")
ax.plot(xmeta, ins_rd, color=C_RAND, lw=1.6, label=f"random (n={len(rand_centers)})")
ax.axvline(0, color="black", ls="--", lw=0.8)
ax.set_xlabel("Distance from boundary (kb)", fontsize=9)
ax.set_ylabel("Insulation score", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("D  Insulation score metaplot\n(dip at boundaries)", loc="left",
             fontsize=10, fontweight="bold")

# P4: DI metaplot
ax = fig.add_subplot(gs[1, 1])
ax.plot(xmeta, di_bd, color=C_WT, lw=1.6, label=f"boundaries (n={len(conf_bins)})")
ax.plot(xmeta, di_rd, color=C_RAND, lw=1.6, label=f"random (n={len(rand_centers)})")
ax.axvline(0, color="black", ls="--", lw=0.8)
ax.axhline(0, color="grey", ls=":", lw=0.7)
ax.set_ylim(-2, 2)
ax.set_xlabel("Distance from boundary (kb)", fontsize=9)
ax.set_ylabel("Mean DI", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("E  DI metaplot at boundaries\n(clean negative→positive transition)",
             loc="left", fontsize=10, fontweight="bold")

# P3b: within vs cross distributions
ax = fig.add_subplot(gs[1, 2])
bp = ax.boxplot([np.log2(w_in), np.log2(w_cross)], labels=["within CID", "cross boundary"],
                patch_artist=True, showfliers=False, widths=0.5,
                medianprops=dict(color="black", lw=1.3))
for patch, c in zip(bp["boxes"], [C_WT, C_RAND]):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.set_ylabel("log2 O/E contact", fontsize=9)
ax.set_title(f"F  Contact O/E: within CID vs cross-boundary\nMann-Whitney p={obs_p:.1e}",
             loc="left", fontsize=10, fontweight="bold")

fig.suptitle("Aggregate validation of re-called CID boundaries (pileup / insulation / permutation)",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig4_pileup_validation.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig4_pileup_validation.png"), dpi=300, bbox_inches="tight")

# ---------------- summary ----------------
summary = dict(
    n_confirmed_boundaries=len(conf_bins),
    P1_insulation_cross=dict(
        boundary_cross_arms_median=arm_b, boundary_corners_median=cor_b,
        boundary_depletion=depl_b,
        random_cross_arms_median=arm_r, random_corners_median=cor_r,
        random_depletion=depl_r),
    P3_within_vs_cross=dict(
        within_median_OE=float(np.nanmedian(w_in)), cross_median_OE=float(np.nanmedian(w_cross)),
        ratio=obs_ratio, mannwhitney_p=float(obs_p),
        permutation_null_mean=float(null_ratios.mean()),
        permutation_null_sd=float(null_ratios.std()),
        empirical_p=float(perm_p)),
)
with open(os.path.join(OUT, "tables/pileup_validation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary, indent=2))
print("done.")
