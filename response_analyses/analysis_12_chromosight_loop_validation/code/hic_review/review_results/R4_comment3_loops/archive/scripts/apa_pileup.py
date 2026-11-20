#!/usr/bin/env python
"""APA (pileup) analysis of Mtb H37Rv Hi-C loops — Response to Reviewer 4, comment 3.

A. Aggregate peak analysis of 71 WT loops on the WT 5-kb matrix (O/E).
   Controls: 71 random distance-matched pairs (seed=42); loops shifted +100 kb.
B. Same WT loops piled up on the napM-KO matrix.
C. Figures (PDF + PNG 300 dpi) and enrichment CSVs.

Run with: ../.venv_cooltools/bin/python apa_pileup.py
"""

import os
import numpy as np
import pandas as pd
import cooler
import cooltools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------- paths ----------------
BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
LOOPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")

BINSIZE = 5000
FLANK = 50000          # +/- 50 kb -> 21 x 21 windows
SEED = 42
N_RANDOM = 71
SHIFT = 100000         # +100 kb shift control
GENOME = 4411532

rng = np.random.default_rng(SEED)

# ---------------- load loops ----------------
loops = pd.read_csv(LOOPS, sep="\t", header=None,
                    names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                           "name", "score", "strand1", "strand2"])
loops["center1"] = ((loops.start1 + loops.end1) // 2).astype(int)
loops["center2"] = ((loops.start2 + loops.end2) // 2).astype(int)
loops["dist"] = loops.center2 - loops.center1

# keep loops whose per-anchor +/- flank window fits inside the (linear) matrix
keep = (loops.center1 - FLANK >= 0) & (loops.center2 + FLANK < GENOME)
n_dropped = int((~keep).sum())
loops = loops[keep].reset_index(drop=True)
print(f"loops kept: {len(loops)} (dropped {n_dropped} near chromosome ends)")

# bedpe features for pileup: cooltools centers each 21x21 window on the two
# anchors, so the loop pixel lands at the exact window center (standard APA)
bedpe = loops[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()

# ---------------- random distance-matched control ----------------
# sample loop distances with replacement; choose anchor1 bins so windows fit
dists = loops.dist.values
random_dists = rng.choice(dists, size=N_RANDOM, replace=True)
rand_rows = []
for d in random_dists:
    lo = FLANK // BINSIZE                      # first usable anchor1 bin
    hi = (GENOME - FLANK) // BINSIZE - d // BINSIZE - 1
    b1 = int(rng.integers(lo, hi + 1)) * BINSIZE
    b2 = b1 + (d // BINSIZE) * BINSIZE
    rand_rows.append(("Chromosome", b1, b1 + BINSIZE, "Chromosome", b2, b2 + BINSIZE))
rand_bedpe = pd.DataFrame(rand_rows, columns=bedpe.columns)

# ---------------- +100 kb shift control ----------------
shift_bins = SHIFT // BINSIZE * BINSIZE
shift_bedpe = bedpe.copy()
shift_bedpe[["start1", "end1", "start2", "end2"]] += shift_bins
ok = shift_bedpe.end2 + FLANK < GENOME
print(f"shifted loops kept: {ok.sum()} of {len(shift_bedpe)}")
shift_bedpe = shift_bedpe[ok].reset_index(drop=True)

# ---------------- load coolers ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)

print("computing expected (cis) ...")
exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)

# ---------------- pileup ----------------
# NOTE: cooltools 0.7.1 pileup returns the full per-feature stack (n, D, D);
# aggregation is a nanmean over the stack.
def aggregate(stack):
    stack = np.where(np.isfinite(stack), stack, np.nan)  # expected==0 -> inf
    return np.nanmean(stack, axis=0)


print("pileup ...")
stack_wt_wt = cooltools.pileup(clr_wt, bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_wt_rand = cooltools.pileup(clr_wt, rand_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_wt_shift = cooltools.pileup(clr_wt, shift_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_ko_wt = cooltools.pileup(clr_ko, bedpe, expected_df=exp_ko, flank=FLANK, nproc=1)

pile_wt_wt = aggregate(stack_wt_wt)
pile_wt_rand = aggregate(stack_wt_rand)
pile_wt_shift = aggregate(stack_wt_shift)
pile_ko_wt = aggregate(stack_ko_wt)


def enrich(m):
    """Center 5x5 mean / mean of four 3x3 corners; also central pixel value."""
    n = m.shape[0]
    c = n // 2
    center = np.nanmean(m[c - 2:c + 3, c - 2:c + 3])
    corners = np.nanmean(np.concatenate([
        m[:3, :3].ravel(), m[:3, -3:].ravel(),
        m[-3:, :3].ravel(), m[-3:, -3:].ravel()]))
    return dict(center_5x5=float(center),
                corners_3x3=float(corners),
                enrichment=float(center / corners),
                center_pixel=float(m[c, c]))


results = {
    "WT loops on WT matrix": enrich(pile_wt_wt),
    "random pairs on WT matrix": enrich(pile_wt_rand),
    "WT loops +100kb shift on WT matrix": enrich(pile_wt_shift),
    "WT loops on napM-KO matrix": enrich(pile_ko_wt),
}
res_df = pd.DataFrame(results).T
res_df.index.name = "condition"
res_df.to_csv(os.path.join(OUT, "tables/apa_enrichment_scores.csv"))
print(res_df.round(3))

# save the aggregate matrices for reuse
np.save(os.path.join(OUT, "tables/apa_wt_loops_on_wt.npy"), pile_wt_wt)
np.save(os.path.join(OUT, "tables/apa_random_on_wt.npy"), pile_wt_rand)
np.save(os.path.join(OUT, "tables/apa_shift100kb_on_wt.npy"), pile_wt_shift)
np.save(os.path.join(OUT, "tables/apa_wt_loops_on_ko.npy"), pile_ko_wt)

# ---------------- per-loop comparison: WT vs napM-KO ----------------
# center 5x5 mean O/E of each individual loop snip in the two conditions
from scipy import stats

def per_loop_center(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    c = s.shape[1] // 2
    return np.nanmean(s[:, c - 2:c + 3, c - 2:c + 3], axis=(1, 2))

pl_wt = per_loop_center(stack_wt_wt)
pl_ko = per_loop_center(stack_ko_wt)
pl = pd.DataFrame({"loop_id": [f"{s}-{e}" for s, e in zip(loops.start1, loops.end2)],
                   "start1": loops.start1, "end2": loops.end2,
                   "OE_center5x5_WT": pl_wt,
                   "OE_center5x5_napMKO": pl_ko})
ok2 = np.isfinite(pl_wt) & np.isfinite(pl_ko)
w_stat, w_p = stats.wilcoxon(pl_wt[ok2], pl_ko[ok2])
pl.to_csv(os.path.join(OUT, "tables/per_loop_oe_wt_vs_ko.csv"), index=False)
print(f"per-loop center O/E (5x5): WT median={np.nanmedian(pl_wt):.3f}, "
      f"KO median={np.nanmedian(pl_ko):.3f}, "
      f"Wilcoxon signed-rank p={w_p:.3e} (n={ok2.sum()})")

# ---------------- figure ----------------
cmap = LinearSegmentedColormap.from_list("redwhite", ["#ffffff", "#d7301f"])
panels = [
    (pile_wt_wt, "WT loops on WT (n=71)", results["WT loops on WT matrix"]),
    (pile_wt_rand, "Random pairs on WT (n=71)", results["random pairs on WT matrix"]),
    (pile_wt_shift, "WT loops +100 kb on WT", results["WT loops +100kb shift on WT matrix"]),
    (pile_ko_wt, "WT loops on napM-KO", results["WT loops on napM-KO matrix"]),
]
vmax = max(np.nanmax(m) for m, *_ in panels)
vmin = min(np.nanmin(m) for m, *_ in panels)

fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.1), constrained_layout=True)
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50 kb", "-25 kb", "0", "+25 kb", "+50 kb"]
for i, (ax, (m, title, enr)) in enumerate(zip(axes, panels)):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    n = m.shape[0]
    c = n // 2
    # outline the center 5x5 region
    ax.add_patch(plt.Rectangle((c - 2.5, c - 2.5), 5, 5, fill=False,
                               edgecolor="black", lw=0.8))
    ax.set_xticks(ticks, ticklabels, fontsize=8)
    ax.set_yticks(ticks, ticklabels, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.text(-0.18, 1.06, chr(ord("A") + i), transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    ax.text(0.5, -0.34,
            f"center/corner = {enr['enrichment']:.2f}\ncenter pixel = {enr['center_pixel']:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, shrink=0.85,
                    label="Aggregate observed / expected")
fig.suptitle("Aggregate peak analysis (APA) of Mtb Hi-C loops (5 kb, $\\pm$50 kb flank)", fontsize=11)
fig.savefig(os.path.join(OUT, "figures/fig1_APA_pileup.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1_APA_pileup.png"), dpi=300, bbox_inches="tight")
print("figures written to", os.path.join(OUT, "figures"))

# ---------------- fig2: per-loop O/E, WT vs napM-KO ----------------
fig2, ax2 = plt.subplots(figsize=(3.6, 3.6), constrained_layout=True)
lo = min(np.nanmin(pl_wt[ok2]), np.nanmin(pl_ko[ok2])) * 0.95
hi = max(np.nanmax(pl_wt[ok2]), np.nanmax(pl_ko[ok2])) * 1.05
ax2.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8, zorder=1)
ax2.scatter(pl_wt[ok2], pl_ko[ok2], s=14, color="#d7301f", alpha=0.75, zorder=2)
ax2.set_xlabel("Loop center O/E (5x5) in WT", fontsize=9)
ax2.set_ylabel("Loop center O/E (5x5) in napM-KO", fontsize=9)
ax2.set_xlim(lo, hi)
ax2.set_ylim(lo, hi)
ax2.set_title(f"Per-loop strength, WT vs napM-KO\nWilcoxon p = {w_p:.1e}", fontsize=9.5)
fig2.savefig(os.path.join(OUT, "figures/fig2_per_loop_oe_wt_vs_ko.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "figures/fig2_per_loop_oe_wt_vs_ko.png"), dpi=300, bbox_inches="tight")
