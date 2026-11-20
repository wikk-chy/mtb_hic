#!/usr/bin/env python
"""Extended loop validation for R4 comment 3.

This script builds on apa_pileup.py and adds:
1. Robust / outlier-excluded statistics (median, IQR, winsorized mean)
2. Representative zoomed-in contact maps with raw contacts and local background
3. Reproducibility across WT replicates
4. An alternative, bacteria-appropriate loop detection using cooltools
   expected-cis background (local enrichment z-score) and comparison to HiCCUPS calls
5. HiCCUPS parameter sensitivity discussion

Output goes to review_results/R4_comment3_loops/ (overwrites previous results).
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
from scipy.stats import mannwhitneyu, wilcoxon

# ---------------- paths ----------------
BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
LOOPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")
REP_DIR = os.path.join(BASE, "review_data/wt_rep_icedmatrix")

os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "scripts"), exist_ok=True)

BINSIZE = 5000
FLANK = 50000
SEED = 42
N_RANDOM = 71
SHIFT = 100000
GENOME = 4411532

rng = np.random.default_rng(SEED)

print("=" * 60)
print("R4_comment3 extended loop validation")
print("=" * 60)

# ---------------- load loops ----------------
loops = pd.read_csv(LOOPS, sep="\t", header=None,
                    names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                           "name", "score", "strand1", "strand2"])
loops["center1"] = ((loops.start1 + loops.end1) // 2).astype(int)
loops["center2"] = ((loops.start2 + loops.end2) // 2).astype(int)
loops["dist"] = loops.center2 - loops.center1

keep = (loops.center1 - FLANK >= 0) & (loops.center2 + FLANK < GENOME)
n_dropped = int((~keep).sum())
loops = loops[keep].reset_index(drop=True)
print(f"loops kept: {len(loops)} (dropped {n_dropped} near chromosome ends)")

bedpe = loops[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()

# ---------------- controls ----------------
dists = loops.dist.values
random_dists = rng.choice(dists, size=N_RANDOM, replace=True)
rand_rows = []
for d in random_dists:
    lo = FLANK // BINSIZE
    hi = (GENOME - FLANK) // BINSIZE - d // BINSIZE - 1
    b1 = int(rng.integers(lo, hi + 1)) * BINSIZE
    b2 = b1 + (d // BINSIZE) * BINSIZE
    rand_rows.append(("Chromosome", b1, b1 + BINSIZE, "Chromosome", b2, b2 + BINSIZE))
rand_bedpe = pd.DataFrame(rand_rows, columns=bedpe.columns)

shift_bins = SHIFT // BINSIZE * BINSIZE
shift_bedpe = bedpe.copy()
shift_bedpe[["start1", "end1", "start2", "end2"]] += shift_bins
ok_shift = shift_bedpe.end2 + FLANK < GENOME
shift_bedpe = shift_bedpe[ok_shift].reset_index(drop=True)

# ---------------- load coolers ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)

print("computing expected (cis) ...")
exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)

# ---------------- pileup helpers ----------------
def aggregate(stack):
    stack = np.where(np.isfinite(stack), stack, np.nan)
    return np.nanmean(stack, axis=0)

def enrich(m):
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

print("pileup ...")
stack_wt_wt = cooltools.pileup(clr_wt, bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_wt_rand = cooltools.pileup(clr_wt, rand_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_wt_shift = cooltools.pileup(clr_wt, shift_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_ko_wt = cooltools.pileup(clr_ko, bedpe, expected_df=exp_ko, flank=FLANK, nproc=1)

pile_wt_wt = aggregate(stack_wt_wt)
pile_wt_rand = aggregate(stack_wt_rand)
pile_wt_shift = aggregate(stack_wt_shift)
pile_ko_wt = aggregate(stack_ko_wt)

# save aggregate matrices
np.save(os.path.join(OUT, "tables/apa_wt_loops_on_wt.npy"), pile_wt_wt)
np.save(os.path.join(OUT, "tables/apa_random_on_wt.npy"), pile_wt_rand)
np.save(os.path.join(OUT, "tables/apa_shift100kb_on_wt.npy"), pile_wt_shift)
np.save(os.path.join(OUT, "tables/apa_wt_loops_on_ko.npy"), pile_ko_wt)

# ---------------- robust / outlier-excluded statistics ----------------
def per_loop_center(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    c = s.shape[1] // 2
    return np.nanmean(s[:, c - 2:c + 3, c - 2:c + 3], axis=(1, 2))

pl_wt = per_loop_center(stack_wt_wt)
pl_ko = per_loop_center(stack_ko_wt)
pl = pd.DataFrame({
    "loop_id": [f"{s}-{e}" for s, e in zip(loops.start1, loops.end2)],
    "start1": loops.start1, "end2": loops.end2,
    "dist_kb": loops.dist // 1000,
    "OE_center5x5_WT": pl_wt,
    "OE_center5x5_napMKO": pl_ko,
})

# identify outliers using IQR
q1, q3 = np.nanpercentile(pl_wt, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
pl["outlier_WT"] = (pl_wt < lower) | (pl_wt > upper)
print(f"outlier loops by IQR: {pl['outlier_WT'].sum()}")

# winsorized mean (5% trimming)
def winsorized_mean(x, limits=(0.05, 0.05)):
    lo, hi = np.nanpercentile(x, [limits[0]*100, (1-limits[1])*100])
    return np.nanmean(np.clip(x, lo, hi))

stats_summary = {
    "WT all loops (n)": int(np.isfinite(pl_wt).sum()),
    "WT mean O/E": float(np.nanmean(pl_wt)),
    "WT median O/E": float(np.nanmedian(pl_wt)),
    "WT IQR O/E": float(iqr),
    "WT winsorized5% O/E": float(winsorized_mean(pl_wt)),
    "WT outlier-excluded mean": float(np.nanmean(pl_wt[~pl['outlier_WT']])),
    "KO all loops (n)": int(np.isfinite(pl_ko).sum()),
    "KO mean O/E": float(np.nanmean(pl_ko)),
    "KO median O/E": float(np.nanmedian(pl_ko)),
    "KO winsorized5% O/E": float(winsorized_mean(pl_ko)),
}

ok2 = np.isfinite(pl_wt) & np.isfinite(pl_ko)
w_stat, w_p = wilcoxon(pl_wt[ok2], pl_ko[ok2])
stats_summary["Wilcoxon p WT vs KO"] = float(w_p)

with open(os.path.join(OUT, "tables/robust_statistics.json"), "w") as f:
    json.dump(stats_summary, f, indent=2)

pl.to_csv(os.path.join(OUT, "tables/per_loop_oe_wt_vs_ko.csv"), index=False)
print(f"per-loop center O/E (5x5): WT median={np.nanmedian(pl_wt):.3f}, "
      f"KO median={np.nanmedian(pl_ko):.3f}, "
      f"Wilcoxon signed-rank p={w_p:.3e} (n={ok2.sum()})")

# ---------------- replicate matrices (load sparse .matrix files) ----------------
def load_rep_matrix(path, n_bins=883):
    mat = np.zeros((n_bins, n_bins))
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                v = float(parts[2])
                if 0 <= i < n_bins and 0 <= j < n_bins:
                    mat[i, j] = v
                    mat[j, i] = v
            except ValueError:
                continue
    return mat

rep_files = sorted([f for f in os.listdir(REP_DIR) if f.endswith(".matrix")])
rep_mats = {f.replace("_5000_iced.matrix", "").replace("wt_", "rep"): load_rep_matrix(os.path.join(REP_DIR, f))
            for f in rep_files}
print(f"loaded {len(rep_mats)} replicate matrices")

# per-replicate recovery of loop anchors
bin1 = loops.center1.values // BINSIZE
bin2 = loops.center2.values // BINSIZE
rep_results = []
for rep_id, mat in rep_mats.items():
    scores = []
    recov = 0
    for b1, b2 in zip(bin1, bin2):
        if b1 >= mat.shape[0] or b2 >= mat.shape[1]:
            scores.append(np.nan)
            continue
        val = mat[b1, b2]
        # local background (3-bin ring around anchor pixel, excluding immediate neighbours)
        bg_vals = []
        for di in range(-3, 4):
            for dj in range(-3, 4):
                if abs(di) <= 1 and abs(dj) <= 1:
                    continue
                ni, nj = b1 + di, b2 + dj
                if 0 <= ni < mat.shape[0] and 0 <= nj < mat.shape[1]:
                    bg_vals.append(mat[ni, nj])
        bg = np.mean(bg_vals) if bg_vals else 1.0
        score = val / bg if bg > 0 else np.nan
        scores.append(score)
        if score > 1.5:
            recov += 1
    rep_results.append({"replicate": rep_id,
                        "recovered": recov,
                        "recovery_rate": recov / len(loops),
                        "median_loop_score": float(np.nanmedian(scores))})

rep_df = pd.DataFrame(rep_results)
rep_df.to_csv(os.path.join(OUT, "tables/replicate_recovery.csv"), index=False)
print(rep_df)

# ---------------- figure 1: APA pileup ----------------
cmap = LinearSegmentedColormap.from_list("redwhite", ["#ffffff", "#d7301f"])
results = {
    "WT loops on WT matrix": enrich(pile_wt_wt),
    "random pairs on WT matrix": enrich(pile_wt_rand),
    "WT loops +100kb shift on WT matrix": enrich(pile_wt_shift),
    "WT loops on napM-KO matrix": enrich(pile_ko_wt),
}
res_df = pd.DataFrame(results).T
res_df.index.name = "condition"
res_df.to_csv(os.path.join(OUT, "tables/apa_enrichment_scores.csv"))

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
fig.suptitle("Aggregate peak analysis (APA) of Mtb Hi-C loops (5 kb, $\pm$50 kb flank)", fontsize=11)
fig.savefig(os.path.join(OUT, "figures/fig1_APA_pileup.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1_APA_pileup.png"), dpi=300, bbox_inches="tight")

# ---------------- figure 2: per-loop O/E WT vs KO with outliers highlighted ----------------
fig2, ax2 = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
lo = min(np.nanmin(pl_wt[ok2]), np.nanmin(pl_ko[ok2])) * 0.95
hi = max(np.nanmax(pl_wt[ok2]), np.nanmax(pl_ko[ok2])) * 1.05
ax2.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8, zorder=1)
colors = np.where(pl.loc[ok2, "outlier_WT"].values, "#cb5148", "#297270")
ax2.scatter(pl_wt[ok2], pl_ko[ok2], s=18, c=colors, alpha=0.75, zorder=2)
ax2.set_xlabel("Loop center O/E (5x5) in WT", fontsize=9)
ax2.set_ylabel("Loop center O/E (5x5) in napM-KO", fontsize=9)
ax2.set_xlim(lo, hi)
ax2.set_ylim(lo, hi)
ax2.set_title(f"Per-loop strength, WT vs napM-KO\nWilcoxon p = {w_p:.1e}\nred = outlier (IQR)", fontsize=9.5)
fig2.savefig(os.path.join(OUT, "figures/fig2_per_loop_oe_wt_vs_ko.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "figures/fig2_per_loop_oe_wt_vs_ko.png"), dpi=300, bbox_inches="tight")

# ---------------- figure 3: zoomed contact maps for representative loops ----------------
# select representative loops: top non-outlier, top outlier, median, weak, long, short
pl_sorted = pl.sort_values("OE_center5x5_WT", ascending=False).reset_index(drop=True)
selected = []
for label, criterion in [
    ("strong non-outlier", pl_sorted[~pl_sorted.outlier_WT].head(1)),
    ("strong outlier", pl_sorted[pl_sorted.outlier_WT].head(1)),
    ("median", pl_sorted.iloc[[len(pl_sorted)//2]]),
    ("weakest", pl_sorted.tail(1)),
]:
    if not criterion.empty:
        row = criterion.iloc[0]
        selected.append((label, row))

# also add a long-distance and short-distance loop
long_loop = pl.sort_values("dist_kb", ascending=False).iloc[[0]]
short_loop = pl.sort_values("dist_kb").iloc[[0]]
selected.append(("long distance", long_loop.iloc[0]))
selected.append(("short distance", short_loop.iloc[0]))

fig3 = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.35, wspace=0.3)

for idx, (label, row) in enumerate(selected[:6]):
    ax = fig3.add_subplot(gs[idx // 3, idx % 3])
    mid1 = int(row["start1"]) + 2500
    mid2 = int(row["end2"]) - 2500
    span = abs(mid2 - mid1)
    window = min(max(span // 2, 50000), 300000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    try:
        mat = clr_wt.matrix(sparse=False).fetch(
            f"Chromosome:{r1_s}-{r1_e}",
            f"Chromosome:{r2_s}-{r2_e}"
        )
        im = ax.imshow(mat, cmap=cmap,
                       vmin=np.nanpercentile(mat[mat > 0], 2),
                       vmax=np.nanpercentile(mat, 98),
                       interpolation="nearest")
        # anchor lines
        a1_rel = (mid1 - r1_s) / (r1_e - r1_s)
        a2_rel = (mid2 - r2_s) / (r2_e - r2_s)
        ax.axhline(a2_rel * mat.shape[0], color="blue", linestyle="--", lw=1)
        ax.axvline(a1_rel * mat.shape[1], color="blue", linestyle="--", lw=1)
        plt.colorbar(im, ax=ax, shrink=0.6, label="Contact freq")
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:50], ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{label}\n{mid1//1000}kb × {mid2//1000}kb  O/E={row['OE_center5x5_WT']:.2f}",
                 fontsize=9)
    ax.set_xlabel("Anchor 1 region (kb)", fontsize=8)
    ax.set_ylabel("Anchor 2 region (kb)", fontsize=8)

fig3.suptitle("Zoomed contact maps: representative WT loops (blue lines = anchors)", fontsize=13)
fig3.savefig(os.path.join(OUT, "figures/fig3_zoomed_loops.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(OUT, "figures/fig3_zoomed_loops.png"), dpi=300, bbox_inches="tight")

# ---------------- figure 4: reproducibility across replicates ----------------
fig4, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

# recovery barplot
axes[0].bar(rep_df["replicate"], rep_df["recovery_rate"] * 100,
            color="#297270", edgecolor="black")
axes[0].axhline(50, color="red", linestyle="--", alpha=0.7)
axes[0].set_xlabel("Replicate", fontsize=10)
axes[0].set_ylabel("Loops recovered (%)", fontsize=10)
axes[0].set_title(f"Loop recovery per replicate\n(threshold >1.5× local background)", fontsize=10)
for i, (rep, r) in enumerate(zip(rep_df["replicate"], rep_df["recovery_rate"])):
    axes[0].text(i, r * 100 + 1, f"{r*100:.1f}%", ha="center", fontsize=9)

# pairwise correlation of loop enrichment scores across replicates
rep_ids = sorted(rep_mats.keys())
n_rep = len(rep_ids)
corr_mat = np.zeros((n_rep, n_rep))
for i, r1 in enumerate(rep_ids):
    for j, r2 in enumerate(rep_ids):
        v1, v2 = [], []
        for b1, b2 in zip(bin1, bin2):
            if b1 < rep_mats[r1].shape[0] and b2 < rep_mats[r1].shape[1]:
                v1.append(rep_mats[r1][b1, b2])
                v2.append(rep_mats[r2][b1, b2])
        if len(v1) > 3:
            corr_mat[i, j], _ = stats.pearsonr(v1, v2)

im = axes[1].imshow(corr_mat, cmap="RdYlGn", vmin=0, vmax=1)
axes[1].set_xticks(np.arange(n_rep))
axes[1].set_yticks(np.arange(n_rep))
axes[1].set_xticklabels(rep_ids)
axes[1].set_yticklabels(rep_ids)
axes[1].set_title("Pearson correlation of loop\nanchor contact frequencies", fontsize=10)
for i in range(n_rep):
    for j in range(n_rep):
        axes[1].text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                     color="white" if corr_mat[i,j] > 0.7 else "black", fontsize=11)
plt.colorbar(im, ax=axes[1], shrink=0.8, label="Pearson r")

fig4.savefig(os.path.join(OUT, "figures/fig4_replicate_reproducibility.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(OUT, "figures/fig4_replicate_reproducibility.png"), dpi=300, bbox_inches="tight")

# ---------------- figure 5: robust statistics comparison ----------------
fig5, ax5 = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
metrics = ["mean", "median", "winsorized5%", "outlier-excluded"]
wt_vals = [stats_summary["WT mean O/E"], stats_summary["WT median O/E"],
           stats_summary["WT winsorized5% O/E"], stats_summary["WT outlier-excluded mean"]]
ko_vals = [stats_summary["KO mean O/E"], stats_summary["KO median O/E"],
           stats_summary["KO winsorized5% O/E"], np.nanmedian(pl_ko)]

x = np.arange(len(metrics))
width = 0.35
ax5.bar(x - width/2, wt_vals, width, label="WT", color="#297270", edgecolor="black")
ax5.bar(x + width/2, ko_vals, width, label="napM-KO", color="#cb5148", edgecolor="black")
ax5.set_xticks(x)
ax5.set_xticklabels(metrics)
ax5.set_ylabel("Center 5x5 O/E", fontsize=10)
ax5.set_title("Robust statistics for loop enrichment\n(all, median, winsorized, outlier-excluded)", fontsize=10)
ax5.legend()
fig5.savefig(os.path.join(OUT, "figures/fig5_robust_statistics.pdf"), bbox_inches="tight")
fig5.savefig(os.path.join(OUT, "figures/fig5_robust_statistics.png"), dpi=300, bbox_inches="tight")

# ---------------- alternative loop detection: local enrichment z-score ----------------
print("running alternative loop detection (local enrichment z-score) ...")

# get expected values per diagonal
exp_wt_diag = exp_wt.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()

# compute O/E matrix for WT
mat_wt = clr_wt.matrix(sparse=False).fetch("Chromosome")
n = mat_wt.shape[0]
oe_wt = np.full_like(mat_wt, np.nan, dtype=float)
for d in range(n):
    exp_val = exp_wt_diag.get(d, np.nan)
    if exp_val and exp_val > 0:
        diag_vals = np.diag(mat_wt, k=d)
        oe_vals = diag_vals / exp_val
        # fill diagonal
        rows = np.arange(n - d)
        cols = rows + d
        oe_wt[rows, cols] = oe_vals
        oe_wt[cols, rows] = oe_vals

# detect local peaks using a simple kernel
def find_local_peaks(oe_mat, min_dist_bins=10, max_dist_bins=200,
                     window=3, z_thresh=2.0):
    """Find pixels whose O/E exceeds local mean + z_thresh*std."""
    peaks = []
    n = oe_mat.shape[0]
    for d in range(min_dist_bins, max_dist_bins):
        for i in range(n - d):
            j = i + d
            val = oe_mat[i, j]
            if not np.isfinite(val):
                continue
            # local background: window around (i,j) excluding center and close diagonal
            bg = []
            for di in range(-window, window + 1):
                for dj in range(-window, window + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or nj < 0 or ni >= n or nj >= n:
                        continue
                    if abs(ni - nj) < min_dist_bins:
                        continue
                    if np.isfinite(oe_mat[ni, nj]):
                        bg.append(oe_mat[ni, nj])
            if len(bg) < 5:
                continue
            bg_mean = np.mean(bg)
            bg_std = np.std(bg)
            if bg_std == 0:
                continue
            z = (val - bg_mean) / bg_std
            if z > z_thresh:
                peaks.append({"bin1": i, "bin2": j, "oe": val, "z": z,
                              "dist_bins": d, "bg_mean": bg_mean, "bg_std": bg_std})
    return pd.DataFrame(peaks)

peaks = find_local_peaks(oe_wt)
peaks.to_csv(os.path.join(OUT, "tables/local_enrichment_peaks.csv"), index=False)
print(f"local enrichment peaks detected: {len(peaks)}")

# compare with HiCCUPS loops: overlap with peaks
hiccups_bins = set()
for _, row in loops.iterrows():
    hiccups_bins.add((row.center1 // BINSIZE, row.center2 // BINSIZE))

peak_bins = set((r.bin1, r.bin2) for _, r in peaks.iterrows())
overlap_direct = len(hiccups_bins & peak_bins)
overlap_tolerant = 0
for hb1, hb2 in hiccups_bins:
    for pb1, pb2 in peak_bins:
        if abs(hb1 - pb1) <= 1 and abs(hb2 - pb2) <= 1:
            overlap_tolerant += 1
            break

overlap_summary = {
    "HiCCUPS loops": len(hiccups_bins),
    "local enrichment peaks": len(peak_bins),
    "direct overlap": overlap_direct,
    "tolerant overlap (±1 bin)": overlap_tolerant,
    "tolerant fraction": overlap_tolerant / len(hiccups_bins) if hiccups_bins else 0,
}
with open(os.path.join(OUT, "tables/method_comparison.json"), "w") as f:
    json.dump(overlap_summary, f, indent=2)
print(overlap_summary)

# ---------------- figure 6: method comparison ----------------
fig6, ax6 = plt.subplots(figsize=(5, 4.2), constrained_layout=True)
labels = ["HiCCUPS loops", "local-enrichment peaks"]
counts = [len(hiccups_bins), len(peak_bins)]
bar_colors = ["#297270", "#8ab07c"]
ax6.bar(labels, counts, color=bar_colors, edgecolor="black")
ax6.set_ylabel("Count", fontsize=10)
ax6.set_title(f"HiCCUPS vs local-enrichment method\noverlap: {overlap_tolerant}/{len(hiccups_bins)} loops (±1 bin)", fontsize=10)
fig6.savefig(os.path.join(OUT, "figures/fig6_method_comparison.pdf"), bbox_inches="tight")
fig6.savefig(os.path.join(OUT, "figures/fig6_method_comparison.png"), dpi=300, bbox_inches="tight")

# ---------------- combined summary figure ----------------
fig7 = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig7, hspace=0.4, wspace=0.3)

# A: APA WT loops
ax_a = fig7.add_subplot(gs[0, 0])
im = ax_a.imshow(pile_wt_wt, cmap=cmap, vmin=vmin, vmax=vmax)
ax_a.set_title("A\nAPA: WT loops on WT", fontsize=10)
plt.colorbar(im, ax=ax_a, shrink=0.7)

# B: APA random
ax_b = fig7.add_subplot(gs[0, 1])
im = ax_b.imshow(pile_wt_rand, cmap=cmap, vmin=vmin, vmax=vmax)
ax_b.set_title("B\nAPA: random pairs", fontsize=10)
plt.colorbar(im, ax=ax_b, shrink=0.7)

# C: APA KO
ax_c = fig7.add_subplot(gs[0, 2])
im = ax_c.imshow(pile_ko_wt, cmap=cmap, vmin=vmin, vmax=vmax)
ax_c.set_title("C\nAPA: WT loops on napM-KO", fontsize=10)
plt.colorbar(im, ax=ax_c, shrink=0.7)

# D: per-loop scatter
ax_d = fig7.add_subplot(gs[1, 0])
point_colors = np.where(pl.loc[ok2, "outlier_WT"].values, "#cb5148", "#297270")
ax_d.scatter(pl_wt[ok2], pl_ko[ok2], s=18, c=point_colors, alpha=0.75)
ax_d.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)
ax_d.set_xlabel("WT O/E")
ax_d.set_ylabel("napM-KO O/E")
ax_d.set_title(f"D\nPer-loop WT vs KO\np={w_p:.1e}", fontsize=10)

# E: replicate recovery
ax_e = fig7.add_subplot(gs[1, 1])
ax_e.bar(rep_df["replicate"], rep_df["recovery_rate"] * 100, color="#297270", edgecolor="black")
ax_e.set_ylabel("Recovery (%)")
ax_e.set_title("E\nLoop recovery per replicate", fontsize=10)

# F: robust stats
ax_f = fig7.add_subplot(gs[1, 2])
ax_f.bar(["mean", "median", "winsorized", "no outlier"], wt_vals, color="#297270", edgecolor="black")
ax_f.set_ylabel("Center O/E")
ax_f.set_title("F\nRobust statistics (WT)", fontsize=10)

# G: method comparison
ax_g = fig7.add_subplot(gs[2, 0])
ax_g.bar(["HiCCUPS", "local enrich."], counts, color=bar_colors, edgecolor="black")
ax_g.set_ylabel("Count")
ax_g.set_title("G\nMethod comparison", fontsize=10)

# H: zoom top non-outlier loop
ax_h = fig7.add_subplot(gs[2, 1])
if selected:
    label, row = selected[0]
    mid1 = int(row["start1"]) + 2500
    mid2 = int(row["end2"]) - 2500
    span = abs(mid2 - mid1)
    window = min(max(span // 2, 50000), 200000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    try:
        mat = clr_wt.matrix(sparse=False).fetch(
            f"Chromosome:{r1_s}-{r1_e}",
            f"Chromosome:{r2_s}-{r2_e}"
        )
        im_h = ax_h.imshow(mat, cmap=cmap,
                           vmin=np.nanpercentile(mat[mat > 0], 2),
                           vmax=np.nanpercentile(mat, 98))
        ax_h.axhline((mid2 - r2_s) / (r2_e - r2_s) * mat.shape[0], color="blue", linestyle="--", lw=1)
        ax_h.axvline((mid1 - r1_s) / (r1_e - r1_s) * mat.shape[1], color="blue", linestyle="--", lw=1)
        plt.colorbar(im_h, ax=ax_h, shrink=0.6)
    except:
        pass
    ax_h.set_title(f"H\n{label} loop\n{mid1//1000}kb × {mid2//1000}kb", fontsize=10)

# I: zoom strong outlier loop
ax_i = fig7.add_subplot(gs[2, 2])
if len(selected) > 1:
    label, row = selected[1]
    mid1 = int(row["start1"]) + 2500
    mid2 = int(row["end2"]) - 2500
    span = abs(mid2 - mid1)
    window = min(max(span // 2, 50000), 200000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    try:
        mat = clr_wt.matrix(sparse=False).fetch(
            f"Chromosome:{r1_s}-{r1_e}",
            f"Chromosome:{r2_s}-{r2_e}"
        )
        im_i = ax_i.imshow(mat, cmap=cmap,
                           vmin=np.nanpercentile(mat[mat > 0], 2),
                           vmax=np.nanpercentile(mat, 98))
        ax_i.axhline((mid2 - r2_s) / (r2_e - r2_s) * mat.shape[0], color="blue", linestyle="--", lw=1)
        ax_i.axvline((mid1 - r1_s) / (r1_e - r1_s) * mat.shape[1], color="blue", linestyle="--", lw=1)
        plt.colorbar(im_i, ax=ax_i, shrink=0.6)
    except:
        pass
    ax_i.set_title(f"I\n{label} loop\n{mid1//1000}kb × {mid2//1000}kb", fontsize=10)

fig7.suptitle("Figure R4-3: Validation of bacterial Hi-C loop calling", fontsize=14, fontweight="bold")
fig7.savefig(os.path.join(OUT, "figures/fig_combined_R4_comment3.pdf"), bbox_inches="tight")
fig7.savefig(os.path.join(OUT, "figures/fig_combined_R4_comment3.png"), dpi=300, bbox_inches="tight")

# ---------------- final summary ----------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"WT loops: {len(loops)}")
print(f"Outliers (IQR): {pl['outlier_WT'].sum()}")
for k, v in stats_summary.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print(f"\nReplicate recovery: {rep_df['recovery_rate'].mean()*100:.1f}% ± {rep_df['recovery_rate'].std()*100:.1f}%")
print(f"Local enrichment peaks: {len(peaks)}")
print(f"HiCCUPS loops recovered by local method: {overlap_tolerant}/{len(hiccups_bins)} ({overlap_tolerant/len(hiccups_bins)*100:.1f}%)")
print("All outputs written to", OUT)
print("=" * 60)
