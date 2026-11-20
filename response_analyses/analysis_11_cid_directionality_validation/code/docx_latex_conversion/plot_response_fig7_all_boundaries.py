#!/usr/bin/env python3
"""Standalone Response Fig. 7 using the complete set of 18 DI-defined CID boundaries.

Panels A-G use the original WT Hi-C data, DI thresholds and 5-kb resolution.
No confirmatory subsetting is applied to the boundary set. This script writes
standalone previews and numerical summaries; it does not alter reviewer replies.
"""

from pathlib import Path
import json

import cooler
import cooltools
from coolpuppy import coolpup
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "hic_review"
RESULTS = BASE / "review_results/R4_comment2_CID"
OUT = ROOT / "docx_latex_conversion/generated"
OUT.mkdir(parents=True, exist_ok=True)
WT_COOL = str(BASE / "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
DI_WT = BASE / "review_data/DI_Score_results/wt.csv"
BOUNDARY_TABLE = RESULTS / "tables/cid_boundaries_recalled.csv"
REP_DIR = BASE / "review_data/rep_icedmatrix"

BINSIZE = 5000
WINDOW_BINS = 20
NEG_THR, POS_THR = -2.0, 2.0
LOOKBACK, LOOKAHEAD = 10, 10
MIN_CID_BINS = 20
SEED = 42
N_PERMUTATIONS = 1000
C_WT, C_RAND = "#0072B2", "#999999"

cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list(
    "oe", ["#2166ac", "#f7f7f7", "#b2182b"])
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def call_boundaries(di, n):
    """Reproduce the DI-based calling used for the source boundary table."""
    candidates = []
    for i in range(1, n - 1):
        left = di[max(0, i - LOOKBACK):i + 1]
        right = di[i + 1:i + 1 + LOOKAHEAD]
        if (left <= NEG_THR).any() and (right >= POS_THR).any():
            candidates.append((i, di[i + 1] - di[i]))
    candidates.sort(key=lambda item: -item[1])
    chosen = []
    for i, jump in candidates:
        if all(abs(i - existing) >= MIN_CID_BINS for existing, _ in chosen):
            chosen.append((i, jump))
    return sorted(chosen)


def load_rep_matrix(path, n_bins):
    matrix = np.zeros((n_bins, n_bins))
    with open(path) as handle:
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) < 3:
                continue
            try:
                i, j, value = int(fields[0]) - 1, int(fields[1]) - 1, float(fields[2])
                if 0 <= i < n_bins and 0 <= j < n_bins:
                    matrix[i, j] = value
                    matrix[j, i] = value
            except ValueError:
                continue
    return matrix


def expected_oe(clr, matrix, n):
    expected = cooltools.expected_cis(clr, nproc=1)
    expected_by_distance = expected.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()
    oe = np.full_like(matrix, np.nan, dtype=float)
    for distance in range(n):
        denominator = expected_by_distance.get(distance, np.nan)
        if denominator and denominator > 0:
            rows = np.arange(n - distance)
            columns = rows + distance
            values = np.diag(matrix, k=distance) / denominator
            oe[rows, columns] = values
            oe[columns, rows] = values
    return oe, expected


def replicate_oe(matrix):
    n = matrix.shape[0]
    oe = np.full_like(matrix, np.nan, dtype=float)
    for distance in range(n):
        diagonal = np.diag(matrix, k=distance)
        positive = diagonal[diagonal > 0]
        denominator = np.mean(positive) if len(positive) else np.nan
        if denominator and denominator > 0:
            rows = np.arange(n - distance)
            columns = rows + distance
            values = diagonal / denominator
            oe[rows, columns] = values
            oe[columns, rows] = values
    return oe


def pileup_windows(centers, matrix, flank=30, diag_mask=3):
    stack = []
    for center in centers:
        if center - flank < 0 or center + flank + 1 > matrix.shape[0]:
            continue
        window = matrix[center - flank:center + flank + 1,
                        center - flank:center + flank + 1].copy()
        size = window.shape[0]
        for distance in range(-diag_mask, diag_mask + 1):
            index = np.arange(max(0, -distance), size - max(0, distance))
            window[index, index + distance] = np.nan
        stack.append(window)
    return np.nanmean(np.asarray(stack), axis=0)


def cross_depletion(pileup, arm=5, corner=10):
    center = pileup.shape[0] // 2
    arms = np.concatenate([
        pileup[center - arm:center + arm + 1, :center - corner].ravel(),
        pileup[center - arm:center + arm + 1, center + corner + 1:].ravel(),
        pileup[:center - corner, center - arm:center + arm + 1].ravel(),
        pileup[center + corner + 1:, center - arm:center + arm + 1].ravel(),
    ])
    corners = np.concatenate([
        pileup[:center - corner, :center - corner].ravel(),
        pileup[:center - corner, center + corner + 1:].ravel(),
        pileup[center + corner + 1:, :center - corner].ravel(),
        pileup[center + corner + 1:, center + corner + 1:].ravel(),
    ])
    arms = arms[np.isfinite(arms)]
    corners = corners[np.isfinite(corners)]
    return float(np.nanmedian(arms) / np.nanmedian(corners))


def insulation_score(matrix, n, window=20):
    insulation = np.full(n, np.nan)
    for i in range(n - 2 * window):
        values = matrix[i:i + window, i + window:i + 2 * window]
        values = values[np.isfinite(values)]
        if len(values) > window * window * 0.5:
            insulation[i] = np.nanmedian(values)
    smoothed = np.copy(insulation)
    for i in range(n):
        smoothed[i] = np.nanmedian(insulation[max(0, i - 2):min(n, i + 3)])
    return smoothed


def meta_profile(centers, track, half=40):
    profiles = [track[center - half:center + half + 1] for center in centers
                if center - half >= 0 and center + half + 1 <= len(track)]
    return np.nanmean(np.asarray(profiles), axis=0)


def panel_label(axis, letter, title, fontsize=10.5):
    axis.set_title(f"{letter}  {title}", loc="left", fontsize=fontsize,
                   fontweight="bold", pad=8)


print("Loading WT matrix and boundary table ...", flush=True)
clr = cooler.Cooler(WT_COOL)
mat = np.nan_to_num(clr.matrix(sparse=False).fetch("Chromosome"), nan=0.0)
n = mat.shape[0]
di = pd.read_csv(DI_WT, header=None).values.flatten()
source = pd.read_csv(BOUNDARY_TABLE)
source = source[source["transition_ok"].astype(str).str.lower().eq("true")].copy()
boundary_bins = sorted(source["bin"].astype(int).tolist())
called_bins = [i for i, _ in call_boundaries(di, n)]
assert len(boundary_bins) == 18, f"Expected 18 boundaries, found {len(boundary_bins)}"
assert boundary_bins == called_bins, "Source table does not match the reproduced DI calls"
print(f"Using all {len(boundary_bins)} DI-defined CID boundaries", flush=True)

print("Calculating distance-normalized contacts ...", flush=True)
oe, expected = expected_oe(clr, mat, n)
rng = np.random.default_rng(SEED)
flank = 30
random_centers = []
while len(random_centers) < len(boundary_bins):
    center = int(rng.integers(flank + 5, n - flank - 5))
    if all(abs(center - boundary) > 10 for boundary in boundary_bins):
        random_centers.append(center)

print("Aggregating contacts at all 18 boundaries ...", flush=True)
boundary_bed = pd.DataFrame({
    "chrom": "Chromosome",
    "start": [boundary * BINSIZE for boundary in boundary_bins],
    "end": [boundary * BINSIZE + BINSIZE for boundary in boundary_bins],
})
coolpup_result = coolpup.pileup(
    clr, boundary_bed, features_format="bed", local=True,
    expected_df=expected, flank=150000, nshifts=0)
pile_cp = np.asarray(coolpup_result["data"].iloc[0]).astype(float)
size = pile_cp.shape[0]
for distance in range(-3, 4):
    index = np.arange(max(0, -distance), size - max(0, distance))
    pile_cp[index, index + distance] = np.nan
depl_cp = cross_depletion(pile_cp)

insulation = insulation_score(oe, n)
ins_boundary = meta_profile(boundary_bins, insulation)
ins_random = meta_profile(random_centers, insulation)
di_boundary = meta_profile(boundary_bins, di)
di_random = meta_profile(random_centers, di)

# Precompute the same 30-145 kb contacts as in the original implementation.
# Counting boundaries by binary search makes the 1,000 permutations efficient
# without changing the within/cross definition: i < boundary <= j.
pair_i, pair_j, pair_values = [], [], []
for distance in range(6, 30):
    rows = np.arange(n - distance)
    columns = rows + distance
    values = oe[rows, columns]
    keep = np.isfinite(values)
    pair_i.append(rows[keep])
    pair_j.append(columns[keep])
    pair_values.append(values[keep])
pair_i = np.concatenate(pair_i)
pair_j = np.concatenate(pair_j)
pair_values = np.concatenate(pair_values)


def within_cross_ratio(boundaries):
    boundaries = np.asarray(sorted(boundaries), dtype=int)
    left_count = np.searchsorted(boundaries, pair_i, side="right")
    right_count = np.searchsorted(boundaries, pair_j, side="right")
    crosses = right_count > left_count
    if np.sum(crosses) <= 100:
        return np.nan, np.nan, np.nan
    within_median = float(np.nanmedian(pair_values[~crosses]))
    cross_median = float(np.nanmedian(pair_values[crosses]))
    return within_median / cross_median, within_median, cross_median


print(f"Running {N_PERMUTATIONS} size-matched partition permutations ...", flush=True)
obs_ratio, within_median, cross_median = within_cross_ratio(boundary_bins)
segment_sizes = np.diff([0] + boundary_bins + [n - 1])
null_ratios = []
for _ in range(N_PERMUTATIONS):
    shuffled = np.concatenate([[0], np.cumsum(rng.permutation(segment_sizes))[:-1]])
    random_boundaries = [int(value) for value in shuffled[1:] if 0 < value < n - 1]
    ratio, _, _ = within_cross_ratio(random_boundaries)
    if np.isfinite(ratio):
        null_ratios.append(ratio)
null_ratios = np.asarray(null_ratios)
perm_p = (np.sum(null_ratios >= obs_ratio) + 1) / (len(null_ratios) + 1)
null_median = float(np.nanmedian(null_ratios))

print("Calculating replicate pileups ...", flush=True)
rep_files = sorted(path for path in REP_DIR.iterdir()
                   if path.name.startswith("wt_") and path.name.endswith(".matrix"))
rep_depl = {}
for path in rep_files:
    rep_id = path.name.replace("_5000_iced.matrix", "")
    rep_matrix = load_rep_matrix(path, n)
    rep_pileup = pileup_windows(boundary_bins, replicate_oe(rep_matrix))
    rep_depl[rep_id] = cross_depletion(rep_pileup)
rep_ids = sorted(rep_depl)
print("Merged cross-arm/corner O/E:", round(depl_cp, 4), flush=True)
print("Within/cross median ratio:", round(obs_ratio, 4), "P =", perm_p, flush=True)
print("Per-replicate values:", {key: round(value, 4) for key, value in rep_depl.items()}, flush=True)

# Numerical outputs retain all 18 boundaries and contain no subset labels.
metrics = source.drop(columns=["confirmed"], errors="ignore").copy()
metrics.to_csv(OUT / "Response_Fig_07_all_boundaries_metrics.csv", index=False)
summary = {
    "figure": "Response Fig. 7",
    "cid_boundaries": len(boundary_bins),
    "boundary_bins": boundary_bins,
    "boundary_positions_kb": [boundary * 5 for boundary in boundary_bins],
    "resolution_bp": BINSIZE,
    "di_thresholds": [NEG_THR, POS_THR],
    "lookback_bins": LOOKBACK,
    "lookahead_bins": LOOKAHEAD,
    "minimum_boundary_spacing_bins": MIN_CID_BINS,
    "cross_arm_over_corner_oe_merged": depl_cp,
    "within_cid_median_oe": within_median,
    "cross_boundary_median_oe": cross_median,
    "within_over_cross_median_ratio": obs_ratio,
    "random_partition_median_ratio": null_median,
    "permutation_p": float(perm_p),
    "permutations_requested": N_PERMUTATIONS,
    "permutations_retained": len(null_ratios),
    "permutation_seed": SEED,
    "contact_distance_range_bins": [6, 29],
    "per_replicate_cross_arm_over_corner_oe": {key: float(value) for key, value in rep_depl.items()},
    "boundary_table": str(BOUNDARY_TABLE),
}
with open(OUT / "Response_Fig_07_all_boundaries_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)

print("Rendering standalone A-G figure ...", flush=True)
fig = plt.figure(figsize=(16, 11.6))
grid = gridspec.GridSpec(
    3, 3, figure=fig, height_ratios=[0.67, 1.06, 1.0],
    hspace=0.48, wspace=0.34, left=0.065, right=0.975, top=0.915, bottom=0.075)
vertices = [0] + boundary_bins + [n - 1]

# A: DI track and the unified 18-boundary set.
axis = fig.add_subplot(grid[0, :])
for k in range(len(vertices) - 1):
    if k % 2 == 0:
        axis.axvspan(vertices[k] * 5, vertices[k + 1] * 5,
                     color="grey", alpha=0.10, zorder=0)
x = np.arange(n) * BINSIZE / 1000
axis.fill_between(x, di, 0, where=(di < 0), color="#2166ac", alpha=0.75, zorder=2)
axis.fill_between(x, di, 0, where=(di >= 0), color="#b2182b", alpha=0.75, zorder=2)
axis.axhline(NEG_THR, color="#2166ac", ls=":", lw=0.8)
axis.axhline(POS_THR, color="#b2182b", ls=":", lw=0.8)
axis.scatter([boundary * 5 for boundary in boundary_bins], [2.06] * len(boundary_bins),
             marker="v", s=48, color="black", zorder=5, clip_on=False)
axis.set_xlim(0, n * 5)
axis.set_ylim(-2, 2)
axis.set_xlabel("Genomic position (kb)", fontsize=10)
axis.set_ylabel("Directional preference (DI)", fontsize=10)
axis.legend(handles=[
    Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
           markersize=8, label=f"CID boundaries (n={len(boundary_bins)})"),
    Line2D([0], [0], color="#2166ac", lw=5, alpha=0.75, label="Upstream preference (DI < 0)"),
    Line2D([0], [0], color="#b2182b", lw=5, alpha=0.75, label="Downstream preference (DI > 0)"),
], fontsize=8.5, loc="upper left", frameon=False, ncol=3)
panel_label(axis, "A", "DI-defined CID boundaries across the M. tuberculosis chromosome")

# B: Contact map and continuous CID blocks.
axis = fig.add_subplot(grid[1, 0])
axis.imshow(mat, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat, 99), interpolation="nearest")
cid_colors = ["#0072B2", "#8fd3f4"]
for k in range(len(vertices) - 1):
    start, end = vertices[k], vertices[k + 1]
    axis.add_patch(Rectangle((start, start), end - start, end - start, fill=True,
                             facecolor=cid_colors[k % 2], alpha=0.30,
                             edgecolor="none", zorder=2))
for boundary in boundary_bins:
    axis.plot([boundary, boundary], [-6, 0], color="#003f63", lw=1.0, clip_on=False, zorder=3)
    axis.plot([n, n + 6], [boundary, boundary], color="#003f63", lw=1.0, clip_on=False, zorder=3)
ticks = np.arange(0, n, 200)
axis.set_xticks(ticks, [f"{value * 5 / 1000:.0f}" for value in ticks], fontsize=8)
axis.set_yticks(ticks, [f"{value * 5 / 1000:.0f}" for value in ticks], fontsize=8)
axis.set_xlabel("Genomic position (Mb)", fontsize=9)
axis.set_ylabel("Genomic position (Mb)", fontsize=9)
panel_label(axis, "B", f"WT contact map partitioned by {len(boundary_bins)} CID boundaries\n"
                       "Alternating shades indicate consecutive CIDs")

# C: Boundary pileup.
axis = fig.add_subplot(grid[1, 1])
image = axis.imshow(pile_cp, cmap=cmap_oe, vmin=0.7, vmax=1.3, interpolation="nearest")
center = pile_cp.shape[0] // 2
axis.axhline(center, color="black", ls="--", lw=0.8)
axis.axvline(center, color="black", ls="--", lw=0.8)
axis.set_xticks([0, center, 2 * center], ["−150 kb", "boundary", "+150 kb"], fontsize=8)
axis.set_yticks([0, center, 2 * center], ["−150 kb", "boundary", "+150 kb"], fontsize=8)
panel_label(axis, "C", f"Contact pileup at {len(boundary_bins)} CID boundaries\n"
                       f"Cross-arm / corner O/E = {depl_cp:.2f}")
colorbar = plt.colorbar(image, ax=axis, shrink=0.68, label="Aggregate O/E")
colorbar.ax.tick_params(labelsize=8)

# D: Size-matched random partitions.
axis = fig.add_subplot(grid[1, 2])
axis.hist(null_ratios, bins=40, color=C_RAND, edgecolor="black", linewidth=0.35, alpha=0.8)
axis.axvline(obs_ratio, color=C_WT, lw=2)
axis.text(0.97, 0.84, f"observed = {obs_ratio:.3f}\nP = {perm_p:.3f}",
          transform=axis.transAxes, color=C_WT, fontsize=9,
          ha="right", va="top", fontweight="bold")
axis.set_xlabel("Within-CID / cross-boundary median O/E ratio", fontsize=9)
axis.set_ylabel("Random partitions", fontsize=9)
panel_label(axis, "D", "Within-CID versus cross-boundary contacts\n"
                       "Observed and 1,000 size-matched random partitions")

# E: DI metaprofile.
axis = fig.add_subplot(grid[2, 0])
meta_x = np.arange(-40, 41) * 5
axis.plot(meta_x, di_boundary, color=C_WT, lw=1.6,
          label=f"CID boundaries (n={len(boundary_bins)})")
axis.plot(meta_x, di_random, color=C_RAND, lw=1.6,
          label=f"Random positions (n={len(random_centers)})")
axis.axvline(0, color="black", ls="--", lw=0.8)
axis.axhline(0, color="grey", ls=":", lw=0.7)
axis.set_xlabel("Distance from boundary (kb)", fontsize=9)
axis.set_ylabel("Mean DI", fontsize=9)
axis.legend(fontsize=8, frameon=False)
panel_label(axis, "E", "Directional-preference profile\nat CID boundaries and random positions")

# F: Insulation metaprofile.
axis = fig.add_subplot(grid[2, 1])
axis.plot(meta_x, ins_boundary, color=C_WT, lw=1.6,
          label=f"CID boundaries (n={len(boundary_bins)})")
axis.plot(meta_x, ins_random, color=C_RAND, lw=1.6,
          label=f"Random positions (n={len(random_centers)})")
axis.axvline(0, color="black", ls="--", lw=0.8)
axis.set_xlabel("Distance from boundary (kb)", fontsize=9)
axis.set_ylabel("Insulation score", fontsize=9)
axis.legend(fontsize=8, frameon=False)
panel_label(axis, "F", "Insulation-score profile\nat CID boundaries and random positions")

# G: Replicate measurements, with no confirmatory classification of boundaries.
axis = fig.add_subplot(grid[2, 2])
replicate_names = [rep_id.replace("wt_", "rep") for rep_id in rep_ids]
replicate_values = [rep_depl[rep_id] for rep_id in rep_ids]
bars = axis.bar(replicate_names, replicate_values, color=C_WT,
                edgecolor="black", alpha=0.85, width=0.55)
axis.axhline(1.0, color="black", ls="--", lw=0.8)
axis.axhline(depl_cp, color="#D55E00", ls=":", lw=1.4,
             label=f"Merged pileup = {depl_cp:.3f}")
for bar, value in zip(bars, replicate_values):
    axis.text(bar.get_x() + bar.get_width() / 2, value - 0.025, f"{value:.3f}",
              ha="center", va="top", fontsize=9, color="white", fontweight="bold")
axis.set_ylabel("Cross-arm / corner O/E\n(< 1 = relative cross depletion)", fontsize=9)
lower = min(replicate_values + [depl_cp, 1.0])
upper = max(replicate_values + [depl_cp, 1.0])
axis.set_ylim(lower - 0.08, upper + 0.05)
axis.legend(fontsize=8, frameon=False, loc="upper right")
panel_label(axis, "G", "Cross-boundary pileup signal in WT replicates\n"
                       f"All {len(boundary_bins)} CID boundaries included")

fig.suptitle("Response Fig. 7 | Chromosome interaction domains and their boundaries in M. tuberculosis",
             fontsize=14, fontweight="bold", y=0.973)
fig.savefig(OUT / "Response_Fig_07_all_boundaries.pdf", bbox_inches="tight")
fig.savefig(OUT / "Response_Fig_07_all_boundaries.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved standalone figure and metrics to {OUT}", flush=True)
