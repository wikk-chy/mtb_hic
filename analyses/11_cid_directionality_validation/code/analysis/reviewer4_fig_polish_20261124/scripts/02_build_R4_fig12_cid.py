#!/usr/bin/env python3
"""
Rebuild Response Fig. 12 (Reviewer 4, Comment 2 — 18 DI-defined CID boundaries):
100% vector-editable TrueType Arial (pdf.fonttype=42), standard print scale (9.6 x 7.6 in),
crisp readable fonts, NO redundant suptitle, and ZERO overlaps (fixes Panel A legend over DI bars
and Panel D text colliding with observed vertical line).
Exact same underlying WT Hi-C data, 18 boundaries, and seed=42 computations.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
import cooltools
from coolpuppy import coolpup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
RESULTS = BASE / "review_results/R4_comment2_CID"
OUT_FIG = ROOT / "analysis/reviewer4_fig_polish_20261124/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

WT_COOL = str(BASE / "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
DI_WT = BASE / "review_data/DI_Score_results/wt.csv"
BOUNDARY_TABLE = RESULTS / "tables/cid_boundaries_recalled.csv"
REP_DIR = BASE / "review_data/rep_icedmatrix"

BINSIZE = 5000
NEG_THR, POS_THR = -2.0, 2.0
LOOKBACK, LOOKAHEAD = 10, 10
MIN_CID_BINS = 20
SEED = 42
N_PERMUTATIONS = 1000
C_WT, C_RAND = "#0072B2", "#8A94A0"
C_TEXT = "#1F2933"

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7.5,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

cmap_hic = LinearSegmentedColormap.from_list("hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list("oe", ["#2166ac", "#f7f7f7", "#b2182b"])


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


# Load data & compute exact metrics
clr = cooler.Cooler(WT_COOL)
mat = np.nan_to_num(clr.matrix(sparse=False).fetch("Chromosome"), nan=0.0)
n = mat.shape[0]
di = pd.read_csv(DI_WT, header=None).values.flatten()
source = pd.read_csv(BOUNDARY_TABLE)
source = source[source["transition_ok"].astype(str).str.lower().eq("true")].copy()
boundary_bins = sorted(source["bin"].astype(int).tolist())
assert len(boundary_bins) == 18

oe, expected = expected_oe(clr, mat, n)
rng = np.random.default_rng(SEED)
flank = 30
random_centers = []
while len(random_centers) < len(boundary_bins):
    center = int(rng.integers(flank + 5, n - flank - 5))
    if all(abs(center - boundary) > 10 for boundary in boundary_bins):
        random_centers.append(center)

boundary_bed = pd.DataFrame({
    "chrom": "Chromosome",
    "start": [boundary * BINSIZE for boundary in boundary_bins],
    "end": [boundary * BINSIZE + BINSIZE for boundary in boundary_bins],
})
coolpup_result = coolpup.pileup(
    clr, boundary_bed, features_format="bed", local=True,
    expected_df=expected, flank=150000, nshifts=0,
)
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

rep_files = sorted(path for path in REP_DIR.iterdir()
                   if path.name.startswith("wt_") and path.name.endswith(".matrix"))
rep_depl = {}
for path in rep_files:
    rep_id = path.name.replace("_5000_iced.matrix", "")
    rep_matrix = load_rep_matrix(path, n)
    rep_pileup = pileup_windows(boundary_bins, replicate_oe(rep_matrix))
    rep_depl[rep_id] = cross_depletion(rep_pileup)
rep_ids = sorted(rep_depl)

# ==================== Build crisp print-scale figure ====================
fig = plt.figure(figsize=(9.6, 7.8))
grid = gridspec.GridSpec(
    3, 3, figure=fig,
    height_ratios=[0.66, 1.05, 0.98],
    hspace=0.48, wspace=0.34,
    left=0.070, right=0.972, top=0.935, bottom=0.068,
)
vertices = [0] + boundary_bins + [n - 1]


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


# A: DI track across chromosome (legend placed cleanly ABOVE plot spine — zero bar overlap!)
axA = fig.add_subplot(grid[0, :])
for k in range(len(vertices) - 1):
    if k % 2 == 0:
        axA.axvspan(vertices[k] * 5, vertices[k + 1] * 5, color="grey", alpha=0.10, zorder=0)
x = np.arange(n) * BINSIZE / 1000
axA.fill_between(x, di, 0, where=(di < 0), color="#2166ac", alpha=0.78, zorder=2)
axA.fill_between(x, di, 0, where=(di >= 0), color="#b2182b", alpha=0.78, zorder=2)
axA.axhline(NEG_THR, color="#2166ac", ls=":", lw=0.75)
axA.axhline(POS_THR, color="#b2182b", ls=":", lw=0.75)
axA.scatter([boundary * 5 for boundary in boundary_bins], [2.10] * len(boundary_bins),
            marker="v", s=26, color="black", zorder=5, clip_on=False)
axA.set_xlim(0, n * 5)
axA.set_ylim(-2.05, 2.05)
axA.set_xlabel("Genomic position (kb)")
axA.set_ylabel("Directional preference (DI)")
axA.set_title("DI-defined CID boundaries across the M. tuberculosis chromosome",
              loc="left", pad=16, fontweight="bold", color=C_TEXT)
axA.legend(
    handles=[
        Line2D([0], [0], marker="v", color="w", markerfacecolor="black", markersize=6.5, label="CID boundaries (n = 18)"),
        Line2D([0], [0], color="#2166ac", lw=4.5, alpha=0.78, label="Upstream (DI < 0)"),
        Line2D([0], [0], color="#b2182b", lw=4.5, alpha=0.78, label="Downstream (DI > 0)"),
    ],
    loc="lower right", bbox_to_anchor=(1.0, 1.03), frameon=False, ncol=3,
    handletextpad=0.4, columnspacing=1.0, fontsize=6.8,
)
axA.tick_params(direction="out", length=2.8)

# B: Contact map partitioned by 18 CID boundaries
axB = fig.add_subplot(grid[1, 0])
axB.imshow(mat, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat, 99), interpolation="nearest")
cid_colors = ["#0072B2", "#8fd3f4"]
for k in range(len(vertices) - 1):
    start, end = vertices[k], vertices[k + 1]
    axB.add_patch(Rectangle((start, start), end - start, end - start, fill=True,
                            facecolor=cid_colors[k % 2], alpha=0.30, edgecolor="none", zorder=2))
for boundary in boundary_bins:
    axB.plot([boundary, boundary], [-12, 0], color="#003f63", lw=0.85, clip_on=False, zorder=3)
    axB.plot([n, n + 12], [boundary, boundary], color="#003f63", lw=0.85, clip_on=False, zorder=3)
ticks = np.arange(0, n, 200)
axB.set_xticks(ticks, [f"{value * 5 / 1000:.0f}" for value in ticks])
axB.set_yticks(ticks, [f"{value * 5 / 1000:.0f}" for value in ticks])
axB.set_xlabel("Genomic position (Mb)")
axB.set_ylabel("Genomic position (Mb)")
axB.set_title("WT contact map (18 CID boundaries)\nAlternating shades = consecutive CIDs",
              loc="left", pad=8, fontweight="bold", color=C_TEXT)

# C: Boundary pileup
axC = fig.add_subplot(grid[1, 1])
image = axC.imshow(pile_cp, cmap=cmap_oe, vmin=0.7, vmax=1.3, interpolation="nearest")
center = pile_cp.shape[0] // 2
axC.axhline(center, color="black", ls="--", lw=0.75)
axC.axvline(center, color="black", ls="--", lw=0.75)
axC.set_xticks([0, center, 2 * center], ["−150 kb", "boundary", "+150 kb"])
axC.set_yticks([0, center, 2 * center], ["−150 kb", "boundary", "+150 kb"])
axC.set_title(f"Contact pileup at 18 CID boundaries\nCross-arm / corner O/E = {depl_cp:.2f}",
              loc="left", pad=8, fontweight="bold", color=C_TEXT)
cb = plt.colorbar(image, ax=axC, fraction=0.048, pad=0.04)
cb.set_label("Aggregate O/E", fontsize=7.0)
cb.ax.tick_params(labelsize=6.6)

# D: Size-matched random partitions (text cleanly LEFT of observed line — ZERO collision!)
axD = fig.add_subplot(grid[1, 2])
axD.hist(null_ratios, bins=38, color=C_RAND, edgecolor="black", linewidth=0.35, alpha=0.82)
axD.axvline(obs_ratio, color=C_WT, lw=1.8)
axD.set_xlim(min(null_ratios.min() - 0.008, 0.935), obs_ratio + 0.014)
axD.text(obs_ratio - 0.005, axD.get_ylim()[1] * 0.88,
         f"Observed = {obs_ratio:.3f}\nP = {perm_p:.3f}",
         color=C_WT, fontsize=7.2, ha="right", va="top", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.88, edgecolor="none"))
axD.set_xlabel("Within-CID / cross-boundary O/E ratio")
axD.set_ylabel("Random partitions")
axD.set_title("Within-CID vs cross-boundary contacts\nObserved & 1,000 size-matched partitions",
              loc="left", pad=8, fontweight="bold", color=C_TEXT)
clean_spines(axD)

# E: DI metaprofile
axE = fig.add_subplot(grid[2, 0])
meta_x = np.arange(-40, 41) * 5
axE.plot(meta_x, di_boundary, color=C_WT, lw=1.5, label="CID boundaries (n = 18)")
axE.plot(meta_x, di_random, color=C_RAND, lw=1.4, label="Random positions (n = 18)")
axE.axvline(0, color="black", ls="--", lw=0.75)
axE.axhline(0, color="grey", ls=":", lw=0.7)
axE.set_xlabel("Distance from boundary (kb)")
axE.set_ylabel("Mean DI")
axE.set_ylim(-3.8, 4.6)
axE.legend(frameon=False, loc="upper left", fontsize=6.5)
axE.set_title("Directional-preference profile\nat CID boundaries & random sites",
              loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axE)

# F: Insulation metaprofile
axF = fig.add_subplot(grid[2, 1])
axF.plot(meta_x, ins_boundary, color=C_WT, lw=1.5, label="CID boundaries (n = 18)")
axF.plot(meta_x, ins_random, color=C_RAND, lw=1.4, label="Random positions (n = 18)")
axF.axvline(0, color="black", ls="--", lw=0.75)
axF.set_xlabel("Distance from boundary (kb)")
axF.set_ylabel("Insulation score")
axF.set_ylim(0.925, 1.135)
axF.legend(frameon=False, loc="upper right", fontsize=6.5)
axF.set_title("Insulation-score profile\nat CID boundaries & random sites",
              loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axF)

# G: Per-replicate pileup signal
axG = fig.add_subplot(grid[2, 2])
replicate_names = [rep_id.replace("wt_", "rep") for rep_id in rep_ids]
replicate_values = [rep_depl[rep_id] for rep_id in rep_ids]
bars = axG.bar(replicate_names, replicate_values, color=C_WT, edgecolor="black", lw=0.55, alpha=0.88, width=0.54)
axG.axhline(1.0, color="black", ls="--", lw=0.8)
axG.axhline(depl_cp, color="#D55E00", ls=":", lw=1.4, label=f"Merged pileup = {depl_cp:.3f}")
for bar, value in zip(bars, replicate_values):
    axG.text(bar.get_x() + bar.get_width() / 2, value - 0.015, f"{value:.3f}",
             ha="center", va="top", fontsize=7.0, color="white", fontweight="bold")
axG.set_ylabel("Cross-arm / corner O/E\n(< 1 = relative cross depletion)")
axG.set_ylim(0.89, 1.055)
axG.legend(frameon=False, loc="upper right", fontsize=6.6)
axG.set_title("Cross-boundary pileup in WT replicates\nAll 18 CID boundaries included",
              loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axG)

# Aligned figure-level panel letters A-G
pA = axA.get_position()
pB, pC, pD = axB.get_position(), axC.get_position(), axD.get_position()
pE, pF, pG = axE.get_position(), axF.get_position(), axG.get_position()
fig.text(0.016, pA.y1 + 0.044, "A", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.016, pB.y1 + 0.046, "B", fontsize=11.5, fontweight="bold", va="top")
fig.text(pC.x0 - 0.056, pC.y1 + 0.046, "C", fontsize=11.5, fontweight="bold", va="top")
fig.text(pD.x0 - 0.048, pD.y1 + 0.046, "D", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.016, pE.y1 + 0.042, "E", fontsize=11.5, fontweight="bold", va="top")
fig.text(pF.x0 - 0.056, pF.y1 + 0.042, "F", fontsize=11.5, fontweight="bold", va="top")
fig.text(pG.x0 - 0.054, pG.y1 + 0.042, "G", fontsize=11.5, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_12.pdf"
png_p = OUT_FIG / "Response_Fig_12.png"
jpg_p = OUT_FIG / "Response_Fig_12.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_12.{pdf,png,jpg}")
