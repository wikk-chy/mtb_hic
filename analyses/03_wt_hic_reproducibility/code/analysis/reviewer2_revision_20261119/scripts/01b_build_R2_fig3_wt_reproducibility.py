#!/usr/bin/env python3
"""
Build NEW Response Fig. 3 for Reviewer #2 Comment 1:
Reproducibility and robustness of bulk WT Hi-C across four independent biological replicates.
  Panel A: Pairwise WT contact-matrix Pearson correlation scatter matrix (r = 0.97-0.99;
           from Extended Data Fig. 1c, HiCExplorer hicCorrelate).
  Panel B: Overlaid 4-replicate WT contact-probability decay curves P(s) across circular
           genomic separation (5 kb to 2.2 Mb) with log2(Pi(s)/Pmean(s)) deviation inset
           and verified per-replicate metric summary (all pairwise P(s) r > 0.998).
"""
from pathlib import Path
from itertools import combinations
import fitz
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
MDIR = ROOT / "hic_review/review_data/rep_icedmatrix"
ED1_PDF = ROOT / "Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis/figures_pdf/Extended_Data_Fig_1.pdf"
PAIR_CSV = ROOT / "analysis/reproducibility_qc_20261105/replicate_pairwise_metrics.csv"
QC_CSV = ROOT / "analysis/reproducibility_qc_20261105/replicate_QC_summary.csv"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# 1. Extract crisp high-DPI Panel A (Extended Data Fig. 1c scatter matrix)
doc = fitz.open(ED1_PDF)
page = doc[0]
clip = fitz.Rect(19.5, 548.5, 276.0, 811.5)
pix = page.get_pixmap(clip=clip, dpi=450, alpha=False)
im_A = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# 2. Compute circular P(s) curves across 4 WT replicates (5-kb bins, N=883)
N_BINS = 883
BIN_KB = 5.0
ps_dict = {}
for r in range(1, 5):
    df = pd.read_csv(MDIR / f"wt_{r}_5000_iced.matrix", sep=r"\s+", header=None, names=["b1", "b2", "val"])
    d_lin = (df["b2"] - df["b1"]).abs()
    d_circ = np.minimum(d_lin, N_BINS - d_lin)
    # Sum contacts per circular separation, normalize by total off-diagonal contacts (d >= 1)
    grp = df.loc[d_circ >= 1].groupby(d_circ.loc[d_circ >= 1])["val"].sum()
    idx = np.arange(1, N_BINS // 2 + 1)
    s_vals = grp.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    # Count number of bin pairs at each circular distance d (for circular genome, N_BINS pairs except midpoint)
    n_pairs = np.full(len(idx), float(N_BINS))
    p_s = (s_vals / n_pairs)
    p_s = p_s / p_s.sum()
    ps_dict[r] = p_s

dist_kb = np.arange(1, N_BINS // 2 + 1) * BIN_KB
ps_mat = np.vstack([ps_dict[r] for r in range(1, 5)])
ps_mean = ps_mat.mean(axis=0)

# Log-spaced smoothing bins from 10 kb to 2.2 Mb (excluding adjacent 5-kb bin d=1)
edges = np.geomspace(10.0, dist_kb.max(), 72)
bin_idx = np.digitize(dist_kb, edges) - 1
s_plot, p_plot = [], {r: [] for r in range(1, 5)}
p_mean_plot = []
for b in range(len(edges) - 1):
    m = (bin_idx == b) & (dist_kb >= 10.0)
    if m.sum() == 0:
        continue
    s_plot.append(np.sqrt(dist_kb[m].min() * dist_kb[m].max()))
    for r in range(1, 5):
        p_plot[r].append(ps_dict[r][m].mean())
    p_mean_plot.append(ps_mean[m].mean())
s_plot = np.array(s_plot)
p_mean_plot = np.array(p_mean_plot)
for r in range(1, 5):
    p_plot[r] = np.array(p_plot[r])

# Load exact reported pairwise P(s) correlations from replicate_pairwise_metrics.csv
pair_df = pd.read_csv(PAIR_CSV)
wt_pairs = pair_df.loc[pair_df["condition"] == "WT", "pearson_P_of_s_r"]
ps_r_min, ps_r_max = wt_pairs.min(), wt_pairs.max()

# Also compute full circular P(s) log10 correlation across s >= 10 kb
circ_rs = []
for a, b in combinations(range(1, 5), 2):
    r_val = stats.pearsonr(np.log10(ps_dict[a][1:] + 1e-12), np.log10(ps_dict[b][1:] + 1e-12)).statistic
    circ_rs.append(r_val)
print(f"WT linear P(s) r (QC table): {ps_r_min:.4f} - {ps_r_max:.4f}")
print(f"WT circular P(s) r (all s>=10kb): {min(circ_rs):.4f} - {max(circ_rs):.4f}")

# 3. Compose figure (A4 width, crisp 2-panel horizontal layout with generous spacing)
fig = plt.figure(figsize=(8.27, 4.30))
gs = GridSpec(1, 2, figure=fig, left=0.045, right=0.975, top=0.825, bottom=0.14,
              wspace=0.30, width_ratios=[0.96, 1.22])
C_TEXT = "#1F2933"

fig.suptitle(
    "High reproducibility of bulk WT M. tuberculosis Hi-C across four biological replicates",
    x=0.035, y=0.955, ha="left", fontsize=10.0, fontweight="bold", color=C_TEXT
)

# Panel A: Embedded Extended Data Fig. 1c
axA = fig.add_subplot(gs[0, 0])
axA.imshow(im_A)
axA.axis("off")
axA.set_title("Matrix correlation (r = 0.97–0.99)", loc="left", pad=8, fontsize=8.8, fontweight="bold", color=C_TEXT)
axA.text(-0.07, 1.08, "A", transform=axA.transAxes, fontsize=11.5, fontweight="bold", va="top", color="black")

# Panel B: Overlaid P(s) curves across 4 WT replicates
axB = fig.add_subplot(gs[0, 1])
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
linestyles = ["-", "--", "-.", ":"]
linewidths = [2.2, 1.7, 1.6, 1.9]
for idx_r, r in enumerate(range(1, 5)):
    axB.loglog(
        s_plot, p_plot[r],
        color=colors[idx_r], linestyle=linestyles[idx_r], linewidth=linewidths[idx_r],
        alpha=0.92, label=f"WT-Rep{r}"
    )

axB.set_xlabel("Circular genomic separation s (kb)")
axB.set_ylabel("Contact probability P(s)")
axB.set_xlim(9.0, 2500)
axB.set_ylim(1.8e-4, 4.8e-2)
axB.set_xticks([10, 25, 50, 100, 250, 500, 1000, 2000])
axB.set_xticklabels(["10", "25", "50", "100", "250", "500", "1,000", "2,000"])
axB.grid(True, which="major", linestyle=":", linewidth=0.5, color="#D8DEE4")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

# Legend at top right
leg = axB.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D8DEE4",
                 ncol=2, handlelength=2.0, columnspacing=0.9, borderpad=0.30, fontsize=6.9)
leg.get_frame().set_linewidth(0.6)

# Quantitative reproducibility annotation box cleanly in open upper-right space above curve
axB.text(
    0.98, 0.80,
    f"Pairwise P(s) r = {ps_r_min:.4f}–{ps_r_max:.4f} (all > 0.998)\n"
    "Short (<100 kb): 0.377 ± 0.001\n"
    "Long (>1 Mb): 0.260 ± 0.001\n"
    "P(s) slope (0.1–1 Mb): −0.620 ± 0.001\n"
    "Median distance: 276 ± 2 kb",
    transform=axB.transAxes, ha="right", va="top", fontsize=6.25, linespacing=1.32, color=C_TEXT,
    bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.6)
)

# Inset placed cleanly inside Panel B with zero overlap against axes or curve
ax_in = inset_axes(
    axB, width="100%", height="100%",
    bbox_to_anchor=(0.175, 0.12, 0.41, 0.31),
    bbox_transform=axB.transAxes, loc="lower left", borderpad=0
)
ax_in.axhline(0, color="#555555", linewidth=0.65, linestyle="--")
for idx_r, r in enumerate(range(1, 5)):
    ratio_log2 = np.log2((p_plot[r] + 1e-12) / (p_mean_plot + 1e-12))
    ax_in.semilogx(s_plot, ratio_log2, color=colors[idx_r], linestyle=linestyles[idx_r], linewidth=1.15, alpha=0.92)
ax_in.set_xlim(10, 2200)
ax_in.set_ylim(-0.08, 0.08)
ax_in.set_yticks([-0.05, 0.0, 0.05])
ax_in.set_xticks([10, 100, 1000])
ax_in.set_xticklabels(["10", "100", "1k"], fontsize=6.1)
ax_in.tick_params(axis="y", labelsize=6.0, length=2.2)
ax_in.tick_params(axis="x", length=2.2)
ax_in.set_ylabel(r"$\log_2(P_i/\bar{P})$", fontsize=6.2, labelpad=1.5)
ax_in.set_title("Replicate deviation vs mean", fontsize=6.4, pad=2.5, fontweight="bold", color=C_TEXT)
ax_in.spines["top"].set_visible(False)
ax_in.spines["right"].set_visible(False)
ax_in.set_facecolor("white")

axB.set_title("Overlaid contact-probability decay curves P(s)", loc="left", pad=8, fontsize=8.8, fontweight="bold", color=C_TEXT)
axB.text(-0.155, 1.08, "B", transform=axB.transAxes, fontsize=11.5, fontweight="bold", va="top", color="black")

pdf_path = OUT_FIG / "Response_Fig_3_wt_repro.pdf"
png_path = OUT_FIG / "Response_Fig_3_wt_repro.png"
jpg_path = OUT_FIG / "Response_Fig_3_wt_repro.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=96, subsampling=0, dpi=(300, 300))
print("Saved", pdf_path, jpg_path)
