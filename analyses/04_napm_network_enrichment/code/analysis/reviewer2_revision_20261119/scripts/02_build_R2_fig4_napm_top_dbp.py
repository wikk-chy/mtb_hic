#!/usr/bin/env python3
"""
Build final Response Fig. 4 for Reviewer #2 Comment 2:
Single-row 4-layer distribution of all 229 M. tuberculosis DNA-binding proteins (anonymous grey dots),
highlighting ONLY NapM (Rv0047c) in purple diamond at the extreme upper tail across:
  - Double-anchor loops (of 41): 17 (Top 1%, 16.4x mean)
  - Linked operons (of 749): 470 (Top 3%, 4.6x mean)
  - Integrated network degree: 441 (Top 3%, 5.7x mean)
  - PPI network degree (log scale): 14 (Top 6%, 3.3x mean)
No suptitle, no competitor protein names.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.0,
    "axes.titlesize": 8.2,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
DBP_CSV = ROOT / "hic_review/review_results/R4_comment6/tables/fig2f_metrics.csv"
CENT_CSV = ROOT / "analysis/reviewer2_revision_20261119/tables/full_network_centrality.csv"
OUT_DIR = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df229 = pd.read_csv(DBP_CSV)
cent = pd.read_csv(CENT_CSV)
m = df229.merge(cent[["node", "degree"]], left_on="protein", right_on="node", how="left")
assert len(m) == 229

C_TEXT = "#1F2933"
C_NAPM = "#6A3D9A"
C_OTHER = "#94A3B8"

fig = plt.figure(figsize=(8.27, 3.45))
gs = GridSpec(1, 4, figure=fig, left=0.075, right=0.975, top=0.86, bottom=0.06, wspace=0.38)

metrics_info = [
    ("loop", "Double-anchor\nloops (of 41)", "Top 1%\n(16.4× mean)", 17, 1.04, False),
    ("operon", "Linked operons\n(of 749)", "Top 3%\n(4.6× mean)", 470, 102.73, False),
    ("degree", "Integrated network\ndegree", "Top 3%\n(5.7× mean)", 441, 77.93, False),
    ("ppi", "PPI network\ndegree (log scale)", "Top 6%\n(3.3× mean)", 14, 4.27, True),
]

rng = np.random.default_rng(42)
for idx, (col, title, badge, napm_v, other_mean, use_symlog) in enumerate(metrics_info):
    ax = fig.add_subplot(gs[0, idx])
    others = m.loc[m["protein"] != "Rv0047c", col].dropna().to_numpy()
    jit = rng.uniform(-0.22, 0.22, size=len(others))
    ax.scatter(jit, others, s=11.5, color=C_OTHER, alpha=0.52, linewidths=0, zorder=2)
    ax.hlines(other_mean, -0.32, 0.32, color="#475569", linewidth=1.4, linestyle="--", zorder=3)
    ax.text(
        0.34, other_mean,
        f"mean\n{other_mean:.1f}" if other_mean >= 10 else f"mean\n{other_mean:.2f}",
        va="center", ha="left", fontsize=6.2, color="#475569"
    )
    ax.scatter([0], [napm_v], s=88, marker="D", facecolor=C_NAPM, edgecolor="black", linewidth=0.9, zorder=5)
    ax.text(0.11, napm_v, f"NapM\n({int(napm_v)})", va="center", ha="left",
            fontsize=7.0, fontweight="bold", color=C_NAPM)
    ax.text(
        0.50, 0.97, badge, transform=ax.transAxes, ha="center", va="top",
        fontsize=6.7, fontweight="bold", color=C_NAPM,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F3E8FF", edgecolor="#C4B5FD", lw=0.6)
    )
    if use_symlog:
        ax.set_yscale("symlog", linthresh=2.0)
        ax.set_yticks([0, 2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["0", "2", "5", "10", "20", "50", "100"])
        ax.set_ylim(-0.35, 340)
    else:
        ymax = max(others.max(), napm_v)
        ax.set_ylim(-ymax * 0.06, ymax * 1.32)
    ax.set_xlim(-0.48, 0.74)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8.0, fontweight="bold", color=C_TEXT, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if idx == 0:
        ax.set_ylabel("Count across 229 DBPs", fontsize=8.0)

pdf_path = OUT_DIR / "Response_Fig_4_napm_top_dbp.pdf"
png_path = OUT_DIR / "Response_Fig_4_napm_top_dbp.png"
jpg_path = OUT_DIR / "Response_Fig_4_napm_top_dbp.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=96, subsampling=0, dpi=(300, 300))
print("Saved final Response Fig. 4:", pdf_path, jpg_path)
