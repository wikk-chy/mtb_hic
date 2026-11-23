#!/usr/bin/env python3
"""Standalone redraw of the 'observed changes vs WT (log2 fold; Welch p)'
heatmap (Panel B of Fig_R4C5_v3) as a single SVG file."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{BASE}/review_results/R4_comment5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "font.size": 7, "axes.linewidth": 0.6,
})

obs = pd.read_csv(f"{OUT}/tables/observed_vs_predictions.csv")

comps = ["latent", "hypoxia"]
mlabels = {"ori_ter_OE": "ori-ter", "sec_diag_OE": "arm-arm",
           "frac_short_lt100kb": "P(s) short",
           "frac_long_gt1Mb": "P(s) long",
           "compaction_slope": "P(s) slope"}
M5 = list(mlabels)
Z = np.array([[np.log2(obs.loc[obs.metric == m, f"{c}_fold"].iloc[0])
               for c in comps] for m in M5])

fig, ax = plt.subplots(figsize=(2.4, 3.2))
im = ax.imshow(Z, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(2))
ax.set_xticklabels([f"{c}/WT" for c in comps])
ax.set_yticks(range(len(M5)))
ax.set_yticklabels([mlabels[m] for m in M5])
for i, m in enumerate(M5):
    for j, c in enumerate(comps):
        p = obs.loc[obs.metric == m, f"{c}_p"].iloc[0]
        fc = obs.loc[obs.metric == m, f"{c}_fold"].iloc[0]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        ax.text(j, i, f"{fc:.2f}×\n{star}", ha="center", va="center",
                fontsize=7)
ax.set_title("observed changes vs WT\n(log2 fold; Welch p)", fontsize=8)
cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.75, pad=0.16)
cb.set_label("log2 fold", fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.savefig(f"{OUT}/figures/observed_changes_vs_WT_heatmap.svg",
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/observed_changes_vs_WT_heatmap.png", dpi=300,
            bbox_inches="tight")
print("saved figures/observed_changes_vs_WT_heatmap.svg")
