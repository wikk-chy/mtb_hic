#!/usr/bin/env python3
"""Standalone redraw of combined-figure panel G (per-metric z-scores vs
latent experiment) as a single SVG file."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLY = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "font.size": 7, "axes.linewidth": 0.6,
})

mc = pd.read_csv(f"{POLY}/tables/model_comparison.csv")
metrics = ["frac_short", "frac_mid", "frac_long", "slope",
           "ori_ter_OE", "arm_arm_OE"]
Z = np.clip(mc[[f"z_{m}" for m in metrics]].values, -10, 10)

fig, ax = plt.subplots(figsize=(4.2, 3.0))
im = ax.imshow(Z, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=7)
ax.set_yticks(range(len(mc)))
ax.set_yticklabels(mc.scenario, fontsize=7)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        ax.text(j, i, f"{mc.iloc[i][f'z_{metrics[j]}']:.1f}", ha="center",
                va="center", fontsize=6.5)
ax.set_title("z-scores vs latent experiment\n(held-out: last two columns)",
             fontsize=8)
cb = fig.colorbar(im, ax=ax, label="z (clipped at ±10)", shrink=0.85)
cb.ax.tick_params(labelsize=6)

fig.savefig(f"{POLY}/figures/fig_G_zscores_heatmap.svg", bbox_inches="tight")
fig.savefig(f"{POLY}/figures/fig_G_zscores_heatmap.png", dpi=300,
            bbox_inches="tight")
print("saved figures/fig_G_zscores_heatmap.svg")
