#!/usr/bin/env python
"""Final zoomed-map figure for R4 comment 3.

Layout: 3 rows x 4 columns = 6 representative loops x (WT | napM-KO).
Selection: 4 strongest long-range loops (>200 kb) + the single strongest loop
(1342 x 1532 kb, O/E 4.12) + one additional 100-200 kb loop.

Run: ../.venv_cooltools/bin/python final_figures.py
"""

import os
import numpy as np
import pandas as pd
import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"

BINSIZE = 5000
cmap = LinearSegmentedColormap.from_list("redwhite", ["#ffffff", "#d7301f"])

clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)

pl = pd.read_csv(os.path.join(OUT, "tables/per_loop_oe_wt_vs_ko.csv"))

# representative loops: 4 strongest >200 kb + the 1342 kb loop + one more 100-200 kb
pl["dist_kb"] = (pl.end2 - pl.start1 - 5000) // 1000
long_top = pl[pl.dist_kb > 200].sort_values("OE_center5x5_WT", ascending=False).head(4)
mid_strong = pl[(pl.dist_kb <= 200) & (pl.dist_kb > 100)].sort_values("OE_center5x5_WT", ascending=False).head(2)
sel = pd.concat([long_top, mid_strong]).reset_index(drop=True)

fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.3)

for idx, (_, row) in enumerate(sel.iterrows()):
    r, c = divmod(idx, 2)
    cond_idx = idx % 2
    mid1 = int(row.start1) + 2500
    mid2 = int(row.end2) - 2500
    span = mid2 - mid1
    window = min(max(span // 2, 100000), 350000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window

    for j, (clr, cond, oe_col) in enumerate([(clr_wt, "WT", "OE_center5x5_WT"),
                                             (clr_ko, "napM-KO", "OE_center5x5_napMKO")]):
        ax = fig.add_subplot(gs[r, c * 2 + j])
        m = clr.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}", f"Chromosome:{r2_s}-{r2_e}")
        im = ax.imshow(m, cmap=cmap,
                       vmin=np.nanpercentile(m[m > 0], 2),
                       vmax=np.nanpercentile(m, 98),
                       interpolation="nearest")
        ax.axhline((mid2 - r2_s) / (r2_e - r2_s) * m.shape[0], color="blue", ls="--", lw=1)
        ax.axvline((mid1 - r1_s) / (r1_e - r1_s) * m.shape[1], color="blue", ls="--", lw=1)
        ax.set_title(f"{cond}\nO/E={row[oe_col]:.2f}", fontsize=9)
        if r == 2:
            ax.set_xlabel("Anchor 1 region", fontsize=8)
        if c * 2 + j == 0:
            ax.set_ylabel("Anchor 2 region", fontsize=8)
        # panel letter
        ax.text(-0.12, 1.08, chr(ord("A") + idx * 2 + j), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Contacts")
    # loop label between the pair
    fig.text(0.02, 1 - (r + 0.55) / 3.3,
             f"{mid1//1000} kb x {mid2//1000} kb ({row.dist_kb} kb)",
             fontsize=10, fontweight="bold", va="center")

fig.suptitle("Representative long-range chromosomal contacts: zoomed contact maps (blue dashes = anchors)",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig12_representative_zoomed_maps.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig12_representative_zoomed_maps.png"), dpi=300, bbox_inches="tight")
print("fig12 saved")
