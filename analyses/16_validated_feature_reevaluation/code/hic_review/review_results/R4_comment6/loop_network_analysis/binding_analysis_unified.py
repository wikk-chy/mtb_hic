#!/usr/bin/env python3
"""binding_analysis_unified.py — redraw figures/binding_analysis.{pdf,png}.

Panel A: the five network-analysis metrics from tables/binding_summary.csv
(loop_network_analysis.py, 1000 distance-matched permutations; NapM binding
= unified interval-overlap definition since 2026-10-06).

Panel B: unified loop-bridging analysis
(R1_mechanism_reanalysis/scripts/s08_loop_bridging_unified.py): circular-shift
null distributions (10,000 shifts, seed 42) for the THREE NapM binding
definitions — primary: interval overlap of peak spans (454 bins), 17/41;
sensitivities: curated centre bins (433 bins), 17/41; network edges
(427 bins), 15/41.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment6/loop_network_analysis")
UNI = os.path.join(BASE, "review_results/R1_mechanism_reanalysis/"
                  "06_permutation_tests")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False})

C_OBS, C_RAND = "#665A7A", "#C9C5C0"
C_PRIMARY, C_SENS = "#665A7A", "#B9A8C9"

summ = pd.read_csv(os.path.join(OUT, "tables/binding_summary.csv"))
uni = pd.read_csv(os.path.join(UNI, "loop_bridging_unified.csv"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# ---------------- A: archived network-analysis metrics (unchanged) ----------
labels = {"rv_anchor": "Rv0047c-bound\nanchors (of 82)",
          "rv_bridge": "Rv0047c-bridged loops\n(both anchors, of 41)"}
ks = list(labels.keys())
x = np.arange(len(ks))
obs_vals = summ.set_index("metric").loc[ks, "observed"].values
null_means = summ.set_index("metric").loc[ks, "null_mean"].values
null_sds = summ.set_index("metric").loc[ks, "null_sd"].values
ps = summ.set_index("metric").loc[ks, "empirical_p"].values

w = 0.38
axes[0].bar(x - w / 2, obs_vals, w, color=C_OBS, edgecolor="#3A3344",
            linewidth=0.8, label="observed (41 loops)")
axes[0].bar(x + w / 2, null_means, w, color=C_RAND, edgecolor="#8A8580",
            linewidth=0.8, yerr=null_sds, capsize=3,
            error_kw=dict(ecolor="#555049", lw=0.9),
            label="distance-matched random pairs\n(1000 permutations)")
axes[0].set_xticks(x, [labels[k] for k in ks], fontsize=8)
axes[0].set_ylabel("Count", fontsize=9)
axes[0].set_ylim(0, max(obs_vals) * 1.28)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
for xi, (o, p) in enumerate(zip(obs_vals, ps)):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    axes[0].text(xi - w / 2, o + max(obs_vals) * 0.03, f"{int(o)} {star}",
                 ha="center", fontsize=8.5)
axes[0].set_title("A  NapM occupancy at loop anchors\nvs distance-matched random pairs",
                  loc="left", fontsize=10, fontweight="bold")

# ---------------- B: unified circular-shift nulls, three definitions --------
ax = axes[1]
bins = np.arange(-0.5, 30.5, 1)
DEFS = [("network", "#C9C5C0", 1, "sensitivity: network edges"),
        ("curated", "#7FA6B8", 2, "sensitivity: curated centre bins"),
        ("interval", C_PRIMARY, 3, "primary: interval overlap (peak span)")]
for defname, col, zd, lab in DEFS:
    null = np.load(os.path.join(UNI, f"null_loop_bridging_{defname}_both.npy"))
    row = uni[(uni.definition == defname) & (uni.metric == "both")].iloc[0]
    ax.hist(null, bins=bins, color=col, edgecolor="none", alpha=0.5, zorder=zd,
            label=f"{lab}\nnull {row.null_mean:.1f}±{row.null_sd:.1f}, "
                  f"observed {int(row.observed)}, p = {row.p_empirical:.4f}")
    ax.axvline(row.observed, color=col, lw=2, zorder=4)
ax.set_xlabel("NapM-bridged loops (both anchors bound) in circular-shift null",
              fontsize=9)
ax.set_ylabel("Shifts (of 10,000)", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_title("B  NapM-bridged loops vs circular-shift null\n(10,000 shifts; three binding definitions)",
             loc="left", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/binding_analysis.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/binding_analysis.png"), dpi=300,
            bbox_inches="tight")
print("wrote figures/binding_analysis.{pdf,png}")
