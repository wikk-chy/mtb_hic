#!/usr/bin/env python3
"""Standalone redraw of the posterior-predictive check (combined-figure
panel H) with per-scenario vertical jitter so clipped/overlapping points
remain visible. Saves SVG (editable text) + PNG."""
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
    "font.size": 8, "axes.linewidth": 0.6,
})

mc = pd.read_csv(f"{POLY}/tables/model_comparison.csv")
exp = pd.read_csv(f"{POLY}/tables/experiment_metrics.csv")
lat = exp[exp.condition == "latent"]

SCEN = ["baseline", "S1", "S2", "S3", "S3app"]
SCEN_LABEL = {"baseline": "baseline (WT-like)", "S1": "S1 global collapse",
              "S2": "S2 ori-ter tether", "S3": "S3 toroid",
              "S3app": "S3 + apposition"}
COLORS = ["C0", "C1", "C2", "C3", "C4"]
metrics = ["frac_short", "frac_mid", "frac_long", "slope",
           "ori_ter_OE", "arm_arm_OE"]
labels = {"frac_short": "frac short", "frac_mid": "frac mid",
          "frac_long": "frac long", "slope": "P(s) slope",
          "ori_ter_OE": "ori-ter O/E *", "arm_arm_OE": "arm-arm O/E *"}
EXP_NAME = dict(frac_short="frac_short_lt100kb", frac_mid="frac_mid_100kb_1Mb",
                frac_long="frac_long_gt1Mb", slope="compaction_slope",
                ori_ter_OE="ori_ter_OE", arm_arm_OE="sec_diag_OE")

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ys = np.arange(len(metrics))[::-1]
jit = np.linspace(-0.22, 0.22, len(SCEN))
for y, m in zip(ys, metrics):
    mu = lat[EXP_NAME[m]].mean()
    sd = lat[EXP_NAME[m]].std(ddof=1)
    lo, hi = mu - 3 * sd, mu + 3 * sd
    span = max(abs(hi - lo), 1e-12)
    ax.plot([1 / 6, 5 / 6], [y, y], color="grey", lw=3, alpha=0.4, zorder=1)
    ax.plot([0.5, 0.5], [y - 0.42, y + 0.42], color="black", lw=1.5, zorder=2)
    for k, (sc, col) in enumerate(zip(SCEN, COLORS)):
        v_ = mc[mc.scenario == sc].iloc[0][f"sim_{m}"]
        x_raw = (v_ - lo) / span / 6 + 0.5
        xc = np.clip(x_raw, -0.35, 1.35)
        clipped = not (-0.35 < x_raw < 1.35)
        ax.plot(xc, y + jit[k], ">", ms=6, color=col, zorder=3,
                marker=">" if (clipped and x_raw > 1.35) else
                      "<" if (clipped and x_raw < -0.35) else "o",
                label=SCEN_LABEL[sc] if m == metrics[0] else None)

ax.set_yticks(ys)
ax.set_yticklabels([labels[m] for m in metrics], fontsize=8)
ax.axvline(0.5, color="black", lw=0.4, alpha=0.3)
ax.set_xlim(-0.55, 1.75)
ax.set_xlabel("position relative to latent experiment\n"
              "(tick = mean, grey band = ±3 SD; triangles = clipped outside)",
              fontsize=7.5)
ax.set_title("posterior-predictive check (* = held-out metrics)", fontsize=9)
ax.legend(fontsize=6.5, loc="lower right", frameon=False)
fig.savefig(f"{POLY}/figures/fig_H_postpred.svg", bbox_inches="tight")
fig.savefig(f"{POLY}/figures/fig_H_postpred.png", dpi=300, bbox_inches="tight")
print("saved figures/fig_H_postpred.svg/.png")
