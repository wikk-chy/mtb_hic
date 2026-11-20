#!/usr/bin/env python3
"""Build Fig_R4C4_replicate_metrics (PDF + 300 dpi PNG).

Panel A : P(s) curves, 4-replicate mean +/- SD band; WT vs napM-KO (large),
          hypoxia/latent vs WT (small).
Panel B : per-replicate dot plots (mean +/- SD) for frac_short, sec_diag_OE,
          ori_ter_OE, compaction_slope; Mann-Whitney p vs WT annotated.
Panel C : reconciliation -- secondary-diagonal RAW mean, depth-normalised
          raw (per 1M contacts), and distance-normalised O/E side by side.
Panel D : 16x16 replicate correlation heatmap (log1p upper triangle).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.linewidth": 0.6,
})

OUT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment4"
COND_ORDER = ["WT", "napM-KO", "hypoxia", "latent"]
COLORS = {"WT": "#1f77b4", "napM-KO": "#d62728", "hypoxia": "#2ca02c",
          "latent": "#9467bd"}

df = pd.read_csv(f"{OUT}/tables/replicate_metrics.csv")
ps = pd.read_csv(f"{OUT}/tables/ps_curves_per_replicate.csv")
stat = pd.read_csv(f"{OUT}/tables/stats_tests.csv")
C = np.load(f"{OUT}/tables/correlation_matrix_16x16.npy")
samples = [f"{c}_{r}" for c in COND_ORDER for r in (1, 2, 3, 4)]

MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 205 * MM))
gs = GridSpec(3, 12, figure=fig, hspace=0.85, wspace=0.9,
              left=0.07, right=0.97, top=0.94, bottom=0.06)


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


# ---------------- Panel A ----------------
axA = fig.add_subplot(gs[0, 0:8])
axA2 = fig.add_subplot(gs[0, 8:12])
ps["center"] = np.sqrt(ps.dist_lo * ps.dist_hi)
# normalise each replicate's P(s) by its total contacts -> contact
# probability, removing sequencing-depth differences between libraries
_tot = df.set_index(["condition", "replicate"])["total_contacts"]
ps["ps"] = ps.apply(lambda r: r.ps / _tot.loc[(r.condition, r.replicate)],
                    axis=1)


def draw_ps(ax, conds, title):
    for cond in conds:
        piv = (ps[ps.condition == cond]
               .groupby("center")["ps"].agg(["mean", "std"]).dropna())
        ax.loglog(piv.index, piv["mean"], "-", lw=1.2, color=COLORS[cond],
                  label=cond)
        ax.fill_between(piv.index, piv["mean"] - piv["std"],
                        piv["mean"] + piv["std"], color=COLORS[cond],
                        alpha=0.2, lw=0)
    ax.set_xlabel("genomic distance s (bp)")
    ax.set_ylabel("P(s)")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.25, which="both", lw=0.3)


draw_ps(axA, ["WT", "napM-KO"], "WT vs napM-KO (mean ± SD, n = 4)")
draw_ps(axA2, ["WT", "hypoxia", "latent"], "stress conditions vs WT")
panel_label(axA, "A")

# ---------------- Panel B ----------------
B_METRICS = [("frac_short_lt100kb", "short-range fraction\n(<100 kb)"),
             ("sec_diag_OE", "secondary-diagonal O/E\n(arm-arm, >500 kb)"),
             ("ori_ter_OE", "ori-ter O/E"),
             ("compaction_slope", "compaction slope\nP(s), 100 kb-1 Mb")]
for k, (m, label) in enumerate(B_METRICS):
    ax = fig.add_subplot(gs[1, k * 3:(k + 1) * 3])
    xs = np.arange(len(COND_ORDER))
    for x, cond in zip(xs, COND_ORDER):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.12, 0.12, 4), y, s=7,
                   color=COLORS[cond], zorder=3, lw=0)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=8,
                    color="black", capsize=2, lw=0.8, zorder=4)
        # Mann-Whitney p vs WT
        if cond != "WT":
            p = stat.query("comparison == @cond + ' vs WT' and metric == @m")["mw_p"].iloc[0]
            ax.text(x, ax.get_ylim()[1], "", fontsize=6)
            ax.annotate(f"p={p:.3f}", (x, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=5.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(COND_ORDER, rotation=30, ha="right", fontsize=6)
    ax.set_title(label, fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.margins(x=0.15, y=0.25)
    if k == 0:
        panel_label(ax, "B")

# ---------------- Panel C ----------------
C_ITEMS = [("sec_diag_raw", "raw mean contact\n(secondary diagonal, >500 kb)", 1),
           ("sec_diag_raw_per1M", "raw / total contacts\n(per 10$^6$ contacts)", 1),
           ("sec_diag_OE", "distance-normalised O/E\n(expected = P(s), same replicate)", 1)]
for k, (m, label, _) in enumerate(C_ITEMS):
    ax = fig.add_subplot(gs[2, k * 3:(k + 1) * 3])
    xs = np.arange(len(COND_ORDER))
    for x, cond in zip(xs, COND_ORDER):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.12, 0.12, 4), y, s=7,
                   color=COLORS[cond], zorder=3, lw=0)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=8,
                    color="black", capsize=2, lw=0.8, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(COND_ORDER, rotation=30, ha="right", fontsize=6)
    ax.set_title(label, fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.margins(x=0.15, y=0.25)
    if k == 0:
        panel_label(ax, "C")

# ---------------- Panel D ----------------
axD = fig.add_subplot(gs[2, 9:12])
im = axD.imshow(C, vmin=0.4, vmax=1.0, cmap="viridis")
axD.set_xticks(range(16)); axD.set_yticks(range(16))
axD.set_xticklabels(samples, rotation=90, fontsize=4)
axD.set_yticklabels(samples, fontsize=4)
for b in (3.5, 7.5, 11.5):
    axD.axhline(b, color="white", lw=0.8)
    axD.axvline(b, color="white", lw=0.8)
cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=5)
cb.set_label("Pearson r (log1p)", fontsize=5.5)
axD.set_title("replicate correlation", fontsize=6.5)
panel_label(axD, "D")

fig.savefig(f"{OUT}/figures/Fig_R4C4_replicate_metrics.pdf")
fig.savefig(f"{OUT}/figures/Fig_R4C4_replicate_metrics.png", dpi=300)
print("Saved figures/Fig_R4C4_replicate_metrics.{pdf,png}")
