#!/usr/bin/env python3
"""
s03_figures.py — three main figures for the NapM reviewer-response analysis.

Inputs : tables/ produced by s01_build_tables.py + s02_stats.py
Outputs: figures/Fig1_spatial_association.{pdf,png}
         figures/Fig2_joint_discordant.{pdf,png}
         figures/Fig3_condition_dependent.{pdf,png}

All results are plotted as estimated (negative results included); no
re-computation of statistics except where stated (bootstrap CIs for Fig 3D
and the display null histogram for Fig 3E use seed 42; the pre-registered
p-values annotated come from the stats tables).

Environment: .venv_cooltools (cooler 0.10.4, numpy 1.26.4, pandas 2.2.2,
scipy 1.13.1, matplotlib 3.8.0). Random seed: 42.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import gridspec
import cooler

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{ROOT}/review_results/R1_mechanism_zhenghe"
TAB = f"{OUT}/tables"
FIG = f"{OUT}/figures"
BIN = 5000
N_BINS = 883

# ------------------------------------------------------------------- style
C = dict(WT="#0072B2", KO="#D55E00", hypoxia="#009E73", latent="#CC79A7",
         KO_hypoxia="#E69F00", chip="#6A51A3", matched="#A6A6A6")
C_CLS = dict(detected_increase="#B2182B", detected_decrease="#2166AC",
             supports_small_change="#969696", cannot_determine="#D9D9D9")
CLS_LABEL = dict(detected_increase="Detected increase",
                 detected_decrease="Detected decrease",
                 supports_small_change="Supports small change",
                 cannot_determine="Cannot determine")
C_RNA = dict(detected_change="#B2182B", supports_small_change="#66C2A5",
             cannot_determine="#BDBDBD", no_data="#F0F0F0")
RNA_LABEL = dict(detected_change="RNA: detected change",
                 supports_small_change="RNA: supports small change",
                 cannot_determine="RNA: cannot determine",
                 no_data="RNA: no gene in bin")

CMAP_HEAT = mcolors.LinearSegmentedColormap.from_list(
    "contact", ["#FFF5F0", "#FB6A4A", "#99000D"])
CMAP_HEAT.set_bad("#BFBFBF")   # masked/missing bins in grey, distinct from 0
CMAP_DIV = mcolors.LinearSegmentedColormap.from_list(
    "effect", ["#2166AC", "#F7F7F7", "#B2182B"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.7, "lines.linewidth": 1.0,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
MM = 1 / 25.4
FIGW = 180 * MM


def panel_label(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def savefig(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------- data
b = pd.read_csv(f"{TAB}/bin_level_integrated.csv")
genes = pd.read_csv(f"{TAB}/gene_level_integrated.csv")
inter = pd.read_csv(f"{TAB}/rna_interaction_model.csv")
regions = pd.read_csv(f"{TAB}/napm_bound_regions.csv")
mb = b.bin.values * BIN / 1e6   # bin midpoint in Mb (start-based)

# ChIP smoothed signal per bin (same rule as s01: 1-kb rolling mean)
bg = pd.read_csv(f"{ROOT}/review_data/chip_seq_data/Rv0047c.bedgraph",
                 sep="\t", header=None, names=["c", "s", "e", "v"])
sig = np.zeros(int(bg.e.max()))
sig[bg.s.values] = bg.v.values
sm = np.convolve(sig, np.ones(100) / 100, mode="same")
chip_bin = np.array([sm[i * BIN:(i + 1) * BIN].mean() for i in range(N_BINS)])

bound = b.chip_bound.values.astype(bool)
matched = b.matched_background.values.astype(bool)

# balanced contact matrices (for maps)
def load_mat(cond):
    fn = dict(WT="wt_5k.cool", KO="napm_ko_5k.cool")[cond]
    c = cooler.Cooler(f"{ROOT}/review_data/cool_file/{fn}::/resolutions/5000")
    w = c.bins()[:]["weight"].values.astype(float)
    m = c.matrix(balance=False, sparse=False)[:].astype(float)
    m = m * w[:, None] * w[None, :]
    m[~np.isfinite(m)] = np.nan
    return m

MAT_WT = load_mat("WT")
MAT_KO = load_mat("KO")

# example loci (fixed choices, documented in README):
#   locus A: bin 34 (~170 kb) bound, detected insulation increase,
#            RNA detected decrease (Rv0144/Rv0145) -> concordant
#   locus B: bin 10 (~50 kb), the napM locus itself: bound, "supports small
#            structural change" although Rv0047c RNA drops ~2^-11 (the
#            deletion itself) -> occupancy/large RNA change without
#            detectable structural change
LOC_CONC = 34
LOC_DISC = 10
print("example loci: concordant bin", LOC_CONC, "| napM-locus bin", LOC_DISC)


def heatmap_ax(ax, b0, b1, vmax_pct=98):
    """WT upper / KO lower triangle, shared sequential colour scale."""
    n = b1 - b0
    a = MAT_WT[b0:b1, b0:b1]
    k = MAT_KO[b0:b1, b0:b1]
    both = np.concatenate([a[np.isfinite(a)], k[np.isfinite(k)]])
    both = both[both > 0]
    vmin, vmax = np.percentile(np.log10(both), [5, vmax_pct])
    la, lk = np.log10(np.where(a > 0, a, np.nan)), np.log10(np.where(k > 0, k, np.nan))
    comb = np.where(np.triu(np.ones((n, n)), k=0).astype(bool), la, lk)
    comb = np.ma.masked_invalid(comb)
    ext = [b0 * BIN / 1e6, b1 * BIN / 1e6, b1 * BIN / 1e6, b0 * BIN / 1e6]
    im = ax.imshow(comb, cmap=CMAP_HEAT, vmin=vmin, vmax=vmax,
                   extent=ext, origin="upper", aspect="auto",
                   interpolation="nearest", rasterized=True)
    ax.plot([ext[0], ext[1]], [ext[2], ext[3]], color="k", lw=0.5)
    return im


def track_chip(ax, b0, b1):
    x = np.arange(b0, b1) * BIN / 1e6
    ax.fill_between(x, 0, chip_bin[b0:b1], color=C["chip"], lw=0, alpha=0.85)
    ax.set_ylabel("NapM ChIP", color=C["chip"])
    ax.set_xlim(x[0], x[-1])
    ax.spines[["top", "right"]].set_visible(False)


def track_ins(ax, b0, b1, show_delta=True):
    x = np.arange(b0, b1) * BIN / 1e6
    ax.plot(x, b.ins_WT.values[b0:b1], color=C["WT"], lw=0.9, label="WT")
    ax.plot(x, b.ins_KO.values[b0:b1], color=C["KO"], lw=0.9, label="napM-KO")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_ylabel("log2 insulation\n(25 kb)")
    ax.set_xlim(x[0], x[-1])
    ax.spines[["top", "right"]].set_visible(False)


def track_delta(ax, b0, b1):
    x = np.arange(b0, b1) * BIN / 1e6
    d = b.d_ins_KO_WT.values[b0:b1]
    ax.fill_between(x, 0, d, where=d >= 0, color="#B2182B", lw=0, alpha=0.8)
    ax.fill_between(x, 0, d, where=d < 0, color="#2166AC", lw=0, alpha=0.8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("Δ insulation\n(KO−WT)")
    ax.set_xlim(x[0], x[-1])
    ax.spines[["top", "right"]].set_visible(False)


# ==================================================================== FIG 1
def fig1():
    fig = plt.figure(figsize=(FIGW, 240 * MM))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.55,
                           height_ratios=[1.5, 1, 1])

    # ---- A: representative region (0-500 kb, contains the napM locus)
    gsA = gridspec.GridSpecFromSubplotSpec(4, 1, gs[0, :],
                                           height_ratios=[2.6, 0.5, 0.65, 0.65],
                                           hspace=0.12)
    b0, b1 = 0, 100
    axm = fig.add_subplot(gsA[0])
    im = heatmap_ax(axm, b0, b1)
    axm.set_title("Upper triangle: WT | lower triangle: napM-KO "
                  "(balanced contacts, log10, shared scale)", pad=3,
                  fontsize=6.5)
    axm.set_xticklabels([])
    cb = fig.colorbar(im, ax=axm, fraction=0.02, pad=0.01)
    cb.set_label("log10 balanced\ncontact", fontsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
    axm.annotate("napM", xy=(0.051, 0.051), xytext=(0.14, 0.14),
                 arrowprops=dict(arrowstyle="-", lw=0.5), fontsize=6)
    axc = fig.add_subplot(gsA[1], sharex=axm)
    track_chip(axc, b0, b1)
    axc.set_ylabel("NapM\nChIP", color=C["chip"], fontsize=6, labelpad=2)
    axc.set_yticks([0])
    plt.setp(axc.get_xticklabels(), visible=False)
    axi = fig.add_subplot(gsA[2], sharex=axm)
    track_ins(axi, b0, b1)
    axi.set_ylabel("log2 ins.", fontsize=6, labelpad=2)
    axi.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2,
               fontsize=5.8, borderaxespad=0.1)
    plt.setp(axi.get_xticklabels(), visible=False)
    axd = fig.add_subplot(gsA[3], sharex=axm)
    track_delta(axd, b0, b1)
    axd.set_ylabel("Δins", fontsize=6, labelpad=2)
    axd.set_xlabel("Genomic position (Mb)")
    panel_label(axm, "A")

    # ---- B: class proportions (100% stacked, k/n annotated)
    prop = pd.read_csv(f"{TAB}/stats_fig1_class_proportions.csv")
    axB = fig.add_subplot(gs[1, 0:2])
    classes = ["detected_increase", "detected_decrease",
               "supports_small_change", "cannot_determine"]
    for xi, sname in enumerate(["NapM_bound", "matched_background"]):
        sub = prop[prop["set"] == sname]
        n = int(sub.n.iloc[0])
        left = 0.0
        for cls in classes:
            r = sub[sub.change_class == cls].iloc[0]
            axB.bar(xi, r.frac, bottom=left, width=0.62, color=C_CLS[cls],
                    edgecolor="white", lw=0.4)
            if r.frac > 0.06:
                axB.text(xi, left + r.frac / 2, f"{r.k}/{n}", ha="center",
                         va="center", fontsize=6,
                         color="white" if "detected" in cls else "black")
            left += r.frac
    axB.set_xticks([0, 1])
    axB.set_xticklabels([f"NapM bound\n(n=104 bins)", f"Matched background\n(n=104 bins)"])
    axB.set_ylabel("Fraction of bins")
    axB.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_CLS[c]) for c in classes]
    axB.legend(handles, [CLS_LABEL[c] for c in classes], frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
               fontsize=5.8, handlelength=1.2, columnspacing=0.8)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("Insulation-change calls after napM deletion", fontsize=6.8)
    panel_label(axB, "B")

    # ---- C: ECDF of continuous effect sizes
    pr = pd.read_csv(f"{TAB}/stats_fig1_paired_bound_vs_matched.csv").iloc[0]
    axC = fig.add_subplot(gs[1, 2:4])
    for vals, col, lab in [(b.d_ins_KO_WT[bound], C["chip"], "NapM bound"),
                           (b.d_ins_KO_WT[matched], C["matched"],
                            "Matched background")]:
        v = np.sort(vals.values)
        axC.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.2,
                 label=lab)
    axC.axvline(0, color="k", lw=0.5, ls="--")
    axC.axvspan(-0.25, 0.25, color="0.92", zorder=0)
    axC.text(0.98, 0.04, "shaded band: equivalence bound ±0.25 log2",
             transform=axC.transAxes, ha="right", fontsize=5.5, color="0.35")
    axC.set_xlabel("Δ insulation KO−WT (log2, 25 kb window)")
    axC.set_ylabel("ECDF")
    axC.set_xlim(-1.2, 1.2)
    axC.legend(frameon=False, loc="upper left", fontsize=6)
    axC.spines[["top", "right"]].set_visible(False)
    axC.set_title(f"median paired diff {pr.median_paired_diff:+.3f} "
                  f"[95% CI {pr.ci_lo:+.3f}, {pr.ci_hi:+.3f}], "
                  f"Wilcoxon p = {pr.wilcoxon_p:.2f} (n = {int(pr.n_pairs)} pairs)",
                  fontsize=6.5)
    panel_label(axC, "C")

    # ---- D: aggregate profile around occupancy centres
    prof = pd.read_csv(f"{TAB}/fig1_aggregate_profile.csv")
    axD = fig.add_subplot(gs[2, 0:2])
    axD.plot(prof.offset_kb, prof.bound_mean, color=C["chip"], lw=1.2,
             label="Bound-region centres (n=110)")
    axD.fill_between(prof.offset_kb, prof.bound_mean - prof.bound_se,
                     prof.bound_mean + prof.bound_se, color=C["chip"],
                     alpha=0.25, lw=0)
    axD.plot(prof.offset_kb, prof.matched_mean, color=C["matched"], lw=1.2,
             label="Matched-bin centres (n=104)")
    axD.fill_between(prof.offset_kb, prof.matched_mean - prof.matched_se,
                     prof.matched_mean + prof.matched_se, color=C["matched"],
                     alpha=0.35, lw=0)
    axD.axvline(0, color="k", lw=0.5, ls=":")
    axD.axhline(0, color="k", lw=0.4)
    axD.set_xlabel("Distance from occupancy centre (kb)")
    axD.set_ylabel("Mean Δ insulation (KO−WT)")
    axD.legend(frameon=False, loc="upper right")
    axD.spines[["top", "right"]].set_visible(False)
    axD.set_title("No concentration of structural change at occupancy centres\n"
                  "(mean ± s.e.m.; centre-enrichment permutation p = 0.999)",
                  fontsize=6.5)
    panel_label(axD, "D")

    # ---- E: observed vs circular-shift null
    perm = pd.read_csv(f"{TAB}/stats_fig1_permutation.csv").iloc[0]
    null = np.load(f"{TAB}/fig1_null_shift.npy")
    axE = fig.add_subplot(gs[2, 2])
    axE.hist(null, bins=40, color="0.75", edgecolor="white", lw=0.3)
    axE.axvline(perm.obs, color=C["chip"], lw=1.4)
    axE.set_xlabel("median Δins(bound) − median Δins(matched)\nunder circular-shift null")
    axE.set_ylabel("Count (1000 shifts)")
    axE.spines[["top", "right"]].set_visible(False)
    axE.set_title(f"observed {perm.obs:+.3f},  p = {perm.p_circular_shift:.2f}",
                  fontsize=6.8)
    axE.text(0.03, 0.95, "CID-boundary distance\npermutation p = 0.73",
             transform=axE.transAxes, fontsize=6, va="top", color="0.3")
    panel_label(axE, "E")

    # ---- F: two example loci
    gsF = gridspec.GridSpecFromSubplotSpec(2, 1, gs[2, 3], hspace=1.2)
    for axF, loc, tag in [
            (fig.add_subplot(gsF[0]), LOC_CONC,
             f"concordant (bin {LOC_CONC}):\nbound + ins. increase + RNA down"),
            (fig.add_subplot(gsF[1]), LOC_DISC,
             f"napM locus (bin {LOC_DISC}):\nbound, small struct. change")]:
        h = 8
        x = np.arange(loc - h, loc + h + 1) * BIN / 1e6
        axF.plot(x, b.ins_WT.values[loc-h:loc+h+1], color=C["WT"], lw=0.9)
        axF.plot(x, b.ins_KO.values[loc-h:loc+h+1], color=C["KO"], lw=0.9)
        axF2 = axF.twinx()
        axF2.fill_between(x, 0, chip_bin[loc-h:loc+h+1], color=C["chip"],
                          alpha=0.3, lw=0)
        axF2.set_yticks([])
        axF2.spines[["top", "right"]].set_visible(False)
        axF.axhline(0, color="k", lw=0.4, ls=":")
        axF.set_title(tag, fontsize=5.8, loc="left")
        axF.spines[["top", "right"]].set_visible(False)
        axF.set_ylabel("ins.", fontsize=6)
        axF.tick_params(labelsize=5.5)
    axF.set_xlabel("Position (Mb)", fontsize=6)
    panel_label(fig.add_subplot(gsF[:], frame_on=False), "F")
    fig.axes[-1].set_xticks([]); fig.axes[-1].set_yticks([])

    fig.suptitle("Fig 1. Spatial association between NapM occupancy and "
                 "chromosome structural changes following napM deletion",
                 fontsize=8.5, y=0.995)
    savefig(fig, "Fig1_spatial_association")


# ==================================================================== FIG 2
def fig2():
    fig = plt.figure(figsize=(FIGW, 235 * MM))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.6, wspace=0.6,
                           height_ratios=[1, 1, 1.1])

    # ---- A: occupancy x structure x RNA classification
    cc = pd.read_csv(f"{TAB}/stats_fig2_cross_classification.csv")
    axA = fig.add_subplot(gs[0, 0:2])
    rna_classes = ["detected_change", "supports_small_change",
                   "cannot_determine", "no_data"]
    xticks, xi = [], 0
    for sname, slab in [("NapM_bound", "NapM bound"), ("matched_background",
                                                      "Matched background")]:
        for sd in [True, False]:
            sub = cc[(cc["set"] == sname) & (cc.structure_detected == sd)]
            n = int(sub.n.iloc[0])
            left = 0.0
            for rc in rna_classes:
                k = int(sub[sub.rna_class == rc].k.iloc[0])
                if k == 0:
                    continue
                axA.bar(xi, k, bottom=left, width=0.62, color=C_RNA[rc],
                        edgecolor="white", lw=0.4)
                if k >= 6:
                    axA.text(xi, left + k / 2, str(k), ha="center",
                             va="center", fontsize=6,
                             color="white" if rc == "detected_change"
                             else "black")
                left += k
            xticks.append((xi, f"{'struct. chg' if sd else 'no det. chg'}\n(n={n})"))
            xi += 1
        xi += 0.4
    axA.set_xticks([t[0] for t in xticks])
    axA.set_xticklabels([t[1] for t in xticks], fontsize=5.5)
    axA.set_ylabel("Number of bins")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_RNA[c]) for c in rna_classes]
    axA.legend(handles, [RNA_LABEL[c] for c in rna_classes], frameon=False,
               fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.3),
               ncol=2, handlelength=1.2, columnspacing=0.8)
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_title("Occupancy × structure × RNA classification",
                  fontsize=6.8)
    panel_label(axA, "A")

    # ---- B: RNA vs structure scatter
    corr = pd.read_csv(f"{TAB}/stats_fig2_rna_structure_corr.csv")
    axB = fig.add_subplot(gs[0, 2:4])
    ok = np.isfinite(b.rna_mean_log2FC_KOvsWT.values)
    axB.scatter(b.d_ins_KO_WT[ok & ~bound & ~matched],
                b.rna_mean_log2FC_KOvsWT[ok & ~bound & ~matched],
                s=3, color="0.85", lw=0, rasterized=True, label="other bins")
    axB.scatter(b.d_ins_KO_WT[ok & matched], b.rna_mean_log2FC_KOvsWT[ok & matched],
                s=10, color=C["matched"], lw=0, label="matched background")
    axB.scatter(b.d_ins_KO_WT[ok & bound], b.rna_mean_log2FC_KOvsWT[ok & bound],
                s=10, color=C["chip"], lw=0, label="NapM bound")
    axB.axhline(0, color="k", lw=0.4); axB.axvline(0, color="k", lw=0.4)
    rb = corr[corr["set"] == "NapM_bound"].iloc[0]
    rm = corr[corr["set"] == "matched_background"].iloc[0]
    axB.set_xlabel("Δ insulation KO−WT (log2)")
    axB.set_ylabel("Mean gene log2FC KO/WT (bin)")
    axB.legend(frameon=False, markerscale=2, loc="lower left", fontsize=6)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title(f"no RNA–structure coupling: Spearman r = {rb.spearman_r:+.2f}\n"
                  f"(bound, p={rb.p:.2f}); r = {rm.spearman_r:+.2f} "
                  f"(matched, p={rm.p:.2f})", fontsize=6.3)
    panel_label(axB, "B")

    # ---- C: structural change with limited RNA change
    lim = pd.read_csv(f"{TAB}/stats_fig2_limited_rna_structure_change.csv").iloc[0]
    axC = fig.add_subplot(gs[1, 0:2])
    sd = b[b.structure_detected]
    counts = [int((sd.rna_bin_class == c).sum()) for c in rna_classes]
    bars = axC.bar(range(4), counts, color=[C_RNA[c] for c in rna_classes],
                   width=0.62, edgecolor="white")
    for i, k in enumerate(counts):
        axC.text(i, k + 1, f"{k}/{len(sd)}", ha="center", fontsize=6.5)
    axC.set_xticks(range(4))
    axC.set_xticklabels(["RNA detected\nchange", "RNA supports\nsmall change",
                         "RNA cannot\ndetermine", "no gene\nin bin"],
                        fontsize=6)
    axC.set_ylabel(f"Bins with detected\nstructural change (n={len(sd)})")
    axC.spines[["top", "right"]].set_visible(False)
    axC.set_title("Structural changes in regions with limited changes in\n"
                  f"measured RNA abundance: {int(lim.n_limitedRNA_structureChanged)}"
                  f"/{len(sd)} bins, none NapM-bound", fontsize=6.8)
    panel_label(axC, "C")

    # ---- D: nested models
    nm = pd.read_csv(f"{TAB}/nested_models.csv") if False else \
        pd.read_csv(f"{TAB}/stats_fig2_nested_models.csv")
    mc = pd.read_csv(f"{TAB}/stats_fig2_model_comparison.csv").iloc[0]
    axD = fig.add_subplot(gs[1, 2])
    axD.bar(range(3), nm.cv_R2, color=["0.55", "0.35", C["chip"]], width=0.6)
    for i, v in enumerate(nm.cv_R2):
        axD.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5)
    axD.set_xticks(range(3))
    axD.set_xticklabels(["M0", "M1\n+RNA", "M2\n+occup."], fontsize=6)
    axD.set_ylim(0, 0.8)
    axD.set_ylabel("Out-of-block CV R²")
    axD.spines[["top", "right"]].set_visible(False)
    axD.set_title(f"ΔR² (M2−M1) = {mc.dR2_M2_vs_M1:+.4f}\n"
                  f"[{mc.ci_lo:+.4f}, {mc.ci_hi:+.4f}], n={int(nm.n_bins.iloc[0])}",
                  fontsize=6)
    panel_label(axD, "D")

    # ---- E: sensitivity
    sens = pd.read_csv(f"{TAB}/stats_sensitivity.csv")
    axE = fig.add_subplot(gs[1, 3])
    ypos = np.arange(len(sens))[::-1]
    axE.errorbar(sens.stat, ypos, fmt="o", color=C["chip"], ms=4,
                 mfc="white", mew=1.2)
    for y, (_, r) in zip(ypos, sens.iterrows()):
        axE.text(0.19, y, f"p={r.p:.2f}", va="center", ha="right",
                 fontsize=5.8)
    axE.axvline(0, color="k", lw=0.5, ls="--")
    axE.set_yticks(ypos)
    axE.set_yticklabels(["25 kb (primary)", "10 kb", "50 kb", "alt. occup."],
                        fontsize=5.8)
    axE.set_xlabel("median Δins diff\n(bound − matched)", fontsize=6)
    axE.set_xlim(-0.16, 0.2)
    axE.spines[["top", "right"]].set_visible(False)
    axE.set_title("Robustness across scales\nand occupancy calls", fontsize=6.5)
    panel_label(axE, "E")

    # ---- F: local instances of two evidence combinations
    gsF = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2, :], wspace=0.35)
    for axF, loc, tag in [
            (fig.add_subplot(gsF[0]), LOC_CONC,
             f"bound + structure change + RNA change\n(bin {LOC_CONC}, "
             f"~{LOC_CONC*5} kb: Rv0144/Rv0145 down)"),
            (fig.add_subplot(gsF[1]), LOC_DISC,
             f"napM locus (bin {LOC_DISC}): bound + supports small\n"
             "structural change despite Rv0047c RNA loss")]:
        h = 8
        x = np.arange(loc - h, loc + h + 1) * BIN / 1e6
        axF.plot(x, b.ins_WT.values[loc-h:loc+h+1], color=C["WT"], lw=0.9,
                 label="WT insulation")
        axF.plot(x, b.ins_KO.values[loc-h:loc+h+1], color=C["KO"], lw=0.9,
                 label="KO insulation")
        axF2 = axF.twinx()
        axF2.fill_between(x, 0, chip_bin[loc-h:loc+h+1], color=C["chip"],
                          alpha=0.3, lw=0, label="NapM ChIP")
        axF2.set_yticks([])
        axF2.spines[["top", "right"]].set_visible(False)
        axF.axhline(0, color="k", lw=0.4, ls=":")
        axF.set_title(tag, fontsize=6.3, loc="left")
        axF.set_xlabel("Position (Mb)")
        axF.set_ylabel("log2 insulation")
        axF.spines[["top", "right"]].set_visible(False)
        h1, l1 = axF.get_legend_handles_labels()
        h2, l2 = axF2.get_legend_handles_labels()
        axF.legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.8,
                   loc="upper right")
    panel_label(fig.add_subplot(gsF[:], frame_on=False), "F")
    fig.axes[-1].set_xticks([]); fig.axes[-1].set_yticks([])

    fig.suptitle("Fig 2. Joint and discordant patterns of NapM occupancy, "
                 "chromosome structure and RNA abundance", fontsize=8.5,
                 y=0.995)
    savefig(fig, "Fig2_joint_discordant")


# ==================================================================== FIG 3
def fig3():
    fig = plt.figure(figsize=(FIGW, 240 * MM))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.65, wspace=0.5,
                           height_ratios=[0.9, 1.1, 1])

    # ---- A: design schematic
    axA = fig.add_subplot(gs[0, 0])
    bw, bh = 0.8, 0.62
    boxes = [("WT", C["WT"], 0, 1.1), ("napM-KO", C["KO"], 1.05, 1.1),
             ("WT", C["hypoxia"], 0, 0.1), ("napM-KO", C["KO_hypoxia"], 1.05, 0.1)]
    for lab, col, x, y in boxes:
        axA.add_patch(plt.Rectangle((x, y), bw, bh, facecolor=col,
                                    alpha=0.9, edgecolor="none"))
        axA.text(x + bw / 2, y + bh / 2, lab, ha="center", va="center",
                 color="white", fontsize=6.2, fontweight="bold")
    axA.text(0.4, 1.86, "baseline", ha="center", fontsize=6)
    axA.text(1.45, 1.86, "baseline", ha="center", fontsize=6)
    axA.text(0.4, -0.02, "hypoxia", ha="center", fontsize=6)
    axA.text(1.45, -0.02, "hypoxia", ha="center", fontsize=6)
    axA.add_patch(plt.Rectangle((2.45, 0.6), bw, bh, facecolor=C["latent"],
                                alpha=0.9, edgecolor="none"))
    axA.text(2.85, 0.91, "WT", ha="center", va="center", color="white",
             fontsize=6.2, fontweight="bold")
    axA.text(2.85, 0.48, "latent", ha="center", fontsize=6)
    axA.text(2.85, 0.22, "no KO-latent:\ndescriptive only", ha="center",
             fontsize=5.5, style="italic")
    axA.text(0, -0.5, "RNA-seq: 2 reps/group → formal\ninteraction test. Hi-C: 4/4/4/3 reps\n→ Welch interaction (unbalanced)",
             ha="left", fontsize=5.5, va="top")
    axA.set_xlim(-0.15, 3.45); axA.set_ylim(-1.35, 2.1)
    axA.axis("off")
    axA.set_title("2 × 2 design (genotype × condition)", fontsize=6.8,
                  loc="left")
    panel_label(axA, "A")

    # ---- B: genome-wide tracks of condition effects
    axB = fig.add_subplot(gs[0, 1:])
    def smooth(v, w=5):
        return pd.Series(v).rolling(w, center=True, min_periods=1).mean().values
    axB.plot(mb, smooth(b.d_ins_KO_WT), color=C["KO"], lw=0.9,
             label="KO − WT (baseline)")
    axB.plot(mb, smooth(b.d_ins_hyp_WT), color=C["hypoxia"], lw=0.9,
             label="hypoxia − WT (WT response)")
    axB.plot(mb, smooth(b.d_ins_hyp_KO), color=C["KO_hypoxia"], lw=0.9,
             label="KO-hypoxia − KO (KO response)")
    axB.axhline(0, color="k", lw=0.5)
    axB.set_xlabel("Genomic position (Mb)")
    axB.set_ylabel("Δ log2 insulation (25 kb)")
    axB.legend(frameon=False, fontsize=6, ncol=3, loc="upper right")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_title("Genome-wide insulation effects (5-bin running mean of "
                  "replicate-based effects;\nHi-C replicates: WT/KO/hypoxia "
                  "n=4, KO-hypoxia n=3)", fontsize=6.8)
    panel_label(axB, "B")

    # ---- C: RNA hypoxia response WT vs KO
    axC = fig.add_subplot(gs[1, 0])
    axC.scatter(inter.hyp_response_wt, inter.hyp_response_ko, s=2.5,
                color="0.75", lw=0, rasterized=True)
    nom = inter.p_int < 0.01
    axC.scatter(inter.hyp_response_wt[nom], inter.hyp_response_ko[nom],
                s=5, color=C["KO_hypoxia"], lw=0, rasterized=True,
                label=f"nominal p(interaction)<0.01 (n={nom.sum()})")
    lim = np.nanpercentile(np.abs(np.concatenate(
        [inter.hyp_response_wt, inter.hyp_response_ko])), 99)
    axC.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls="--")
    axC.set_xlim(-lim, lim); axC.set_ylim(-lim, lim)
    axC.set_xlabel("Hypoxia response in WT (log2FC)")
    axC.set_ylabel("Hypoxia response in KO (log2FC)")
    axC.legend(frameon=False, fontsize=5.8, loc="lower right")
    axC.spines[["top", "right"]].set_visible(False)
    axC.set_title("RNA hypoxia response, WT vs KO\n"
                  f"(n={len(inter)} genes; 0 genes at interaction FDR<0.05;\n"
                  "unmoderated model, df=4 — low power)", fontsize=6.3)
    panel_label(axC, "C")

    # ---- D: condition effects by occupancy (effect size + bootstrap CI)
    ce = pd.read_csv(f"{TAB}/stats_fig3_condition_effects_by_occupancy.csv")
    axD = fig.add_subplot(gs[1, 1])
    metrics = [("d_ins_hyp_WT", "WT hypoxia\nresponse"),
               ("d_ins_hyp_KO", "KO hypoxia\nresponse"),
               ("d_ins_interaction", "interaction")]
    for j, (coln, lab) in enumerate(metrics):
        for k, (labset, colr) in enumerate([("NapM_bound", C["chip"]),
                                            ("matched_background", C["matched"])]):
            mask = bound if labset == "NapM_bound" else matched
            vals = b[coln].values[mask]
            vals = vals[np.isfinite(vals)]
            med = np.median(vals)
            bs = RNG.choice(vals, size=(2000, len(vals)), replace=True)
            lo, hi = np.percentile(np.median(bs, axis=1), [2.5, 97.5])
            x = j + (k - 0.5) * 0.22
            axD.errorbar(x, med, yerr=[[med - lo], [hi - med]], fmt="o",
                         color=colr, ms=4.5, capsize=2.5, lw=1,
                         label="NapM bound" if (j == 0 and k == 0) else
                         ("matched" if (j == 0 and k == 1) else None))
    axD.axhline(0, color="k", lw=0.5, ls="--")
    axD.set_xlim(-0.5, 2.5)
    axD.set_xticks(range(3))
    axD.set_xticklabels([m[1] for m in metrics], fontsize=5.6)
    axD.set_ylabel("Median Δ log2 insulation\n[95% bootstrap CI]")
    axD.legend(frameon=False, fontsize=5.8, loc="lower left")
    axD.spines[["top", "right"]].set_visible(False)
    axD.set_title("Condition effects at bound vs matched bins (n=104 each;\n"
                  "Welch estimates, unbalanced 4/4/4/3 replicates)",
                  fontsize=6.5)
    panel_label(axD, "D")

    # ---- E: occupancy vs condition-dependent change (permutation)
    ip = pd.read_csv(f"{TAB}/stats_fig3_interaction_permutation.csv").iloc[0]
    null_i = np.load(f"{TAB}/fig3_null_interaction.npy")
    axE = fig.add_subplot(gs[1, 2])
    axE.hist(null_i, bins=40, color="0.75", edgecolor="white", lw=0.3)
    axE.axvline(ip.obs, color=C["chip"], lw=1.4)
    axE.set_xlabel("median interaction(bound) − (matched)\nunder circular-shift null")
    axE.set_ylabel("Count (1000 shifts)")
    axE.spines[["top", "right"]].set_visible(False)
    axE.set_title(f"baseline occupancy vs condition-dependent\nstructural change: "
                  f"observed {ip.obs:+.3f}, p = {ip.p_circular_shift:.2f}\n"
                  "(replicate-based interaction, Welch)", fontsize=6.2)
    panel_label(axE, "E")

    # ---- F: latent (descriptive only)
    gsF = gridspec.GridSpecFromSubplotSpec(1, 2, gs[2, :], wspace=0.35)
    axF1 = fig.add_subplot(gsF[0])
    axF1.plot(mb, smooth(b.ins_latent - b.ins_WT), color=C["latent"], lw=0.9)
    axF1.axhline(0, color="k", lw=0.5)
    axF1.set_xlabel("Genomic position (Mb)")
    axF1.set_ylabel("Δ log2 insulation (latent − WT)")
    axF1.spines[["top", "right"]].set_visible(False)
    axF1.set_title("latent − WT insulation (merged matrices; median "
                   f"{float(pd.read_csv(f'{TAB}/stats_fig3_latent_descriptive.csv').median_d_ins_latent_WT.iloc[0]):+.3f})",
                   fontsize=6.5)
    axF2 = fig.add_subplot(gsF[1])
    both = np.isfinite(genes.log2FC_hypoxiavsWT) & np.isfinite(genes.log2FC_latentvsWT)
    axF2.scatter(genes.log2FC_hypoxiavsWT[both], genes.log2FC_latentvsWT[both],
                 s=2.5, color="0.75", lw=0, rasterized=True)
    r_lat = np.corrcoef(genes.log2FC_hypoxiavsWT[both],
                        genes.log2FC_latentvsWT[both])[0, 1]
    l2 = np.nanpercentile(np.abs(np.concatenate(
        [genes.log2FC_hypoxiavsWT[both], genes.log2FC_latentvsWT[both]])), 99)
    axF2.plot([-l2, l2], [-l2, l2], color="k", lw=0.6, ls="--")
    axF2.set_xlabel("Hypoxia vs WT (log2FC)")
    axF2.set_ylabel("Latent vs WT (log2FC)")
    axF2.spines[["top", "right"]].set_visible(False)
    axF2.set_title(f"latent vs hypoxia expression (r = {r_lat:.2f});\n"
                   "no KO-latent exists — no napM-dependence inference",
                   fontsize=6.3)
    for axF in (axF1, axF2):
        axF.text(0.02, 0.04, "DESCRIPTIVE ONLY", transform=axF.transAxes,
                 fontsize=6, color=C["latent"], fontweight="bold")
    panel_label(axF1, "F")

    fig.suptitle("Fig 3. Condition-dependent molecular effects associated "
                 "with napM deletion", fontsize=8.5, y=0.995)
    savefig(fig, "Fig3_condition_dependent")


if __name__ == "__main__":
    import os
    os.makedirs(FIG, exist_ok=True)
    fig1()
    fig2()
    fig3()
