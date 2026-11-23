#!/usr/bin/env python3
"""Fig3_corrected — incremental local RNA explanatory power is weak.

Panels A, B and D assemble pre-computed results. Panel C uses the recomputed
feature-pair effects from 05_fig3c_matched_effects.py: the legacy gene-level
spatial test indexed gene arrays with bin coordinates and is not carried
forward. Gene and operon matched effects therefore have ordinary matched-pair
bootstrap intervals and no feature-level spatial P values. These intervals
are descriptive, not spatial-autocorrelation-corrected tests.

Seed 42. RA/01-08 are read-only inputs. Outputs are versioned as
supplementary_analysis/figures/Fig3_corrected.{pdf,png}; source CSVs are in
tables/corrected_figure_sources/. Original preliminary figures are preserved.
Panel D reports RNA-undetermined and no-data bins explicitly rather than
silently treating them as unchanged.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

RA = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R1_mechanism_reanalysis"
sys.path.insert(0, f"{RA}/07_figures/scripts")
import fc  # shared style (Arial, fonttype 42, white bg, 180 mm width)

SEED = 42
RNG = np.random.default_rng(SEED)

OUT = f"{RA}/09_combined/R1_review_delivery_20261027/supplementary_analysis"
FIG = f"{OUT}/figures"
TAB = f"{OUT}/tables/corrected_figure_sources"
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

F5 = f"{RA}/08_extended/fig5_rna/tables"
F6 = f"{RA}/08_extended/fig6_operon/tables"

C_PURPLE = "#665A7A"
C_GREY = "#8C8C8C"
CMAP_CHIP = mcolors.LinearSegmentedColormap.from_list("chip", ["#F0EDF4", C_PURPLE])


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)


# ---------------- load pre-computed and recomputed sources ----------------
srcA = pd.read_csv(f"{F5}/fig5A_source.csv")          # bin, rna, d_ins, chip
lowess = pd.read_csv(f"{F5}/fig5F_lowess.csv")        # LOWESS curve
qfits = pd.read_csv(f"{F5}/fig5F_quantile_fits.csv")  # quantile-regression fits
srcD = pd.read_csv(f"{F5}/fig5D_source.csv")          # regression coefficients
srcE = pd.read_csv(f"{OUT}/tables/nested_cv_complete_case_models.csv")
cmpE = pd.read_csv(f"{OUT}/tables/nested_cv_complete_case_comparison.csv")
srcC2 = pd.read_csv(f"{F5}/fig5C_source.csv")         # 2x2 + unresolved counts
st5 = pd.read_csv(f"{F5}/fig5_stats_summary.csv")     # non-panel-C statistics
matchedC = pd.read_csv(f"{OUT}/tables/fig3c_matched_effects_recomputed.csv")
ecdf_raw = pd.read_csv(f"{F5}/fig5B_source.csv", header=[0, 1])


def matched_row(level, metric):
    hit = matchedC[(matchedC.level == level) & (matchedC.metric == metric)]
    assert len(hit) == 1, (level, metric)
    return hit.iloc[0]


spA = st5[st5.test == "Fig5A_spearman_rna_vs_dins"].iloc[0]
g_fc = matched_row("gene", "|RNA log2FC|")
g_deg = matched_row("gene", "P(DEG)")
op_fc = matched_row("operon", "|RNA log2FC|")
op_deg = matched_row("operon", "DEG fraction")
fish = st5[st5.test == "Fig5C_fisher_2x2"].iloc[0]

cnt2x2 = srcC2.set_index("category")["n"]

# ---------------- per-panel source tables ----------------
srcA.to_csv(f"{TAB}/fig3_final_panelA_source.csv", index=False)
srcD.to_csv(f"{TAB}/fig3_final_panelB_regression.csv", index=False)
srcE.to_csv(f"{TAB}/fig3_final_panelB_nested_cv.csv", index=False)
cmpE.to_csv(f"{TAB}/fig3_final_panelB_nested_cv_comparison.csv", index=False)
gene_ecdf = pd.DataFrame(dict(
    bound_abslog2FC=ecdf_raw[("|RNA log2FC| (gene)", "bound")],
    matched_abslog2FC=ecdf_raw[("|RNA log2FC| (gene)", "matched")]))
gene_ecdf.to_csv(f"{TAB}/fig3_final_panelC_gene_ecdf.csv", index=False)
matchedC.to_csv(f"{TAB}/fig3_final_panelC_matched_summary.csv", index=False)

# Reconstruct the unresolved classification composition from the bin table.
bin_level = pd.read_csv(f"{RA}/03_structure_metrics/bin_level_v2.csv")
sd = bin_level.structure_detected.astype(bool)
rc = bin_level.rna_bin_class
expected = {
    "struct+RNA+": int((sd & (rc == "detected_change")).sum()),
    "struct+RNA-": int((sd & (rc == "supports_small_change")).sum()),
    "struct-RNA+": int((~sd & (rc == "detected_change")).sum()),
    "struct-RNA-": int((~sd & (rc == "supports_small_change")).sum()),
    "struct+undet": int((sd & rc.isin(["cannot_determine", "no_data"])).sum()),
    "struct-undet": int((~sd & rc.isin(["cannot_determine", "no_data"])).sum()),
}
assert all(int(cnt2x2[k]) == v for k, v in expected.items())
composition = {k: "" for k in expected}
for prefix, mask in (("struct+", sd), ("struct-", ~sd)):
    n_cannot = int((mask & (rc == "cannot_determine")).sum())
    n_no_data = int((mask & (rc == "no_data")).sum())
    composition[f"{prefix}undet"] = (
        f"{n_cannot} cannot_determine + {n_no_data} no_data"
    )
srcC2.assign(
    note=("Fisher OR/P use the RNA-determined 2x2 only; the third category "
          "combines RNA-undetermined and no-data bins and is not unchanged"),
    composition=srcC2.category.map(composition),
).to_csv(f"{TAB}/fig3_final_panelD_classification.csv", index=False)

# ---------------- figure ----------------
fig = plt.figure(figsize=(fc.FIGW, 112 * fc.MM))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.93, bottom=0.115,
                      wspace=0.55, hspace=0.60,
                      height_ratios=[1.0, 1.0])

# ---- Panel a: bin-level scatter + LOWESS + quantile-regression band ----
axA = fig.add_subplot(gs[0, 0])
x = srcA["rna_mean_log2FC"].values
y = srcA["d_ins_KO_WT"].values
ch = srcA["chip_bin_score_max"].values
sc = axA.scatter(x, y, c=ch, cmap=CMAP_CHIP, s=3.5, lw=0, alpha=0.85,
                 vmin=0, vmax=1, zorder=2)
xs = np.array([x.min(), x.max()])
q = qfits.set_index("quantile")
axA.fill_between(xs, q.loc[0.1, "intercept"] + q.loc[0.1, "slope"] * xs,
                 q.loc[0.9, "intercept"] + q.loc[0.9, "slope"] * xs,
                 color=C_GREY, alpha=0.18, lw=0, zorder=1)
axA.plot(xs, q.loc[0.5, "intercept"] + q.loc[0.5, "slope"] * xs,
         color=C_GREY, ls=":", lw=0.9, zorder=3, label="median QR")
axA.plot(lowess["grid_x"], lowess["lowess_y"], color=fc.C["neg"], lw=1.4,
         zorder=4, label="LOWESS (f=0.35)")
axA.axhline(0, color="#DDDDDD", lw=0.5, zorder=1)
cb = fig.colorbar(sc, ax=axA, pad=0.02, fraction=0.07)
cb.set_label("NapM ChIP score (max)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)
axA.set_xlabel("Bin mean RNA log2FC (KO / WT)")
axA.set_ylabel("Δ insulation (KO − WT)")
axA.legend(fontsize=6, frameon=False, loc="upper right", borderaxespad=0.1)
axA.text(0.03, 0.03,
         f"Spearman ρ = {spA['observed']:.3f}, p = {spA['p']:.2f}\n"
         f"n = {int(spA['n'])} bins\n"
         "LOWESS/QR: descriptive",
         transform=axA.transAxes, fontsize=6.1, va="bottom")
fc.panel_label(axA, "A", x=-0.22)

# ---- Panel b left: multiple-regression coefficients ----
axB1 = fig.add_subplot(gs[0, 1])
preds = srcD.iloc[1:].reset_index(drop=True)   # drop intercept
labels = ["RNA log2FC", "expr WT", "ins WT", "weight WT", "n genes"]
ypos = np.arange(len(preds))[::-1]
for i, r in preds.iterrows():
    col = C_PURPLE if r["predictor"] == "rna_mean_log2FC_KOvsWT" else C_GREY
    axB1.errorbar(r["beta"], ypos[i],
                  xerr=[[r["beta"] - r["ci_lo"]], [r["ci_hi"] - r["beta"]]],
                  fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.0,
                  capsize=2)
rna_row = preds[preds.predictor == "rna_mean_log2FC_KOvsWT"].iloc[0]
pr2 = st5[st5.test == "Fig5D_partialR2_rna"].iloc[0]
axB1.axvline(0, color="#BBBBBB", lw=0.7, ls=":")
axB1.set_yticks(ypos)
axB1.set_yticklabels(labels, fontsize=7)
axB1.set_xlabel("Standardized coefficient (95% CI)")
axB1.set_title("multiple regression on Δins (KO−WT)", fontsize=6.8)
axB1.text(0.03, 0.97,
          f"RNA coef = {rna_row['beta']:.4f}\n"
          f"[{rna_row['ci_lo']:.4f}, {rna_row['ci_hi']:.4f}]\n"
          f"partial R²(RNA) = {pr2['observed']:.4f}\n"
          f"F p = {pr2['p']:.2f}, n = {int(pr2['n'])}",
          transform=axB1.transAxes, fontsize=6.3, ha="left", va="top")
fc.panel_label(axB1, "B", x=-0.30)

# ---- Panel b right: nested-model block-CV R2 ----
axB2 = fig.add_subplot(gs[0, 2])
r2s = srcE["cv_R2"].values
mlabels = ["M0\nbaseline", "M1\n+RNA", "M2\n+occupancy"]
mcols = [C_GREY, "#2166AC", C_PURPLE]
axB2.plot(range(3), r2s, "-", color="#CCCCCC", lw=0.8, zorder=1)
for i, (r2v, col) in enumerate(zip(r2s, mcols)):
    axB2.plot(i, r2v, "o", ms=6, color=col, zorder=2)
    axB2.annotate(f"{r2v:.4f}", (i, r2v), textcoords="offset points",
                  xytext=(0, 7), ha="center", fontsize=6.5)
axB2.set_xticks(range(3))
axB2.set_xticklabels(mlabels, fontsize=7)
axB2.set_xlim(-0.5, 2.5)
pad = 0.0012
axB2.set_ylim(r2s.min() - pad, r2s.max() + pad * 2.6)
axB2.set_ylabel("10-block CV R²")
dR10, lo10, hi10 = cmpE.loc[0, ["dR2", "ci_lo", "ci_hi"]]
dR21, lo21, hi21 = cmpE.loc[1, ["dR2", "ci_lo", "ci_hi"]]
axB2.set_title("complete-case nested models (10-block CV)", fontsize=6.8)
axB2.text(0.03, 0.03,
          f"ΔR²(RNA) = {dR10:+.4f} [{lo10:+.4f},{hi10:+.4f}]\n"
          f"ΔR²(occupancy) = {dR21:+.4f}\n"
          f"[{lo21:+.4f},{hi21:+.4f}]; n = {int(cmpE.n_bins.iloc[0])}",
          transform=axB2.transAxes, fontsize=5.8, va="bottom")

# ---- Panel c: matched controls at gene and operon level ----
gsC = gs[1, 0:2].subgridspec(1, 2, wspace=0.62, width_ratios=[1.0, 1.15])
axC1 = fig.add_subplot(gsC[0, 0])
vb = gene_ecdf["bound_abslog2FC"].dropna().values
vm = gene_ecdf["matched_abslog2FC"].dropna().values
for v, col, lab in ((vb, C_PURPLE, "NapM-bound genes"),
                    (vm, "#9A9D9F", "matched control genes")):
    xs_e = np.sort(v)
    ys_e = np.arange(1, len(xs_e) + 1) / len(xs_e)
    axC1.plot(xs_e, ys_e, color=col, lw=1.4, label=lab)
axC1.set_xlabel("|RNA log2FC (KO / WT)| per gene (truncated at 3)")
axC1.set_ylabel("ECDF")
axC1.set_xlim(0, 3)
axC1.legend(fontsize=6.3, loc="upper left", framealpha=0.9, edgecolor="none")
axC1.text(0.30, 0.60,
          f"gene: {int(g_fc['n_pairs'])} matched pairs\n"
          f"Δ|log2FC| = {g_fc['observed']:+.3f}\n"
          f"[{g_fc['ci_lo']:+.3f}, {g_fc['ci_hi']:+.3f}]\n"
          f"ΔP(DEG) = {g_deg['observed']:+.3f}\n"
          f"[{g_deg['ci_lo']:+.3f}, {g_deg['ci_hi']:+.3f}]\n"
          "paired-bootstrap 95% intervals\n"
          "(no spatial P reported)",
          transform=axC1.transAxes, fontsize=5.5, va="top", linespacing=1.22)
fc.panel_label(axC1, "C", x=-0.24)

axC2 = fig.add_subplot(gsC[0, 1])
rows = [
    ("gene |log2FC|", g_fc, C_PURPLE),
    ("gene P(DEG)", g_deg, C_PURPLE),
    ("operon |log2FC|", op_fc, "#2166AC"),
    ("operon DEG frac", op_deg, "#2166AC"),
]
yp = np.arange(len(rows))[::-1].astype(float) + 0.3
for (lab, r, col), yy in zip(rows, yp):
    axC2.errorbar(r["observed"], yy,
                  xerr=[[r["observed"] - r["ci_lo"]], [r["ci_hi"] - r["observed"]]],
                  fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.0,
                  capsize=2)
axC2.axvline(0, color="#BBBBBB", lw=0.7, ls=":")
axC2.set_yticks(yp)
axC2.set_yticklabels([r[0] for r in rows], fontsize=6.8)
axC2.set_ylim(-0.6, 3.9)
axC2.set_xlim(-0.16, 0.16)
axC2.set_xlabel("paired Δ (bound − matched), 95% bootstrap interval")
axC2.set_title("matched effects (ordinary paired bootstrap)", fontsize=6.8)
axC2.text(-0.15, -0.45,
          f"gene: {int(g_fc['n_pairs'])} pairs; operon: {int(op_fc['n_pairs'])} pairs",
          fontsize=6, color="#555555", va="center")

# ---- Panel d: compact structure x RNA classification, unresolved retained ----
axD = fig.add_subplot(gs[1, 2])
cats = ["RNA\ndetected", "RNA not\ndetected", "RNA undetermined/\nno data"]
s_pos = [cnt2x2["struct+RNA+"], cnt2x2["struct+RNA-"], cnt2x2["struct+undet"]]
s_neg = [cnt2x2["struct-RNA+"], cnt2x2["struct-RNA-"], cnt2x2["struct-undet"]]
xs2 = np.arange(3)
w = 0.55
axD.bar(xs2, s_pos, w, color=fc.C["KO"], lw=0, label="structure detected")
axD.bar(xs2, s_neg, w, bottom=s_pos, color=fc.C["ns"], lw=0,
        label="structure not detected")
for xi, (a, bb) in enumerate(zip(s_pos, s_neg)):
    if a < 15:
        axD.text(xi - 0.11, a + bb + 22, str(int(a)), ha="center",
                 va="center", fontsize=6.2, color=fc.C["KO"])
    else:
        axD.text(xi, a / 2, str(int(a)), ha="center", va="center",
                 fontsize=6.5, color="white")
    if bb < 15:
        axD.text(xi + 0.11, a + bb + 22, str(int(bb)), ha="center",
                 va="center", fontsize=6.2, color="#555555")
    else:
        axD.text(xi, a + bb / 2, str(int(bb)), ha="center", va="center",
                 fontsize=6.5, color="#555555")
axD.set_xticks(xs2)
axD.set_xticklabels(cats, fontsize=6.2)
axD.set_ylabel("bins")
axD.set_ylim(0, 560)
axD.legend(fontsize=5.8, frameon=False, loc="upper left", borderaxespad=0.1,
           handlelength=1.2, labelspacing=0.3)
axD.set_title("structure × RNA joint classification\n(unresolved states retained)",
              fontsize=6.8)
axD.text(0.97, 0.82,
         f"OR = {fish['observed']:.2f}, Fisher p = {fish['p']:.2f}\n"
         "(RNA-determined 2×2 only)\n"
         "unresolved ≠ unchanged",
         transform=axD.transAxes, fontsize=5.3, va="top", ha="right",
         linespacing=1.35)
fc.panel_label(axD, "D", x=-0.26)

save(fig, "Fig3_corrected")
print("Fig3_corrected done")
