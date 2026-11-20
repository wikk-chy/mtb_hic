#!/usr/bin/env python3
"""Fig1_corrected — local NapM occupancy and chromosome architecture.

Five-panel layout:
  A  Genome-wide NapM peak scores, 41 loop arcs (17 both-anchor-bound),
     Delta insulation (KO-WT), and RNA log2 fold change.
  B  Peak-centre metaprofiles (433 unique centre bins) for absolute and
     signed Delta insulation and local-contact change, with 10,000 circular
     shifts and a 95% null band.
  C  Matched association estimates, structure-overlap odds ratio, CID-boundary
     distance analysis, and an 18-variant robustness summary (16/18 intervals
     cross zero). Spatially aware shift P values are distinguished from naive
     paired tests.
  D  Both-anchor positional association (17/41 versus null mean 10.9,
     P = 0.0328) plus stratified sensitivity analyses (P = 0.022-0.032).
     These results concern occupancy at called loop anchors, not physical
     bridging or loop anchoring by NapM.
  E  WT versus KO loop O/E for 39 ICE-covered loops; the between-group
     difference in Delta log2 O/E is +0.059 [-0.203, +0.335], label
     permutation P = 0.685. L35/L36 are excluded for missing ICE coverage.

This script assembles pre-computed results; it does not rerun the underlying
statistics. Seed 42. Original preliminary figures are not overwritten.
Outputs: supplementary_analysis/figures/Fig1_corrected.{pdf,png} and
per-panel source CSVs in tables/corrected_figure_sources/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
                "hic_review/review_results/R1_mechanism_reanalysis/"
                "09_combined/scripts")
import fc9  # noqa: E402
import fc   # noqa: E402  (via fc9 sys.path)

SEED = 42
rng = np.random.default_rng(SEED)

d = fc.load_tables()
b, pk, loops = d["b"], d["peaks"], d["loops"]
stats13 = d["stats13"]
x_mb = (b["bin"] * fc.BIN + fc.BIN / 2) / 1e6

EXT1 = f"{fc.RA}/08_extended/fig1_revise/tables"
EXT2 = f"{fc.RA}/08_extended/fig2_loops/tables"
SUPP = f"{fc.RA}/09_combined/R1_review_delivery_20261027/supplementary_analysis"
TAB9 = f"{SUPP}/tables/corrected_figure_sources"
fc9.FIG = f"{SUPP}/figures"
os.makedirs(TAB9, exist_ok=True)
os.makedirs(fc9.FIG, exist_ok=True)
GREY = "#6E7072"


def src(name, df):
    df.to_csv(f"{TAB9}/Fig1_final_{name}.csv", index=False)
    print("wrote", f"{TAB9}/Fig1_final_{name}.csv")


fig = plt.figure(figsize=(fc.FIGW, 215 * fc.MM))
gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 1.42, 1.0],
                      width_ratios=[1.5, 1.0], hspace=0.62, wspace=0.40,
                      left=0.085, right=0.99, top=0.97, bottom=0.055)

# ---------------- A: genome-wide overview tracks ----------------
gsA = gs[0, :].subgridspec(4, 1, hspace=0.30, height_ratios=[0.8, 1.15, 1, 1])
axP = fig.add_subplot(gsA[0])   # NapM peaks
axL = fig.add_subplot(gsA[1], sharex=axP)   # loop arcs
axI = fig.add_subplot(gsA[2], sharex=axP)   # d insulation
axR = fig.add_subplot(gsA[3], sharex=axP)   # RNA

axP.vlines(pk["center"] / 1e6, 0, pk["score"], color=fc.C["chip"], lw=0.4,
           alpha=0.8)
axP.set_ylim(0, 1.05); axP.set_yticks([0, 0.5, 1.0])
axP.tick_params(axis="y", labelsize=6)
axP.set_ylabel("NapM ChIP\npeak score\n(634 peaks)", fontsize=6.5)

# loop arcs: half-ellipse between anchors, height ~ sqrt(genomic span);
# loops with both anchors NapM-bound (17, primary definition) in purple
span = (loops["bin2"] - loops["bin1"]).values.astype(float)
arc_h = np.sqrt(span / span.max())
both = loops["both_bound"].values.astype(bool)
tt = np.linspace(0, np.pi, 120)
for i in range(len(loops)):
    c1, c2 = loops["center1"].iat[i] / 1e6, loops["center2"].iat[i] / 1e6
    x_arc = c1 + (c2 - c1) * (1 - np.cos(tt)) / 2
    y_arc = arc_h[i] * np.sin(tt)
    col = fc.C["chip"] if both[i] else "#8C8C8C"
    axL.plot(x_arc, y_arc, color=col, lw=0.9 if both[i] else 0.6,
             alpha=0.95 if both[i] else 0.8, zorder=2 if both[i] else 1)
axL.set_ylim(-0.12, 1.12); axL.set_yticks([])
axL.set_ylabel("loops (n=41)\narc height ~\n\u221aspan", fontsize=6.5)
axL.plot([], [], color=fc.C["chip"], lw=0.9,
         label="both anchors NapM-bound (n=17)")
axL.plot([], [], color="#8C8C8C", lw=0.6, label="other loops (n=24)")
axL.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.4,
           bbox_to_anchor=(0.02, 1.0), borderaxespad=0.1, labelspacing=0.25)

di = b["d_ins_KO_WT"].values
axI.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
axI.bar(x_mb, di, width=0.005, color=np.where(di >= 0, fc.C["pos"], fc.C["neg"]),
        alpha=0.85, lw=0)
axI.set_ylabel("\u0394 insulation\n(KO \u2212 WT)", fontsize=6.5)

lr = b["rna_mean_log2FC_KOvsWT"].values
cols = np.where(np.isnan(lr), fc.C["ns"], np.where(lr >= 0, fc.C["pos"], fc.C["neg"]))
axR.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
axR.bar(x_mb, np.nan_to_num(lr), width=0.005, color=cols, alpha=0.85, lw=0)
axR.set_ylabel("RNA log2FC\n(KO/WT)", fontsize=6.5)
axR.set_xlabel("genome position (Mb)")

for ax in (axP, axL, axI, axR):
    ax.set_xlim(0, fc.GENOME / 1e6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axP.set_xticklabels([]); axL.set_xticklabels([]); axI.set_xticklabels([])
fc9.panel_label(axP, "A", x=-0.095)

src("A_peaks", pk[["chrom", "start", "end", "center", "target_gene",
                   "score", "vpm"]])
src("A_loop_anchors", loops[["loop_id", "center1", "center2", "bin1", "bin2",
                             "anchor1_bound", "anchor2_bound", "both_bound"]])
src("A_dins_perbin", b[["bin", "start", "end", "d_ins_KO_WT"]])
src("A_rna_perbin", b[["bin", "start", "end", "rna_mean_log2FC_KOvsWT"]])

# ---------------- B: peak-centred metaprofiles, 2x2 ----------------
metrics = [
    ("absdins", "mean |\u0394 insulation| (KO \u2212 WT)"),
    ("signed_dins", "mean \u0394 insulation (KO \u2212 WT)"),
    ("signed_dlocalcont", "mean \u0394 local contact (KO \u2212 WT)"),
    ("abs_dlocalcont", "mean |\u0394 local contact| (KO \u2212 WT)"),
]
wstats = pd.read_csv(f"{EXT1}/fig1b_extended_stats.csv").set_index("metric")
n_peaks = int(wstats["n_peaks"].iloc[0])

gsB = gs[1, 0].subgridspec(2, 2, hspace=0.66, wspace=0.34)
for k, (key, ylab) in enumerate(metrics):
    ax = fig.add_subplot(gsB[k // 2, k % 2])
    mp = pd.read_csv(f"{EXT1}/fig1b_extended_metaprofile_{key}.csv")
    ax.fill_between(mp["offset_kb"], mp["null_lo95"], mp["null_hi95"],
                    color=fc.C["bg"], lw=0,
                    label="circular-shift null 95% (10,000)")
    ax.plot(mp["offset_kb"], mp["null_mean"], color="#8a8a8a", lw=0.8,
            ls="--", label="null mean")
    ax.plot(mp["offset_kb"], mp["observed"], color=fc.C["chip"], lw=1.3,
            label="observed (peak-centred)")
    ax.axvline(0, color=fc.C["chip"], lw=0.5, ls=":")
    if "signed" in key:
        ax.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
    r = wstats.loc[key]
    p = r["p"]
    ptxt = "p = 1.0" if p >= 0.99995 else (f"p = {p:.2f}" if p >= 0.01
                                           else f"p = {p:.4f}")
    ax.text(0.03, 0.04,
            f"\u00b15 kb: obs {r['observed']:.4f} vs null {r['null_mean']:.4f}\n"
            f"circular-shift {ptxt} (n={n_peaks} peak-centre bins)",
            transform=ax.transAxes, fontsize=5.8, va="bottom")
    ax.set_xlabel("offset from NapM peak centre (kb)", fontsize=6.5)
    ax.set_ylabel(ylab, fontsize=6.5)
    ax.set_xlim(-100, 100)
    ax.tick_params(labelsize=6)
    if k == 0:
        ax.legend(frameon=False, loc="upper right", handlelength=1.4,
                  fontsize=5.6)
        fc9.panel_label(ax, "B", x=-0.30)
    src(f"B_metaprofile_{key}", mp)
src("B_window_stats", wstats.reset_index())

# ---------------- C: forest (left) + robustness strip (right) ----------------
gsC = gs[1, 1].subgridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.42)

rows = []
rows.append(dict(key="a", label="a  |\u0394ins| paired\n"
                 "\u22120.009 [\u22120.029, +0.010], p = 0.53",
                 eff=fc.stat(stats13, "Fig1Da_paired_absdins"), kind="ci",
                 ref=0.0))
rs = fc.stat(stats13, "Fig1Db_paired_dlocalcont_signed")
rows.append(dict(key="b1", label="b1  \u0394 local contact (signed)\n"
                 "\u22120.020 [\u22120.040, \u22120.000], p = 0.32*",
                 eff=rs, kind="ci", ref=0.0))
rows.append(dict(key="b2", label="b2  |\u0394 local contact|\n"
                 "\u22120.004 [\u22120.019, +0.011], p = 0.73",
                 eff=fc.stat(stats13, "Fig1Db_paired_abslocalcontmag"),
                 kind="ci", ref=0.0))
rd = fc.stat(stats13, "Fig1Dd_bound_x_structure_detected")
rows.append(dict(key="d", label="d  bound \u00d7 structure OR\n"
                 "0.84 [0.63, 1.12], Fisher p = 0.27",
                 eff=rd, kind="or", ref=1.0))
cidt = pd.read_csv(f"{fc.RA}/06_permutation_tests/fig1c_cid_distance_test.csv")
cm = cidt[cidt["statistic"] == "median"].iloc[0]
cmean = cidt[cidt["statistic"] == "mean"].iloc[0]
rows.append(dict(key="e", label="e  CID boundary dist. (median)\n"
                 "13 vs null 12.4, p = 0.42",
                 eff=dict(observed=cm["obs"], null_mean=cm["null_med"],
                          effect=cm["obs"] - cm["null_med"],
                          ci_lo=np.nan, ci_hi=np.nan, p=cm["p_farther"],
                          test="one-sided farther, median statistic",
                          null_model="circular-shift of bound mask (10,000)"),
                 kind="nullband", band=(cm["lo"], cm["hi"])))

gsCf = gsC[0].subgridspec(len(rows), 1, hspace=2.1)
forest_rows = []
for k, row in enumerate(rows):
    ax = fig.add_subplot(gsCf[k])
    r = row["eff"]
    if row["kind"] == "ci":
        ax.axvline(row["ref"], color=fc.C["ns"], lw=0.6, zorder=0)
        ax.errorbar([r["effect"]], [0], xerr=[[r["effect"] - r["ci_lo"]],
                    [r["ci_hi"] - r["effect"]]], fmt="o", ms=3.5, color=GREY,
                    ecolor=GREY, elinewidth=1.0, capsize=2)
        pad = 0.45 * (r["ci_hi"] - r["ci_lo"])
        ax.set_xlim(r["ci_lo"] - pad, r["ci_hi"] + pad)
    elif row["kind"] == "or":
        ax.axvline(1.0, color=fc.C["ns"], lw=0.6, zorder=0)
        ax.errorbar([r["effect"]], [0], xerr=[[r["effect"] - r["ci_lo"]],
                    [r["ci_hi"] - r["effect"]]], fmt="o", ms=3.5, color=GREY,
                    ecolor=GREY, elinewidth=1.0, capsize=2)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 2.0)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.minorticks_off()
    else:  # nullband
        lo, hi = row["band"]
        ax.axvspan(lo, hi, color=fc.C["bg"], lw=0, zorder=0)
        ax.axvline(r["null_mean"], color="#8a8a8a", lw=0.7, ls="--", zorder=1)
        ax.plot([r["observed"]], [0], "o", ms=4, color=GREY, zorder=2)
        sp = max(hi - lo, abs(r["observed"] - lo), abs(r["observed"] - hi))
        ax.set_xlim(min(lo, r["observed"]) - 0.35 * sp,
                    max(hi, r["observed"]) + 0.35 * sp)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, 0.7)
    ax.text(0.0, 1.15, row["label"], transform=ax.transAxes,
            fontsize=5.6, ha="left", va="bottom", clip_on=False)
    ax.tick_params(axis="x", labelsize=5.5, pad=1)
    forest_rows.append(dict(panel_row=row["key"], quantity=row["label"],
                            observed=r["observed"], null_mean=r["null_mean"],
                            effect=r["effect"], ci_lo=r["ci_lo"],
                            ci_hi=r["ci_hi"], p=r["p"], test=r["test"],
                            null_model=r["null_model"]))
fc9.panel_label(fig.axes[-len(rows)], "C", x=-0.28, y=1.35)
fig.axes[-1].text(
    0.0, -1.55,
    "* naive paired Wilcoxon p = 0.047;\n"
    "displayed p from circular-shift (10,000)\n"
    "unless stated. Row e: median statistic;\n"
    f"mean-based statistic marginal (farther\n"
    f"p = {cmean['p_farther']:.3f}; trim-robust but\nnot median/quartile-supported.",
    transform=fig.axes[-1].transAxes, fontsize=5.4, va="top")
src("C_forest", pd.DataFrame(forest_rows))
src("C_cid_distance_test", cidt)

# C-right: compact robustness strip (18 variants; statistics from
# 08_extended/fig1_revise/tables/fig1e_robustness_stats.csv, not re-run)
rob = pd.read_csv(f"{EXT1}/fig1e_robustness_stats.csv")
axS = fig.add_subplot(gsC[1])
yy = np.arange(len(rob))[::-1]
for y, (_, r) in zip(yy, rob.iterrows()):
    axS.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=GREY, lw=0.55, alpha=0.75)
    axS.plot(r["effect"], y, "o", ms=1.8, color=GREY)
axS.axvline(0, color=fc.C["ns"], lw=0.7, ls="--", zorder=0)
axS.set_yticks([])
axS.spines["left"].set_visible(False)
axS.set_xlabel("mean paired |\u0394ins|\n(bound \u2212 matched)",
               fontsize=6.0)
axS.tick_params(axis="x", labelsize=5.5, pad=1)
axS.set_title("robustness: 18 variants\n(3 wins \u00d7 3 ChIP \u00d7 2 trims)",
              fontsize=5.8, pad=3)
n_cross = int(((rob["ci_lo"] <= 0) & (rob["ci_hi"] >= 0)).sum())
rawmin = rob.loc[rob["wilcoxon_p"].idxmin()]
axS.text(0.03, 0.03,
         f"effects {rob['effect'].min():+.3f} to\n"
         f"{rob['effect'].max():+.3f}; {n_cross}/18 CIs\n"
         "cross 0; both exceptions\n"
         f"negative (min raw p =\n{rawmin['wilcoxon_p']:.3f}, Bonf. ns)",
         transform=axS.transAxes, fontsize=5.2, va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
src("C_robustness_18variants", rob)

# ---------------- D: loop positional association ----------------
gsD = gs[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.62)

# D-left: both-anchor enrichment vs circular-shift null
null_both = np.load(f"{fc.RA}/06_permutation_tests/"
                    "null_loop_bridging_interval_both.npy")
brid = pd.read_csv(f"{EXT2}/panelA_bridging_primary.csv")
rb = brid[(brid["definition"] == "interval") & (brid["metric"] == "both")].iloc[0]
axD1 = fig.add_subplot(gsD[0])
axD1.hist(null_both, bins=np.arange(-0.5, null_both.max() + 1.5),
          color=fc.C["bg"], edgecolor="white", lw=0.3)
axD1.axvline(rb["observed"], color=fc.C["chip"], lw=1.3)
axD1.text(0.97, 0.97,
          f"both anchors bound:\n{int(rb['observed'])}/41 loops vs null "
          f"{rb['null_mean']:.1f}\nfold = {rb['fold_enrichment']:.2f}, "
          f"p = {rb['p_empirical']:.3f}",
          transform=axD1.transAxes, ha="right", va="top", fontsize=5.6,
          color=fc.C["chip"], fontweight="bold",
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
axD1.set_xlabel("count under circular-shift null (10,000)", fontsize=6.5)
axD1.set_ylabel("frequency", fontsize=6.5)
axD1.tick_params(labelsize=6)
fc9.panel_label(axD1, "D", x=-0.28)
src("D_loop_bridging_primary", brid)
src("D_null_both_anchor_counts",
    pd.DataFrame({"null_count": null_both}))

# D-right: confounder-control mini-forest + sensitivity note
strat = pd.read_csv(f"{EXT2}/panelC_stratified_shift.csv")
sens = pd.read_csv(f"{EXT2}/panelB_sensitivity_grid.csv")
sens_both = sens[sens["metric"] == "both"]
shortC = {"unstratified (reference)": "unstratified (ref.)",
          "loop span (distance terciles)": "loop span",
          "anchor gene density (terciles)": "gene density",
          "anchor baseline expression (terciles)": "baseline expr.",
          "flanking ChIP density \u00b11\u20132 bins (terciles)":
          "flanking ChIP"}
axD2 = fig.add_subplot(gsD[1])
sv = strat.iloc[::-1].reset_index(drop=True)
yp = np.arange(len(sv))
for y, (_, r) in zip(yp, sv.iterrows()):
    axD2.plot(r["fold"], y, "o", ms=4, color=fc.C["chip"], zorder=3)
    axD2.text(0.985, y, f"fold {r['fold']:.2f}, p = {r['p_empirical']:.3f}",
              transform=axD2.get_yaxis_transform(), va="center", ha="right",
              fontsize=5.6, color=fc.C["chip"])
axD2.axvline(1.0, color="0.6", lw=0.6, ls="--")
axD2.set_yticks(yp)
axD2.set_yticklabels([shortC.get(c, c) for c in sv["control"]], fontsize=5.8)
axD2.set_xlabel("fold enrichment, both anchors bound", fontsize=6.5)
axD2.set_xlim(0.6, 3.2)
axD2.tick_params(axis="x", labelsize=6)
axD2.set_title("stratified circular-shift controls", fontsize=6.0, pad=3)
axD2.text(0.0, -0.34,
          "sensitivity (2 binding definitions \u00d7 2 anchor windows): "
          f"all p < 0.05 (p = {sens_both['p_empirical'].min():.3f}\u2013"
          f"{sens_both['p_empirical'].max():.3f})",
          transform=axD2.transAxes, fontsize=5.4, va="top")
src("D_stratified_controls", strat)
src("D_sensitivity_grid", sens_both)

# ---------------- E: loop functional test ----------------
loe = pd.read_csv(f"{fc.RA}/06_permutation_tests/loop_oe_wt_ko_check.csv")
loe = loe.dropna(subset=["d_log2OE"])
gb = loe.loc[loe["both_bound"], "d_log2OE"].values
ub = loe.loc[~loe["both_bound"], "d_log2OE"].values
# statistics from pre-computed table (08 fig2_loops panel E; seed-42
# permutation/bootstrap of the original run), not re-run here
estat = pd.read_csv(f"{EXT2}/panelE_bound_vs_other_dlog2oe.csv").iloc[0]
obsD, ciD = estat["mean_diff"], (estat["CI95_lo"], estat["CI95_hi"])
pD = estat["p_label_permutation"]

gsE = gs[2, 1].subgridspec(1, 2, wspace=0.42)
# E-left: WT vs KO loop O/E scatter
axE1 = fig.add_subplot(gsE[0])
wt_l = np.log2(loe["WT_mean"].values)
ko_l = np.log2(loe["KO_mean"].values)
mD = loe["both_bound"].values
lims = [min(wt_l.min(), ko_l.min()) - 0.1, max(wt_l.max(), ko_l.max()) + 0.1]
axE1.plot(lims, lims, color="0.5", lw=0.6, ls="--", zorder=1)
axE1.scatter(wt_l[~mD], ko_l[~mD], s=13, color="#8C8C8C", alpha=0.85, lw=0,
             label=f"other loops (n={int((~mD).sum())})", zorder=2)
axE1.scatter(wt_l[mD], ko_l[mD], s=13, color=fc.C["chip"], alpha=0.9, lw=0,
             label=f"both anchors NapM-bound (n={int(mD.sum())})", zorder=3)
axE1.set_xlim(lims); axE1.set_ylim(lims)
axE1.set_xlabel("WT loop O/E (log2)", fontsize=6.5)
axE1.set_ylabel("napM-KO loop O/E (log2)", fontsize=6.5)
axE1.tick_params(labelsize=5.5)
leg = axE1.legend(fontsize=5.4, loc="lower right", handletextpad=0.1,
                  borderpad=0.3, labelspacing=0.25, framealpha=0.9,
                  edgecolor="none")
fc9.panel_label(axE1, "E", x=-0.28)
# E-right: effect-size strip
axE2 = fig.add_subplot(gsE[1])
axE2.axhline(0, color="0.5", lw=0.5, ls="--")
j1 = rng.uniform(-0.14, 0.14, len(gb))
j2 = rng.uniform(-0.14, 0.14, len(ub))
axE2.scatter(0 + j1, gb, s=12, color=fc.C["chip"], alpha=0.85, lw=0, zorder=3)
axE2.scatter(1 + j2, ub, s=12, color="#8C8C8C", alpha=0.85, lw=0, zorder=3)
for x0, v in [(0, gb), (1, ub)]:
    axE2.plot([x0 - 0.26, x0 + 0.26], [np.median(v)] * 2, color="k", lw=1.1,
              zorder=4)
axE2.set_xlim(-0.55, 1.55)
axE2.set_ylim(-1.35, 1.2)
axE2.set_yticks([-1, -0.5, 0, 0.5])
axE2.set_xticks([0, 1])
axE2.set_xticklabels(["bound", "other"], fontsize=6)
axE2.set_ylabel("\u0394 log2 loop O/E\n(KO / WT)", fontsize=6)
axE2.tick_params(labelsize=5.5)
axE2.text(0.06, 0.97, f"mean diff = {obsD:+.3f}\n[{ciD[0]:+.3f}, "
          f"{ciD[1]:+.3f}]\nlabel-permutation\np = {pD:.2f} (10,000)\n"
          "L35/L36 excluded:\nICE-uncovered anchors\n(technical)",
          transform=axE2.transAxes, fontsize=5.6, va="top", ha="left",
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
src("E_loop_wt_vs_ko_scatter", loe[["loop_id", "distance_kb", "both_bound",
                                    "WT_mean", "KO_mean", "d_log2OE"]])
src("E_loop_strength_stats", pd.DataFrame([dict(
    n_bound=len(gb), n_other=len(ub), mean_diff=obsD,
    boot_ci_lo=ciD[0], boot_ci_hi=ciD[1], permutation_p=pD,
    excluded_loops="L35,L36",
    exclusion_reason="ICE-uncovered anchor bins near 3.93 Mb",
    sensitivity_3x3="bound -0.265 vs other -0.244, diff -0.022 "
                    "[-0.186,+0.144], p=0.81 (n=39)")]))

fc9.save(fig, "Fig1_corrected")

# console digest (verification vs stated numbers)
print("B p-values:", {k: round(float(wstats.loc[k, 'p']), 3)
                      for k, *_ in metrics})
print("C forest:", [(r["panel_row"], round(float(r["effect"]), 4),
                     round(float(r["p"]), 3)) for r in forest_rows])
print("C robustness: effect range",
      round(rob["effect"].min(), 4), "to", round(rob["effect"].max(), 4),
      "| CIs crossing 0:", n_cross, "/18")
print("D: both-anchor", int(rb["observed"]), "vs null",
      round(float(rb["null_mean"]), 2), "p =", round(float(rb["p_empirical"]), 4),
      "| stratified p range",
      round(strat["p_empirical"][1:].min(), 4), "-",
      round(strat["p_empirical"][1:].max(), 4),
      "| sensitivity p range",
      round(sens_both["p_empirical"].min(), 4), "-",
      round(sens_both["p_empirical"].max(), 4))
print(f"E: mean diff {obsD:+.3f} [{ciD[0]:+.3f},{ciD[1]:+.3f}] p={pD:.3f}")
