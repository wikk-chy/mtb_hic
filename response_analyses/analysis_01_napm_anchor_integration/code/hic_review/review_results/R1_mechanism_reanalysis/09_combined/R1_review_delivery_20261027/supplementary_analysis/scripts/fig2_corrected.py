#!/usr/bin/env python3
"""Fig2_corrected — NapM-dependent differences in the hypoxia response.

Six-panel layout assembled from pre-computed results; seed 42.
  A  Factorial design: four independent samples each for WT, KO and WT-hypoxia,
     and three for KO-hypoxia. WT-latent is descriptive only; no KO-latent
     Hi-C exists because the author-reported latency observation is loss of
     survival during model establishment.
  B  Genome-wide WT and KO hypoxia responses and their genotype-by-hypoxia
     interaction. A five-bin circular rolling median is a visual guide only.
  C  Unified condition-mean response estimator for both genotypes (n = 881):
     concordant 154, attenuated 78, enhanced 187 and sign-reversed 462.
  D  Descriptive pairwise replicate consistency plus the independent-sample
     genome-mean analysis: WT response +0.04098, KO response -0.06701,
     difference-in-differences -0.10799 [sample-bootstrap 95% CI -0.11403,
     -0.10150], exact oxygen-label permutation P = 0.000408.
  E  Multivariate fingerprint PCA for 19 samples and contribution decomposition.
     The response magnitude differs (Euclidean P = 0.00080), whereas response
     direction alone does not exceed its null (cosine P = 0.365).
  F  Interval-binding matched sensitivity analysis: paired hypoxia-interaction
     difference +0.01135 [-0.01289, +0.03663], circular-shift P = 0.651,
     n = 406 matched pairs.

Original preliminary figures are not overwritten. Outputs:
supplementary_analysis/figures/Fig2_corrected.{pdf,png} and per-panel source
CSVs in tables/corrected_figure_sources/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
                "hic_review/review_results/R1_mechanism_reanalysis/"
                "09_combined/scripts")
import fc9  # noqa: E402
import fc   # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)

F3T = f"{fc.RA}/08_extended/fig3_hypoxia/tables"
F4T = f"{fc.RA}/08_extended/fig4_fingerprint/tables"
SUPP = f"{fc.RA}/09_combined/R1_review_delivery_20261027/supplementary_analysis"
TAB9 = f"{SUPP}/tables/corrected_figure_sources"
fc9.FIG = f"{SUPP}/figures"
os.makedirs(TAB9, exist_ok=True)
os.makedirs(fc9.FIG, exist_ok=True)


def src(name, df):
    df.to_csv(f"{TAB9}/Fig2_final_{name}.csv", index=False)
    print("wrote", f"{TAB9}/Fig2_final_{name}.csv")


b = pd.read_csv(f"{fc.RA}/03_structure_metrics/bin_level_v2.csv")
unified = pd.read_csv(f"{SUPP}/tables/unified_mean_response_perbin.csv")
qdf = pd.read_csv(f"{SUPP}/tables/unified_mean_response_quadrant_counts.csv")
edf = pd.read_csv(f"{F3T}/Fig3_v2_E_replicate_consistency.csv")
factorial = pd.read_csv(f"{SUPP}/tables/independent_sample_factorial_summary.csv")
factorial_row = factorial[factorial.test == "genotype contrast of genome-mean hypoxia responses"].iloc[0]
interval_stats = pd.read_csv(f"{SUPP}/tables/interval_binding_matched_sensitivity.csv")
r3 = interval_stats[(interval_stats.binding_definition == "interval_454") &
                    (interval_stats.metric == "hypoxia_interaction_signed")].iloc[0].copy()
r3["p"] = r3["circular_shift_p"]
ip = pd.read_csv(f"{SUPP}/tables/binding_definition_matched_pairs.csv")
ip = ip[(ip.binding_definition == "interval_454")].dropna(subset=["matched_bin"])
ib = ip.bound_bin.to_numpy(int)
im = ip.matched_bin.to_numpy(int)
vb0 = b.d_ins_interaction.to_numpy()[ib]
vm0 = b.d_ins_interaction.to_numpy()[im]
okf = np.isfinite(vb0) & np.isfinite(vm0)
fdf = pd.DataFrame(dict(bound_bin=ib[okf], matched_bin=im[okf],
                        interaction_bound=vb0[okf], interaction_matched=vm0[okf],
                        paired_diff=vb0[okf] - vm0[okf]))
pca_scores = pd.read_csv(f"{F4T}/fig4_pca_scores.csv")
pca_var = pd.read_csv(f"{F4T}/fig4_pca_variance.csv")
contrib = pd.read_csv(f"{F4T}/fig4_response_vectors_contribution.csv")
rstats = pd.read_csv(f"{F4T}/fig4_response_stats.csv").set_index("statistic")["observed"]

xw = unified["WT_hypoxia_response_mean_difference"].values
yk = unified["KO_hypoxia_response_mean_difference"].values
di = unified["genotype_by_hypoxia_interaction"].values
unified["d_ins_hyp_WT"] = xw
unified["d_ins_hyp_KO"] = yk
unified["d_ins_interaction"] = di
x_mb = (b["bin"] * fc.BIN + fc.BIN / 2) / 1e6

COND_COLOR = {"WT": "#355F7A", "KO": "#A5574D", "HYP": "#4F7C73",
              "KO-HYP": "#75566F", "latent": "#8A7567"}
COND_ORDER = ["WT", "KO", "HYP", "KO-HYP", "latent"]


def roll_med(v, w=5):
    """Circular rolling median, window w (visual guide only)."""
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, np.nan)
    h = w // 2
    for i in range(len(v)):
        idx = (np.arange(i - h, i + h + 1)) % len(v)
        out[i] = np.nanmedian(v[idx])
    return out


fig = plt.figure(figsize=(fc.FIGW, 205 * fc.MM))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0],
                      width_ratios=[1.0, 1.0, 1.0],
                      hspace=0.60, wspace=0.45,
                      left=0.075, right=0.985, top=0.955, bottom=0.06)

# ---------------- A: 2x2 factorial design ----------------
axA = fig.add_subplot(gs[0, 0])
axA.axis("off")
cells = [("WT\nnormoxia", "n = 4", fc.C["WT"], 0, 1),
         ("KO\nnormoxia", "n = 4", fc.C["KO"], 1, 1),
         ("WT\nhypoxia", "n = 4", fc.C["hypoxia"], 0, 0),
         ("KO\nhypoxia", "n = 3", "#75566F", 1, 0)]
for name, nn, col, ix, iy in cells:
    axA.add_patch(Rectangle((ix, iy), 0.92, 0.92, facecolor=col, alpha=0.30,
                            edgecolor=col, lw=0.8))
    axA.text(ix + 0.46, iy + 0.60, name, ha="center", va="center", fontsize=6)
    axA.text(ix + 0.46, iy + 0.22, nn, ha="center", va="center", fontsize=5.5,
             color="#555555")
# latent cell: no KO-latent (napM-KO cells fail to survive latent stress)
axA.add_patch(Rectangle((2.1, 0.5), 0.92, 0.92, facecolor=fc.C["ns"],
                        alpha=0.25, edgecolor=fc.C["latent"], lw=0.8,
                        hatch="//////"))
axA.text(2.56, 1.10, "WT-latent\nn = 4", ha="center", va="center", fontsize=6,
         color=fc.C["latent"])
axA.text(2.56, 0.68, "DESCRIPTIVE\nONLY", ha="center", va="center",
         fontsize=5, color=fc.C["latent"], fontweight="bold")
axA.text(2.56, 0.40, "KO-latent unavailable\n"
                     "(author observation;\n"
                     "see legend)",
         ha="center", va="top", fontsize=4.5, color=fc.C["latent"])
axA.text(0.0, 2.2, "interaction = (KO\u00b7hyp \u2212 KO) \u2212 (WT\u00b7hyp \u2212 WT)",
         fontsize=6.5, ha="left")
axA.set_xlim(-0.05, 3.15)
axA.set_ylim(-0.75, 2.45)
fc9.panel_label(axA, "A", x=-0.28, y=1.08)
src("A_design", pd.DataFrame([
    dict(condition="WT", n=4, role="hypothesis test"),
    dict(condition="KO", n=4, role="hypothesis test"),
    dict(condition="HYP", n=4, role="hypothesis test"),
    dict(condition="KO-HYP", n=3, role="hypothesis test"),
    dict(condition="latent", n=4, role="descriptive only; KO-latent unavailable; "
         "author-reported loss of viability; no CFU/survival quantitative data "
         "in current package; WT-latent descriptive only")]))

# ---------------- B: three genome-wide tracks ----------------
gsB = gs[0, 1:].subgridspec(3, 1, hspace=0.42)
tracks = [(xw, fc.C["WT"], "WT response"),
          (yk, "#75566F", "KO response"),
          (di, fc.C["text"], "interaction")]
axB = []
for k, (v, col, lab) in enumerate(tracks):
    ax = fig.add_subplot(gsB[k], sharex=axB[0] if axB else None)
    ax.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
    ax.bar(x_mb, v, width=0.005,
           color=np.where(v >= 0, fc.C["pos"], fc.C["neg"]), alpha=0.55, lw=0)
    ax.plot(x_mb, roll_med(v, 5), color=col, lw=0.9)
    ax.set_ylabel(lab, fontsize=6)
    ax.set_xlim(0, fc.GENOME / 1e6)
    ax.tick_params(labelsize=6)
    if k < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
    axB.append(ax)
axB[0].text(0.99, 0.92, "coloured line: circular rolling median\n"
            "(5 bins / 25 kb), visual guide only", transform=axB[0].transAxes,
            fontsize=5, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
axB[-1].set_xlabel("genome position (Mb)")
axB[0].set_title("genome-wide \u0394insulation tracks", fontsize=6.5, pad=3)
fc9.panel_label(axB[0], "B", x=-0.16, y=1.12)
src("B_tracks", unified[["bin", "start", "end", "d_ins_hyp_WT",
                       "d_ins_hyp_KO", "d_ins_interaction"]])

# ---------------- C: quadrant scatter ----------------
axC = fig.add_subplot(gs[1, 0])
quad = unified["quadrant_class"].values
okc = np.isfinite(xw) & np.isfinite(yk)
n_ok = int(okc.sum())
qcol = dict(concordant=fc.C["hypoxia"], attenuated=fc.C["WT"],
            enhanced="#B2182B", sign_reversed="#75566F")
for q in ["concordant", "attenuated", "enhanced", "sign_reversed"]:
    m = okc & (quad == q)
    axC.scatter(xw[m], yk[m], s=2.5, color=qcol[q], lw=0, alpha=0.6,
                rasterized=True,
                label=f"{q.replace('_', ' ')} {m.sum()} ({m.sum()/n_ok*100:.0f}%)")
lim = np.nanmax(np.abs(np.concatenate([xw[okc], yk[okc]]))) * 1.05
axC.plot([-lim, lim], [-lim, lim], color=fc.C["ns"], lw=0.6, ls="--")
axC.axhline(0, color=fc.C["ns"], lw=0.4)
axC.axvline(0, color=fc.C["ns"], lw=0.4)
axC.set_xlim(-lim, lim)
axC.set_ylim(-lim, lim)
axC.set_xlabel("WT hypoxia response (\u0394ins)")
axC.set_ylabel("KO hypoxia response (\u0394ins)")
axC.legend(fontsize=5, loc="upper left", markerscale=2,
           handletextpad=0.2, borderaxespad=0.1, framealpha=0.9,
           edgecolor="none", borderpad=0.3)
n_sr = int(qdf.loc[qdf["quadrant"] == "sign_reversed", "n_bins"].iloc[0])
axC.text(0.98, 0.02, f"sign-reversed {n_sr}/{n_ok} "
         f"({n_sr/n_ok*100:.1f}%)", transform=axC.transAxes, fontsize=5.8,
         ha="right", va="bottom", color="#75566F", fontweight="bold")
fc9.panel_label(axC, "C")
src("C_quadrant_counts", qdf)
src("C_quadrant_perbin", unified[["bin", "start", "end", "d_ins_hyp_WT",
                                "d_ins_hyp_KO", "quadrant_class"]])

# ---------------- D: replicate consistency ----------------
axD = fig.add_subplot(gs[1, 1])
groups = [("within_WT", "within WT\n(4\u00d74, 120 pairs)", fc.C["WT"]),
          ("between_WT_KO", "WT \u00d7 KO\n(192 pairs)", fc.C["ns"]),
          ("within_KO", "within KO\n(4\u00d73, 66 pairs)", "#75566F")]
meds = {}
for pos, (g, lab, col) in enumerate(groups):
    vv = edf.loc[edf["group"] == g, "spearman"].values
    meds[g] = np.median(vv)
    jit = rng.uniform(-0.10, 0.10, len(vv))
    axD.scatter(np.full(len(vv), pos) + jit, vv, s=2, color=col, alpha=0.30,
                lw=0, rasterized=True)
    med = meds[g]
    lo, hi = np.percentile(vv, [2.5, 97.5])
    axD.errorbar([pos + 0.24], [med], yerr=[[med - lo], [hi - med]], fmt="o",
                 ms=4, color=col, ecolor=col, elinewidth=1.1, capsize=2.5)
axD.set_xticks(range(3))
axD.set_xticklabels([g[1] for g in groups], fontsize=5.8)
axD.set_ylabel("pairwise Spearman \u03c1 of\ngenome-wide response vectors")
axD.text(0.03, 0.03,
         f"median \u03c1: within WT = {meds['within_WT']:.3f}, "
         f"within KO = {meds['within_KO']:.3f}\n"
         f"between groups = {meds['between_WT_KO']:.3f}\n"
         "pairwise distributions descriptive (shared replicates)\n"
         "genome-mean response difference, 15 independent samples:\n"
         f"{factorial_row['effect']:.3f} [{factorial_row['ci_lo']:.3f}, "
         f"{factorial_row['ci_hi']:.3f}], exact p = {factorial_row['p']:.1e}",
         transform=axD.transAxes, fontsize=4.8, va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
axD.set_ylim(-0.35, 1.05)
fc9.panel_label(axD, "D")
src("D_replicate_consistency", edf)
src("D_independent_sample_factorial", factorial)

# ---------------- E: multivariate fingerprint ----------------
# E-left: PCA, 19 samples coloured by condition
axE1 = fig.add_subplot(gs[1, 2])
frac1 = pca_var.loc[pca_var["PC"] == "PC1", "variance_fraction"].iloc[0]
frac2 = pca_var.loc[pca_var["PC"] == "PC2", "variance_fraction"].iloc[0]
for c in COND_ORDER:
    sel = (pca_scores["condition"] == c).values
    axE1.scatter(pca_scores.loc[sel, "PC1"], pca_scores.loc[sel, "PC2"], s=16,
                 color=COND_COLOR[c], edgecolor="white", linewidth=0.4,
                 label=f"{c} (n={sel.sum()})", zorder=3)
axE1.axhline(0, color="#CCCCCC", lw=0.5, zorder=1)
axE1.axvline(0, color="#CCCCCC", lw=0.5, zorder=1)
axE1.set_xlabel(f"PC1 ({frac1*100:.1f}% var)")
axE1.set_ylabel(f"PC2 ({frac2*100:.1f}% var)")
axE1.legend(frameon=False, loc="best", handletextpad=0.2, borderpad=0.2,
            fontsize=5.5)
axE1.set_title("PCA of 9-metric fingerprint, 19 samples\n(latent descriptive)",
               fontsize=6)
fc9.panel_label(axE1, "E")
src("E_pca_scores", pca_scores)
src("E_pca_variance", pca_var)

# E-right (row 3, left): per-metric contribution |R_WT - R_KO|
axE2 = fig.add_subplot(gs[2, 0])
csorted = contrib.sort_values("abs_diff", ascending=False).reset_index(drop=True)
top3 = csorted["metric"].head(3).tolist()   # entropy, short_frac, long_frac
bcols = [fc.C["chip"] if m in top3 else fc.C["light"]
         for m in csorted["metric"]]
axE2.bar(range(len(csorted)), csorted["abs_diff"], color=bcols, width=0.65)
axE2.set_xticks(range(len(csorted)))
axE2.set_xticklabels(csorted["metric_label"], rotation=32, ha="right",
                     fontsize=5.4)
axE2.set_ylabel("$|R_{WT} - R_{KO}|$ (z units)", fontsize=6.5)
axE2.set_ylim(0, csorted["abs_diff"].max() * 1.75)
axE2.tick_params(axis="y", labelsize=6)
axE2.text(0.98, 0.97,
          "purple = top3 contributors;\n"
          "z over 15 test samples\n"
          "(latent excluded)\n"
          "response magnitude differs\n"
          f"(Euclidean p = {rstats['perm_p_euclidean']:.4f});\n"
          "direction alone not beyond null\n"
          f"(cosine p = {rstats['perm_p_cosine']:.2f})",
          transform=axE2.transAxes, fontsize=5.4, ha="right", va="top")
src("E_contribution", contrib)
src("E_response_stats", rstats.reset_index())

# ---------------- F: bound vs matched interaction + null inset ----------------
gsF = gs[2, 1:].subgridspec(1, 2, width_ratios=[1.0, 0.85], wspace=0.40)
axF1 = fig.add_subplot(gsF[0])
vb = fdf["interaction_bound"].values
vm = fdf["interaction_matched"].values
pdiff = fdf["paired_diff"].values
# r3 is the interval-overlap matched sensitivity loaded above.


def boot_ci(vals, nboot=10000):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    bs = np.nanmean(rng.choice(vals, size=(nboot, len(vals)), replace=True),
                    axis=1)
    return np.percentile(bs, [2.5, 97.5])


for grp, vals, col in [(0, vb, fc.C["chip"]), (1, vm, fc.C["matched"])]:
    jit = rng.uniform(-0.12, 0.12, len(vals))
    axF1.scatter(np.full(len(vals), grp) + jit, vals, s=1.5, color=col,
                 alpha=0.25, lw=0, rasterized=True)
    m = np.nanmean(vals)
    lo, hi = boot_ci(vals)
    axF1.errorbar([grp + 0.28], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=4,
                  color=col, ecolor=col, elinewidth=1.1, capsize=2.5)
axF1.set_xticks([0, 1])
axF1.set_xticklabels([f"NapM-bound\n(n={len(vb)})", f"matched\n(n={len(vm)})"],
                     fontsize=6)
axF1.set_ylabel("insulation interaction")
axF1.set_title(f"interval matched diff = {r3['effect']:+.3f} "
               f"[{r3['ci_lo']:+.3f}, {r3['ci_hi']:+.3f}]\n"
               f"circular-shift p = {r3['p']:.2f}, n = {int(r3['n_pairs'])} pairs "
               f"(not significant)", fontsize=6)
fc9.panel_label(axF1, "F", x=-0.30)
axin = axF1.inset_axes([0.50, 0.52, 0.46, 0.36])
axin.set_facecolor("white")
with np.load(f"{SUPP}/tables/interval_binding_all_shift_nulls.npz") as null_zip:
    null = null_zip["interval_454__hypoxia_interaction_signed"]
axin.hist(null, bins=60, color=fc.C["bg"], lw=0)
axin.axvline(r3["effect"], color=fc.C["chip"], lw=1.0)
axin.set_title("circular-shift null", fontsize=5.5)
axin.set_xlabel("mean paired diff", fontsize=5.5, labelpad=1)
axin.tick_params(labelsize=5, length=1.5)
for spn in ("top", "right"):
    axin.spines[spn].set_visible(False)

axF2 = fig.add_subplot(gsF[1])
jit = rng.uniform(-0.10, 0.10, len(pdiff))
axF2.scatter(np.full(len(pdiff), 0) + jit, pdiff, s=1.5, color=fc.C["chip"],
             alpha=0.25, lw=0, rasterized=True)
md = np.mean(pdiff)
lo, hi = boot_ci(pdiff)
axF2.errorbar([0.24], [md], yerr=[[md - lo], [hi - md]], fmt="o", ms=4,
              color=fc.C["chip"], ecolor=fc.C["chip"], elinewidth=1.1,
              capsize=2.5)
axF2.axhline(0, color=fc.C["ns"], lw=0.6, ls="--")
axF2.set_xticks([0])
axF2.set_xticklabels(["bound \u2212 matched\npaired difference"], fontsize=6)
axF2.set_ylabel("paired \u0394 interaction")
axF2.set_title(f"mean paired diff = {md:+.4f} [{lo:+.3f}, {hi:+.3f}]\n"
               f"median = {np.median(pdiff):+.4f}; "
               f"{np.mean(pdiff > 0)*100:.0f}% of pairs positive", fontsize=6)
src("F_paired_interaction", fdf)
src("F_null_paired_diff",
    pd.DataFrame({"null_mean_paired_diff": null}))

fc9.save(fig, "Fig2_corrected")

# console digest (verification vs stated numbers)
print("C: sign-reversed", n_sr, "/", n_ok, f"({n_sr/n_ok*100:.1f}%)")
print("D: medians", {k: round(v, 3) for k, v in meds.items()},
      "| independent-sample DID", round(factorial_row["effect"], 3),
      [round(factorial_row["ci_lo"], 3), round(factorial_row["ci_hi"], 3)],
      f"exact p={factorial_row['p']:.1e}")
print("E: top3 contributors", top3,
      "| euclid p", rstats["perm_p_euclidean"],
      "| cosine p", round(rstats["perm_p_cosine"], 3))
print("F: paired diff", round(r3["effect"], 4), "p", round(r3["p"], 4))
