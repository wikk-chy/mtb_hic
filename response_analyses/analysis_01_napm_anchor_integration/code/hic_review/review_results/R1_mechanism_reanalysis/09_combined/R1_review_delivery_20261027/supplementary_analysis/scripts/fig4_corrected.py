#!/usr/bin/env python3
"""Fig4_corrected — baseline structural change and hypoxia modulation.

Assembly only: statistics are read from pre-computed tables in
RA/03_structure_metrics and RA/08_extended (fig7_stress and fig8_loci).
Browser tracks use the same cooler inputs as RA/08_extended/scripts/fig8.py.
Seed 42. Original preliminary figures are not overwritten. Outputs:
supplementary_analysis/figures/Fig4_corrected.{pdf,png} and source CSVs in
tables/corrected_figure_sources/.

Layout: A-D are quantitative or explicit limitation panels; E-H show four
representative loci in the same six-track format.

A  Baseline KO insulation change versus the WT hypoxia response:
   Spearman rho = 0.5265, paired circular moving-block 95% CI [0.4381,
   0.6092] (100-kb blocks), circular-shift P = 0.00110, n = 881 bins.
B  Exploratory response classes based on 5th–95th percentile intervals across
   replicate-pair differences excluding zero: 594/684 hypoxia-responsive bins
   are also changed at baseline (86.8%); these are not independent-sample CIs.
C  Interaction classification additionally applies the upstream 90% CI and
   +/-0.25 effect rule; median |interaction| = 0.2003. The 315 detected
   increases/decreases are distinct from the 736 FDR-significant bins.
D  The data package contains no survival/CFU time course. It records only the
   author-provided observation that napM-KO fails to survive establishment of
   the latency model. KO-latent architecture therefore cannot be measured or
   inferred from these data; WT-latent results remain descriptive.
E-H  L2, L3, L4/L16 and L5 illustrate distinct combinations of NapM occupancy,
   RNA change, baseline structural change and hypoxia modulation. They are
   examples, not causal tests.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import cooler

RA = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R1_mechanism_reanalysis"
sys.path.insert(0, f"{RA}/09_combined/scripts")
import fc9  # style adapters (fc palette, panel_label, MM, FIGW, save)
fc = fc9.fc

SEED = 42
RNG = np.random.default_rng(SEED)

OUT = f"{RA}/09_combined/R1_review_delivery_20261027/supplementary_analysis"
FIG = f"{OUT}/figures"
TAB = f"{OUT}/tables/corrected_figure_sources"
fc9.FIG = FIG
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

F7 = f"{RA}/08_extended/fig7_stress/tables"
F8 = f"{RA}/08_extended/fig8_loci/tables"

CLRS = dict(
    WT=cooler.Cooler(f"{fc.COOL}/wt_5k.cool::/resolutions/5000"),
    KO=cooler.Cooler(f"{fc.COOL}/napm_ko_5k.cool::/resolutions/5000"))
COL_WT = "#355F7A"
COL_KO = "#A5574A"
COL_CHIP = "#665A7A"
COL_LOOP_BOUND = "#665A7A"   # both-anchor-bound loop arcs (purple)
COL_LOOP_OTHER = "#999999"   # thin grey

# ---------------- load pre-computed sources ----------------
b = pd.read_csv(f"{RA}/03_structure_metrics/bin_level_v2.csv")
assert len(b) == fc.N_BINS
sp = pd.read_csv(f"{F7}/fig7A_spearman_matrix.csv")
ovl = pd.read_csv(f"{F7}/fig7B_overlap_counts.csv")
sel = pd.read_csv(f"{F8}/fig8_locus_selection.csv")
ins_tr = pd.read_csv(f"{F8}/fig8_insulation_tracks.csv")
genes8 = pd.read_csv(f"{F8}/fig8_genes.csv")
peaks8 = pd.read_csv(f"{F8}/fig8_peaks.csv")
loops8 = pd.read_csv(f"{F8}/fig8_loops.csv")

spatial = pd.read_csv(f"{OUT}/tables/fig4A_spatial_correlation_summary.csv")
spatial_row = spatial[spatial.analysis == "primary_pairwise_median_contrasts"].iloc[0]
rho_row = dict(spearman_rho=spatial_row.estimator, ci_lo=spatial_row.ci_lo,
               ci_hi=spatial_row.ci_hi, p=spatial_row.p, n=spatial_row.n_bins)

# ---------------- panel A source + stats ----------------
srcA = b[["bin", "d_ins_hyp_WT", "d_ins_KO_WT"]].dropna()
srcA.to_csv(f"{TAB}/Fig4_v2_A_baseline_vs_hypoxia.csv", index=False)

# ---------------- panel B: response-class composition ----------------
# latent_responsive_descriptive is DESCRIPTIVE ONLY: collapsed out.
grp = (ovl.groupby(["baseline_changed", "hypoxia_responsive"])["n_bins"]
       .sum())
cnt = {
    "neither": int(grp.get((False, False), 0)),
    "baseline only": int(grp.get((True, False), 0)),
    "hypoxia only": int(grp.get((False, True), 0)),
    "both (shared)": int(grp.get((True, True), 0)),
}
assert [cnt[g] for g in
        ("neither", "baseline only", "hypoxia only", "both (shared)")] \
    == [35, 164, 90, 594], cnt
srcB = pd.DataFrame(dict(category=list(cnt), n_bins=list(cnt.values())))
srcB.to_csv(f"{TAB}/Fig4_v2_B_response_class_counts.csv", index=False)
n_hyp_resp = cnt["both (shared)"] + cnt["hypoxia only"]

# ---------------- panel C: interaction summary ----------------
ICL_ORDER = ["detected_increase", "detected_decrease",
             "supports_small_change", "cannot_determine"]
ICL_COL = dict(detected_increase=fc.C["pos"], detected_decrease=fc.C["neg"],
               supports_small_change=fc.C["ns"],
               cannot_determine=fc.C["matched"])
icount = b["interaction_class"].value_counts()
assert [int(icount[c]) for c in ICL_ORDER] == [79, 236, 455, 113], icount
dint = b["d_ins_interaction"].dropna()
med_abs_int = float(np.median(np.abs(dint)))
srcC = b[["bin", "d_ins_interaction", "d_ins_interaction_lo",
          "d_ins_interaction_hi", "interaction_padj",
          "interaction_class"]]
srcC.to_csv(f"{TAB}/Fig4_v2_C_interaction_per_bin.csv", index=False)

# ---------------- panel D: latent text box (no survival data found) ----
D_MAIN = ("Author observation: napM-KO\nfails to survive establishment\n"
          "of the latency model. No KO-latent\nHi-C sample was available;\n"
          "architecture cannot be measured\nor inferred.")
D_SUB = ("No CFU/survival quantification\nin the current data package.\n"
         "WT-latent data (not shown):\ndescriptive only.")
pd.DataFrame([dict(
    panel="D", kind="text_box",
    note=("author-provided phenotype observation; no survival/CFU/growth "
          "quantitative data found in review_data/; panel kept as text box"),
    main_text=D_MAIN.replace("\n", " "), sub_text=D_SUB.replace("\n", " "))]
).to_csv(f"{TAB}/Fig4_v2_D_latent_statement.csv", index=False)

# ---------------- loci E-H (windows from the selection record) ---------
LOCI = [("E", "L2", "high occupancy, RNA change,\nsmall structural change"),
        ("F", "L3", "unbound, structural and\nRNA change"),
        ("G", "L4", "loop L16, both anchors\nNapM-bound"),
        ("H", "L5", "strong hypoxia\nsign-reversal")]
loci = []
for panel, key, head in LOCI:
    r = sel[sel.locus == key].iloc[0]
    i0, i1 = int(r["window_start"]) // fc.BIN, int(r["window_end"]) // fc.BIN
    loci.append(dict(panel=panel, key=key, head=head, i0=i0, i1=i1, row=r))
    r.to_frame().T.to_csv(f"{TAB}/Fig4_v2_{panel}_locus_selection.csv",
                          index=False)
    sub = b.iloc[i0:i1][["bin", "start", "ins_WT", "ins_KO", "d_ins_KO_WT",
                         "d_ins_interaction"]]
    sub.to_csv(f"{TAB}/Fig4_v2_{panel}_bin_tracks.csv", index=False)
    genes8[genes8.locus == key].to_csv(f"{TAB}/Fig4_v2_{panel}_genes.csv",
                                       index=False)
    peaks8[peaks8.locus == key].to_csv(f"{TAB}/Fig4_v2_{panel}_peaks.csv",
                                       index=False)
    loops8[loops8.locus == key].to_csv(f"{TAB}/Fig4_v2_{panel}_loops.csv",
                                       index=False)

# shared O/E colour scale across ALL loci and both genotypes
TRI = {}
allv = []
for loc in loci:
    for c in ("WT", "KO"):
        img, ext = fc.oe_triangle(CLRS[c], loc["i0"], loc["i1"])
        TRI[(loc["key"], c)] = (img, ext)
        allv.append(img.compressed())
allv = np.concatenate(allv)
vmin_oe = float(np.nanpercentile(allv, 2))
vmax_oe = float(np.nanpercentile(allv, 98))
vmin_oe = min(vmin_oe, 0.98)
vmax_oe = max(vmax_oe, 1.02)
norm_oe = TwoSlopeNorm(vmin=vmin_oe, vcenter=1, vmax=vmax_oe)

# ---------------- figure ----------------
fig = plt.figure(figsize=(fc.FIGW, 152 * fc.MM))
outer = fig.add_gridspec(2, 1, left=0.055, right=0.925, top=0.97,
                         bottom=0.115, hspace=0.45,
                         height_ratios=[1.0, 3.3])

# ======== top row: panels A-D (equal width) ========
top = outer[0].subgridspec(1, 4, wspace=0.45)

# ---- Panel A: baseline KO defect vs WT hypoxia response ----
axA = fig.add_subplot(top[0, 0])
hb = axA.hexbin(srcA["d_ins_hyp_WT"], srcA["d_ins_KO_WT"], gridsize=45,
                cmap="Greys", mincnt=1, linewidths=0)
cb = fig.colorbar(hb, ax=axA, pad=0.02, fraction=0.07)
cb.set_label("bins", fontsize=6)
cb.ax.tick_params(labelsize=5.5)
cb.outline.set_linewidth(0.5)
lim = float(np.nanmax(np.abs([srcA["d_ins_hyp_WT"], srcA["d_ins_KO_WT"]])))
axA.plot([-lim, lim], [-lim, lim], ls="--", lw=0.7, color="#BBBBBB")
axA.set_xlim(-lim, lim)
axA.set_ylim(-lim, lim)
axA.set_xlabel("WT hypoxia response,\nΔins (HYP − WT)", fontsize=6.5)
axA.set_ylabel("baseline KO effect,\nΔins (KO − WT)", fontsize=6.5)
axA.tick_params(labelsize=6)
axA.text(0.03, 0.97,
         f"Spearman ρ = {rho_row['spearman_rho']:.2f}\n"
         f"100-kb block 95% CI [{rho_row['ci_lo']:.2f}, {rho_row['ci_hi']:.2f}]\n"
         f"circular-shift p = {rho_row['p']:.4f}\n"
         f"n = {int(rho_row['n'])} bins",
         transform=axA.transAxes, fontsize=5.1, va="top")
fc.panel_label(axA, "A", x=-0.30)

# ---- Panel B: response-class composition ----
axB = fig.add_subplot(top[0, 1])
GCOL = {"neither": fc.C["ns"], "baseline only": COL_KO,
        "hypoxia only": fc.C["hypoxia"], "both (shared)": "#7A5C6E"}
GROUPS = ["neither", "baseline only", "hypoxia only", "both (shared)"]
xs = np.arange(4)
axB.bar(xs, [cnt[g] for g in GROUPS], width=0.62,
        color=[GCOL[g] for g in GROUPS], lw=0)
for x, g in zip(xs, GROUPS):
    axB.text(x, cnt[g] + 15, str(cnt[g]), ha="center", fontsize=6.5)
axB.set_xticks(xs)
axB.set_xticklabels(["neither", "baseline\nonly", "hypoxia\nonly",
                     "both\n(shared)"], fontsize=5.2, rotation=28,
                    ha="right")
axB.set_ylabel("bins", fontsize=6.5)
axB.tick_params(labelsize=6)
axB.set_ylim(0, 800)
axB.text(0.04, 0.97,
         f"{cnt['both (shared)'] / n_hyp_resp * 100:.0f}% of "
         f"hypoxia-\nresponsive bins\n({cnt['both (shared)']}/{n_hyp_resp}) "
         "already baseline-\nchanged (5th–95th\npairwise interval)",
         transform=axB.transAxes, fontsize=5.2, ha="left", va="top")
fc.panel_label(axB, "B", x=-0.30)

# ---- Panel C: genotype x hypoxia interaction distribution ----
axC = fig.add_subplot(top[0, 1 + 1])
edges = np.linspace(float(dint.min()), float(dint.max()), 41)
ICL_SHORT = dict(detected_increase="increase", detected_decrease="decrease",
                 supports_small_change="supp. small",
                 cannot_determine="undetermined")
bot = np.zeros(len(edges) - 1)
for cls in ICL_ORDER:
    v = b.loc[b["interaction_class"] == cls, "d_ins_interaction"].dropna()
    h, _ = np.histogram(v, bins=edges)
    axC.bar(edges[:-1], h, width=np.diff(edges), bottom=bot,
            color=ICL_COL[cls], lw=0, align="edge",
            label=f"{ICL_SHORT[cls]} ({int(icount[cls])})")
    bot += h
axC.axvline(0, color="k", lw=0.5)
axC.set_xlabel("genotype × hypoxia\ninteraction, Δins per bin", fontsize=6.5)
axC.set_ylabel("bins", fontsize=6.5)
axC.tick_params(labelsize=6)
axC.set_ylim(0, 80)
axC.legend(fontsize=4.4, frameon=False, loc="upper left", ncol=2,
           borderaxespad=0.05, handlelength=0.9, handletextpad=0.4,
           labelspacing=0.25, columnspacing=0.5)
axC.text(0.98, 0.80,
         f"median |interaction| = {med_abs_int:.2f}\n"
         "315/883 detected beyond ±0.25\n(90% CI rule)",
         transform=axC.transAxes, fontsize=5.0, ha="right", va="top",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                   pad=0.3))
fc.panel_label(axC, "C", x=-0.30)

# ---- Panel D: latent phenotype evidence (text box, no survival data) ----
axD = fig.add_subplot(top[0, 3])
axD.axis("off")
axD.add_patch(plt.Rectangle((0.03, 0.03), 0.94, 0.90,
                            transform=axD.transAxes, facecolor="white",
                            edgecolor=fc.C["text"], lw=0.8, zorder=1,
                            clip_on=False))
axD.text(0.5, 0.88, D_MAIN, transform=axD.transAxes, fontsize=5.0,
         ha="center", va="top", linespacing=1.4, zorder=2)
axD.text(0.5, 0.36, D_SUB, transform=axD.transAxes, fontsize=4.4,
         ha="center", va="top", color="#555555", linespacing=1.3,
         zorder=2)
fc.panel_label(axD, "D", x=-0.10, y=1.06)

# ======== bottom row: panels E-H, four loci, identical format ========
bot_grid = outer[1].subgridspec(
    6, 4, height_ratios=[1.0, 1.0, 0.55, 0.55, 0.50, 0.45],
    hspace=0.10, wspace=0.30)
tri_axes = []
im0 = None
for k, loc in enumerate(loci):
    i0, i1w, key = loc["i0"], loc["i1"], loc["key"]
    x0m = i0 * fc.BIN / 1e6
    x1m = (i1w - 1) * fc.BIN / 1e6  # matches triangle extent exactly

    # -- tracks 1-2: O/E triangles (WT, KO), shared scale across figure --
    for j, c in enumerate(("WT", "KO")):
        ax = fig.add_subplot(bot_grid[j, k])
        img, ext = TRI[(key, c)]
        arr = img.copy()
        arr[0, :] = np.nan  # diagonal excluded
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=fc.CMAP_DIV,
                       norm=norm_oe, extent=ext, origin="lower",
                       aspect="auto", interpolation="none")
        if im0 is None:
            im0 = im
        ax.text(0.04, 0.96, "WT" if c == "WT" else "napM-KO",
                transform=ax.transAxes, fontsize=5.5, fontweight="bold",
                color=COL_WT if c == "WT" else COL_KO, va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_visible(True)
            spn.set_linewidth(0.5)
        tri_axes.append(ax)
        if j == 0:
            ax_first = ax
    fc.panel_label(ax_first, loc["panel"], x=-0.30 if k == 0 else -0.16,
                   y=1.10)
    ax_first.text(0.02, 1.05, loc["head"], transform=ax_first.transAxes,
                  fontsize=4.8, color="#555555", va="bottom",
                  linespacing=1.25)

    # -- track 3: insulation WT + KO --
    axI = fig.add_subplot(bot_grid[2, k])
    tr = ins_tr[ins_tr.locus == key]
    xb = (tr["bin"] * fc.BIN + 2500) / 1e6
    axI.plot(xb, tr["ins_WT"], color=COL_WT, lw=0.9, label="WT")
    axI.plot(xb, tr["ins_KO"], color=COL_KO, lw=0.9, label="napM-KO")
    foc = loc["row"]["focal_bin"]
    if np.isfinite(foc):
        fx = (foc * fc.BIN + 2500) / 1e6
        axI.axvline(fx, color="k", lw=0.5, ls=":")
    axI.set_ylabel("insulation", fontsize=5.5)
    axI.tick_params(labelsize=5.5)
    axI.set_xticks([])
    axI.set_xlim(x0m, x1m)
    axI.yaxis.set_major_locator(MaxNLocator(3))
    if k == 0:
        lg = axI.legend(fontsize=4.8, loc="lower right", ncol=2, frameon=True,
                        handlelength=1.1, borderaxespad=0.05,
                        columnspacing=0.6)
        lg.get_frame().set(edgecolor="none", facecolor="white", alpha=0.75,
                           lw=0)

    # -- track 4: Δins(KO−WT) black + d_ins_interaction grey, zero line --
    axDl = fig.add_subplot(bot_grid[3, k], sharex=axI)
    sub_b = b.iloc[i0:i1w]
    axDl.axhline(0, color="k", lw=0.4)
    axDl.plot(xb.values, sub_b["d_ins_KO_WT"].values, color="k", lw=0.9,
              label="Δins (KO−WT)")
    axDl.plot(xb.values, sub_b["d_ins_interaction"].values,
              color="#999999", lw=0.9, label="interaction")
    if np.isfinite(foc):
        axDl.axvline(fx, color="k", lw=0.5, ls=":")
    axDl.set_ylabel("Δins", fontsize=5.5)
    axDl.tick_params(labelsize=5.5)
    axDl.set_xticks([])
    axDl.yaxis.set_major_locator(MaxNLocator(3))
    if k == 0:
        lg = axDl.legend(fontsize=4.8, loc="lower left", ncol=1,
                         frameon=True, handlelength=1.1, borderaxespad=0.05)
        lg.get_frame().set(edgecolor="none", facecolor="white", alpha=0.75,
                           lw=0)

    # -- track 5: NapM ChIP stems + loop arcs --
    axP = fig.add_subplot(bot_grid[4, k], sharex=axI)
    pp = peaks8[peaks8.locus == key]
    axP.vlines(pp["center"] / 1e6, 0, pp["score"], color=COL_CHIP, lw=0.9)
    ll = loops8[loops8.locus == key]
    for _, r in ll.iterrows():
        c1, c2 = r["center1"] / 1e6, r["center2"] / 1e6
        t = np.linspace(0, np.pi, 60)
        xc = (c1 + c2) / 2 + (c2 - c1) / 2 * np.cos(t)
        yc = 1.18 + 0.34 * np.sin(t)
        bound = bool(r["both_bound"])
        axP.plot(xc, yc, color=COL_LOOP_BOUND if bound else COL_LOOP_OTHER,
                 lw=0.9 if bound else 0.5)
        axP.plot([c1, c2], [1.16, 1.16], "o", ms=1.6, mfc="none",
                 mec=COL_LOOP_BOUND if bound else COL_LOOP_OTHER, mew=0.5)
    axP.set_ylim(0, 1.62)
    axP.set_yticks([])
    axP.set_xticks([])
    axP.spines["left"].set_visible(False)
    if k == 0:
        axP.text(0.01, 0.92, "NapM ChIP (height = score); arcs = loops",
                 transform=axP.transAxes, fontsize=4.6, color=COL_CHIP,
                 va="top")

    # -- track 6: genes, +/− rows, DEG red/blue, else light grey --
    axG = fig.add_subplot(bot_grid[5, k], sharex=axI)
    gg = genes8[genes8.locus == key]
    for _, r in gg.iterrows():
        yy = 0.72 if r["strand"] == "+" else 0.28
        col = (fc.C["pos"] if r["class_KOvsWT"] == "detected_increase"
               else fc.C["neg"] if r["class_KOvsWT"] == "detected_decrease"
               else fc.C["ns"])
        axG.plot([r["start"] / 1e6, r["end"] / 1e6], [yy, yy], color=col,
                 lw=1.2, solid_capstyle="butt")
    axG.text(x0m, 0.78, "+", fontsize=5, va="center", color="#555555")
    axG.text(x0m, 0.10, "−", fontsize=5, va="center", color="#555555")
    axG.set_ylim(0, 1)
    axG.set_yticks([])
    axG.set_xlabel("genome position (Mb)", fontsize=6)
    axG.tick_params(labelsize=5.5)
    axG.spines["left"].set_visible(False)
    axG.xaxis.set_major_locator(MaxNLocator(3))

# one shared O/E colorbar for all eight triangles
cboe = fig.colorbar(im0, ax=tri_axes, fraction=0.025, pad=0.012)
cboe.set_label("O/E (balanced Hi-C)", fontsize=6)
cboe.ax.tick_params(labelsize=5.5)
cboe.outline.set_linewidth(0.5)

fig.text(0.5, 0.054,
         "E–H: identical track order per locus — WT and napM-KO O/E 45° "
         "triangles (shared scale: red = O/E > 1, blue = O/E < 1, grey = "
         "masked), insulation, Δins (black)\nand genotype × hypoxia "
         "interaction (grey), NapM ChIP + loops (purple = both anchors "
         "NapM-bound), genes (+/− strand; red/blue = detected up/down, "
         "grey = not detected).",
         ha="center", va="top", fontsize=4.8, color="#555555",
         linespacing=1.4)

fc9.save(fig, "Fig4_corrected")
print("fig4_v2 done | O/E scale", round(vmin_oe, 3), "to", round(vmax_oe, 3),
      "| median |interaction| =", round(med_abs_int, 4))
