#!/usr/bin/env python3
"""Figure 3 — napM genotype alters the hypoxia structural response genome-wide."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import fc

d = fc.load_tables()
b, pk, loops, pairs = d["b"], d["peaks"], d["loops"], d["pairs"]
stats13 = d["stats13"]
x_mb = (b["bin"] * fc.BIN + fc.BIN / 2) / 1e6

sd = fc.SourceData("Fig3")

fig = plt.figure(figsize=(fc.FIGW, 170 * fc.MM))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.72, 1.05], hspace=0.55,
                      wspace=0.45, left=0.075, right=0.985, top=0.93, bottom=0.08)

# ---------------- A: 2x2 design schematic ----------------
axA = fig.add_subplot(gs[0, 0])
axA.axis("off")
cells = [("WT\nnormoxia", "n = 4", fc.C["WT"], False, 0, 1),
         ("KO\nnormoxia", "n = 4", fc.C["KO"], False, 1, 1),
         ("WT\nhypoxia", "n = 4", fc.C["hypoxia"], False, 0, 0),
         ("KO\nhypoxia", "n = 3", "#75566F", False, 1, 0)]
for name, nn, col, hatch, ix, iy in cells:
    axA.add_patch(Rectangle((ix, iy), 0.92, 0.92, facecolor=col, alpha=0.30,
                            edgecolor=col, lw=0.8))
    axA.text(ix + 0.46, iy + 0.60, name, ha="center", va="center", fontsize=6)
    axA.text(ix + 0.46, iy + 0.22, nn, ha="center", va="center", fontsize=5.5,
             color="#555555")
axA.add_patch(Rectangle((2.1, 0.5), 0.92, 0.92, facecolor=fc.C["latent"],
                        alpha=0.15, edgecolor=fc.C["latent"], lw=0.8,
                        hatch="//////"))
axA.text(2.56, 1.02, "latent\nn = 4", ha="center", va="center", fontsize=6,
         color=fc.C["latent"])
axA.text(2.56, 0.62, "DESCRIPTIVE\nONLY", ha="center", va="center", fontsize=5,
         color=fc.C["latent"], fontweight="bold")
axA.text(0.46, 2.15, "interaction = (KO·hyp − KO) − (WT·hyp − WT)",
         fontsize=6.5, ha="left")
axA.set_xlim(-0.05, 3.15); axA.set_ylim(-0.1, 2.35)
fc.panel_label(axA, "A", x=-0.28, y=1.10)
sd.add("A_design", pd.DataFrame(dict(
    group=["WT", "KO", "WT_hypoxia", "KO_hypoxia", "latent"],
    n_replicates=[4, 4, 4, 3, 4],
    role=["factorial", "factorial", "factorial", "factorial",
          "descriptive only"])))

# ---------------- B: genome-wide interaction track ----------------
gsB = gs[0, 1:].subgridspec(2, 1, hspace=0.06, height_ratios=[0.28, 1])
axT = fig.add_subplot(gsB[0])
axB = fig.add_subplot(gsB[1], sharex=axT)
axT.vlines(pk["center"] / 1e6, 0, 1, color=fc.C["chip"], lw=0.35, alpha=0.7)
anchors = np.sort(np.concatenate([loops["center1"].values, loops["center2"].values]))
axT.scatter(anchors / 1e6, np.full_like(anchors, -0.25, dtype=float), marker="o",
            s=4, facecolors="none", edgecolors="k", linewidths=0.45)
axT.set_ylim(-0.7, 1); axT.set_yticks([])
axT.set_ylabel("peaks /\nanchors", fontsize=6)
axT.spines["left"].set_visible(False)
plt.setp(axT.get_xticklabels(), visible=False)
di = b["d_ins_interaction"].values
axB.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
axB.bar(x_mb, di, width=0.005, color=np.where(di >= 0, fc.C["pos"], fc.C["neg"]),
        alpha=0.85, lw=0)
axB.set_ylabel("insulation interaction\n(KO·hyp−KO) − (WT·hyp−WT)", fontsize=6.5)
axB.set_xlabel("genome position (Mb)")
axB.set_xlim(0, fc.GENOME / 1e6)
frac_small = (b["interaction_class"] == "supports_small_change").mean()
axB.text(0.99, 0.04, f"interaction widespread but mostly small: "
                     f"{b['interaction_class'].value_counts().get('detected_increase', 0)} increase, "
                     f"{b['interaction_class'].value_counts().get('detected_decrease', 0)} decrease, "
                     f"{b['interaction_class'].value_counts().get('supports_small_change', 0)} small, "
                     f"{b['interaction_class'].value_counts().get('cannot_determine', 0)} undetermined",
         transform=axB.transAxes, fontsize=5.8, ha="right", va="bottom")
fc.panel_label(axT, "B")
sd.add("B_interaction_track", b[["bin", "start", "end", "d_ins_interaction",
                                 "d_ins_interaction_lo", "d_ins_interaction_hi",
                                 "interaction_class", "interaction_padj"]])

# ---------------- C: WT vs KO hypoxia response ----------------
axC = fig.add_subplot(gs[1, 0])
xw, yk = b["d_ins_hyp_WT"].values, b["d_ins_hyp_KO"].values
sc = b["chip_bin_score_max"].values
unb = sc == 0
axC.scatter(xw[unb], yk[unb], s=2, color=fc.C["light"], lw=0, rasterized=True,
            label="unbound")
scat = axC.scatter(xw[~unb], yk[~unb], s=3.5, c=sc[~unb], cmap="Purples",
                   vmin=0.7, vmax=1.0, lw=0, rasterized=True,
                   label="NapM-bound (colour = ChIP score)")
lim = np.nanmax(np.abs(np.concatenate([xw, yk]))) * 1.05
axC.plot([-lim, lim], [-lim, lim], color=fc.C["ns"], lw=0.6, ls="--")
axC.axhline(0, color=fc.C["ns"], lw=0.4); axC.axvline(0, color=fc.C["ns"], lw=0.4)
axC.set_xlim(-lim, lim); axC.set_ylim(-lim, lim)
axC.set_xlabel("WT hypoxia response (Δins hyp − WT)")
axC.set_ylabel("KO hypoxia response (Δins KO·hyp − KO)")
axC.set_title(f"genome means: WT {np.nanmean(xw):+.3f}, KO {np.nanmean(yk):+.3f}",
              fontsize=6)
cb = fig.colorbar(scat, ax=axC, shrink=0.75, pad=0.02)
cb.set_label("ChIP score", fontsize=6); cb.ax.tick_params(labelsize=5.5)
cb.outline.set_linewidth(0.5)
fc.panel_label(axC, "C")
sd.add("C_hypoxia_responses", b[["bin", "d_ins_hyp_WT", "d_ins_hyp_KO",
                                 "chip_bin_score_max", "chip_bound"]])

# ---------------- D: bound vs matched interaction + null inset ----------------
axD = fig.add_subplot(gs[1, 1])
B = pairs["bound_bin"].values; M = pairs["matched_bin"].values
vb, vm = di[B], di[M]
okp = ~(np.isnan(vb) | np.isnan(vm))
vb, vm = vb[okp], vm[okp]
for grp, vals, col in [(0, vb, fc.C["chip"]), (1, vm, fc.C["matched"])]:
    jit = fc.RNG.uniform(-0.12, 0.12, len(vals))
    axD.scatter(np.full(len(vals), grp) + jit, vals, s=1.5, color=col, alpha=0.25,
                lw=0, rasterized=True)
    m = np.nanmean(vals)
    bs = fc.RNG.choice(vals, size=(10000, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    axD.errorbar([grp + 0.28], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=4,
                 color=col, ecolor=col, elinewidth=1.1, capsize=2.5)
r3 = fc.stat(stats13, "Fig3_paired_d_ins_interaction")
axD.set_xticks([0, 1]); axD.set_xticklabels(["NapM-bound", "matched"])
axD.set_ylabel("insulation interaction")
axD.set_title(f"paired diff = {r3['effect']:+.4f} [{r3['ci_lo']:+.3f}, {r3['ci_hi']:+.3f}]\n"
              f"circular-shift p = {r3['p']:.2f}", fontsize=6)
fc.panel_label(axD, "D")
axin = axD.inset_axes([0.52, 0.50, 0.45, 0.34])
axin.set_facecolor("white")
null = np.load(f"{fc.RA}/06_permutation_tests/null_Fig3_paired_d_ins_interaction.npy")
axin.hist(null, bins=60, color=fc.C["bg"], lw=0)
axin.axvline(r3["effect"], color=fc.C["chip"], lw=1.0)
axin.set_title("circular-shift null", fontsize=5.5)
axin.set_xlabel("mean paired diff", fontsize=5.5, labelpad=1)
axin.tick_params(labelsize=5, length=1.5)
for sp in ("top", "right"):
    axin.spines[sp].set_visible(False)
sd.add("D_paired_interaction", pd.DataFrame(dict(
    bound_bin=B[okp], matched_bin=M[okp], interaction_bound=vb,
    interaction_matched=vm)))

# ---------------- E: long-range contact response slope ----------------
axE = fig.add_subplot(gs[1, 2])
lr_wt = b["lrfrac_WT"].values
lr_ko = b["lrfrac_KO"].values
lr_h = b[[f"lrfrac_hypoxia_rep{r}" for r in range(1, 5)]].mean(axis=1).values
lr_kh = b[[f"lrfrac_KO_hypoxia_rep{r}" for r in range(1, 4)]].mean(axis=1).values
wt_resp = lr_h - lr_wt
ko_resp = lr_kh - lr_ko
# per-replicate paired genome means (reps 1-3 available for both arms)
rep_wt = [np.nanmean(b[f"lrfrac_hypoxia_rep{r}"].values - b[f"lrfrac_WT_rep{r}"].values)
          for r in range(1, 4)]
rep_ko = [np.nanmean(b[f"lrfrac_KO_hypoxia_rep{r}"].values - b[f"lrfrac_KO_rep{r}"].values)
          for r in range(1, 4)]
for r in range(3):
    axE.plot([0, 1], [rep_wt[r], rep_ko[r]], color=fc.C["ns"], lw=0.7, zorder=1)
axE.plot([0, 1], [np.nanmean(wt_resp), np.nanmean(ko_resp)], color=fc.C["hypoxia"],
         lw=1.8, marker="o", ms=5, zorder=2,
         markerfacecolor=fc.C["hypoxia"])
axE.scatter([0], [np.nanmean(wt_resp)], s=40, color=fc.C["WT"], zorder=3)
axE.scatter([1], [np.nanmean(ko_resp)], s=40, color=fc.C["KO"], zorder=3)
axE.axhline(0, color=fc.C["ns"], lw=0.4, ls=":")
axE.set_xticks([0, 1]); axE.set_xticklabels(["WT", "napM-KO"])
axE.set_ylabel("hypoxia response of\nlong-range contact fraction")
axE.set_title(f"WT {np.nanmean(wt_resp):+.3f} → KO {np.nanmean(ko_resp):+.3f}\n"
              f"(grey: replicate-paired genome means, reps 1–3)", fontsize=6)
fc.panel_label(axE, "E")
sd.add("E_lrfrac_response", pd.DataFrame(dict(
    replicate=[1, 2, 3], WT_response=rep_wt, KO_response=rep_ko)))

fc.save(fig, "Fig3_hypoxia_interaction")
sd.write()
