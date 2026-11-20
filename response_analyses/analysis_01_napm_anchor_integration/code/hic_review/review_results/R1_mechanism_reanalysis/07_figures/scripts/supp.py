#!/usr/bin/env python3
"""Supplementary figures — ChIP QC, locus browsers, latent descriptive panel."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cooler
import fc

d = fc.load_tables()
b, pk, loops, g = d["b"], d["peaks"], d["loops"], d["genes"]

# ================= SuppFig 1: ChIP QC =================
fig = plt.figure(figsize=(fc.FIGW, 120 * fc.MM))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.40, left=0.07, right=0.985,
                      top=0.88, bottom=0.13)

ax = fig.add_subplot(gs[0, 0])
ax.hist(pk["score"], bins=40, color=fc.C["chip"], lw=0)
ax.set_xlabel("peak score"); ax.set_ylabel("peaks")
ax.set_title(f"score: median 0.932\n[0.701–0.998]", fontsize=6.5)
fc.panel_label(ax, "A")

ax = fig.add_subplot(gs[0, 1])
ax.hist(np.log10(pk["vpm"]), bins=40, color=fc.C["chip"], lw=0)
ax.axvline(np.log10(408.5), color="k", lw=0.7, ls="--")
ax.set_xlabel("log10 VPM"); ax.set_ylabel("peaks")
ax.set_title("VPM: median 408.5 [96–24,957]", fontsize=6.5)
fc.panel_label(ax, "B")

ax = fig.add_subplot(gs[0, 2])
loc = pd.read_csv(f"{fc.RA}/01_chip_curated/qc_peak_location_counts.csv")
ax.bar(range(len(loc)), loc["count"], color=[fc.C["chip"], "#9B8FA8", fc.C["ns"]],
       width=0.6)
for i, v in enumerate(loc["count"]):
    ax.text(i, v + 8, str(v), ha="center", fontsize=7)
ax.set_xticks(range(len(loc)))
ax.set_xticklabels(["gene body", "promoter\n(−300 bp)", "intergenic"], fontsize=6.5)
ax.set_ylabel("peaks")
ax.set_title("peak location (n = 634)", fontsize=6.5)
fc.panel_label(ax, "C")

ax = fig.add_subplot(gs[1, 0])
gaps = pd.read_csv(f"{fc.RA}/01_chip_curated/qc_adjacent_peak_gaps.csv")["gap_bp"]
ax.hist(gaps / 1e3, bins=60, color="#9B8FA8", lw=0)
ax.set_xlabel("adjacent-peak gap (kb)"); ax.set_ylabel("peak pairs")
ax.set_title("gaps: median 4.0 kb [0.03–50.0]", fontsize=6.5)
fc.panel_label(ax, "D")

ax = fig.add_subplot(gs[1, 1])
pb = pd.read_csv(f"{fc.RA}/01_chip_curated/qc_peaks_per_bin_counts.csv")
ax.bar(pb.iloc[:, 0], pb["count"], color="#9B8FA8", width=0.6)
ax.set_xlabel("peaks per 5-kb bin"); ax.set_ylabel("bins")
ax.set_title("433 bound bins (49% of genome)", fontsize=6.5)
for i, r in pb.iterrows():
    ax.text(r.iloc[0], r["count"] + 4, str(r["count"]), ha="center", fontsize=6.5)
fc.panel_label(ax, "E")

ax = fig.add_subplot(gs[1, 2])
w = (pk["end"] - pk["start"]) 
ax.hist(w, bins=40, color="#9B8FA8", lw=0)
ax.set_xlabel("peak width (bp)"); ax.set_ylabel("peaks")
ax.set_title("peak width: median 363 bp", fontsize=6.5)
fc.panel_label(ax, "F")

fc.save(fig, "SuppFig1_ChIP_QC")

# ================= SuppFig 2: locus browsers =================
def cls3(s):
    return s.map(lambda v: "detected" if str(v).startswith("detected")
                 else ("supports_small" if str(v).startswith("supports_small")
                       else "cannot_determine"))
b["struct3"] = cls3(b["ins_change_class"])
b["rna3"] = b["rna_bin_class"].map(
    lambda v: "detected" if v == "detected_change"
    else ("supports_small" if v == "supports_small_change" else "cannot_determine"))

m1 = (b["chip_bound"]) & (b["struct3"] == "detected") & (b["rna3"] == "detected")
i1 = int(b.loc[m1, "d_ins_KO_WT"].abs().idxmax())
m2 = (b["chip_bound"]) & (b["struct3"] == "supports_small") & (b["rna3"] == "detected")
i2 = int(b.loc[m2, "chip_bin_score_max"].idxmax())
m3 = (~b["chip_bound"]) & (b["struct3"] == "detected") & (b["rna3"] == "detected")
m3v = m3 & (np.abs(b["bin"] - i1) > 60)
i3 = int(b.loc[m3v, "d_ins_KO_WT"].abs().idxmax())
r4 = loops[loops["both_bound"]].sort_values("score", ascending=False).iloc[0]

loci = [
    dict(name=f"locus 1 (bin {i1})", i0=i1 - 20, i1=i1 + 21,
         crit="bound & struct.-detected & RNA-detected, max |Δins| (n = 65)"),
    dict(name=f"locus 2 (bin {i2})", i0=i2 - 20, i1=i2 + 21,
         crit="bound & struct.-small & RNA-detected, max ChIP score (n = 117)"),
    dict(name=f"locus 3 (bin {i3})", i0=i3 - 20, i1=i3 + 21,
         crit="unbound & struct.-detected & RNA-detected, max |Δins|, >300 kb from locus 1 (n = 80)"),
    dict(name=f"locus 4 (loop {r4['loop_id']})",
         i0=int(r4["center1"]) // fc.BIN - 15, i1=int(r4["center2"]) // fc.BIN + 16,
         crit="highest-score loop with both anchors NapM-bound (n = 17)"),
]
loci = [dict(l, i0=max(0, l["i0"]), i1=min(fc.N_BINS, l["i1"])) for l in loci]

genes = fc.load_genes_gtf().merge(
    g[["gene_id", "log2FC_KOvsWT", "class_KOvsWT"]], on="gene_id", how="left")

clr_wt = cooler.Cooler(f"{fc.COOL}/wt_5k.cool::/resolutions/5000")
clr_ko = cooler.Cooler(f"{fc.COOL}/napm_ko_5k.cool::/resolutions/5000")

fig = plt.figure(figsize=(fc.FIGW, 205 * fc.MM))
outer = fig.add_gridspec(2, 2, hspace=0.72, wspace=0.30, left=0.06, right=0.985,
                         top=0.94, bottom=0.115)
panels = "ABCD"
sd_rows = []
for k, loc in enumerate(loci):
    i0, i1 = loc["i0"], loc["i1"]
    sub = outer[k // 2, k % 2].subgridspec(3, 2, height_ratios=[1, 1, 0.55],
                                           hspace=0.05, wspace=0.12)
    imgs = []
    for j, clr in enumerate([clr_wt, clr_ko]):
        imgs.append(fc.oe_triangle(clr, i0, i1))
    allv = np.concatenate([im[0].compressed() for im in imgs])
    vmax = np.nanpercentile(np.abs(np.log2(allv)), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for j, (clr, ttl) in enumerate([(clr_wt, "WT"), (clr_ko, "napM-KO")]):
        ax = fig.add_subplot(sub[0, j])
        if j == 0:
            ax_wt = ax
        img, ext = imgs[j]
        arr = np.log2(img)
        arr[0, :] = np.nan  # diagonal excluded
        ax.imshow(np.ma.masked_invalid(arr), cmap=fc.CMAP_DIV, norm=norm,
                  extent=ext, origin="lower", aspect="auto",
                  interpolation="none")
        ax.set_title(ttl, fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    # peaks + genes strip spanning both columns
    axg = fig.add_subplot(sub[1:, :])
    x0, x1 = i0 * fc.BIN / 1e6, i1 * fc.BIN / 1e6
    pks = pk[(pk["center"] / 1e6 >= x0) & (pk["center"] / 1e6 <= x1)]
    axg.vlines(pks["center"] / 1e6, 2.15, 2.95, color=fc.C["chip"], lw=1.0)
    axg.text(x0, 3.05, "NapM peaks", fontsize=5.5, color=fc.C["chip"], va="bottom")
    gl = genes[(genes["end"] / 1e6 >= x0) & (genes["start"] / 1e6 <= x1)]
    for _, r in gl.iterrows():
        y = 1.6 if r["strand"] == "+" else 0.6
        cls = r["class_KOvsWT"]
        col = (fc.C["pos"] if cls == "detected_increase"
               else fc.C["neg"] if cls == "detected_decrease" else fc.C["ns"])
        axg.plot([r["start"] / 1e6, r["end"] / 1e6], [y, y], color=col, lw=1.4,
                 solid_capstyle="butt")
    # thin loop arcs
    for _, r in loops.iterrows():
        c1, c2 = r["center1"] / 1e6, r["center2"] / 1e6
        if c1 >= x0 and c2 <= x1:
            t = np.linspace(0, np.pi, 60)
            xc = (c1 + c2) / 2 + (c2 - c1) / 2 * np.cos(t)
            yc = 2.15 + 0.75 * np.sin(t)
            axg.plot(xc, yc, color="k", lw=0.5)
            axg.plot([c1, c2], [2.12, 2.12], "o", ms=2, mfc="none", mec="k",
                     mew=0.5)
    axg.text(x0, 1.75, "+ strand", fontsize=5, va="bottom", color="#555555")
    axg.text(x0, 0.75, "− strand", fontsize=5, va="bottom", color="#555555")
    axg.set_ylim(0.2, 3.6); axg.set_yticks([])
    axg.set_xlim(x0, x1)
    axg.set_xlabel(f"genome position (Mb) — {loc['name']}", fontsize=6)
    axg.spines["left"].set_visible(False)
    axg.text(0.0, -0.48, "selection: " + loc["crit"], transform=axg.transAxes,
             fontsize=5.3, color="#555555", va="top")
    fc.panel_label(ax_wt, panels[k], x=-0.18, y=1.10)
    sd_rows.append(dict(panel=panels[k], locus=loc["name"], i0=i0, i1=i1,
                        start=i0 * fc.BIN, end=i1 * fc.BIN, selection=loc["crit"],
                        n_peaks_in_window=len(pks), n_genes_in_window=len(gl)))

# shared colorbar
cax = fig.add_axes([0.32, 0.500, 0.36, 0.009])
sm = plt.cm.ScalarMappable(cmap=fc.CMAP_DIV, norm=norm)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("log2(O/E), balanced Hi-C, white = 1; grey = masked/missing",
             fontsize=6)
cb.ax.tick_params(labelsize=5.5); cb.outline.set_linewidth(0.5)
fc.save(fig, "SuppFig2_locus_browsers")
pd.DataFrame(sd_rows).to_csv(f"{fc.SD}/SuppFig2_locus_selection.csv", index=False)

# ================= SuppFig 3: latent descriptive =================
fig = plt.figure(figsize=(fc.FIGW1 * 1.6, 70 * fc.MM))
gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.10, right=0.97, top=0.80,
                      bottom=0.16)
axA = fig.add_subplot(gs[0])
d_lat = b["ins_latent"] - b["ins_WT"]
axA.axhline(0, color=fc.C["ns"], lw=0.4)
axA.bar(x := (b["bin"] * fc.BIN + 2500) / 1e6, d_lat, width=0.005,
        color=fc.C["latent"], alpha=0.8, lw=0)
axA.set_xlabel("genome position (Mb)")
axA.set_ylabel("Δ insulation (latent − WT)")
axA.set_xlim(0, fc.GENOME / 1e6)
fc.panel_label(axA, "A", x=-0.22)

axB = fig.add_subplot(gs[1])
axB.scatter(b["d_ins_KO_WT"], d_lat, s=2.5, color=fc.C["latent"], alpha=0.4,
            lw=0, rasterized=True)
axB.axhline(0, color=fc.C["ns"], lw=0.4); axB.axvline(0, color=fc.C["ns"], lw=0.4)
axB.set_xlabel("Δ insulation (KO − WT)")
axB.set_ylabel("Δ insulation (latent − WT)")
fc.panel_label(axB, "B", x=-0.22)
for ax in (axA, axB):
    ax.set_title("latent condition — DESCRIPTIVE ONLY", fontsize=6.5,
                 color=fc.C["latent"])
    for pos in ("top", "right", "bottom", "left"):
        ax.spines[pos].set_visible(True)
        ax.spines[pos].set_linestyle((0, (2, 2)))
        ax.spines[pos].set_color(fc.C["latent"])
fc.save(fig, "SuppFig3_latent_descriptive")
print("done supp")
