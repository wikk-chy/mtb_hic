#!/usr/bin/env python3
"""Figure 1 — NapM occupancy is widespread but does not predict local remodeling."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fc

d = fc.load_tables()
b, pk, loops = d["b"], d["peaks"], d["loops"]
meta, stats13 = d["meta"], d["stats13"]
x_mb = (b["bin"] * fc.BIN + fc.BIN / 2) / 1e6

sd = fc.SourceData("Fig1")

fig = plt.figure(figsize=(fc.FIGW, 165 * fc.MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.42,
                      left=0.075, right=0.985, top=0.94, bottom=0.09)

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

# loop arcs: half-ellipse between anchors, height ∝ sqrt(genomic span);
# loops with both anchors NapM-bound (17, primary definition) in purple
span = (loops["bin2"] - loops["bin1"]).values.astype(float)
arc_h = np.sqrt(span / span.max())          # sqrt compression, max height = 1
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
axL.set_ylabel("loops (n=41)\narc height ~\n√span", fontsize=6.5)
axL.plot([], [], color=fc.C["chip"], lw=0.9,
         label="both anchors NapM-bound (n=17)")
axL.plot([], [], color="#8C8C8C", lw=0.6, label="other loops (n=24)")
axL.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.4,
           borderaxespad=0.1, labelspacing=0.25)

di = b["d_ins_KO_WT"].values
axI.axhline(0, color=fc.C["ns"], lw=0.4, zorder=0)
axI.bar(x_mb, di, width=0.005, color=np.where(di >= 0, fc.C["pos"], fc.C["neg"]),
        alpha=0.85, lw=0)
axI.set_ylabel("Δ insulation\n(KO − WT)", fontsize=6.5)

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
fc.panel_label(axP, "A")

sd.add("A_peaks", pk[["chrom", "start", "end", "center", "target_gene", "score", "vpm"]])
sd.add("A_loop_anchors", loops[["loop_id", "center1", "center2", "bin1", "bin2",
                                "anchor1_bound", "anchor2_bound"]])
sd.add("A_dins_perbin", b[["bin", "start", "end", "d_ins_KO_WT"]])
sd.add("A_rna_perbin", b[["bin", "start", "end", "rna_mean_log2FC_KOvsWT"]])

# ---------------- B: peak-centred metaprofile ----------------
axB = fig.add_subplot(gs[1, 0])
axB.fill_between(meta["offset_kb"], meta["null_lo95"], meta["null_hi95"],
                 color=fc.C["bg"], lw=0, label="circular-shift null 95% (10,000)")
axB.plot(meta["offset_kb"], meta["null_mean"], color="#8a8a8a", lw=0.8, ls="--",
         label="null mean")
axB.plot(meta["offset_kb"], meta["observed"], color=fc.C["chip"], lw=1.3,
         label="observed (peak-centred)")
axB.axvline(0, color=fc.C["chip"], lw=0.5, ls=":")
r5 = fc.stat(stats13, "Fig1B_window_pm5kb")
r25 = fc.stat(stats13, "Fig1B_window_pm25kb")
axB.text(0.03, 0.04, f"±5 kb: obs {r5['observed']:.3f} vs null {r5['null_mean']:.3f}\n"
                     f"circular-shift p = {r5['p']:.2f} (depleted side)\n"
                     f"±25 kb window saturates (p = {r25['p']:.2f}, uninformative)",
         transform=axB.transAxes, fontsize=6, va="bottom")
axB.set_xlabel("offset from NapM peak centre (kb)")
axB.set_ylabel("mean |Δ insulation|")
axB.legend(frameon=False, loc="upper right", handlelength=1.4)
fc.panel_label(axB, "B")
sd.add("B_metaprofile", meta)

# ---------------- C: forest of matched/shift tests ----------------
rows = []
rows.append(dict(key="a", label="|Δ insulation| (paired), p = 0.53",
                 eff=fc.stat(stats13, "Fig1Da_paired_absdins"), kind="ci", ref=0.0,
                 ptxt=lambda r: f"shift p = {r['p']:.2f}"))
rs = fc.stat(stats13, "Fig1Db_paired_dlocalcont_signed")
rows.append(dict(key="b1", label="Δ local contact (signed), p = 0.32*",
                 eff=rs, kind="ci", ref=0.0,
                 ptxt=lambda r: f"shift p = 0.32; naive p = 0.047"))
rows.append(dict(key="b2", label="|Δ local contact|, p = 0.73",
                 eff=fc.stat(stats13, "Fig1Db_paired_abslocalcontmag"), kind="ci",
                 ref=0.0, ptxt=lambda r: f"shift p = {r['p']:.2f}"))
rl = fc.stat(stats13, "Fig1Dc_loops_both_bound")
rows.append(dict(key="c", label="loops both anchors bound, p = 0.011",
                 eff=rl, kind="nullband", band=(5, 16), accent=True,
                 ptxt=lambda r: f"17 vs null 9.9; p = 0.011"))
rd = fc.stat(stats13, "Fig1Dd_bound_x_structure_detected")
rows.append(dict(key="d", label="bound × structure (OR), p = 0.27",
                 eff=rd, kind="or", ref=1.0,
                 ptxt=lambda r: f"Fisher p = 0.27; shift p = 0.89"))
# e: NapM occupancy vs CID boundaries — formal farther-than-null test,
#    median statistic (robust); mean-based statistic is marginal and tail-driven
cidt = pd.read_csv(f"{fc.RA}/06_permutation_tests/fig1c_cid_distance_test.csv")
cm = cidt[cidt["statistic"] == "median"].iloc[0]
cmean = cidt[cidt["statistic"] == "mean"].iloc[0]
rows.append(dict(key="e", label="distance to CID boundary (median), p = 0.42",
                 eff=dict(observed=cm["obs"], null_mean=cm["null_med"],
                          effect=cm["obs"] - cm["null_med"],
                          ci_lo=np.nan, ci_hi=np.nan, p=cm["p_farther"],
                          test="one-sided farther, median statistic",
                          null_model="circular-shift of bound mask (10,000)"),
                 kind="nullband", band=(cm["lo"], cm["hi"]),
                 ptxt=lambda r: "obs 13 vs null 12.4; farther p = 0.42"))

gsC = gs[1, 1].subgridspec(len(rows), 1, hspace=1.05)
forest_rows = []
for k, row in enumerate(rows):
    ax = fig.add_subplot(gsC[k])
    r = row["eff"]
    accent = row.get("accent", False)
    col = fc.C["chip"] if accent else "#6E7072"
    if row["kind"] == "ci":
        ax.axvline(row["ref"], color=fc.C["ns"], lw=0.6, zorder=0)
        ax.errorbar([r["effect"]], [0], xerr=[[r["effect"] - r["ci_lo"]],
                    [r["ci_hi"] - r["effect"]]], fmt="o", ms=3.5, color=col,
                    ecolor=col, elinewidth=1.0, capsize=2)
        pad = 0.45 * (r["ci_hi"] - r["ci_lo"])
        ax.set_xlim(r["ci_lo"] - pad, r["ci_hi"] + pad)
    elif row["kind"] == "or":
        ax.axvline(1.0, color=fc.C["ns"], lw=0.6, zorder=0)
        ax.errorbar([r["effect"]], [0], xerr=[[r["effect"] - r["ci_lo"]],
                    [r["ci_hi"] - r["effect"]]], fmt="o", ms=3.5, color=col,
                    ecolor=col, elinewidth=1.0, capsize=2)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 2.0)
        ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
        ax.minorticks_off()
    else:  # nullband
        lo, hi = row["band"]
        ax.axvspan(lo, hi, color=fc.C["bg"], lw=0, zorder=0)
        ax.axvline(r["null_mean"], color="#8a8a8a", lw=0.7, ls="--", zorder=1)
        ax.plot([r["observed"]], [0], "o", ms=4, color=col, zorder=2)
        span = max(hi - lo, abs(r["observed"] - lo), abs(r["observed"] - hi))
        ax.set_xlim(min(lo, r["observed"]) - 0.35 * span,
                    max(hi, r["observed"]) + 0.35 * span)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, 0.7)
    ax.text(0.0, 1.18, f"{row['key']}  {row['label']}", transform=ax.transAxes,
            fontsize=5.8, ha="left", va="bottom", clip_on=False,
            color=col if accent else fc.C["text"],
            fontweight="bold" if accent else "normal")
    ax.tick_params(axis="x", labelsize=5.5, pad=1)
    forest_rows.append(dict(panel_row=row["key"], quantity=row["label"],
                            observed=r["observed"], null_mean=r["null_mean"],
                            effect=r["effect"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                            p=r["p"], test=r["test"], null_model=r["null_model"]))
fc.panel_label(fig.axes[-len(rows)], "C", x=-0.32, y=1.30)
fig.axes[-1].text(0.0, -0.75, "* naive paired Wilcoxon p = 0.047; p from circular-shift\n"
                  f"(10,000) unless stated. Row e: median statistic; mean-based\n"
                  f"statistic is marginal (farther p = {cmean['p_farther']:.3f}), tail-driven",
                  transform=fig.axes[-1].transAxes, fontsize=5.6, va="top")
sd.add("C_forest", pd.DataFrame(forest_rows))

# ---------------- D: loop functional analysis ----------------
# Do loops with both anchors NapM-bound preferentially change after deletion?
loe = pd.read_csv(f"{fc.RA}/06_permutation_tests/loop_oe_wt_ko_check.csv")
loe = loe.dropna(subset=["d_log2OE"])
gb = loe.loc[loe["both_bound"], "d_log2OE"].values
ub = loe.loc[~loe["both_bound"], "d_log2OE"].values
rngD = np.random.default_rng(42)
valsD = loe["d_log2OE"].values
obsD = valsD[loe["both_bound"].values].mean() - valsD[~loe["both_bound"].values].mean()
nbD = int(loe["both_bound"].sum())
permsD = np.empty(10000)
for i in range(10000):
    pidx = rngD.permutation(len(valsD))
    permsD[i] = valsD[pidx[:nbD]].mean() - valsD[pidx[nbD:]].mean()
pD = (np.sum(np.abs(permsD) >= abs(obsD)) + 1) / 10001
bootsD = np.empty(10000)
for i in range(10000):
    bootsD[i] = (rngD.choice(gb, len(gb), replace=True).mean()
                 - rngD.choice(ub, len(ub), replace=True).mean())
ciD = np.percentile(bootsD, [2.5, 97.5])

gsD = gs[1, 2].subgridspec(2, 1, hspace=0.55, height_ratios=[1, 1])
# D-top: WT vs KO loop O/E scatter
axD1 = fig.add_subplot(gsD[0])
wt_l = np.log2(loe["WT_mean"].values)
ko_l = np.log2(loe["KO_mean"].values)
mD = loe["both_bound"].values
lims = [min(wt_l.min(), ko_l.min()) - 0.1, max(wt_l.max(), ko_l.max()) + 0.1]
axD1.plot(lims, lims, color="0.5", lw=0.6, ls="--", zorder=1)
axD1.scatter(wt_l[~mD], ko_l[~mD], s=13, color="#8C8C8C", alpha=0.85, lw=0,
             label=f"other loops (n={int((~mD).sum())})", zorder=2)
axD1.scatter(wt_l[mD], ko_l[mD], s=13, color=fc.C["chip"], alpha=0.9, lw=0,
             label=f"both anchors NapM-bound (n={int(mD.sum())})", zorder=3)
axD1.set_xlim(lims); axD1.set_ylim(lims)
axD1.set_xlabel("WT loop O/E (log2)", fontsize=6.5)
axD1.set_ylabel("napM-KO loop O/E (log2)", fontsize=6.5)
axD1.tick_params(labelsize=5.5)
axD1.legend(frameon=False, fontsize=5.6, loc="lower right", handletextpad=0.1,
            borderpad=0.1, labelspacing=0.25)
axD1.text(0.02, 0.98, "L35/L36 excluded: ICE-uncovered\nanchor bins (technical)",
          transform=axD1.transAxes, fontsize=5.4, va="top", color="#6E7072")
fc.panel_label(axD1, "D", x=-0.5)
# D-bottom: effect-size comparison
axD2 = fig.add_subplot(gsD[1])
axD2.axhline(0, color="0.5", lw=0.5, ls="--")
j1 = rngD.uniform(-0.14, 0.14, len(gb))
j2 = rngD.uniform(-0.14, 0.14, len(ub))
axD2.scatter(0 + j1, gb, s=12, color=fc.C["chip"], alpha=0.85, lw=0, zorder=3)
axD2.scatter(1 + j2, ub, s=12, color="#8C8C8C", alpha=0.85, lw=0, zorder=3)
for x0, v in [(0, gb), (1, ub)]:
    axD2.plot([x0 - 0.26, x0 + 0.26], [np.median(v)] * 2, color="k", lw=1.1, zorder=4)
axD2.set_xlim(-0.55, 1.55)
axD2.set_ylim(-1.35, 1.2)
axD2.set_yticks([-1, -0.5, 0, 0.5])
axD2.set_xticks([0, 1])
axD2.set_xticklabels(["bound", "other"], fontsize=6)
axD2.set_ylabel("Δ log2 loop O/E\n(KO / WT)", fontsize=6)
axD2.tick_params(labelsize=5.5)
axD2.text(0.03, 0.97, f"mean diff = {obsD:+.3f} [{ciD[0]:+.3f}, {ciD[1]:+.3f}]\n"
                      f"label-permutation p = {pD:.2f} (10,000)",
          transform=axD2.transAxes, fontsize=5.8, va="top", ha="left")
sd.add("D_loop_wt_vs_ko_scatter", loe[["loop_id", "distance_kb", "both_bound",
                                       "WT_mean", "KO_mean", "d_log2OE"]])
sd.add("D_loop_strength_stats", pd.DataFrame([dict(
    n_bound=len(gb), n_other=len(ub), mean_diff=obsD,
    boot_ci_lo=ciD[0], boot_ci_hi=ciD[1], permutation_p=pD,
    excluded_loops="L35,L36", exclusion_reason="ICE-uncovered anchor bins near 3.93 Mb",
    sensitivity_3x3="bound -0.265 vs other -0.244, diff -0.022 [-0.186,+0.144], p=0.81 (n=39)")]))

fc.save(fig, "Fig1_NapM_occupancy_vs_local_remodeling")
sd.write()
