#!/usr/bin/env python3
"""Figure 2 — Heterogeneous response classes, no dominant direct class."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats
import fc

d = fc.load_tables()
b, g, gpairs = d["b"], d["genes"], d["gpairs"]
classes, stats2 = d["classes"], d["stats2"]

sd = fc.SourceData("Fig2")

fig = plt.figure(figsize=(fc.FIGW, 150 * fc.MM))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.40,
                      left=0.08, right=0.985, top=0.92, bottom=0.10,
                      height_ratios=[1, 0.9])

# ---------------- A: 3x3 classification glyph matrices ----------------
gsA = gs[0, 0].subgridspec(1, 2, wspace=0.55)
struct_order = ["detected", "supports_small", "cannot_determine"]
rna_order = ["detected", "supports_small", "cannot_determine"]
struct_lab = ["detected", "supports small", "undetermined"]
rna_lab = ["detected", "supports small", "undetermined"]
ct = classes.copy()
maxn = ct["n_bins"].max()
sd_A = []
for k, (bval, ttl) in enumerate([(True, "NapM-bound bins (n = 433)"),
                                 (False, "unbound bins (n = 450)")]):
    ax = fig.add_subplot(gsA[k])
    sub = ct[ct["chip_bound"] == bval]
    for i, s3 in enumerate(struct_order):
        for j, r3 in enumerate(rna_order):
            m = sub[(sub["struct3"] == s3) & (sub["rna3"] == r3)]
            n = int(m["n_bins"].sum())
            sd_A.append(dict(chip_bound=bval, struct3=s3, rna3=r3, n_bins=n))
            size = max(60, 900 * n / maxn)
            col = fc.C["chip"] if (s3 == "detected" and r3 == "detected") else "#9B939E"
            ax.scatter(j, 2 - i, s=size, color=col, alpha=0.85, lw=0)
            if n / maxn > 0.18:
                ax.text(j, 2 - i, str(n), ha="center", va="center", fontsize=5.5,
                        color="white")
            else:
                ax.text(j + 0.16, 2 - i + 0.16, str(n), ha="left", va="bottom",
                        fontsize=5.5, color=fc.C["text"])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.6, 2.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(rna_lab, fontsize=5.8, rotation=30,
                                                ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(struct_lab[::-1], fontsize=5.8)
    ax.set_title(ttl, fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    if k == 0:
        ax.set_ylabel("structure class (Δins)", fontsize=6.5)
        ax.set_xlabel("RNA class (KO/WT)", fontsize=6.5)
    else:
        ax.set_xlabel("RNA class (KO/WT)", fontsize=6.5)
fc.panel_label(fig.axes[-2], "A", x=-0.42, y=1.14)
sd.add("A_class_matrix", pd.DataFrame(sd_A))

# ---------------- B: RNA log2FC vs Δins ----------------
axB = fig.add_subplot(gs[0, 1])
gg = g.dropna(subset=["log2FC_KOvsWT", "bin_d_ins_KO_WT"]).copy()
gg["gene_bound"] = gg["binding_class"].isin(["promoter_bound", "gene_body_bound"])
matched_genes = set(gpairs["matched_gene"])
gg["is_matched"] = gg["gene_id"].isin(matched_genes)
rest = gg[~gg["gene_bound"] & ~gg["is_matched"]]
axB.scatter(rest["bin_d_ins_KO_WT"], rest["log2FC_KOvsWT"], s=2, color=fc.C["light"],
            lw=0, rasterized=True, label=f"other genes (n={len(rest)})")
mm = gg[gg["is_matched"]]
axB.scatter(mm["bin_d_ins_KO_WT"], mm["log2FC_KOvsWT"], s=3, color=fc.C["matched"],
            lw=0, rasterized=True, label=f"matched unbound (n={len(mm)})")
bb_ = gg[gg["gene_bound"]]
axB.scatter(bb_["bin_d_ins_KO_WT"], bb_["log2FC_KOvsWT"], s=3.5, color=fc.C["chip"],
            lw=0, alpha=0.8, rasterized=True, label=f"NapM-bound (n={len(bb_)})")
med_absd = b["d_ins_KO_WT"].abs().median()
axB.axhline(1, color=fc.C["ns"], lw=0.5, ls="--")
axB.axhline(-1, color=fc.C["ns"], lw=0.5, ls="--")
axB.axvline(med_absd, color=fc.C["ns"], lw=0.5, ls=":")
axB.axvline(-med_absd, color=fc.C["ns"], lw=0.5, ls=":")
axB.text(med_absd + 0.01, axB.get_ylim()[0], "", fontsize=5)
r_all = fc.stat(stats2, "Fig2d_spearman_rna_vs_dins_all")
r_b = fc.stat(stats2, "Fig2d_spearman_rna_vs_dins_bound")
r_m = fc.stat(stats2, "Fig2d_spearman_rna_vs_dins_matched")
axB.text(0.02, 0.98,
         f"Spearman all: ρ = {r_all['effect']:.3f}, p = {r_all['p']:.2f}\n"
         f"bound: ρ = {r_b['effect']:.3f}, p = {r_b['p']:.2f}; "
         f"matched: ρ = {r_m['effect']:.3f}, p = {r_m['p']:.2f}\n"
         f"dashed: |log2FC| = 1; dotted: median |Δins| = {med_absd:.2f}",
         transform=axB.transAxes, fontsize=5.8, va="top")
axB.set_xlabel("Δ insulation of the TSS bin (KO − WT)")
axB.set_ylabel("RNA log2FC (KO/WT)")
axB.legend(frameon=False, loc="lower right", markerscale=2.2, fontsize=6,
           borderaxespad=0.1)
fc.panel_label(axB, "B")
sd.add("B_rna_vs_dins", gg[["gene_id", "gene_name", "binding_class", "is_matched",
                            "log2FC_KOvsWT", "padj_KOvsWT", "class_KOvsWT",
                            "bin_d_ins_KO_WT"]])

# ---------------- C: bound vs matched genes ----------------
gsC = gs[1, 0].subgridspec(1, 2, wspace=0.55)
r_deg = fc.stat(stats2, "Fig2b_DEGfrac_bound_vs_matched_genes")
r_fc = fc.stat(stats2, "Fig2b_abslog2FC_bound_vs_matched_genes")
gm = gpairs.merge(g[["gene_id", "log2FC_KOvsWT", "class_KOvsWT"]],
                  left_on="bound_gene", right_on="gene_id").drop(columns=["gene_id"])
gm = gm.merge(g[["gene_id", "log2FC_KOvsWT", "class_KOvsWT"]],
              left_on="matched_gene", right_on="gene_id",
              suffixes=("_bound", "_matched"))
ok = gm["class_KOvsWT_bound"].notna() & gm["class_KOvsWT_matched"].notna()
gm = gm[ok]
deg_b = gm["class_KOvsWT_bound"].isin(["detected_increase", "detected_decrease"])
deg_m = gm["class_KOvsWT_matched"].isin(["detected_increase", "detected_decrease"])

axC1 = fig.add_subplot(gsC[0])
fracs = [deg_b.mean(), deg_m.mean()]
cis = [sstats.binomtest(int(x.sum()), len(x)).proportion_ci(0.05)
       for x in (deg_b, deg_m)]
for i, (fr, ci, col) in enumerate(zip(fracs, cis, [fc.C["chip"], fc.C["matched"]])):
    axC1.errorbar([i], [fr * 100],
                  yerr=[[(fr - ci.low) * 100], [(ci.high - fr) * 100]],
                  fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.0, capsize=2.5)
axC1.set_xticks([0, 1]); axC1.set_xticklabels(["NapM-bound", "matched"], fontsize=6.5)
axC1.set_ylabel("DEG fraction (%)")
axC1.set_ylim(0, 30)
axC1.set_title(f"DEG rate: {int(deg_b.sum())}/825 vs {int(deg_m.sum())}/825\n"
               f"OR = {r_deg['effect']:.2f} [{r_deg['ci_lo']:.2f}, {r_deg['ci_hi']:.2f}], "
               f"Fisher p = {r_deg['p']:.2f}", fontsize=6)
fc.panel_label(axC1, "C", x=-0.42, y=1.14)

axC2 = fig.add_subplot(gsC[1])
for vals, col, lab in [(gm["log2FC_KOvsWT_bound"].abs().values, fc.C["chip"], "NapM-bound"),
                       (gm["log2FC_KOvsWT_matched"].abs().values, fc.C["matched"], "matched")]:
    v = np.sort(vals); y = np.arange(1, len(v) + 1) / len(v)
    axC2.plot(v, y, color=col, lw=1.1, label=lab)
axC2.set_xlabel("|log2FC| (KO/WT)")
axC2.set_ylabel("ECDF")
axC2.set_xlim(0, 3)
axC2.legend(frameon=False, loc="lower right", fontsize=6)
axC2.set_title(f"median |log2FC| diff = {r_fc['effect']:.3f} "
               f"[{r_fc['ci_lo']:.3f}, {r_fc['ci_hi']:.3f}]\n"
               f"paired Wilcoxon p = {r_fc['p']:.2f}", fontsize=6)
sd.add("C_gene_pairs", gm[["bound_gene", "matched_gene", "log2FC_KOvsWT_bound",
                           "log2FC_KOvsWT_matched", "class_KOvsWT_bound",
                           "class_KOvsWT_matched"]])

# ---------------- D: joint enrichment forest ----------------
axD = fig.add_subplot(gs[1, 1])
tests = [("Fig2c_OR_boundSC_vs_unboundSC",
          "bound vs unbound | structure-changed\n(45/227 vs 224/998)"),
         ("Fig2c_OR_boundSC_vs_boundSS",
          "structure-changed vs stable | bound\n(45/227 vs 109/598)"),
         ("Fig2c_OR_boundSS_vs_unboundSS",
          "bound vs unbound | structure-stable\n(109/598 vs 442/2258)")]
rows = []
for i, (t, lab) in enumerate(tests):
    r = fc.stat(stats2, t)
    y = len(tests) - 1 - i
    axD.errorbar([r["effect"]], [y], xerr=[[r["effect"] - r["ci_lo"]],
                 [r["ci_hi"] - r["effect"]]], fmt="o", ms=4, color="#6E7072",
                 ecolor="#6E7072", elinewidth=1.0, capsize=2.5)
    axD.text(3.3, y + 0.38, f"{r['effect']:.2f} [{r['ci_lo']:.2f}\u2013{r['ci_hi']:.2f}], "
                     f"p = {r['p']:.2f}", fontsize=6, va="center", ha="right")
    rows.append(dict(contrast=t, OR=r["effect"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                     p=r["p"], label=lab.replace("\n", " ")))
axD.axvline(1.0, color=fc.C["ns"], lw=0.6)
axD.set_ylim(-0.6, len(tests) - 1 + 0.75)
axD.set_yticks(range(len(tests)))
axD.set_yticklabels([t[1] for t in tests][::-1], fontsize=6.2)
axD.set_xscale("log"); axD.set_xlim(0.4, 3.4)
axD.set_xticks([0.5, 1, 2, 3]); axD.set_xticklabels(["0.5", "1", "2", "3"])
axD.minorticks_off()
axD.set_xlabel("odds ratio for DEG (log scale)")
fc.panel_label(axD, "D")
sd.add("D_joint_enrichment", pd.DataFrame(rows))

fc.save(fig, "Fig2_response_classes")
sd.write()
