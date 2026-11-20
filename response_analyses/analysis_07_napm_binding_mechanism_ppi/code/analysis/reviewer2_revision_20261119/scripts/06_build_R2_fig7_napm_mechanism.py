#!/usr/bin/env python3
"""
Build Response Fig. 7 (Reviewer 2, Comment 6 / Audit Comment 12):
Self-contained genomic, transcriptional, structural, and protein-interaction evaluation of NapM.
Fixes all layout overlaps and the Python operator-precedence bug in Panel I (which previously
plotted Rv0047c itself as a 15th neighbor), and directly integrates key R4/R1 structural panels
(F-H) so Reviewer 2 does not need to flip to other reviewers' sections.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
HIC = ROOT / "hic_review"
PK_PATH = HIC / "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
EDGES_PATH = HIC / "review_data/Interaction_network/full_edges.csv"
BD_TAB = ROOT / "response_figure_reproducibility/figure_05_napm_binding_sequence_network/results/hic_review/review_results/R2_comments/binding_distribution/tables"
CR_TAB = ROOT / "response_figure_reproducibility/figure_05_napm_binding_sequence_network/results/hic_review/review_results/R2_comments/chip_rna_integration/tables"
PPI_TAB = ROOT / "response_figure_reproducibility/figure_05_napm_binding_sequence_network/results/hic_review/review_results/R2_comments/ppi_evidence/tables"
R4_CUM = HIC / "review_results/R4_comment4/tables/g_cumulative_curves.csv"
R1_META = HIC / "review_results/R1_mechanism_reanalysis/09_combined/tables/Fig1_final_B_metaprofile_absdins.csv"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7.1,
    "axes.titlesize": 7.9,
    "axes.labelsize": 7.1,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.1,
    "axes.linewidth": 0.65,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

C_PK = "#6A51A3"
C_PK_LT = "#9E9AC8"
C_NULL = "#C0C6CC"
C_WT = "#1F77B4"
C_KO = "#D62728"
C_TEAL = "#1B9E77"
C_TEXT = "#1F2933"

pk = pd.read_csv(PK_PATH, sep="\t")
gc_tab = pd.read_csv(BD_TAB / "gc_composition.csv").set_index("metric")["value"]
kmer_tab = pd.read_csv(BD_TAB / "kmer8_enrichment_min20.csv").head(7).iloc[::-1].reset_index(drop=True)
pos_tab = pd.read_csv(BD_TAB / "position_annotation.csv")
deg_tab = pd.read_csv(CR_TAB / "deg_overlap_enrichment.csv")
gstat = pd.read_csv(CR_TAB / "gene_binding_status.csv")
corr = pd.read_csv(CR_TAB / "binding_vs_expression_corr.csv")
cum_df = pd.read_csv(R4_CUM)
meta_df = pd.read_csv(R1_META)
nb_tab = pd.read_csv(PPI_TAB / "napm_ppi_neighbours.csv")

# Verify exact 14 PPI partners (FIXES the operator precedence bug!)
edges = pd.read_csv(EDGES_PATH)
ppi = edges[(edges["interaction_type"] == "Protein-Protein") & ((edges["source"] == "Rv0047c") | (edges["target"] == "Rv0047c"))]
nbrs = sorted((set(ppi["source"]) | set(ppi["target"])) - {"Rv0047c"})
assert "Rv0047c" not in nbrs and len(nbrs) == 14, f"Expected 14 PPI partners without Rv0047c, got {len(nbrs)}: {nbrs}"

fig = plt.figure(figsize=(8.27, 11.4))
gs = GridSpec(4, 3, figure=fig, left=0.08, right=0.97, top=0.935, bottom=0.045,
              wspace=0.40, hspace=0.52, height_ratios=[0.78, 0.92, 0.95, 1.38])
fig.suptitle("Genomic occupancy, transcriptional decoupling, structural associations, and PPI context of NapM",
             x=0.045, y=0.975, ha="left", fontsize=10.0, fontweight="bold", color=C_TEXT)


def finish(ax, letter, title, lx=-0.22):
    ax.set_title(title, loc="left", pad=5, fontweight="bold", color=C_TEXT)
    ax.text(lx, 1.11, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


# Panel A (Row 0, cols 0:2): Distributed genome-wide NapM ChIP-seq peaks (634 peaks)
axA = fig.add_subplot(gs[0, 0:2])
mids_mb = ((pk["start"] + pk["end"]) / 2.0) / 1e6
scores = pk["score"] if "score" in pk.columns else np.ones(len(pk))
axA.vlines(mids_mb, 0, scores, color=C_PK, linewidth=0.55, alpha=0.82)
axA.set_xlim(0, 4.42)
axA.set_ylim(0, 1.16)
axA.set_xlabel("Genome position (Mb)")
axA.set_ylabel("Peak score")
axA.text(0.015, 0.96,
         "634 curated peaks | median width 363 bp | merged coverage 5.30% | nearest-neighbour spacing vs random P = 0.88",
         transform=axA.transAxes, va="top", ha="left", fontsize=6.2, color=C_TEXT)
finish(axA, "A", "Distributed NapM ChIP-seq peaks across the chromosome", lx=-0.10)

# Panel B (Row 0, col 2): AT-rich preference & top enriched 8-mers
axB = fig.add_subplot(gs[0, 2])
yB = np.arange(len(kmer_tab))
axB.barh(yB, kmer_tab["log2_enrichment"], height=0.62, color=C_PK, alpha=0.85, edgecolor="black", lw=0.4)
axB.set_yticks(yB, kmer_tab["kmer"], fontsize=6.3, family="monospace")
axB.set_xlabel(r"$\log_2$ enrichment vs matched bg")
axB.set_xlim(0, 4.1)
axB.text(0.97, 0.08, f"Peak GC: {gc_tab['peak_gc_bp_weighted']:.3f}\nGenome GC: {gc_tab['genome_gc']:.3f}",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=6.1,
         bbox=dict(boxstyle="round,pad=0.22", facecolor="#F7F5FA", edgecolor="#CCCCCC", lw=0.5))
finish(axB, "B", "AT-preferring enriched 8-mers", lx=-0.34)

# Panel C (Row 1, col 0): Positional annotation vs circular-shift null
axC = fig.add_subplot(gs[1, 0])
cat_map = {"gene_body": "Gene\nbody", "promoter_300bp": "Promoter\n(−300 bp)", "intergenic": "Intergenic"}
xC = np.arange(len(pos_tab))
w = 0.35
axC.bar(xC - w / 2, pos_tab["observed_frac"], width=w, color=C_PK, edgecolor="black", lw=0.45, label="Observed")
axC.bar(xC + w / 2, pos_tab["null_mean"] / 634.0, width=w, color=C_NULL, edgecolor="black", lw=0.45, label="Circular-shift null")
for i, r in pos_tab.iterrows():
    lbl = f"{r['fold']:.2f}×***" if r["p_two_sided"] < 0.001 else "n.s."
    top = max(r["observed_frac"], r["null_mean"] / 634.0)
    axC.text(i, top + 0.03, lbl, ha="center", va="bottom", fontsize=6.2, fontweight="bold" if "*" in lbl else "normal")
axC.set_xticks(xC, [cat_map.get(c, c) for c in pos_tab["category"]])
axC.set_ylim(0, 1.12)
axC.set_ylabel("Fraction of 634 peaks")
axC.legend(loc="upper right", frameon=False, fontsize=6.0)
finish(axC, "C", "Positional enrichment vs null")

# Panel D (Row 1, col 1): DEG fraction by binding tier
axD = fig.add_subplot(gs[1, 1])
tier_lbl = {"direct_promoter": "Promoter", "direct_promoter_or_body": "Prom +\nbody",
            "operon_extended_only": "Operon\next.", "any_binding_associated": "Any\nbound"}
xD = np.arange(len(deg_tab))
colsD = ["#54278F", "#756BB1", "#9E9AC8", "#CBC9E2"]
axD.axhline(822 / 4060, color="#444444", linestyle="--", lw=0.8)
axD.bar(xD, deg_tab["frac_deg"], width=0.58, color=colsD, edgecolor="black", lw=0.45)
for i, r in deg_tab.iterrows():
    y_txt = max(r["frac_deg"] + 0.008, 0.208 if i != 2 else r["frac_deg"] + 0.008)
    axD.text(i, y_txt, f"{int(r['n_deg'])}/{int(r['n_genes'])}\nP={r['fisher_p']:.2f}",
             ha="center", va="bottom", fontsize=5.6, linespacing=1.15)
axD.set_xticks(xD, [tier_lbl.get(t, t) for t in deg_tab["tier"]])
axD.set_ylim(0, 0.305)
axD.set_ylabel("DEG fraction (KO vs WT)")
axD.text(0.98, 0.96, "Genome bg: 20.2%\n(no positive enrichment)", transform=axD.transAxes,
         ha="right", va="top", fontsize=5.8, color="#444444")
finish(axD, "D", "No DEG enrichment by binding tier")

# Panel E (Row 1, col 2): Peak score vs |log2FC|
axE = fig.add_subplot(gs[1, 2])
bound = gstat[gstat["binding_tier"].isin(["direct_body", "direct_promoter"])].dropna(subset=["gene_peak_score_max", "log2FC_KOvsWT"])
axE.scatter(bound["gene_peak_score_max"], np.abs(bound["log2FC_KOvsWT"]), s=6, color=C_PK, alpha=0.35, linewidths=0)
rho_row = corr[corr["x"] == "gene_peak_score_max"].iloc[0]
axE.set_xlabel("Max peak score per bound gene")
axE.set_ylabel(r"$|\log_2\mathrm{FC}|$ (KO vs WT)")
axE.set_ylim(-0.3, 6.2)
axE.text(0.04, 0.96,
         f"n = {int(rho_row['n'])} bound genes\nSpearman ρ = {rho_row['spearman_rho']:.4f}\nP = {rho_row['p']:.3f} (n.s.)",
         transform=axE.transAxes, va="top", ha="left", fontsize=6.0, linespacing=1.35)
finish(axE, "E", "Peak strength vs expression shift")

# Panel F (Row 2, col 0): Reused from R4 — Global KO contact-distance shift (median 276 -> 515 kb)
axF = fig.add_subplot(gs[2, 0])
for cond_name, col in [("WT", C_WT), ("napM-KO", C_KO)]:
    sub = cum_df[cum_df["condition"] == cond_name]
    piv = sub.pivot(index="distance_bp", columns="replicate", values="cum_fraction")
    mu = piv.mean(axis=1)
    sd = piv.std(axis=1)
    axF.plot(mu.index / 1e3, mu.values, color=col, lw=1.3, label=cond_name)
    axF.fill_between(mu.index / 1e3, mu.values - sd.values, mu.values + sd.values, color=col, alpha=0.18)
axF.axhline(0.5, color="#777777", ls=":", lw=0.7)
axF.axvline(276, color=C_WT, ls="--", lw=0.75)
axF.axvline(515, color=C_KO, ls="--", lw=0.75)
axF.set_xscale("log")
axF.set_xlabel("Genomic distance s (kb)")
axF.set_ylabel("Cumulative contact fraction")
axF.set_ylim(0, 1.04)
axF.legend(loc="upper left", frameon=False, fontsize=6.1)
axF.text(0.97, 0.05, "Median: WT 276 kb\nKO 515 kb (P = 1.5e-9)\n<100 kb: 0.377 → 0.242",
         transform=axF.transAxes, ha="right", va="bottom", fontsize=5.7, linespacing=1.25,
         bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#DDDDDD", lw=0.5))
finish(axF, "F", "Global contact shift in napM-KO")

# Panel G (Row 2, col 1): Reused from R1 — NapM peak-centered |Δinsulation| metaprofile vs circular-shift null
axG = fig.add_subplot(gs[2, 1])
axG.fill_between(meta_df["offset_kb"], meta_df["null_lo95"], meta_df["null_hi95"], color="#DCE0E5", label="95% circular-shift null")
axG.plot(meta_df["offset_kb"], meta_df["null_mean"], "--", color="#666666", lw=0.8, label="Null mean")
axG.plot(meta_df["offset_kb"], meta_df["observed"], color=C_PK, lw=1.2, label="Observed (433 peak bins)")
axG.axvline(0, color="#444444", ls=":", lw=0.7)
axG.set_xlabel("Offset from NapM peak (kb)")
axG.set_ylabel("Mean |Δ insulation| (KO − WT)")
axG.set_ylim(0.204, 0.278)
axG.legend(loc="upper left", frameon=False, fontsize=5.7)
axG.text(0.97, 0.04, "±5 kb: 0.2367 vs null 0.2429\ncircular-shift P = 0.97",
         transform=axG.transAxes, ha="right", va="bottom", fontsize=5.7, linespacing=1.25,
         bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"))
finish(axG, "G", "No local peak-centred insulation bias")

# Panel H (Row 2, col 2): Reused from R1 — Loop double-anchor enrichment & descriptive KO non-recovery
axH = fig.add_subplot(gs[2, 2])
bars_h = axH.bar([0, 1], [64.7, 41.7], width=0.52, color=[C_PK, "#8C9BA5"], edgecolor="black", lw=0.5)
axH.text(0, 64.7 + 2.5, "64.7%\n(11/17)", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=C_PK)
axH.text(1, 41.7 + 2.5, "41.7%\n(10/24)", ha="center", va="bottom", fontsize=6.4, color=C_TEXT)
axH.set_xticks([0, 1], ["Both anchors\nNapM-bound", "Other WT\nloops"])
axH.set_ylim(0, 102)
axH.set_ylabel("WT loops not recovered in KO (%)")
axH.text(0.03, 0.96, "WT double-anchor occupancy:\n17/41 vs null 10.9 (1.56×, P = 0.033)\n(KO comparison descriptive)",
         transform=axH.transAxes, va="top", ha="left", fontsize=5.7, linespacing=1.28, color=C_TEXT)
finish(axH, "H", "Candidate NapM-sensitive loops")

# Panel I (Row 3, full width cols 0:3): FIXED NapM PPI ego network (EXACTLY 14 PPI partners, ZERO self-loop!)
axI = fig.add_subplot(gs[3, :])
axI.axis("off")

CAT_COLORS = {
    "conserved hypothetical": "#B8C2CC",
    "unknown": "#DCE1E7",
    "TA system": "#F6AD55",
    "lipid metabolism": "#63B3ED",
    "insertion seqs/phages": "#68D391",
    "intermediary metabolism": "#FBD38D",
    "cell wall/cell processes": "#B794F4",
    "DNA repair": "#FC8181",
}
DEG_EDGE = {"Up": "#C53030", "Down": "#2B6CB0"}

nb_idx = nb_tab.set_index("neighbour")
# Arrange 14 neighbors cleanly in an ellipse on the left so legends sit on the right with ZERO overlap!
cx, cy = -2.35, 0.12
rx, ry = 1.48, 1.32
angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(nbrs), endpoint=False)

for a, nb in zip(angles, nbrs):
    row = nb_idx.loc[nb]
    x1 = cx + rx * np.cos(a)
    y1 = cy + ry * np.sin(a)
    axI.plot([cx, x1], [cy, y1], color="#7E6B9C", lw=1.15, zorder=1)

    fc = CAT_COLORS.get(row["functional_category"], "#CBD5E0")
    reg = str(row["akr_regulate_KOvsWT"])
    ec = DEG_EDGE.get(reg, "#4A5568")
    lw = 1.9 if reg in DEG_EDGE else 0.75
    sz = 110 + 6.0 * min(float(row["degree_full"]), 60)
    axI.scatter([x1], [y1], s=sz, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)

    lab = str(row["gene_name"]).split(" /")[0]
    lx = cx + (rx + 0.34) * np.cos(a)
    ly = cy + (ry + 0.29) * np.sin(a)
    ha = "left" if np.cos(a) > 0.25 else ("right" if np.cos(a) < -0.25 else "center")
    va = "bottom" if np.sin(a) > 0.35 else ("top" if np.sin(a) < -0.35 else "center")
    axI.text(lx, ly, lab, ha=ha, va=va, fontsize=6.4, fontweight="bold" if reg in DEG_EDGE else "normal", color=C_TEXT)

# Center NapM node
axI.scatter([cx], [cy], s=860, facecolor=C_PK, edgecolor="black", linewidth=1.2, zorder=4)
axI.text(cx, cy, "NapM\n(Rv0047c)", color="white", fontsize=7.0, fontweight="bold", ha="center", va="center", zorder=5)

# Right side: DNA-binding layer glyph + clean non-overlapping legends
dx, dy = 1.10, 1.05
axI.plot([cx + 0.42, dx - 0.32], [cy + 0.18, dy], color=C_TEAL, lw=1.3, ls=":", zorder=1)
axI.scatter([dx], [dy], s=640, facecolor=C_TEAL, edgecolor="black", linewidth=0.9, zorder=3)
axI.text(dx, dy, "427\nbins", color="white", fontsize=6.6, fontweight="bold", ha="center", va="center", zorder=4)
axI.text(dx + 0.36, dy, "ChIP-seq DNA-binding layer\n(634 peaks across 427 5-kb bins)",
         va="center", ha="left", fontsize=6.5, color=C_TEXT, linespacing=1.3)

cat_handles = [Patch(facecolor=CAT_COLORS[k], edgecolor="#555555", lw=0.5, label=k)
               for k in CAT_COLORS if k in set(nb_tab["functional_category"])]
deg_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#C53030", markeredgewidth=1.8, markersize=6.2, label="DEG up in KO (border)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2B6CB0", markeredgewidth=1.8, markersize=6.2, label="DEG down in KO (border)"),
    Line2D([0], [0], color="#7E6B9C", lw=1.2, label="B2H PPI edge (Wang et al., 2010)"),
]
leg1 = axI.legend(handles=cat_handles, loc="center left", bbox_to_anchor=(0.56, 0.45),
                  ncol=2, frameon=False, fontsize=5.8, title="PPI partner category (14 partners; no self-loop in B2H)",
                  title_fontsize=6.2, handletextpad=0.35, columnspacing=0.8)
axI.add_artist(leg1)
axI.legend(handles=deg_handles, loc="lower left", bbox_to_anchor=(0.56, 0.05),
           ncol=1, frameon=False, fontsize=5.8, handletextpad=0.35)

axI.text(0.0, -0.05,
         "Note: High-throughput B2H screens carry false negatives (e.g., published NapM–DnaA binding, Liu et al. 2019, is not present in Wang et al. 2010);\n"
         "therefore, absence of a B2H self-interaction entry does not exclude NapM self-association or additional functional partners in vivo.",
         transform=axI.transAxes, ha="left", va="top", fontsize=5.9, color="#4A5568", style="italic", linespacing=1.3)

axI.set_xlim(-4.75, 4.15)
axI.set_ylim(-1.85, 1.98)
axI.set_title("NapM ego network: 14 B2H PPI partners (no Rv0047c self-loop) and DNA-binding layer",
              loc="left", pad=4, fontweight="bold", color=C_TEXT)
axI.text(-0.06, 1.07, "I", transform=axI.transAxes, fontsize=10.5, fontweight="bold", va="top", color="black")

pdf_path = OUT_FIG / "Response_Fig_7.pdf"
png_path = OUT_FIG / "Response_Fig_7.png"
jpg_path = OUT_FIG / "Response_Fig_7.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved Response_Fig_7.{pdf,png,jpg}")
