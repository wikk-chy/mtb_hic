#!/usr/bin/env python3
"""
Build Response Fig. 9 (Reviewer 2, Comments 8 & 10 / Audit Comments 14 & 16):
Upgraded, transparent multi-pathway transcriptional and structural comparison of the
acid/antibiotic-treated non-replicating persistence (NRP) model versus antibiotic-free hypoxia.
  A: Four-module heatmap comparing Hypoxia vs Antibiotic-NRP log2FC:
       (i)   RecA/LexA SOS & mutasome (lexA, recA, dnaE2, imuA', imuB, ruvA, ruvC)
       (ii)  PafBC / DSB resection & DNA repair (adnA, adnB, pafB, pafC, uvrA, uvrB, mku, ligD, ogt)
       (iii) Oxidative / ROS & stress sigma factors (furA, katG, sigH, sigE, whiB3, ahpD, sodA, sodC)
       (iv)  Canonical DosR dormancy regulon (devR, devS, dosT, hspX, tgs1, narK2, narX, fdxA + 48-gene summary)
     Fixes the misleading 'No coordinated SOS activation / No DEG enrichment (P=0.0064)' label.
  B: Genome-wide log2FC concordance across 4,065 genes (rho = 0.603; 896 shared DEGs, 96.3% concordant).
  C: Six global chromosome-organization metrics fold change vs WT (5/6 concordant; unique appearance).
  D: Bin-level WT insulation response comparison across 881 bins (rho = 0.250; weak positive correlation).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
STRESS = ROOT / "figures/stress_response_analysis_20261105"
LATENT_SIG = ROOT / "hic_review/review_results/R4_comment5/tables/hypoxia_vs_latent_signature.csv"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_TAB = ROOT / "analysis/reviewer2_revision_20261119/tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7.4,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.65,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

C_TEXT = "#1F2933"

df_gene = pd.read_csv(STRESS / "stress_gene_level.csv").set_index("gene")
shared_sum = pd.read_csv(STRESS / "shared_response_summary.csv").set_index("metric")["value"]
struct_sum = pd.read_csv(STRESS / "shared_structural_response_summary.csv").set_index("metric")["value"]
dna_sum = pd.read_csv(STRESS / "dna_damage_response_summary.csv")

# Four-module panel A genes
modules = [
    ("SOS / mutasome", [
        ("Rv2720", "lexA"),
        ("Rv2737c", "recA"),
        ("Rv3370c", "dnaE2"),
        ("Rv3395c", "imuA'"),
        ("Rv3394c", "imuB"),
        ("Rv2593c", "ruvA"),
        ("Rv2594c", "ruvC"),
    ]),
    ("PafBC / DSB repair", [
        ("Rv3202c", "adnA"),
        ("Rv3201c", "adnB"),
        ("Rv2096c", "pafB"),
        ("Rv2095c", "pafC"),
        ("Rv1638", "uvrA"),
        ("Rv1633", "uvrB"),
        ("Rv0937c", "mku"),
        ("Rv0938", "ligD"),
        ("Rv1316c", "ogt"),
    ]),
    ("ROS / redox stress", [
        ("Rv1909c", "furA"),
        ("Rv1908c", "katG"),
        ("Rv3223c", "sigH"),
        ("Rv1221", "sigE"),
        ("Rv3416", "whiB3"),
        ("Rv2429", "ahpD"),
        ("Rv3846", "sodA"),
        ("Rv0432", "sodC"),
    ]),
    ("DosR dormancy", [
        ("Rv3133c", "devR/dosR"),
        ("Rv3132c", "devS/dosS"),
        ("Rv2027c", "dosT"),
        ("Rv2031c", "hspX"),
        ("Rv3130c", "tgs1"),
        ("Rv1737c", "narK2"),
        ("Rv1736c", "narX"),
        ("Rv2007c", "fdxA"),
    ]),
]

# Build matrix with blank separator rows? Better: keep compact solid matrix + horizontal black group dividers & left bracket labels!
gene_rows = []
group_bounds = []
cur = 0
for mod_name, glist in modules:
    start = cur
    for gid, lbl in glist:
        r = df_gene.loc[gid]
        gene_rows.append({
            "module": mod_name, "gene": gid, "label": lbl,
            "hyp_fc": float(r["hypoxia_log2FC"]), "hyp_reg": str(r["hypoxia_regulate"]),
            "nrp_fc": float(r["latent_log2FC"]), "nrp_reg": str(r["latent_regulate"]),
        })
        cur += 1
    group_bounds.append((mod_name, start, cur - 1))

mod_df = pd.DataFrame(gene_rows)
mod_df.to_csv(OUT_TAB / "antibiotic_nrp_four_module_genes.csv", index=False)

# Load bin-level WT insulation for Panel D
bin_hyp = pd.read_csv(ROOT / "hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Hypoxia_bin_level.csv")
bin_lat = pd.read_csv(ROOT / "hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Latency_bin_level.csv")
b_m = bin_hyp[["bin", "d_ins_hyp_WT"]].merge(bin_lat[["bin", "d_ins_latent_WT"]], on="bin").dropna()

# Global 6 metrics for Panel C
lat_sig = pd.read_csv(LATENT_SIG)

# Figure layout: A on left (tall, spanning both rows so all 32 genes are crisp and readable!),
# B at top-right, C at middle-right, D at bottom-right!
fig = plt.figure(figsize=(8.27, 10.1))
gs = GridSpec(3, 2, figure=fig, left=0.175, right=0.96, top=0.885, bottom=0.068,
              wspace=0.38, hspace=0.44, width_ratios=[0.92, 1.15], height_ratios=[1.08, 0.72, 1.0])
fig.suptitle("Mixed DNA-damage/ROS signatures and shared stress responses in acid/antibiotic-induced latency-like stress",
             x=0.04, y=0.972, ha="left", fontsize=9.1, fontweight="bold", color=C_TEXT)

# Panel A (left column spanning all 3 rows!)
axA = fig.add_subplot(gs[:, 0])
matA = mod_df[["hyp_fc", "nrp_fc"]].to_numpy()
normA = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.5)
imA = axA.imshow(matA, cmap="RdBu_r", norm=normA, aspect="auto")

axA.set_xticks([0, 1], ["Hypoxia\n(no drug)", "Acid/antibiotic\n(Latent)"], fontsize=7.5, fontweight="bold")
axA.xaxis.tick_top()
axA.set_yticks(range(len(mod_df)), mod_df["label"], fontsize=6.8)

for i, r in mod_df.iterrows():
    for j, (fc_col, reg_col) in enumerate([("hyp_fc", "hyp_reg"), ("nrp_fc", "nrp_reg")]):
        val = r[fc_col]
        star = "*" if r[reg_col] in ("Up", "Down") else ""
        tcol = "white" if (val >= 2.0 or val <= -1.8) else "#111111"
        axA.text(j, i, f"{val:+.2f}{star}", ha="center", va="center", fontsize=6.3,
                 fontweight="bold" if star else "normal", color=tcol)

# Module dividers & crisp rotated group headers on the left (cleanly inside page margin!)
for idx, (mname, s_i, e_i) in enumerate(group_bounds):
    if idx > 0:
        axA.axhline(s_i - 0.5, color="#1F2933", linewidth=1.5)
    mid = (s_i + e_i) / 2.0
    axA.text(-0.36, mid, mname, transform=axA.get_yaxis_transform(),
             rotation=90, ha="center", va="center", fontsize=6.6, fontweight="bold", color="#2B4C7E")

axA.set_xticks([-0.5, 0.5, 1.5], minor=True)
axA.set_yticks(np.arange(-0.5, len(mod_df), 1), minor=True)
axA.grid(which="minor", color="white", linewidth=0.7)
axA.tick_params(which="minor", bottom=False, left=False, top=False)

cbarA = fig.colorbar(imA, ax=axA, orientation="horizontal", fraction=0.028, pad=0.045, aspect=24)
cbarA.set_label(r"$\log_2$FC vs WT (* = DEG: $|\log_2\mathrm{FC}| > 1,\ q < 0.05$)", fontsize=6.4)
cbarA.ax.tick_params(labelsize=6.3)

# Accurate statistical footnote below Panel A
fig.text(0.04, 0.012,
         "LexA 22-gene set: Hypoxia depleted (2/22 DEGs, OR = 0.17, P = 0.0064); Acid/antibiotic (Latent) no positive enrichment (9/22 DEGs, OR = 0.78, P = 0.67).\n"
         "DosR 48-gene core regulon: Hypoxia 43/48 Up (0 Down, median +2.06); Acid/antibiotic (Latent) 15/48 Up, 7 Down (median +0.44).",
         fontsize=6.05, color="#333333", linespacing=1.35)

axA.text(-0.40, 1.075, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="top")

# Panel B (top-right): Genome-wide log2FC concordance (4,065 genes)
axB = fig.add_subplot(gs[0, 1])
both = df_gene.dropna(subset=["hypoxia_log2FC", "latent_log2FC"])
is_shared = both["hypoxia_regulate"].isin(["Up", "Down"]) & both["latent_regulate"].isin(["Up", "Down"])
axB.axhline(0, color="#B0B8C0", lw=0.6)
axB.axvline(0, color="#B0B8C0", lw=0.6)
axB.plot([-7, 7], [-7, 7], "--", color="#777777", lw=0.75)
axB.scatter(both.loc[~is_shared, "hypoxia_log2FC"], both.loc[~is_shared, "latent_log2FC"],
            s=4, color="#C8CFD6", alpha=0.45, linewidths=0, label=f"Other genes (n = {(~is_shared).sum():,})")
axB.scatter(both.loc[is_shared, "hypoxia_log2FC"], both.loc[is_shared, "latent_log2FC"],
            s=6.5, color="#D97724", alpha=0.75, linewidths=0, label=f"Shared DEGs (n = {is_shared.sum()})")
axB.set_xlim(-7.2, 7.2)
axB.set_ylim(-7.2, 7.2)
axB.set_xlabel(r"Hypoxia $\log_2$FC vs WT")
axB.set_ylabel(r"Acid/antibiotic (Latent) $\log_2$FC vs WT")
axB.text(0.03, 0.96,
         f"Spearman ρ = {shared_sum['Spearman_rho_log2FC']:.3f} (n = {int(shared_sum['genes_tested_in_both']):,})\n"
         f"All genes concordant: {shared_sum['direction_concordant_percent']:.1f}%\n"
         f"Shared DEGs concordant: {int(shared_sum['shared_DEGs_same_direction'])}/{int(shared_sum['shared_DEGs'])} ({shared_sum['shared_DEGs_same_direction_percent']:.1f}%)",
         transform=axB.transAxes, va="top", ha="left", fontsize=6.2, linespacing=1.35, color=C_TEXT)
axB.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=6.0, markerscale=1.4)
axB.set_title("Substantial shared transcriptional program", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axB.text(-0.18, 1.09, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="top")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

# Panel C (middle-right): 6 global Hi-C metrics fold change vs WT
axC = fig.add_subplot(gs[1, 1])
metric_order = [
    ("frac_short_lt100kb", "Short\n<100 kb"),
    ("frac_mid_100kb_1Mb", "Mid\n0.1–1 Mb"),
    ("frac_long_gt1Mb", "Long\n>1 Mb"),
    ("sec_diag_OE", "Arm–arm\nO/E"),
    ("ori_ter_OE", "ori–ter\nO/E"),
    ("compaction_slope", "P(s)\nslope"),
]
sig_idx = lat_sig.set_index("metric")
matC = np.array([
    [sig_idx.loc[m, "hypoxia_fold"] for m, _ in metric_order],
    [sig_idx.loc[m, "latent_fold"] for m, _ in metric_order],
])
normC = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.36)
imC = axC.imshow(matC, cmap="RdBu_r", norm=normC, aspect="auto")
axC.set_xticks(range(len(metric_order)), [lbl for _, lbl in metric_order], fontsize=6.4)
axC.set_yticks([0, 1], ["Hypoxia", "Acid/antibiotic\n(Latent)"], fontsize=6.5)
for r_i in range(2):
    for c_i in range(len(metric_order)):
        val = matC[r_i, c_i]
        tcol = "white" if (val < 0.72 or val > 1.24) else "#111111"
        axC.text(c_i, r_i, f"{val:.2f}×", ha="center", va="center", fontsize=6.6, fontweight="bold", color=tcol)
axC.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
axC.set_yticks([-0.5, 0.5, 1.5], minor=True)
axC.grid(which="minor", color="white", linewidth=1.2)
axC.tick_params(which="minor", bottom=False, left=False)
cbarC = fig.colorbar(imC, ax=axC, fraction=0.045, pad=0.03)
cbarC.set_label("Fold vs WT", fontsize=6.4)
cbarC.ax.tick_params(labelsize=6.1)
axC.set_title("Global Hi-C metrics (5 of 6 same direction)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axC.text(-0.18, 1.14, "C", transform=axC.transAxes, fontsize=11, fontweight="bold", va="top")

# Panel D (bottom-right): Bin-level WT insulation concordance (881 bins)
axD = fig.add_subplot(gs[2, 1])
axD.axhline(0, color="#B0B8C0", lw=0.6)
axD.axvline(0, color="#B0B8C0", lw=0.6)
axD.scatter(b_m["d_ins_hyp_WT"], b_m["d_ins_latent_WT"], s=7.5, color="#6A51A3", alpha=0.38, linewidths=0)
axD.set_xlabel("WT hypoxia insulation response")
axD.set_ylabel("WT acid/antibiotic (Latent) insulation response")
axD.text(0.03, 0.96,
         f"Spearman ρ = {struct_sum['Spearman_rho_hypoxia_vs_latent_insulation']:.3f}\n"
         f"Same direction: {struct_sum['same_direction_percent']:.1f}% (n = {int(struct_sum['bins_tested'])})\n"
         "Related but non-identical local remodelling",
         transform=axD.transAxes, va="top", ha="left", fontsize=6.15, linespacing=1.35, color=C_TEXT,
         bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88, edgecolor="#E5E7EB", lw=0.5))
axD.set_title("Weak bin-level local insulation concordance", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axD.text(-0.18, 1.09, "D", transform=axD.transAxes, fontsize=11, fontweight="bold", va="top")
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)

pdf_path = OUT_FIG / "Response_Fig_9.pdf"
png_path = OUT_FIG / "Response_Fig_9.png"
jpg_path = OUT_FIG / "Response_Fig_9.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved Response_Fig_9.{pdf,png,jpg}")
