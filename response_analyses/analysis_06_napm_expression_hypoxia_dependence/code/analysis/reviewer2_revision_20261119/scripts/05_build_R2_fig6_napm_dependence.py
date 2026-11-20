#!/usr/bin/env python3
"""
Build Response Fig. 6 (Reviewer 2, Comment 5 / Audit Comment 11):
NapM expression, other core nucleoid-associated/architectural proteins, and genotype-associated
chromosome reorganization under stress.
  A: WT napM expression across normoxia, hypoxia, and acid/antibiotic-NRP (CPM + limma log2FC).
  B: Unified genome-wide genotype x hypoxia insulation interaction (strict 315 + moderate 122 = 437/883
     structure-responsive bins, matching Reviewer 1's Response Fig. 2D).
  C: Independent-sample genome-mean insulation across 15 samples (WT +0.041, KO -0.067, DiD -0.108).
  D: FIXED 9-metric z-scored hypoxia response fingerprint (fixes the Jensen-inequality ins_mean
     sign-flip bug and aligns distance thresholds to <100 kb, 100 kb-1 Mb, >1 Mb; zero legend overlap).
  E: Spatial association between baseline KO effect and WT hypoxia response (881 bins, rho = 0.527).
  F: NEW expression log2FC heatmap of 12 core Mtb NAPs, supercoiling enzymes, and chromosome
     segregation proteins across KO/WT, WT-Hyp/WT, KO-Hyp/WT-Hyp, and WT-NRP/WT.
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
HIC = ROOT / "hic_review"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_TAB = ROOT / "analysis/reviewer2_revision_20261119/tables"
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
    "font.size": 7.4,
    "axes.titlesize": 8.3,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.65,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

C_WT = "#0072B2"
C_HYP = "#56B4E9"
C_KO = "#D55E00"
C_KOHYP = "#E69F00"
C_NRP = "#009E73"
C_STRICT = "#54278F"
C_MOD = "#9E9AC8"
C_GREY = "#C4C9CE"
C_TEXT = "#1F2933"

# Source data
RNA = HIC / "review_data/rnaseq/AKR_RNA_seq_merged.csv"
B437 = ROOT / "docx_latex_conversion/generated/structure_020_response_modes/Response_Fig2E_structure_020_bins.csv"
FINAL_SD = HIC / "review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/final_response_figure_20261027/source_data"
FP_VEC = OUT_TAB / "fingerprint_response_vectors_fixed.csv"
FP_STAT = OUT_TAB / "fingerprint_response_stats_fixed.csv"
COUNTS = HIC / "review_results/R1_mechanism_reanalysis/09_combined/tables/Fig4_v2_B_response_class_counts.csv"

# A: napM CPM
rna = pd.read_csv(RNA).set_index("gene")
lib = rna.sum(axis=0)
rv = rna.loc["Rv0047c"]
cpm = rv / lib * 1e6
expr_reps = {
    "WT\n(normoxia)": np.array([cpm["WT-rep1"], cpm["WT-rep2"]]),
    "Hypoxia": np.array([cpm["Hypoxia-rep1"], cpm["Hypoxia-rep2"]]),
    "Acid/antibiotic\n(Latent)": np.array([cpm["Latent-rep1"], cpm["Latent-rep2"]]),
}

# B: unified 315 + 122 = 437 bins
b437 = pd.read_csv(B437)
is_strict = b437["interaction_ins_class"].isin(["detected_increase", "detected_decrease"])
is_resp = b437["structure_020_responsive"].astype(bool)
is_mod = is_resp & (~is_strict)
is_other = ~is_resp
assert int(is_strict.sum()) == 315 and int(is_mod.sum()) == 122 and int(is_resp.sum()) == 437

# C: independent-sample genome-mean insulation
ins = pd.read_csv(FINAL_SD / "E_independent_sample_genome_mean_insulation.csv")

# D: fixed 9-metric fingerprint
fp_vec = pd.read_csv(FP_VEC)
fp_stat = pd.read_csv(FP_STAT).set_index("statistic")

# E: baseline KO vs WT hypoxia spatial association
assoc = pd.read_csv(FINAL_SD / "G_original_per_bin_association.csv")
spatial = pd.read_csv(FINAL_SD / "G_original_spatial_association.csv").iloc[0]
class_counts = pd.read_csv(COUNTS).set_index("category")["n_bins"]
shared = int(class_counts["both (shared)"])
hypoxia_responsive = shared + int(class_counts["hypoxia only"])

# F: 12 core NAPs / supercoiling / segregation genes across 4 limma contrasts
deg_ko = pd.read_csv(HIC / "review_data/rnaseq/KOvsWT_supo_DEG_limma.csv", index_col=0)
deg_hyp = pd.read_csv(HIC / "review_data/rnaseq/hypoxiaWTvsWT_supo_DEG_limma.csv", index_col=0)
deg_kohyp = pd.read_csv(HIC / "review_data/rnaseq/hypoxia_KOvsWT_supo_DEG_limma.csv", index_col=0)
deg_nrp = pd.read_csv(HIC / "review_data/rnaseq/LatentvsWT_supo_DEG_limma.csv", index_col=0)

arch_genes = [
    ("Rv0047c", "napM (NAP)"),
    ("Rv3597c", "lsr2 (NAP)"),
    ("Rv2986c", "hupB (NAP)"),
    ("Rv3849", "espR (NAP)"),
    ("Rv1388", "mIHF (NAP)"),
    ("Rv0430", "napA (NAP)"),
    ("Rv2922c", "smc (SMC)"),
    ("Rv3918c", "parA (ParAB)"),
    ("Rv3917c", "parB (ParAB)"),
    ("Rv0006", "gyrA (Gyrase)"),
    ("Rv0005", "gyrB (Gyrase)"),
    ("Rv3646c", "topA (Topo I)"),
]
contrasts = [
    ("KO / WT", deg_ko),
    ("WT Hyp / WT", deg_hyp),
    ("KO Hyp /\nWT Hyp", deg_kohyp),
    ("WT Latent / WT", deg_nrp),
]
mat_fc = np.zeros((len(arch_genes), len(contrasts)))
mat_deg = np.empty((len(arch_genes), len(contrasts)), dtype=object)
for r_i, (gid, _) in enumerate(arch_genes):
    for c_i, (_, dtab) in enumerate(contrasts):
        row = dtab.loc[gid]
        mat_fc[r_i, c_i] = float(row["avg_log2FC"])
        mat_deg[r_i, c_i] = str(row["regulate"])

# Save Panel F source table for reproducibility
pf_rows = []
for r_i, (gid, glbl) in enumerate(arch_genes):
    d = {"gene": gid, "label": glbl}
    for c_i, (cname, _) in enumerate(contrasts):
        cn = cname.replace("\n", " ")
        d[f"{cn}_log2FC"] = mat_fc[r_i, c_i]
        d[f"{cn}_regulate"] = mat_deg[r_i, c_i]
    pf_rows.append(d)
pd.DataFrame(pf_rows).to_csv(OUT_TAB / "architectural_proteins_expression.csv", index=False)

# Build A4 figure
fig = plt.figure(figsize=(8.27, 11.4))
gs = GridSpec(3, 2, figure=fig, left=0.10, right=0.945, top=0.935, bottom=0.06,
              wspace=0.45, hspace=0.46, height_ratios=[0.88, 1.12, 1.15])
fig.suptitle("NapM expression, architectural-protein remodelling, and genotype-dependent hypoxia response",
             x=0.05, y=0.976, ha="left", fontsize=10.5, fontweight="bold", color=C_TEXT)


def finish(ax, letter, title, lx=-0.18):
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color=C_TEXT)
    ax.text(lx, 1.085, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


# Panel A
axA = fig.add_subplot(gs[0, 0])
conds_A = list(expr_reps.keys())
cols_A = [C_WT, C_HYP, C_NRP]
means_A = [expr_reps[c].mean() for c in conds_A]
xA = np.arange(3)
axA.bar(xA, means_A, width=0.58, color=cols_A, edgecolor="black", linewidth=0.6, alpha=0.88)
for i, c in enumerate(conds_A):
    axA.scatter(np.full(2, i) + np.array([-0.06, 0.06]), expr_reps[c], s=18, color="black", zorder=3)
    if i == 1:
        axA.text(i + 0.14, means_A[i] + 14, f"{means_A[i]:.0f}", ha="left", va="bottom", fontsize=7.0, fontweight="bold")
    else:
        axA.text(i, max(expr_reps[c].max(), means_A[i]) + 18, f"{means_A[i]:.0f}", ha="center", va="bottom", fontsize=7.0)
axA.axhline(means_A[0], color="#666666", linestyle=(0, (3, 2)), linewidth=0.65)
axA.set_xticks(xA, conds_A)
axA.set_ylim(0, 960)
axA.set_ylabel("napM expression (CPM)")
axA.text(0.03, 0.97,
         r"Hypoxia: $\log_2$FC = 0.93, $q$ = 2.45$\times 10^{-4}$" "\n"
         r"Acid/antibiotic (Latent): $\log_2$FC = 0.15, $q$ = 8.04$\times 10^{-3}$",
         transform=axA.transAxes, va="top", ha="left", fontsize=6.4, linespacing=1.4)
finish(axA, "A", "Condition-dependent napM expression in WT")

# Panel B: Unified 315 strict + 122 moderate = 437/883
axB = fig.add_subplot(gs[0, 1])
pos = (b437["start"] + b437["end"]) / 2e6
eff = b437["d_ins_interaction"]
axB.axhline(0, color="#666666", linewidth=0.6)
axB.scatter(pos[is_other], eff[is_other], s=4.0, color=C_GREY, alpha=0.65, linewidths=0, label="Other (446)")
axB.scatter(pos[is_mod], eff[is_mod], s=7.0, color=C_MOD, alpha=0.85, linewidths=0, label="Moderate |Δins| ≥ 0.20 (+122)")
axB.scatter(pos[is_strict], eff[is_strict], s=7.5, color=C_STRICT, alpha=0.90, linewidths=0, label="Strict detected (315)")
axB.set_xlabel("Genome position (Mb)")
axB.set_ylabel("Insulation interaction\n(KO response − WT response)")
axB.set_ylim(-1.62, 1.15)
axB.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=5.9, ncol=1, markerscale=1.4)
axB.text(0.02, 0.04, "Total structure-responsive: 437/883 (49.5%)\n(strict 315 + moderate 122; operational tiers)",
         transform=axB.transAxes, va="bottom", ha="left", fontsize=6.2, color=C_TEXT)
finish(axB, "B", "Genotype × hypoxia insulation interaction")

# Panel C: Independent-sample genome-mean insulation
axC = fig.add_subplot(gs[1, 0])
orderC = ["WT", "HYP", "KO", "KO-HYP"]
labelsC = ["WT\nnormoxia", "WT\nhypoxia", "KO\nnormoxia", "KO\nhypoxia"]
colsC = [C_WT, C_HYP, C_KO, C_KOHYP]
for i, c in enumerate(orderC):
    vals = ins.loc[ins["condition"].eq(c), "genome_mean_insulation"].to_numpy()
    jit = np.linspace(-0.06, 0.06, len(vals))
    axC.scatter(np.full(len(vals), i) + jit, vals, s=24, facecolor=colsC[i], edgecolor="black", linewidth=0.5, zorder=3)
    axC.plot([i - 0.18, i + 0.18], [vals.mean(), vals.mean()], color="black", linewidth=1.2)
axC.set_xticks(range(4), labelsC)
axC.set_ylabel("Genome-mean insulation")
axC.set_ylim(-0.152, 0.028)
axC.text(0.03, 0.97,
         "WT response = +0.041; KO response = −0.067\n"
         "Difference-in-differences = −0.108\n"
         r"95% CI [−0.114, −0.102]; exact $P$ = 4.08$\times 10^{-4}$",
         transform=axC.transAxes, va="top", fontsize=6.25, linespacing=1.4)
finish(axC, "C", "Independent samples confirm response contrast")

# Panel D: FIXED 9-metric fingerprint (zero sign-flip, zero legend overlap!)
axD = fig.add_subplot(gs[1, 1])
yD = np.arange(len(fp_vec))
h = 0.36
axD.axvline(0, color="#555555", linewidth=0.65)
axD.barh(yD - h / 2, fp_vec["R_WT"], height=h, color=C_HYP, edgecolor="none", label="WT hypoxia response")
axD.barh(yD + h / 2, fp_vec["R_KO"], height=h, color=C_KOHYP, edgecolor="none", label="KO hypoxia response")
axD.set_yticks(yD, fp_vec["metric_label"])
axD.invert_yaxis()
# Extra headroom at top so stats (left) and legend (right) sit cleanly above all bars with zero overlap
axD.set_ylim(len(fp_vec) - 0.3, -2.95)
axD.set_xlim(-2.65, 2.55)
axD.set_xlabel("Hypoxia response (z-scored metric units)")
euc_v = fp_stat.loc["euclidean_distance", "observed"]
euc_p = fp_stat.loc["euclidean_distance", "exact_perm_P"]
cos_v = fp_stat.loc["cosine_similarity", "observed"]
cos_p = fp_stat.loc["cosine_similarity", "exact_perm_P"]
axD.text(0.015, 0.985,
         f"Magnitude: d = {euc_v:.2f}, P = {euc_p:.2e}\n"
         f"Direction: cos = {cos_v:.3f}, P = {cos_p:.3f} (n.s.)",
         transform=axD.transAxes, va="top", ha="left", fontsize=5.95, linespacing=1.35, color=C_TEXT)
axD.legend(loc="upper right", bbox_to_anchor=(1.02, 0.995), frameon=False, fontsize=5.95, handlelength=1.1, handletextpad=0.35)
finish(axD, "D", "Nine-metric hypoxia response fingerprint", lx=-0.38)

# Panel E: Spatial association KO baseline vs WT hypoxia
axE = fig.add_subplot(gs[2, 0])
axE.axhline(0, color="#B0B8C0", linewidth=0.6)
axE.axvline(0, color="#B0B8C0", linewidth=0.6)
xE = assoc["d_ins_KO_WT"].to_numpy()
yE = assoc["d_ins_hyp_WT"].to_numpy()
axE.scatter(xE, yE, s=7, color=C_HYP, alpha=0.45, linewidths=0)
# Linear descriptive fit
slope_e, intercept_e = np.polyfit(xE, yE, 1)
xs = np.linspace(xE.min(), xE.max(), 100)
axE.plot(xs, slope_e * xs + intercept_e, color=C_KO, linewidth=1.2)
axE.set_xlabel("Baseline KO effect (KO − WT)")
axE.set_ylabel("WT hypoxia response (HYP − WT)")
axE.text(0.03, 0.97,
         f"Spearman ρ = {spatial['estimator']:.3f}\n"
         f"95% CI [{spatial['ci_lo']:.3f}, {spatial['ci_hi']:.3f}]\n"
         f"Circular-shift P = {spatial['p']:.4f}",
         transform=axE.transAxes, va="top", ha="left", fontsize=6.3, linespacing=1.4, color=C_TEXT)
axE.text(0.97, 0.04, f"Exploratory overlap: {shared}/{hypoxia_responsive} ({100*shared/hypoxia_responsive:.1f}%)",
         transform=axE.transAxes, va="bottom", ha="right", fontsize=6.0, color="#555555")
finish(axE, "E", "Baseline KO effect vs WT hypoxia response (n = 881)")

# Panel F: NEW 12 core NAPs / supercoiling / segregation genes heatmap
axF = fig.add_subplot(gs[2, 1])
normF = TwoSlopeNorm(vmin=-3.2, vcenter=0.0, vmax=2.0)
im = axF.imshow(mat_fc, cmap="RdBu_r", norm=normF, aspect="auto")
axF.set_xticks(range(len(contrasts)), [c[0] for c in contrasts], fontsize=6.5)
axF.set_yticks(range(len(arch_genes)), [g[1] for g in arch_genes], fontsize=6.6)
for r_i in range(len(arch_genes)):
    for c_i in range(len(contrasts)):
        v = mat_fc[r_i, c_i]
        reg = mat_deg[r_i, c_i]
        star = "*" if reg in ("Up", "Down") else ""
        # Cap display for complete knockout (-11.15, -12.76) so text stays crisp
        if abs(v) > 8:
            lbl = "KO*"
            tcol = "white"
        else:
            lbl = f"{v:+.2f}{star}"
            tcol = "white" if (v < -1.8 or v > 1.25) else "#111111"
        axF.text(c_i, r_i, lbl, ha="center", va="center", fontsize=6.0,
                 fontweight="bold" if star else "normal", color=tcol)
# White cell borders
axF.set_xticks(np.arange(-0.5, len(contrasts), 1), minor=True)
axF.set_yticks(np.arange(-0.5, len(arch_genes), 1), minor=True)
axF.grid(which="minor", color="white", linewidth=1.0)
axF.tick_params(which="minor", bottom=False, left=False)
cbar = fig.colorbar(im, ax=axF, fraction=0.046, pad=0.03)
cbar.set_label(r"$\log_2$FC (* = DEG)", fontsize=6.6)
cbar.ax.tick_params(labelsize=6.2)
axF.set_title("Expression of Mtb NAPs, SMC/ParAB & topoisomerases", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axF.text(-0.32, 1.085, "F", transform=axF.transAxes, fontsize=11, fontweight="bold", va="top", color="black")

pdf_path = OUT_FIG / "Response_Fig_6.pdf"
png_path = OUT_FIG / "Response_Fig_6.png"
jpg_path = OUT_FIG / "Response_Fig_6.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved Response_Fig_6.{pdf,png,jpg}")
