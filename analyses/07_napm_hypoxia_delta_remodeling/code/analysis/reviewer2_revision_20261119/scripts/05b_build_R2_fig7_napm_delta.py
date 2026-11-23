#!/usr/bin/env python3
"""
Build redesigned 4-panel Response Fig. 7 (Reviewer 2, Comment 5):
Focused on WT napM expression, normal WT vs napM-KO Hi-C delta trajectories under hypoxia,
and co-regulated architectural proteins supporting why NapM is necessary yet not sufficient.
  A: WT napM expression across normoxia, hypoxia, and acid/antibiotic (Latent).
  B: Independent-sample genome-mean insulation trajectories (n = 15) with explicit WT Δ vs KO Δ arrows.
  C: Nine-metric Hi-C hypoxia Δ vectors (WT Δ vs KO Δ), grouped into Reversed (6), Blunted/shifted (2),
     and Preserved / NapM-independent (1) metrics.
  D: Expression log2FC heatmap of 12 core Mtb NAPs, SMC/ParAB, and topoisomerases across 4 limma contrasts.
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
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.7,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

C_WT = "#0072B2"
C_HYP = "#56B4E9"
C_KO = "#D55E00"
C_KOHYP = "#E69F00"
C_LAT = "#009E73"
C_TEXT = "#1F2933"
C_WT_DELTA = "#2B7BBA"
C_KO_DELTA = "#DF6B1B"

# Source data
RNA = HIC / "review_data/rnaseq/AKR_RNA_seq_merged.csv"
B437 = ROOT / "docx_latex_conversion/generated/structure_020_response_modes/Response_Fig2E_structure_020_bins.csv"
FINAL_SD = HIC / "review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/final_response_figure_20261027/source_data"
FP_VEC = OUT_TAB / "fingerprint_response_vectors_fixed.csv"
FP_STAT = OUT_TAB / "fingerprint_response_stats_fixed.csv"

# Panel A data: napM CPM in WT conditions
rna = pd.read_csv(RNA).set_index("gene")
lib = rna.sum(axis=0)
rv = rna.loc["Rv0047c"]
cpm = rv / lib * 1e6
expr_reps = {
    "WT\n(normoxia)": np.array([cpm["WT-rep1"], cpm["WT-rep2"]]),
    "Hypoxia": np.array([cpm["Hypoxia-rep1"], cpm["Hypoxia-rep2"]]),
    "Acid/antibiotic\n(Latent)": np.array([cpm["Latent-rep1"], cpm["Latent-rep2"]]),
}

# Panel B data: 15 independent Hi-C samples genome-mean insulation
ins = pd.read_csv(FINAL_SD / "E_independent_sample_genome_mean_insulation.csv")
b437 = pd.read_csv(B437)
n_resp_bins = int(b437["structure_020_responsive"].sum())
n_tot_bins = len(b437)
assert n_resp_bins == 437 and n_tot_bins == 883

# Panel C data: 9-metric hypoxia delta vectors grouped by response class
fp_vec_raw = pd.read_csv(FP_VEC).set_index("metric")
fp_stat = pd.read_csv(FP_STAT).set_index("statistic")

ordered_metrics = [
    # Reversed in KO (6)
    "short_frac",
    "cid_intra_frac",
    "loop_oe",
    "ins_mean",
    "entropy",
    "boundary_depth",
    # Blunted / shifted in KO (2)
    "long_frac",
    "inter_frac",
    # Preserved in KO / NapM-independent (1)
    "ps_slope",
]
fp_vec = fp_vec_raw.loc[ordered_metrics].reset_index()

# Panel D data: 12 core architectural proteins across 4 limma contrasts
deg_ko = pd.read_csv(HIC / "review_data/rnaseq/KOvsWT_supo_DEG_limma.csv", index_col=0)
deg_hyp = pd.read_csv(HIC / "review_data/rnaseq/hypoxiaWTvsWT_supo_DEG_limma.csv", index_col=0)
deg_kohyp = pd.read_csv(HIC / "review_data/rnaseq/hypoxia_KOvsWT_supo_DEG_limma.csv", index_col=0)
deg_lat = pd.read_csv(HIC / "review_data/rnaseq/LatentvsWT_supo_DEG_limma.csv", index_col=0)

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
    ("WT Latent / WT", deg_lat),
]
mat_fc = np.zeros((len(arch_genes), len(contrasts)))
mat_deg = np.empty((len(arch_genes), len(contrasts)), dtype=object)
for r_i, (gid, _) in enumerate(arch_genes):
    for c_i, (_, dtab) in enumerate(contrasts):
        row = dtab.loc[gid]
        mat_fc[r_i, c_i] = float(row["avg_log2FC"])
        mat_deg[r_i, c_i] = str(row["regulate"])

# Build streamlined 2x2 figure (no redundant suptitle)
fig = plt.figure(figsize=(9.0, 7.6))
gs = GridSpec(
    2, 2, figure=fig,
    left=0.17, right=0.935, top=0.942, bottom=0.075,
    wspace=0.40, hspace=0.35,
    height_ratios=[0.88, 1.28],
    width_ratios=[1.16, 1.00],
)


def style_ax(ax, title):
    ax.set_title(title, loc="left", pad=7, fontweight="bold", color=C_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


# ---------------- Panel A: WT napM expression ----------------
axA = fig.add_subplot(gs[0, 0])
conds_A = list(expr_reps.keys())
cols_A = [C_WT, C_HYP, C_LAT]
means_A = [expr_reps[c].mean() for c in conds_A]
xA = np.arange(3)
axA.bar(xA, means_A, width=0.52, color=cols_A, edgecolor="black", linewidth=0.65, alpha=0.88)
for i, c in enumerate(conds_A):
    axA.scatter(np.full(2, i) + np.array([-0.06, 0.06]), expr_reps[c], s=20, color="black", zorder=3)
    if i == 1:
        axA.text(i + 0.14, means_A[i] + 14, f"{means_A[i]:.0f}", ha="left", va="bottom", fontsize=7.2, fontweight="bold")
    else:
        axA.text(i, max(expr_reps[c].max(), means_A[i]) + 18, f"{means_A[i]:.0f}", ha="center", va="bottom", fontsize=7.2)
axA.axhline(means_A[0], color="#666666", linestyle=(0, (3, 2)), linewidth=0.7)
axA.set_xticks(xA, conds_A)
axA.set_ylim(0, 1020)
axA.set_ylabel(r"$\mathit{napM}$ expression (CPM)")
axA.text(
    0.03, 0.96,
    r"Hypoxia: $\log_2$FC = +0.93 ($q$ = 2.45$\times 10^{-4}$)" "\n"
    r"Latent: $\log_2$FC = +0.15 ($q$ = 8.04$\times 10^{-3}$)" "\n"
    r"Neither reaches $|\log_2\mathrm{FC}| > 1$ DEG threshold",
    transform=axA.transAxes, va="top", ha="left", fontsize=6.45, linespacing=1.38, color=C_TEXT,
)
style_ax(axA, r"WT $\mathit{napM}$ expression changes modestly under stress")

# ---------------- Panel B: Genome-mean insulation trajectory (WT Δ vs KO Δ) ----------------
axB = fig.add_subplot(gs[0, 1])
orderB = ["WT", "HYP", "KO", "KO-HYP"]
xposB = [0.0, 1.0, 2.35, 3.35]
labelsB = ["WT\nnormoxia", "WT\nhypoxia", "KO\nnormoxia", "KO\nhypoxia"]
colsB = [C_WT, C_HYP, C_KO, C_KOHYP]
meansB = {}
for i, c in enumerate(orderB):
    vals = ins.loc[ins["condition"].eq(c), "genome_mean_insulation"].to_numpy()
    meansB[c] = vals.mean()
    jit = np.linspace(-0.07, 0.07, len(vals))
    axB.scatter(np.full(len(vals), xposB[i]) + jit, vals, s=26, facecolor=colsB[i], edgecolor="black", linewidth=0.55, zorder=4)
    axB.plot([xposB[i] - 0.20, xposB[i] + 0.20], [meansB[c], meansB[c]], color="black", linewidth=1.35, zorder=5)

# Trajectory arrows connecting WT normoxia -> WT hypoxia and KO normoxia -> KO hypoxia
axB.annotate(
    "", xy=(xposB[1] - 0.11, meansB["HYP"]), xytext=(xposB[0] + 0.11, meansB["WT"]),
    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=C_WT_DELTA, lw=1.8),
    zorder=3,
)
axB.text(
    0.58, -0.119,
    "WT Δ = +0.041", ha="left", va="top", fontsize=6.85, fontweight="bold", color=C_WT_DELTA,
)

axB.annotate(
    "", xy=(xposB[3] - 0.11, meansB["KO-HYP"]), xytext=(xposB[2] + 0.11, meansB["KO"]),
    arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=C_KO_DELTA, lw=1.8),
    zorder=3,
)
axB.text(
    2.94, -0.072,
    "KO Δ = −0.067", ha="left", va="center", fontsize=6.85, fontweight="bold", color=C_KO_DELTA,
)

axB.axvline(1.675, color="#D8DEE4", linestyle="--", linewidth=0.75, zorder=1)
axB.set_xticks(xposB, labelsB)
axB.set_xlim(-0.5, 4.18)
axB.set_ylim(-0.152, 0.018)
axB.set_ylabel("Genome-mean insulation")
axB.text(
    0.03, 0.96,
    r"DiD ($\Delta_{\mathrm{KO}} - \Delta_{\mathrm{WT}}$) = −0.108" "\n"
    r"95% CI [−0.114, −0.102]; exact $P$ = 4.08$\times 10^{-4}$",
    transform=axB.transAxes, va="top", ha="left", fontsize=6.45, linespacing=1.38, color=C_TEXT,
)
style_ax(axB, r"Hypoxia insulation $\Delta$ reverses in $\mathit{napM}$-KO (n = 15)")

# ---------------- Panel C: Grouped 9-metric Hi-C hypoxia Δ vectors ----------------
axC = fig.add_subplot(gs[1, 0])
yC = np.arange(len(fp_vec))
h = 0.36
axC.plot([0, 0], [-0.48, 8.48], color="#555555", linewidth=0.75, zorder=2)

# Subtle background shading for the 3 behavioral classes
axC.axhspan(-0.48, 5.48, color="#FFF5EE", alpha=0.55, zorder=0)
axC.axhspan(5.52, 7.48, color="#F4F6F8", alpha=0.75, zorder=0)
axC.axhspan(7.52, 8.48, color="#EEF8F4", alpha=0.65, zorder=0)
axC.axhline(5.5, color="#B8C2CC", linestyle="--", linewidth=0.65, zorder=1)
axC.axhline(7.5, color="#B8C2CC", linestyle="--", linewidth=0.65, zorder=1)

axC.barh(yC - h / 2, fp_vec["R_WT"], height=h, color=C_WT_DELTA, edgecolor="none", label="WT Δ (WT-Hyp − WT)", zorder=3)
axC.barh(yC + h / 2, fp_vec["R_KO"], height=h, color=C_KO_DELTA, edgecolor="none", label="KO Δ (KO-Hyp − KO)", zorder=3)
axC.set_yticks(yC, fp_vec["metric_label"])
axC.invert_yaxis()
axC.set_ylim(8.65, -2.35)
axC.set_xlim(-2.40, 3.55)
axC.set_xlabel("Hypoxia Δ (z-scored Hi-C metric units)")

# Right-hand class brackets & crisp labels
def draw_bracket(ax, y0, y1, x_b, text, col):
    ax.plot([x_b - 0.08, x_b, x_b, x_b - 0.08], [y0, y0, y1, y1], color=col, lw=0.85, zorder=4)
    ax.text(x_b + 0.10, (y0 + y1) / 2, text, ha="left", va="center", fontsize=6.15,
            fontweight="bold", color=col, linespacing=1.2)

draw_bracket(axC, -0.25, 5.25, 2.02, "Reversed\nin KO\n(6/9)", "#B44700")
draw_bracket(axC, 5.75, 7.25, 2.02, "Blunted /\nshifted\n(2/9)", "#4A5568")
draw_bracket(axC, 7.75, 8.25, 2.02, "Preserved\n(1/9)", "#007A57")

euc_v = fp_stat.loc["euclidean_distance", "observed"]
axC.text(
    0.015, 0.985,
    rf"9-metric $\Delta$: $d$ = {euc_v:.2f}, exact $P$ = 4.08$\times 10^{{-4}}$" "\n"
    rf"5-kb bins: {n_resp_bins}/{n_tot_bins} ({100*n_resp_bins/n_tot_bins:.1f}%) $|\Delta_{{\mathrm{{KO}}}} - \Delta_{{\mathrm{{WT}}}}| \geq 0.20$",
    transform=axC.transAxes, va="top", ha="left", fontsize=5.85, linespacing=1.38, color=C_TEXT,
)
axC.legend(
    loc="upper right", bbox_to_anchor=(1.02, 1.00),
    frameon=False, fontsize=5.95, handlelength=1.05, handletextpad=0.35,
)
style_ax(axC, r"WT $\Delta$ versus $\mathit{napM}$-KO $\Delta$ across nine Hi-C metrics")

# ---------------- Panel D: 12 core architectural proteins heatmap ----------------
axD = fig.add_subplot(gs[1, 1])
normD = TwoSlopeNorm(vmin=-3.2, vcenter=0.0, vmax=2.0)
im = axD.imshow(mat_fc, cmap="RdBu_r", norm=normD, aspect="auto")
axD.set_xticks(range(len(contrasts)), [c[0] for c in contrasts], fontsize=6.6)
axD.set_yticks(range(len(arch_genes)), [g[1] for g in arch_genes], fontsize=6.7)
for r_i in range(len(arch_genes)):
    for c_i in range(len(contrasts)):
        v = mat_fc[r_i, c_i]
        reg = mat_deg[r_i, c_i]
        star = "*" if reg in ("Up", "Down") else ""
        if abs(v) > 8:
            lbl = "KO*"
            tcol = "white"
        else:
            lbl = f"{v:+.2f}{star}"
            tcol = "white" if (v < -1.8 or v > 1.25) else "#111111"
        axD.text(
            c_i, r_i, lbl, ha="center", va="center", fontsize=6.15,
            fontweight="bold" if star else "normal", color=tcol,
        )
axD.set_xticks(np.arange(-0.5, len(contrasts), 1), minor=True)
axD.set_yticks(np.arange(-0.5, len(arch_genes), 1), minor=True)
axD.grid(which="minor", color="white", linewidth=1.0)
axD.tick_params(which="minor", bottom=False, left=False)
cbar = fig.colorbar(im, ax=axD, fraction=0.048, pad=0.03)
cbar.set_label(r"$\log_2$FC (* = DEG)", fontsize=6.7)
cbar.ax.tick_params(labelsize=6.3)
axD.set_title("Co-regulated Mtb NAPs, SMC/ParAB & topoisomerases", loc="left", pad=7, fontweight="bold", color=C_TEXT)

# Perfectly aligned figure-level panel letters A, B, C, D
posA, posB, posC, posD = axA.get_position(), axB.get_position(), axC.get_position(), axD.get_position()
fig.text(0.025, posA.y1 + 0.032, "A", fontsize=12, fontweight="bold", va="top", color="black")
fig.text(posB.x0 - 0.075, posB.y1 + 0.032, "B", fontsize=12, fontweight="bold", va="top", color="black")
fig.text(0.025, posC.y1 + 0.032, "C", fontsize=12, fontweight="bold", va="top", color="black")
fig.text(posB.x0 - 0.075, posD.y1 + 0.032, "D", fontsize=12, fontweight="bold", va="top", color="black")

pdf_path = OUT_FIG / "Response_Fig_7_delta.pdf"
png_path = OUT_FIG / "Response_Fig_7_delta.png"
jpg_path = OUT_FIG / "Response_Fig_7_delta.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved", pdf_path, png_path, jpg_path)
