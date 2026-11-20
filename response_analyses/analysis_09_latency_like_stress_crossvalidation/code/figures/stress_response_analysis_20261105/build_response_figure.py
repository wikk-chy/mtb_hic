from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
OUT = ROOT / "figures/stress_response_analysis_20261105"
for family in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        FONT = family
        break
    except ValueError:
        continue
matplotlib.rcParams.update({
    "font.family": FONT, "font.size": 7.5, "axes.titlesize": 8.5,
    "axes.labelsize": 7.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "legend.fontsize": 6.6,
})
C_HYP = "#56B4E9"
C_LAT = "#009E73"
C_SHARED = "#D55E00"
C_GREY = "#B9BEC5"
C_TEXT = "#263238"

core = pd.read_csv(OUT / "core_SOS_gene_results.csv")
genes = pd.read_csv(OUT / "stress_gene_level.csv", index_col=0)
shared = pd.read_csv(OUT / "shared_response_summary.csv").set_index("metric")["value"]
struct_summary = pd.read_csv(OUT / "shared_structural_response_summary.csv").set_index("metric")["value"]
metrics = pd.read_csv(ROOT / "hic_review/review_results/R4_comment5/tables/hypoxia_vs_latent_signature.csv")
base = ROOT / "hic_review/review_results/hypoxia_latency_integration_20261030/source_data"
hyp = pd.read_csv(base / "Hypoxia_bin_level.csv", usecols=["bin", "d_ins_hyp_WT"])
lat = pd.read_csv(base / "Latency_bin_level.csv", usecols=["bin", "d_ins_latent_WT"])
bins = hyp.merge(lat, on="bin").dropna()

assert int(shared["shared_DEGs"]) == 896
assert int(shared["shared_DEGs_same_direction"]) == 863
assert int(struct_summary["bins_tested"]) == 881

fig = plt.figure(figsize=(8.27, 9.25))
gs = GridSpec(2, 2, figure=fig, left=0.105, right=0.965, top=0.91, bottom=0.09,
              wspace=0.43, hspace=0.58, height_ratios=[1.0, 1.08])
fig.suptitle("Evidence relevant to the antibiotic-treated persistence model",
             x=0.105, y=0.97, ha="left", fontsize=11.2, fontweight="bold", color=C_TEXT)

def finish(ax, letter, title):
    ax.set_title(title, loc="left", pad=7, fontweight="bold", color=C_TEXT)
    ax.text(-0.17, 1.07, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)

# A: core SOS responses.
ax = fig.add_subplot(gs[0, 0])
values = core[["hypoxia_log2FC", "latent_log2FC"]].to_numpy(float)
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
labels = ["lexA", "dnaE2", "imuA", "imuB", "recA", "ruvA", "ruvC"]
ax.set_yticks(range(len(labels)), labels)
ax.set_xticks([0, 1], ["Hypoxia", "Latent"])
for i in range(values.shape[0]):
    for j in range(2):
        ax.text(j, i, f"{values[i,j]:+.2f}", ha="center", va="center", fontsize=6.4,
                color="black")
ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("log$_2$FC vs WT", fontsize=6.6)
cb.ax.tick_params(labelsize=6)
ax.text(0.0, -0.25,
        "22 LexA-associated genes: 2 DEGs in hypoxia; 9 in latent\nNo DEG enrichment (P = 0.0064 and 0.67); mixed core SOS responses",
        transform=ax.transAxes, fontsize=6.2, va="top", linespacing=1.45)
finish(ax, "A", "No coordinated SOS activation")

# B: shared transcriptional response.
ax = fig.add_subplot(gs[0, 1])
hyp_deg = genes["hypoxia_regulate"].isin(["Up", "Down"])
lat_deg = genes["latent_regulate"].isin(["Up", "Down"])
shared_mask = hyp_deg & lat_deg
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
ax.scatter(genes["hypoxia_log2FC"], genes["latent_log2FC"], s=4,
           color=C_GREY, alpha=0.30, linewidths=0, label="All tested genes")
ax.scatter(genes.loc[shared_mask, "hypoxia_log2FC"], genes.loc[shared_mask, "latent_log2FC"],
           s=5, color=C_SHARED, alpha=0.42, linewidths=0, label="Shared DEGs (n = 896)")
lim = 7.5
ax.plot([-lim, lim], [-lim, lim], color="#555555", lw=0.65, ls="--")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Hypoxia log$_2$FC vs WT")
ax.set_ylabel("Latent log$_2$FC vs WT")
ax.text(0.04, 0.96,
        f"Spearman ρ = {shared['Spearman_rho_log2FC']:.3f}\nSame direction: {shared['direction_concordant_percent']:.1f}%\nShared DEGs same direction: 863/896 (96.3%)",
        transform=ax.transAxes, va="top", fontsize=6.5, linespacing=1.45)
ax.legend(frameon=False, loc="lower right")
finish(ax, "B", "Substantial shared transcriptional response")

# C: global structural metrics.
ax = fig.add_subplot(gs[1, 0])
values = metrics[["hypoxia_fold", "latent_fold"]].to_numpy(float).T
norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.35)
im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
metric_names = ["Short contacts\n<100 kb", "Intermediate contacts\n100 kb–1 Mb", "Long contacts\n>1 Mb",
                "Arm–arm O/E", "ori–ter O/E", "Compaction slope"]
ax.set_xticks(range(6), metric_names, rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks([0, 1], ["Hypoxia", "Latent"])
for i in range(2):
    for j in range(6):
        ax.text(j, i, f"{values[i,j]:.2f}×", ha="center", va="center", fontsize=6.3)
ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.set_label("Fold vs WT", fontsize=6.6)
cb.ax.tick_params(labelsize=6)
finish(ax, "C", "Five of six global structural metrics align")

# D: per-bin structural response.
ax = fig.add_subplot(gs[1, 1])
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
ax.scatter(bins["d_ins_hyp_WT"], bins["d_ins_latent_WT"], s=6,
           color="#6A51A3", alpha=0.30, linewidths=0)
ax.set_xlabel("WT hypoxia insulation response")
ax.set_ylabel("WT latent insulation response")
ax.text(0.04, 0.96,
        f"Spearman ρ = {struct_summary['Spearman_rho_hypoxia_vs_latent_insulation']:.3f}\nSame direction: {struct_summary['same_direction_percent']:.1f}% of bins\n(n = {int(struct_summary['bins_tested'])})",
        transform=ax.transAxes, va="top", fontsize=6.5, linespacing=1.45)
finish(ax, "D", "Limited bin-level structural concordance")

fig.savefig(OUT / "Response_Fig12_Persistence_confounders.pdf")
fig.savefig(OUT / "Response_Fig12_Persistence_confounders.svg")
fig.savefig(OUT / "Response_Fig12_Persistence_confounders.png", dpi=400)
plt.close(fig)

legend = (
    "Response Fig. 12 | Transcriptional and structural evidence relevant to the antibiotic-treated persistence model. "
    "(A) Core SOS/LexA-associated gene responses in hypoxia and the antibiotic-treated latent model. "
    "Numbers show log2FC relative to WT. Across 22 LexA-associated DNA-damage genes, only 2 and 9 met the original DEG calls in hypoxia and latency, respectively, without DEG enrichment. "
    "(B) Genome-wide comparison of hypoxia- and latency-associated log2FC values for 4,065 genes. Orange points denote 896 shared DEGs; 863 (96.3%) changed in the same direction. "
    "(C) Fold changes for six global chromosome-organization metrics, five of which changed in the same direction under hypoxia and latency. "
    "(D) Per-bin comparison of WT insulation responses between hypoxia and latency (n = 881 bins). The responses are positively but weakly correlated (Spearman rho = 0.250)."
)
(OUT / "Response_Fig12_legend.txt").write_text(legend, encoding="utf-8")
(OUT / "figure_build_info.txt").write_text(f"Font: {FONT}\nWidth: 8.27 in (A4, 210 mm)\nHeight: 9.25 in\n", encoding="utf-8")
print(f"Saved Response Fig. 12 to {OUT}; font={FONT}")
