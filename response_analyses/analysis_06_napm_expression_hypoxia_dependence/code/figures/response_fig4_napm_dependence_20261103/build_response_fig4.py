from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
HIC = ROOT / "hic_review"
OUT = ROOT / "figures/response_fig4_napm_dependence_20261103"
OUT.mkdir(parents=True, exist_ok=True)

# Editable text in vector outputs.
for family in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        FONT = family
        break
    except ValueError:
        continue
matplotlib.rcParams.update({
    "font.family": FONT,
    "font.size": 7.4,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.7,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})

C_WT = "#0072B2"
C_HYP = "#56B4E9"
C_KO = "#D55E00"
C_KOHYP = "#E69F00"
C_LAT = "#009E73"
C_SIG = "#6A51A3"
C_GREY = "#B8BDC4"
C_TEXT = "#263238"

# Source paths.
RNA = HIC / "review_data/rnaseq/AKR_RNA_seq_merged.csv"
BIN = HIC / "review_results/hypoxia_latency_integration_20261030/source_data/Hypoxia_bin_level.csv"
FINAL_SD = HIC / "review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/final_response_figure_20261027/source_data"
FINGER = HIC / "review_results/R1_mechanism_reanalysis/08_extended/fig4_fingerprint/tables"
LATENT = HIC / "review_results/R4_comment5/tables/hypoxia_vs_latent_signature.csv"
COUNTS = HIC / "review_results/R1_mechanism_reanalysis/09_combined/tables/Fig4_v2_B_response_class_counts.csv"

# A: per-replicate napM CPM.
rna = pd.read_csv(RNA).set_index("gene")
lib = rna.sum(axis=0)
rv = rna.loc["Rv0047c"]
cpm = rv / lib * 1e6
expr_reps = {
    "WT": np.array([cpm["WT-rep1"], cpm["WT-rep2"]]),
    "Hypoxia": np.array([cpm["Hypoxia-rep1"], cpm["Hypoxia-rep2"]]),
    "Latent": np.array([cpm["Latent-rep1"], cpm["Latent-rep2"]]),
}

# B: genome-wide genotype x hypoxia structural interaction.
bins = pd.read_csv(BIN)
detected = bins["interaction_ins_class"].isin(["detected_increase", "detected_decrease"])
n_total = len(bins)
n_detected = int(detected.sum())
pct_detected = 100 * n_detected / n_total
bins[["bin", "start", "end", "d_ins_interaction", "interaction_padj", "interaction_ins_class"]].to_csv(
    OUT / "B_genotype_hypoxia_interaction.csv", index=False)
with (OUT / "B_interaction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "value"])
    writer.writerow(["total_bins", n_total])
    writer.writerow(["detected_genotype_by_hypoxia_interaction_bins", n_detected])
    writer.writerow(["percent_bins", pct_detected])

# C: independent-sample genome-mean insulation.
ins = pd.read_csv(FINAL_SD / "E_independent_sample_genome_mean_insulation.csv")

# D: nine-metric z-scored response vectors and global tests.
response = pd.read_csv(FINGER / "fig4_response_vectors_contribution.csv")
stats = pd.read_csv(FINGER / "fig4_response_stats.csv").set_index("statistic")["observed"]

# E: baseline association and exploratory shared-call counts.
association = pd.read_csv(FINAL_SD / "G_original_per_bin_association.csv")
spatial = pd.read_csv(FINAL_SD / "G_original_spatial_association.csv").iloc[0]
class_counts = pd.read_csv(COUNTS).set_index("category")["n_bins"]
shared = int(class_counts["both (shared)"])
hypoxia_responsive = shared + int(class_counts["hypoxia only"])

# F: hypoxia and latency structural signatures in WT cells.
latent = pd.read_csv(LATENT)

# Figure setup: A4 width, flexible height, coordinated 2 x 3 panels.
fig = plt.figure(figsize=(8.27, 11.6))
gs = GridSpec(3, 2, figure=fig, left=0.095, right=0.97, top=0.935, bottom=0.065,
              wspace=0.42, hspace=0.55, height_ratios=[0.92, 1.05, 1.03])
fig.suptitle("NapM expression and genotype-associated chromosome reorganization under stress",
             x=0.095, y=0.978, ha="left", fontsize=11.2, fontweight="bold", color=C_TEXT)


def finish_panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=7, fontweight="bold", color=C_TEXT)
    ax.text(-0.17, 1.075, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)

# A
ax = fig.add_subplot(gs[0, 0])
conditions = ["WT", "Hypoxia", "Latent"]
colors = [C_WT, C_HYP, C_LAT]
means = [expr_reps[c].mean() for c in conditions]
x = np.arange(3)
ax.bar(x, means, width=0.62, color=colors, edgecolor="black", linewidth=0.6, alpha=0.88)
rng = np.random.default_rng(12)
for i, condition in enumerate(conditions):
    jitter = np.array([-0.055, 0.055])
    ax.scatter(np.full(2, i) + jitter, expr_reps[condition], s=15, color="black", zorder=3)
    ax.text(i, means[i] + 17, f"{means[i]:.0f}", ha="center", va="bottom", fontsize=7)
ax.axhline(means[0], color="#666666", linestyle=(0, (3, 2)), linewidth=0.65)
ax.set_xticks(x, conditions)
ax.set_ylim(0, 790)
ax.set_ylabel("napM expression (CPM)")
ax.text(0.02, 0.98,
        "Hypoxia: log$_2$FC = 0.93, q = 2.45e-4\nLatent: log$_2$FC = 0.15, q = 8.04e-3",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.5, linespacing=1.45)
finish_panel(ax, "A", "Condition-dependent napM expression in WT")

# B
ax = fig.add_subplot(gs[0, 1])
pos = (bins["start"] + bins["end"]) / 2e6
effect = bins["d_ins_interaction"]
ax.axhline(0, color="#666666", linewidth=0.6)
ax.scatter(pos[~detected], effect[~detected], s=3.5, color=C_GREY, alpha=0.65, linewidths=0)
ax.scatter(pos[detected], effect[detected], s=7, color=C_SIG, alpha=0.85, linewidths=0)
ax.set_xlabel("Genome position (Mb)")
ax.set_ylabel("Insulation interaction\n(KO response − WT response)")
ax.text(0.02, 0.97, f"Detected interaction bins: {n_detected}/{n_total} ({pct_detected:.1f}%)",
        transform=ax.transAxes, va="top", fontsize=6.8, color=C_TEXT)
finish_panel(ax, "B", "Genome-wide genotype-dependent structural response")

# C
ax = fig.add_subplot(gs[1, 0])
order = ["WT", "HYP", "KO", "KO-HYP"]
labels = ["WT\nnormoxia", "WT\nhypoxia", "KO\nnormoxia", "KO\nhypoxia"]
cond_colors = [C_WT, C_HYP, C_KO, C_KOHYP]
for i, condition in enumerate(order):
    values = ins.loc[ins["condition"].eq(condition), "genome_mean_insulation"].to_numpy()
    jitter = np.linspace(-0.055, 0.055, len(values))
    ax.scatter(np.full(len(values), i) + jitter, values, s=22,
               facecolor=cond_colors[i], edgecolor="black", linewidth=0.45, zorder=3)
    ax.plot([i - 0.17, i + 0.17], [values.mean(), values.mean()], color="black", linewidth=1.1)
ax.set_xticks(range(4), labels)
ax.set_ylabel("Genome-mean insulation")
ax.set_ylim(-0.15, 0.025)
ax.text(0.02, 0.98,
        "WT response = +0.041; KO response = −0.067\nDifference-in-differences = −0.108\n95% CI [−0.114, −0.102]; permutation P = 4.08e-4",
        transform=ax.transAxes, va="top", fontsize=6.25, linespacing=1.42)
finish_panel(ax, "C", "Independent samples confirm altered response magnitude")

# D
ax = fig.add_subplot(gs[1, 1])
metric_labels = [
    "P(s) slope", "Short-range fraction", "Intermediate fraction",
    "Long-range fraction", "Mean insulation", "CID boundary depth",
    "Loop O/E", "CID intra-domain fraction", "Distance entropy"
]
y = np.arange(len(response)) + 2.2
h = 0.35
ax.barh(y - h / 2, response["R_WT"], height=h, color=C_HYP, label="WT response", edgecolor="none")
ax.barh(y + h / 2, response["R_KO"], height=h, color=C_KOHYP, label="KO response", edgecolor="none")
ax.axvline(0, color="#555555", linewidth=0.65)
ax.set_yticks(y, metric_labels)
ax.invert_yaxis()
ax.set_xlabel("Hypoxia response (z-scored metric units)")
ax.legend(frameon=False, loc="upper right", ncol=1)
ax.text(0.02, 0.98,
        f"Magnitude: distance = {stats['euclidean_dist']:.2f}, P = 8.0e-4\nDirection: cosine = {stats['cosine_similarity']:.3f}, P = 0.365",
        transform=ax.transAxes, va="top", fontsize=6.3, linespacing=1.45)
finish_panel(ax, "D", "NapM affects response magnitude more than direction")

# E
ax = fig.add_subplot(gs[2, 0])
xv = association["d_ins_KO_WT"].to_numpy()
yv = association["d_ins_hyp_WT"].to_numpy()
ax.axhline(0, color="#AAAAAA", linewidth=0.5)
ax.axvline(0, color="#AAAAAA", linewidth=0.5)
ax.scatter(xv, yv, s=7, color=C_WT, alpha=0.30, edgecolors="none")
coef = np.polyfit(xv, yv, 1)
xx = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
ax.plot(xx, coef[0] * xx + coef[1], color=C_KO, linewidth=1.0)
ax.set_xlabel("Baseline KO effect (KO − WT)")
ax.set_ylabel("WT hypoxia response (HYP − WT)")
ax.text(0.03, 0.97,
        f"Spearman ρ = {spatial['estimator']:.3f}\n95% CI [{spatial['ci_lo']:.3f}, {spatial['ci_hi']:.3f}]\nCircular-shift P = 0.0011",
        transform=ax.transAxes, va="top", fontsize=6.5, linespacing=1.4)
ax.text(0.97, 0.03, f"Shared exploratory calls: {shared}/{hypoxia_responsive} ({100*shared/hypoxia_responsive:.1f}%)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=C_TEXT)
finish_panel(ax, "E", "Hypoxia response overlaps a pre-existing NapM-associated state")

# F
ax = fig.add_subplot(gs[2, 1])
metric_names = ["Short contacts\n<100 kb", "Intermediate contacts\n100 kb–1 Mb", "Long contacts\n>1 Mb",
                "Arm–arm O/E", "ori–ter O/E", "Compaction slope"]
values = latent[["hypoxia_fold", "latent_fold"]].to_numpy(dtype=float).T
norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.35)
im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(j, i, f"{values[i, j]:.2f}×", ha="center", va="center", fontsize=6.5,
                color="black")
ax.set_xticks(range(6), metric_names, rotation=32, ha="right", rotation_mode="anchor")
ax.set_yticks([0, 1], ["Hypoxia", "Latent"])
ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.1)
ax.tick_params(which="minor", bottom=False, left=False)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.set_label("Fold vs WT", fontsize=6.6)
cbar.ax.tick_params(labelsize=6)
finish_panel(ax, "F", "Latency shows a related but stronger WT signature\n5/6 metrics share direction; KO-latent Hi-C unavailable")

metadata = {"Title": "NapM expression and genotype-associated chromosome reorganization under stress",
            "Author": "", "Subject": "Response figure; editable vector text"}
fig.savefig(OUT / "Response_Fig4_NapM_dependence.pdf", metadata=metadata)
fig.savefig(OUT / "Response_Fig4_NapM_dependence.svg", metadata={"Title": metadata["Title"]})
fig.savefig(OUT / "Response_Fig4_NapM_dependence.png", dpi=400)
plt.close(fig)

legend = f"""Response Fig. 4 | NapM expression and genotype-associated chromosome reorganization under stress. (A) napM expression in WT cells under control, hypoxia and latency. Bars show mean CPM and points show two biological replicates. (B) Genotype-by-hypoxia insulation interaction at 5-kb resolution. Purple denotes detected interaction bins ({n_detected}/{n_total}, {pct_detected:.1f}%); grey denotes other bins. This proportion is not a causal fraction attributable to NapM. (C) Genome-mean insulation in 15 independent samples (WT and WT-hypoxia, n = 4 each; KO, n = 4; KO-hypoxia, n = 3). Points show samples and lines show means. The genotype contrast is reported with a bootstrap 95% CI and exact permutation P value. (D) Nine-metric z-scored hypoxia-response vectors in WT and KO. Response magnitude and direction were tested separately. (E) Association between the baseline KO effect and WT hypoxia response across 881 bins. The red line is a descriptive linear fit; shared calls are exploratory. (F) Hypoxia- and latency-associated fold changes for six structural metrics. Five of six metrics show the same direction. KO-latent Hi-C is unavailable, precluding quantification of NapM dependence during latency.
"""
(OUT / "Response_Fig4_legend.txt").write_text(legend, encoding="utf-8")
(OUT / "figure_build_info.txt").write_text(
    f"Font: {FONT}\nFigure width: 8.27 in (A4, 210 mm)\nFigure height: 11.6 in\nDetected interaction bins: {n_detected}/{n_total} ({pct_detected:.1f}%)\n",
    encoding="utf-8")

print(f"Detected interaction bins: {n_detected}/{n_total} ({pct_detected:.1f}%)")
print(f"Font: {FONT}")
print(f"Saved figure and legend to {OUT}")
