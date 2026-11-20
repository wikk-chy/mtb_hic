#!/usr/bin/env python3
"""Redraw Response Fig. 2 at A4 width with editable text and corrected G/H tracks.

Panel A is intentionally blank except for its 14-pt label. All other panels are
redrawn from source tables at their final physical size; no legacy PDF is used.
"""
from pathlib import Path
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle, FancyArrow
import pymupdf

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
SRC = ROOT / "response_figure_reproducibility/figure_02_hypoxia_factorial_integration/results/hic_review/review_results/hypoxia_latency_integration_20261030/source_data"
SAMPLES = ROOT / "response_figure_reproducibility/figure_01_napm_anchor_integration/results/hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/source_data/Fig2A_independent_samples.csv"
GENERATED = ROOT / "docx_latex_conversion/generated"
MODES = GENERATED / "structure_020_response_modes"
OUT = ROOT / "Response_Fig/version2_responsefigure/A4_redraw"
PANELS = OUT / "panels"
DATA_OUT = OUT / "source_data"
for directory in [OUT, PANELS, DATA_OUT]:
    directory.mkdir(parents=True, exist_ok=True)

W, H = 595.276, 1200.0  # points: A4 width, flexible height
N, BIN, GENOME = 883, 5000, 4411532
FONT = "DejaVu Sans"
WT_COLOR, KO_COLOR = "#3B6FB6", "#B26A3B"
PURPLE, ORANGE, GREEN, GREY = "#6F4E9C", "#D97925", "#2A9D8F", "#C8CDD1"
COLORS = {"RNA only": GREEN, "Structure only": PURPLE,
          "Joint response": ORANGE, "Other / unresolved": GREY}
LABELS = list(COLORS)

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "axes.titlesize": 7.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")


def axes_at(x, y, w, h):
    """Add axes using top-left page coordinates in points."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def text_top(x, y, text, size=7.2, ha="left", color="#202A30", **kwargs):
    return fig.text(x / W, 1 - y / H, text, fontsize=size, ha=ha, va="top",
                    color=color, **kwargs)


def heading(letter, title, x, y, title_x=None):
    text_top(x, y, letter, size=14, fontweight="bold")
    if title:
        text_top(title_x if title_x is not None else x + 23, y + 1, title,
                 size=9, fontweight="bold")


def relative_kb(values, center):
    center_bp = center * BIN + BIN / 2
    return ((np.asarray(values, float) - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000


# A is intentionally reserved for the user's own schematic.
heading("A", "", 20, 18)

# ---------------- B: genotype-by-hypoxia RNA interaction ----------------
genes = pd.read_csv(SRC / "Hypoxia_gene_level.csv")
positive = genes.hypoxia_interaction_RNA_class.eq("detected_increase")
negative = genes.hypoxia_interaction_RNA_class.eq("detected_decrease")
assert int(positive.sum()) == 212 and int(negative.sum()) == 518
heading("B", "Genotype × hypoxia RNA interaction", 20, 165)
ax = axes_at(52, 198, 300, 130)
background = ~positive & ~negative
ax.scatter(genes.loc[background, "interaction"],
           -np.log10(genes.loc[background, "padj_int"]),
           s=5, color="#BFC5CA", alpha=0.45, linewidths=0, rasterized=False)
ax.scatter(genes.loc[positive, "interaction"], -np.log10(genes.loc[positive, "padj_int"]),
           s=8, color=ORANGE, alpha=0.65, linewidths=0,
           label="212 positive interactions", rasterized=False)
ax.scatter(genes.loc[negative, "interaction"], -np.log10(genes.loc[negative, "padj_int"]),
           s=8, color=PURPLE, alpha=0.65, linewidths=0,
           label="518 negative interactions", rasterized=False)
ax.axhline(-np.log10(0.05), color="#999999", lw=0.5, ls="--")
ax.axvline(-1, color="#BBBBBB", lw=0.45, ls="--")
ax.axvline(1, color="#BBBBBB", lw=0.45, ls="--")
for gene_name, offset in [("devR", (5, 5)), ("devS", (5, -12)), ("sigE", (5, 5))]:
    row = genes[genes.gene_name.str.lower().eq(gene_name.lower())].iloc[0]
    display = {"devr": "DosR/DevR", "devs": "DosS/DevS", "sige": "SigE"}[gene_name.lower()]
    ax.scatter(row.interaction, -np.log10(row.padj_int), s=24, color="#202A30", zorder=5)
    ax.annotate(display, (row.interaction, -np.log10(row.padj_int)),
                xytext=offset, textcoords="offset points", fontsize=7.0,
                fontweight="bold", color="#202A30")
ax.text(0.03, 0.97, "730 interaction DEGs", transform=ax.transAxes,
        fontsize=8, fontweight="bold", va="top")
ax.set_xlabel("(KO hypoxia − KO) − (WT hypoxia − WT) RNA log₂FC")
ax.set_ylabel("−log₁₀ BH FDR")
ax.legend(frameon=False, fontsize=7.0, loc="lower left", handletextpad=0.3)
ax.tick_params(pad=1.5)

# ---------------- C: insulation response distributions ----------------
bins_source = pd.read_csv(SRC / "Mechanism_bin_evidence_classes.csv")
wt = bins_source.d_ins_hyp_WT.dropna().to_numpy(float)
ko = bins_source.d_ins_hyp_KO.dropna().to_numpy(float)
heading("C", "Insulation response to hypoxia", 385, 165, title_x=408)
ax = axes_at(437, 195, 118, 100)
parts = ax.violinplot([wt, ko], positions=[1, 2], widths=0.68,
                      showmeans=False, showmedians=False, showextrema=False)
for body, color in zip(parts["bodies"], [WT_COLOR, ORANGE]):
    body.set_facecolor(color)
    body.set_alpha(0.26)
    body.set_edgecolor("none")
rng = np.random.default_rng(42)
for i, (values, color) in enumerate([(wt, WT_COLOR), (ko, ORANGE)], 1):
    x = i + rng.uniform(-0.09, 0.09, len(values))
    ax.scatter(x, values, s=2.2, color=color, alpha=0.10, edgecolors="none", rasterized=False)
    mean, median = np.mean(values), np.median(values)
    ax.scatter(i, mean, marker="D", s=25, color=color, edgecolors="white", linewidths=0.6, zorder=5)
    ax.plot([i - 0.24, i + 0.24], [median, median], color="#263238", lw=1.0, zorder=4)
    ax.text(i, mean + 0.10, f"mean {mean:+.3f}", ha="center", fontsize=7.0, color=color)
ax.axhline(0, color="#999999", lw=0.55, ls="--")
ax.set_xticks([1, 2], ["WT", "ΔnapM"])
ax.set_ylabel("Δ insulation\n(hypoxia − normoxia)")
ax.tick_params(pad=1.5)
samples = pd.read_csv(SAMPLES)
means = samples.groupby("condition").genome_mean_insulation.mean()
wt_response = means["HYP"] - means["WT"]
ko_response = means["KO-HYP"] - means["KO"]
contrast = ko_response - wt_response
assert np.isclose(wt_response, 0.04097577530480988)
assert np.isclose(ko_response, -0.0670136905187206)
text_top(390, 312,
         f"Genome-mean responses: WT {wt_response:+.3f}; ΔnapM {ko_response:+.3f}\n"
         f"Genotype contrast = {contrast:+.3f}\n"
         "Exact oxygen-label P = 4.08 × 10⁻⁴",
         size=7.0, linespacing=1.25)

# ---------------- D/E: final response-mode classification ----------------
df = pd.read_csv(MODES / "Response_Fig2E_structure_020_bins.csv")
tests = pd.read_csv(MODES / "Response_Fig2F_structure_020_enrichment_tests.csv")
observed = np.array([df.structure_020_response_mode.eq(label).sum() for label in LABELS])
assert observed.tolist() == [81, 329, 108, 365]
rna = df.majority_RNA_responsive.to_numpy(dtype=bool)
structure = df.structure_020_responsive.to_numpy(dtype=bool)


def class_counts(rna_mask, structure_mask):
    return np.array([
        np.sum(rna_mask & ~structure_mask), np.sum(structure_mask & ~rna_mask),
        np.sum(rna_mask & structure_mask), np.sum(~rna_mask & ~structure_mask),
    ], dtype=float)


null = np.empty((10000, 4))
rng = np.random.default_rng(42)
for index in range(10000):
    shift = int(rng.integers(1, len(df)))
    null[index] = class_counts(np.roll(rna, shift), structure)
null_mean = null.mean(axis=0)
assert np.allclose(null_mean, tests.null_mean_n.to_numpy(), atol=0.002)

heading("D", "Response-mode counts and proportions", 20, 350)
ax = axes_at(55, 385, 190, 130)
x = np.arange(len(LABELS))
for index, label in enumerate(LABELS):
    ax.bar(x[index], observed[index], width=0.60, color=COLORS[label], alpha=0.84)
    ax.text(x[index], observed[index] + 8,
            f"{int(observed[index])}\n({100 * observed[index] / len(df):.1f}%)",
            ha="center", va="bottom", fontsize=7.0, linespacing=1.2)
ax.set_xticks(x, ["RNA\nonly", "Structure\nonly", "Joint", "Other /\nunresolved"])
ax.set_ylim(0, observed.max() * 1.28)
ax.set_ylabel("Bins, n")
ax.tick_params(axis="x", pad=3)

heading("E", "RNA–structure responses", 280, 350, title_x=303)
ax = axes_at(325, 385, 225, 130)
styles = {
    "Other / unresolved": dict(size=5, alpha=0.38, zorder=1),
    "RNA only": dict(size=9, alpha=0.68, zorder=2),
    "Structure only": dict(size=12, alpha=0.68, zorder=3),
    "Joint response": dict(size=14, alpha=0.75, zorder=4),
}
short_labels = {"Other / unresolved": "Other", "RNA only": "RNA",
                "Structure only": "Structure", "Joint response": "Joint"}
for key in LABELS[::-1]:
    subset = df[df.structure_020_response_mode.eq(key)]
    style = styles[key]
    ax.scatter(subset.median_interaction_log2FC, subset.d_ins_interaction,
               s=style["size"], color=COLORS[key], alpha=style["alpha"],
               edgecolors="none", zorder=style["zorder"], label=short_labels[key],
               rasterized=False)
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
for cutoff in [-0.5, 0.5]:
    ax.axvline(cutoff, color="#BBBBBB", lw=0.45, ls="--")
ax.set_xlabel("Bin-median RNA interaction log₂FC", labelpad=2)
ax.set_ylabel("Insulation interaction (Δ log₂)", labelpad=2)
handles, legend_labels = ax.get_legend_handles_labels()
legend_order = [legend_labels.index(short_labels[key]) for key in LABELS]
ax.legend([handles[i] for i in legend_order], [legend_labels[i] for i in legend_order],
          frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncol=4,
          fontsize=7.0, handletextpad=0.2, columnspacing=0.75, borderaxespad=0)
ax.margins(x=0.055, y=0.10)
ax.tick_params(pad=1.5)

# ---------------- F: spatial null and directional enrichment ----------------
heading("F", "Joint-count spatial null and enrichment tests", 20, 550)
ax = axes_at(52, 590, 220, 100)
joint_null = null[:, 2]
ax.hist(joint_null, bins=36, color=GREY, edgecolor="white", linewidth=0.35)
ax.axvline(null_mean[2], color="#555555", ls="--", lw=1.0,
           label=f"null mean = {null_mean[2]:.1f}")
ax.axvline(observed[2], color=ORANGE, lw=1.6, label=f"observed = {int(observed[2])}")
ax.text(0.97, 0.95,
        f"two-sided P = {tests.P_two_sided_uncorrected.iloc[2]:.4f} {tests.two_sided_star.iloc[2]}\n"
        f"higher P = {tests.P_higher_uncorrected.iloc[2]:.4f} {tests.enrichment_star_higher.iloc[2]}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
        color=ORANGE, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.0))
ax.set_xlabel("Joint-response bins under circular shifts", labelpad=2)
ax.set_ylabel("Randomizations", labelpad=2)
ax.set_title("Circular-shift null", loc="left", fontsize=7.6, fontweight="bold", pad=4)
ax.legend(frameon=False, fontsize=7.0, loc="upper left")
ax.tick_params(pad=1.5)

ax = axes_at(355, 590, 195, 100)
pathway_labels = LABELS[:3]
x = np.arange(len(pathway_labels))
enrichment_p = tests.P_higher_uncorrected.to_numpy()[:3]
ax.bar(x, -np.log10(enrichment_p), color=[COLORS[label] for label in pathway_labels],
       width=0.58, alpha=0.84)
ax.axhline(-np.log10(0.05), color="black", ls="--", lw=0.7)
for index, label in enumerate(pathway_labels):
    ax.text(x[index], -np.log10(enrichment_p[index]) + 0.06,
            f"P={enrichment_p[index]:.3g}\n{tests.enrichment_star_higher.iloc[index]}",
            ha="center", va="bottom", fontsize=7.0, linespacing=1.2)
ax.set_xticks(x, ["RNA only", "Structure only", "Joint"])
ax.set_ylabel("−log₁₀(higher P)", labelpad=2)
ax.set_ylim(0, max(2.7, -np.log10(enrichment_p.min()) + 0.65))
ax.set_title("Higher-than-null tests", loc="left", fontsize=7.6, fontweight="bold", pad=4)
ax.tick_params(pad=1.5)
text_top(30, 710,
         "Empirical P values: 10,000 circular shifts of RNA labels; higher-than-null tests assess enrichment only.",
         size=7.0, color="#52616B")

# ---------------- G/H: validated locus maps and replicate-mean tracks ----------------
orig = pd.read_csv(ROOT / "hic_review/review_results/R1_mechanism_reanalysis/03_structure_metrics/bin_level_v2.csv").set_index("bin")
map_source = pd.read_csv(GENERATED / "Response_Fig2_FI_locus_response_maps.csv")
rep_columns = {
    "ins_WT": [f"ins_WT_rep{i}" for i in range(1, 5)],
    "ins_hypoxia": [f"ins_hypoxia_rep{i}" for i in range(1, 5)],
    "ins_KO": [f"ins_KO_rep{i}" for i in range(1, 5)],
    "ins_KO_hypoxia": [f"ins_KO_hypoxia_rep{i}" for i in range(1, 4)],
}
track_rows = []


def draw_locus(letter, slug, title, center, mode, y0):
    heading(letter, f"{title} | {mode}", 20, y0)
    selected_bins = np.array([(center + i) % N for i in range(-6, 7)], dtype=int)
    pos = relative_kb(selected_bins * BIN + BIN / 2, center)
    row = df[df["bin"].eq(center)].iloc[0]
    assert row.structure_020_response_mode == mode
    text_top(570, y0 + 2,
             f"RNA int. = {row.RNA_interaction_mean_log2FC:+.2f}; insulation int. = {row.d_ins_interaction:+.2f}",
             size=7.0, ha="right", color="#3E474D")

    matrices = {}
    for genotype in ["WT", "ΔnapM"]:
        subset = map_source[(map_source.locus == slug) & (map_source.genotype == genotype)]
        matrix = subset.pivot(index="bin1", columns="bin2", values="log2_hypoxia_normoxia_OE")
        matrices[genotype] = matrix.reindex(index=selected_bins, columns=selected_bins).to_numpy(float)
        assert matrices[genotype].shape == (13, 13)
    finite = np.concatenate([matrices[g][np.isfinite(matrices[g])] for g in matrices])
    limit = max(0.35, float(np.percentile(np.abs(finite), 96)))
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    mesh = None
    for genotype, x0, show_ylabel in [("WT", 50, True), ("ΔnapM", 190, False)]:
        ax_map = axes_at(x0, y0 + 50, 115, 115)
        mesh = ax_map.pcolormesh(pos, pos, matrices[genotype], cmap="RdBu_r", norm=norm,
                                 shading="nearest", rasterized=False)
        ax_map.set_aspect("equal")
        ax_map.set_title(f"{genotype} | hypoxia/normoxia O/E", fontsize=7.2, pad=3)
        ax_map.set_xticks([-25, 0, 25])
        ax_map.set_yticks([-25, 0, 25])
        ax_map.set_xlabel("Position (kb)", labelpad=1)
        if show_ylabel:
            ax_map.set_ylabel("Position (kb)", labelpad=1)
        else:
            ax_map.set_yticklabels([])
        ax_map.tick_params(length=1.8, pad=1)
        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
    cax = axes_at(330, y0 + 60, 8, 82)
    colorbar = fig.colorbar(mesh, cax=cax, ticks=[-limit, 0, limit])
    colorbar.ax.set_yticklabels([f"−{limit:.1f}", "0", f"+{limit:.1f}"])
    colorbar.set_label("O/E", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=7.0, length=1.8, pad=1)
    colorbar.outline.set_linewidth(0.4)

    track = pd.DataFrame({"bin": selected_bins, "position_kb": pos})
    for condition, columns in rep_columns.items():
        track[condition] = orig.loc[selected_bins, columns].mean(axis=1).to_numpy(float)
    focal = track[track["bin"].eq(center)].iloc[0]
    wt_change = focal.ins_hypoxia - focal.ins_WT
    ko_change = focal.ins_KO_hypoxia - focal.ins_KO
    interaction = ko_change - wt_change
    assert np.isclose(interaction, row.d_ins_interaction, atol=1e-8), (slug, interaction, row.d_ins_interaction)
    track.insert(0, "locus", slug)
    track_rows.append(track)

    ax_ins = axes_at(390, y0 + 45, 165, 48)
    conditions = [("ins_WT", WT_COLOR, "-"), ("ins_hypoxia", WT_COLOR, "--"),
                  ("ins_KO", KO_COLOR, "-"), ("ins_KO_hypoxia", KO_COLOR, "--")]
    for condition, color, linestyle in conditions:
        ax_ins.plot(pos, track[condition], color=color, ls=linestyle, lw=0.9,
                    label={"ins_WT": "WT", "ins_hypoxia": "WT hypoxia",
                           "ins_KO": "ΔnapM", "ins_KO_hypoxia": "ΔnapM hypoxia"}[condition])
    ax_ins.set_xlim(-32.5, 32.5)
    ax_ins.margins(y=0.22)
    y_low, y_high = ax_ins.get_ylim()
    ax_ins.add_patch(Rectangle((-2.5, y_low), 5, y_high - y_low, fill=False,
                               ls="--", lw=0.55, edgecolor="#555555"))
    ax_ins.set_ylabel("Ins.", fontsize=7.0, rotation=0, labelpad=5)
    ax_ins.set_xticks([-25, 0, 25])
    ax_ins.tick_params(length=1.8, pad=1, labelbottom=False)
    ax_ins.legend(frameon=False, fontsize=7.0, ncol=2, loc="lower left",
                  bbox_to_anchor=(0, 1.02), handlelength=1.2, handletextpad=0.3,
                  columnspacing=0.6, borderaxespad=0, labelspacing=0.15)
    ax_ins.axhline(0, color="#DDDDDD", lw=0.35, zorder=0)

    local_genes = genes[genes.bin_tss.isin(selected_bins)].copy()
    annotation_genes = local_genes[local_genes.bin_tss.eq(center)].copy()
    named = annotation_genes[annotation_genes.gene_name.fillna("").str.strip().ne("")].drop_duplicates("gene_name")
    if not named.empty:
        annotation_genes = named
    elif annotation_genes.empty:
        annotation_genes = local_genes.assign(abs_interaction=local_genes.interaction.abs()).nlargest(3, "abs_interaction")
    elif len(annotation_genes) > 4:
        annotation_genes = annotation_genes.assign(abs_interaction=annotation_genes.interaction.abs()).nlargest(4, "abs_interaction")

    ax_rna = axes_at(390, y0 + 108, 165, 40)
    offsets = [(-20, 4), (0, 5), (20, 4), (0, -12)]
    for index, gene in enumerate(annotation_genes.itertuples()):
        x_gene = relative_kb(gene.tss, center)
        value = gene.interaction
        color = GREEN if value >= 0 else PURPLE
        ax_rna.vlines(x_gene, 0, value, color=color, lw=1.15, zorder=3)
        ax_rna.scatter(x_gene, value, s=11, color=color, zorder=4)
        ax_rna.annotate(f"{value:+.2f}", (x_gene, value), xytext=offsets[index % 4],
                        textcoords="offset points", fontsize=7.0, ha="center", color=color)
    max_abs = float(np.nanmax(np.abs(annotation_genes.interaction.to_numpy(float))))
    ax_rna.set_ylim(-max(1.3, max_abs * 1.28), max(1.3, max_abs * 1.28))
    ax_rna.set_xlim(-32.5, 32.5)
    ax_rna.axhline(0, color="#999999", lw=0.45)
    ax_rna.axhline(-1, color="#DDDDDD", lw=0.4, ls="--")
    ax_rna.axhline(1, color="#DDDDDD", lw=0.4, ls="--")
    ax_rna.set_ylabel("RNA", fontsize=7.0, rotation=0, labelpad=5)
    ax_rna.set_xticks([-25, 0, 25])
    ax_rna.tick_params(length=1.8, pad=1, labelbottom=False)

    ax_gene = axes_at(390, y0 + 162, 165, 25)
    ax_gene.set_ylim(-1, 1)
    ax_gene.set_xlim(-32.5, 32.5)
    ax_gene.set_yticks([])
    gene_label_offsets = [-20, 0, 20, 0] if len(annotation_genes) > 1 else [0]
    for index, gene in enumerate(annotation_genes.itertuples()):
        left = max(-32.5, relative_kb(gene.start, center))
        right = min(32.5, relative_kb(gene.end, center))
        if right < left:
            left, right = right, left
        y = 0.35 if index % 2 == 0 else -0.15
        direction = 1 if gene.strand == "+" else -1
        start = left if direction == 1 else right
        width = max(right - left, 0.35) * direction
        ax_gene.add_patch(FancyArrow(start, y, width, 0, width=0.10,
                                     head_width=0.24, head_length=min(0.8, max(abs(width) * 0.25, 0.18)),
                                     length_includes_head=True, facecolor="#687983",
                                     edgecolor="none", alpha=0.88))
        name = gene.gene_name if isinstance(gene.gene_name, str) and gene.gene_name else gene.gene_id
        ax_gene.annotate(name, ((left + right) / 2, y),
                         xytext=(gene_label_offsets[index % len(gene_label_offsets)],
                                 4 if index % 2 == 0 else -3),
                         textcoords="offset points", ha="center",
                         va="bottom" if index % 2 == 0 else "top",
                         fontsize=7.0, color="#34444D")
    ax_gene.set_xticks([-25, 0, 25])
    ax_gene.set_xlabel("Position relative to focal bin (kb)", fontsize=7.0, labelpad=1)
    ax_gene.tick_params(length=1.8, pad=1)
    for side in ["top", "right", "left"]:
        ax_gene.spines[side].set_visible(False)


draw_locus("G", "recR", "recR", 832, "Structure only", 740)
draw_locus("H", "accA2_accD2_fadE13", "accA2/accD2/fadE13", 217, "Joint response", 970)

# ---------------- Save complete vector figure ----------------
pdf_path = OUT / "Response_Fig_02_A4_redraw.pdf"
svg_path = OUT / "Response_Fig_02_A4_redraw.svg"
png_path = OUT / "Response_Fig_02_A4_redraw.png"
fig.savefig(pdf_path)
fig.savefig(svg_path)
fig.savefig(png_path, dpi=220)
plt.close(fig)

pd.concat(track_rows, ignore_index=True).to_csv(DATA_OUT / "Response_Fig2_GH_replicate_mean_insulation.csv", index=False)
for filename in ["Response_Fig2E_structure_020_bins.csv", "Response_Fig2F_structure_020_enrichment_tests.csv"]:
    shutil.copy2(MODES / filename, DATA_OUT / filename)
for source, target in [
    (GENERATED / "Response_Fig2_FI_locus_response_maps.csv", "Response_Fig2_GH_response_maps.csv"),
    (SRC / "Hypoxia_gene_level.csv", "Response_Fig2_B_and_GH_gene_level.csv"),
    (SAMPLES, "Response_Fig2_C_independent_samples.csv"),
]:
    shutil.copy2(source, DATA_OUT / target)
shutil.copy2(Path(__file__), OUT / "redraw_response_fig2_a4.py")

# Independent, editable B-H panels cropped without rasterization.
clips = {
    "B": (10, 155, 375, 340),
    "C": (380, 155, 585, 340),
    "D": (10, 340, 265, 535),
    "E": (275, 340, 585, 535),
    "F": (10, 540, 585, 730),
    "G": (10, 730, 585, 960),
    "H": (10, 960, 585, 1195),
}
with pymupdf.open(pdf_path) as source:
    for letter, coordinates in clips.items():
        rect = pymupdf.Rect(coordinates)
        document = pymupdf.open()
        page = document.new_page(width=rect.width, height=rect.height)
        page.show_pdf_page(page.rect, source, 0, clip=rect, keep_proportion=False)
        panel_pdf = PANELS / f"Response_Fig_02{letter}_editable.pdf"
        document.save(panel_pdf, garbage=4, clean=True, deflate=True)
        document.close()
        with pymupdf.open(panel_pdf) as panel_document:
            svg = panel_document[0].get_svg_image(text_as_path=False)
            assert "<text" in svg, f"No editable SVG text in panel {letter}"
            (PANELS / f"Response_Fig_02{letter}_editable.svg").write_text(svg, encoding="utf-8")

# QA: physical dimensions, editable text, font sizes and panel-label consistency.
with pymupdf.open(pdf_path) as document:
    page = document[0]
    spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
             for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    sizes = [float(span["size"]) for span in spans]
    label_sizes = {letter: [float(span["size"]) for span in spans if span["text"].strip() == letter]
                   for letter in "ABCDEFGH"}
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
    preview = OUT / "Response_Fig_02_A4_preview.png"
    pix.save(preview)
    report = {
        "pdf": str(pdf_path),
        "page_width_pt": page.rect.width,
        "page_height_pt": page.rect.height,
        "A4_width": bool(np.isclose(page.rect.width, 595.276, atol=0.01)),
        "pages": len(document),
        "text_span_count": len(spans),
        "minimum_text_size_pt": min(sizes),
        "all_text_at_least_7pt": min(sizes) >= 6.99,
        "panel_label_sizes_pt": label_sizes,
        "all_panel_labels_14pt_bold": all(
            len(label_sizes[letter]) == 1 and np.isclose(label_sizes[letter][0], 14.0, atol=0.01)
            for letter in "ABCDEFGH"),
        "raster_image_count_in_pdf": len(page.get_images(full=True)),
        "optional_content_groups": document.get_ocgs(),
        "panel_A_content_omitted": True,
        "GH_track_rule": "Arithmetic mean of per-replicate circular diamond insulation scores; w=5 bins",
        "GH_interactions": {
            "G_recR": float(df.loc[df["bin"].eq(832), "d_ins_interaction"].iloc[0]),
            "H_accA2_accD2_fadE13": float(df.loc[df["bin"].eq(217), "d_ins_interaction"].iloc[0]),
        },
        "mode_counts": dict(zip(LABELS, observed.astype(int).tolist())),
    }
assert report["A4_width"]
assert report["all_text_at_least_7pt"], report["minimum_text_size_pt"]
assert report["all_panel_labels_14pt_bold"], label_sizes
assert report["text_span_count"] > 100
(OUT / "qa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

archive = OUT / "Response_Fig_02_A4_editable_bundle.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in [pdf_path, svg_path, OUT / "redraw_response_fig2_a4.py", OUT / "qa_report.json"]:
        bundle.write(path, path.relative_to(OUT))
    for directory in [PANELS, DATA_OUT]:
        for path in sorted(directory.glob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(OUT))

print(json.dumps(report, indent=2, ensure_ascii=False))
print("Saved:", pdf_path)
print("Saved:", svg_path)
print("Saved:", archive)
