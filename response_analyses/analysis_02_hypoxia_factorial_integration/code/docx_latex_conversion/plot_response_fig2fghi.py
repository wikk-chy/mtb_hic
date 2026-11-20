#!/usr/bin/env python3
"""Draw new Response Fig. 2F–I representative loci for the three response modes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
SRC = ROOT / "response_figure_reproducibility/figure_02_hypoxia_factorial_integration/results/hic_review/review_results/hypoxia_latency_integration_20261030/source_data"
OUT = ROOT / "docx_latex_conversion/generated"
RA = ROOT / "hic_review/review_results/R1_mechanism_reanalysis"
N, BIN, GENOME = 883, 5000, 4411532
DET = ["detected_increase", "detected_decrease"]

bins = pd.read_csv(SRC / "Mechanism_bin_evidence_classes.csv")
genes = pd.read_csv(SRC / "Hypoxia_gene_level.csv")
orig = pd.read_csv(RA / "03_structure_metrics/bin_level_v2.csv").set_index("bin")
bound = bins.NapM_bound.fillna(False).astype(bool)
structure = bins.interaction_ins_class.isin(DET)
rna = bins.RNA_interaction_any_DEG.fillna(False).astype(bool)
any_rna = bins[["RNA_WT_hypoxia_any_DEG", "RNA_KO_hypoxia_any_DEG",
                "RNA_interaction_any_DEG"]].fillna(False).astype(bool).any(axis=1)
full = pd.read_csv(SRC / "Hypoxia_bin_level.csv")
tested = full[["RNA_WT_hypoxia_n_tested_genes", "RNA_KO_hypoxia_n_tested_genes",
               "RNA_interaction_n_tested_genes"]].min(axis=1).fillna(0).gt(0)
small = full[["RNA_WT_hypoxia_mean_abs_log2FC", "RNA_KO_hypoxia_mean_abs_log2FC",
              "RNA_interaction_mean_abs_log2FC"]].max(axis=1).lt(0.5)
stringent = structure & tested & small & ~any_rna

loci = [
    dict(letter="F", slug="dinG", title="dinG", center=299, mode="RNA only",
         rationale="Strong positive RNA interaction with minimal focal insulation change."),
    dict(letter="G", slug="recR", title="recR", center=832, mode="Structure only",
         rationale="Strong negative insulation interaction with stringent absence of a local RNA response."),
    dict(letter="H", slug="accA2_accD2_fadE13", title="accA2/accD2/fadE13", center=217,
         mode="Joint response", rationale="Concordant negative RNA and insulation interactions."),
]
for loc in loci:
    center = loc["center"]
    loc["bins"] = np.array([(center + i) % N for i in range(-6, 7)], dtype=int)
    center_bp = center * BIN + BIN / 2
    loc["pos_kb"] = ((loc["bins"] * BIN + BIN / 2 - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000
    row = bins[bins["bin"].eq(center)].iloc[0]
    loc["row"] = row
    if loc["mode"] == "RNA only":
        assert rna.loc[row.name] and not structure.loc[row.name]
    elif loc["mode"] == "Structure only":
        assert stringent.loc[row.name]
    else:
        assert rna.loc[row.name] and structure.loc[row.name]


def load_matrix(path):
    d = pd.read_csv(path, sep=r"\s+", header=None, names=["b1", "b2", "v"])
    m = np.zeros((N, N))
    i, j = d.b1.to_numpy(int) - 1, d.b2.to_numpy(int) - 1
    m[i, j] = d.v.to_numpy(float)
    m[j, i] = m[i, j]
    return m


def circular_expected(m):
    idx = np.arange(N)
    expected = np.full(N // 2 + 1, np.nan)
    for offset in range(N // 2 + 1):
        vals = m[idx, (idx + offset) % N]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        expected[offset] = vals.mean() if len(vals) else np.nan
    return expected


def local_oe(m, expected, selected):
    i, j = np.meshgrid(selected, selected, indexing="ij")
    delta = np.abs(j - i)
    dist = np.minimum(delta, N - delta)
    values = m[np.ix_(selected, selected)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((values > 0) & np.isfinite(values) & (expected[dist] > 0),
                        values / expected[dist], np.nan)


def relative_kb(values, center):
    center_bp = center * BIN + BIN / 2
    return ((np.asarray(values, float) - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000


cond_files = {
    "WT": ("wt", 4), "KO": ("napm_ko", 4),
    "WT_hypoxia": ("hypoxia", 4), "KO_hypoxia": ("ko-hypoxia", 3),
}
mean_maps = {cond: {} for cond in cond_files}
for cond, (prefix, nrep) in cond_files.items():
    for loc in loci:
        rep_maps = []
        for rep in range(nrep):
            path = ROOT / f"hic_review/review_data/rep_icedmatrix/{prefix}_{rep+1}_5000_iced.matrix"
            matrix = load_matrix(path)
            expected = circular_expected(matrix)
            rep_maps.append(local_oe(matrix, expected, loc["bins"]))
        mean_maps[cond][loc["slug"]] = np.nanmean(np.stack(rep_maps), axis=0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0,
    "axes.labelsize": 7.0, "axes.titlesize": 7.4,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
WT_COLOR, KO_COLOR = "#3B6FB6", "#B26A3B"
map_rows, ins_rows, rna_rows, selection_rows = [], [], [], []
for loc in loci[1:]:
    slug, center, pos = loc["slug"], loc["center"], loc["pos_kb"]
    wt_response = np.log2(mean_maps["WT_hypoxia"][slug] / mean_maps["WT"][slug])
    ko_response = np.log2(mean_maps["KO_hypoxia"][slug] / mean_maps["KO"][slug])
    finite = np.concatenate([wt_response[np.isfinite(wt_response)], ko_response[np.isfinite(ko_response)]])
    print(f"{slug}: finite response pixels = {finite.size}/{wt_response.size + ko_response.size} ({100 * finite.size / (wt_response.size + ko_response.size):.1f}%)")
    limit = max(0.35, float(np.nanpercentile(np.abs(finite), 96)) if finite.size else 0.5)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig = plt.figure(figsize=(9.0, 2.40))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.08, 1.08, 2.30],
                          left=0.045, right=0.985, top=0.75, bottom=0.15,
                          wspace=0.28, hspace=0.68)
    fig.text(0.045, 0.94, f"{loc['letter']}  {loc['title']}  |  {loc['mode']}",
             fontsize=7.4, fontweight="bold", va="top")
    fig.text(0.985, 0.94,
             f"RNA int. = {loc['row'].RNA_interaction_mean_log2FC:+.2f}; "
             f"insulation int. = {loc['row'].d_ins_interaction:+.2f}",
             fontsize=7.0, ha="right", va="top", color="#3E474D")
    for col, (label, values) in enumerate([("WT", wt_response), ("ΔnapM", ko_response)]):
        ax = fig.add_subplot(gs[:, col])
        mesh = ax.pcolormesh(pos, pos, values, cmap="RdBu_r", norm=norm,
                             shading="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  |  hypoxia/normoxia O/E", fontsize=7.0, pad=2)
        ax.set_xticks([-25, 0, 25])
        ax.set_yticks([-25, 0, 25])
        ax.set_xlabel("Position (kb)", labelpad=1)
        if col == 0:
            ax.set_ylabel("Position (kb)", labelpad=1)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
        ax.tick_params(length=1.5, pad=1)
        for a, b1 in enumerate(loc["bins"]):
            for c, b2 in enumerate(loc["bins"]):
                map_rows.append(dict(locus=slug, genotype=label, bin1=int(b1), bin2=int(b2),
                                     position1_kb=pos[a], position2_kb=pos[c],
                                     log2_hypoxia_normoxia_OE=values[a, c]))
    axins = fig.add_subplot(gs[0, 2])
    conditions = [("ins_WT", WT_COLOR, "-"), ("ins_hypoxia", WT_COLOR, "--"),
                  ("ins_KO", KO_COLOR, "-"), ("ins_KO_hypoxia", KO_COLOR, "--")]
    for cond, color, ls in conditions:
        values = orig.loc[loc["bins"], cond].to_numpy(float)
        axins.plot(pos, values, color=color, ls=ls, lw=0.85,
                   label={"ins_WT": "WT normoxia", "ins_hypoxia": "WT hypoxia",
                          "ins_KO": "ΔnapM normoxia", "ins_KO_hypoxia": "ΔnapM hypoxia"}[cond])
    axins.add_patch(Rectangle((-2.5, axins.get_ylim()[0]), 5,
                              axins.get_ylim()[1] - axins.get_ylim()[0],
                              fill=False, ls="--", lw=0.55, edgecolor="#555555"))
    axins.set_title("Insulation score (log2)", fontsize=7.0, pad=2)
    axins.set_xticks([-25, 0, 25])
    axins.tick_params(length=1.5, pad=1)
    axins.legend(frameon=True, framealpha=0.90, edgecolor="none", fontsize=7.0,
                 ncol=2, loc="lower left", handlelength=1.2)
    axins.axhline(0, color="#CCCCCC", lw=0.35)
    local_genes = genes[genes.bin_tss.isin(loc["bins"])].copy()
    for row in local_genes.itertuples():
        rna_rows.append(dict(locus=slug, gene_id=row.gene_id, gene_name=row.gene_name,
                             bin_tss=int(row.bin_tss), position_kb=relative_kb(row.tss, center),
                             RNA_interaction_log2FC=row.interaction, FDR=row.padj_int))
    annotation_genes = local_genes[local_genes.bin_tss.eq(center)].copy()
    named = annotation_genes[annotation_genes.gene_name.fillna("").str.strip().ne("")].drop_duplicates("gene_name")
    if not named.empty:
        annotation_genes = named
    elif annotation_genes.empty:
        annotation_genes = local_genes.assign(abs_interaction=local_genes.interaction.abs()).nlargest(3, "abs_interaction")
    elif len(annotation_genes) > 4:
        annotation_genes = annotation_genes.assign(abs_interaction=annotation_genes.interaction.abs()).nlargest(4, "abs_interaction")

    # TSS-aligned lollipops share the genomic x-axis with insulation and gene models.
    axrna = fig.add_subplot(gs[1, 2])
    rna_x_offsets = [-18, 0, 18, 0]
    for idx, row in enumerate(annotation_genes.itertuples()):
        x = relative_kb(row.tss, center)
        value = row.interaction
        color = "#2A9D8F" if value >= 0 else "#6F4E9C"
        axrna.vlines(x, 0, value, color=color, lw=1.25, zorder=3)
        axrna.scatter(x, value, s=12, color=color, zorder=4)
        axrna.annotate(f"{value:+.2f}", (x, value),
                       xytext=(rna_x_offsets[idx % 4], 5), textcoords="offset points",
                       ha="center", va="bottom", fontsize=7.0, color="#3E474D")
    axrna.axhline(0, color="#777777", lw=0.45)
    axrna.axhline(1, color="#BBBBBB", lw=0.45, ls="--")
    axrna.axhline(-1, color="#BBBBBB", lw=0.45, ls="--")
    axrna.set_xticks([-25, 0, 25])
    axrna.set_ylim(-max(1.3, float(np.nanmax(np.abs(annotation_genes.interaction.to_numpy(float)))) * 1.22),
                   max(1.3, float(np.nanmax(np.abs(annotation_genes.interaction.to_numpy(float)))) * 1.22))
    axrna.set_title("Focal-gene RNA interaction log2FC", fontsize=7.0, pad=2)
    axrna.tick_params(length=1.5, pad=1)

    axgene = fig.add_subplot(gs[2, 2])
    axgene.set_ylim(-1.0, 1.0)
    axgene.set_yticks([])
    gene_x_offsets = [-16, 0, 16, 0]
    for idx, row in enumerate(annotation_genes.itertuples()):
        left = relative_kb(row.start, center)
        right = relative_kb(row.end, center)
        y = 0.60 if idx % 2 == 0 else -0.60
        axgene.plot([left, right], [y, y], color="#6F787D", lw=0.8, solid_capstyle="butt")
        label = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
        axgene.annotate(label, ((left + right) / 2, y),
                        xytext=(gene_x_offsets[idx % 4], 12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.0, color="#3E474D")
    axgene.set_title("Gene annotation", fontsize=7.0, pad=2)
    axgene.set_xticks([-25, 0, 25])
    axgene.tick_params(length=1.5, pad=1)
    for ax in (axins, axrna, axgene):
        ax.set_xlim(pos.min() - 1.5, pos.max() + 1.5)
    output = ROOT / "Response_Fig/version2_responsefigure/editable/panels" / f"Response_Fig_02{loc['letter']}_editable.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    selection_rows.append(dict(panel=loc["letter"], locus=loc["title"], center_bin=center,
                               response_mode=loc["mode"], NapM_bound=bool(loc["row"].NapM_bound),
                               RNA_interaction=loc["row"].RNA_interaction_mean_log2FC,
                               insulation_interaction=loc["row"].d_ins_interaction,
                               insulation_FDR=loc["row"].interaction_padj,
                               selection_rationale=loc["rationale"]))

pd.DataFrame(map_rows).to_csv(OUT / "Response_Fig2_FI_locus_response_maps.csv", index=False)
pd.DataFrame(rna_rows).to_csv(OUT / "Response_Fig2_FI_locus_RNA_tracks.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT / "Response_Fig2_FI_locus_selection.csv", index=False)
print(pd.DataFrame(selection_rows)[["panel", "locus", "center_bin", "response_mode",
                                    "RNA_interaction", "insulation_interaction"]].to_string(index=False))
