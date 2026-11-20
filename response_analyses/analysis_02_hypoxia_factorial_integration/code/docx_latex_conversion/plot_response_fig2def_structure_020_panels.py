#!/usr/bin/env python3
"""Draw final-size Response Fig. 2D/E/F using the 0.20 structure threshold.

All text is >=7 pt, fonts remain editable (TrueType PDF and text-based SVG), and
canvas sizes exactly match the destination rectangles so no panel is rescaled.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
INPUT = ROOT / "docx_latex_conversion/generated/structure_020_response_modes/Response_Fig2E_structure_020_bins.csv"
TESTS = ROOT / "docx_latex_conversion/generated/structure_020_response_modes/Response_Fig2F_structure_020_enrichment_tests.csv"
OUT = ROOT / "docx_latex_conversion/generated/structure_020_final_panels"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["RNA only", "Structure only", "Joint response", "Other / unresolved"]
COLORS = {"RNA only": "#2A9D8F", "Structure only": "#6F4E9C",
          "Joint response": "#D97925", "Other / unresolved": "#C8CDD1"}
N_SHIFT, SEED = 10000, 42

df = pd.read_csv(INPUT)
tests = pd.read_csv(TESTS)
observed = np.array([df.structure_020_response_mode.eq(label).sum() for label in LABELS])
rna = df.majority_RNA_responsive.to_numpy(dtype=bool)
structure = df.structure_020_responsive.to_numpy(dtype=bool)


def class_counts(rna_mask, structure_mask):
    return np.array([
        np.sum(rna_mask & ~structure_mask), np.sum(structure_mask & ~rna_mask),
        np.sum(rna_mask & structure_mask), np.sum(~rna_mask & ~structure_mask),
    ], dtype=float)


rng = np.random.default_rng(SEED)
null = np.empty((N_SHIFT, 4))
for index in range(N_SHIFT):
    shift = int(rng.integers(1, len(df)))
    null[index] = class_counts(np.roll(rna, shift), structure)
null_mean = null.mean(axis=0)
null_lo, null_hi = np.percentile(null, [2.5, 97.5], axis=0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.labelsize": 7.4, "axes.titlesize": 9.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def save_panel(fig, stem):
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{stem}.{extension}", dpi=400 if extension == "png" else None)
    plt.close(fig)


# ---------------- D: exact 166.276 x 193 pt ----------------
fig, ax = plt.subplots(figsize=(166.276 / 72, 193 / 72))
x = np.arange(len(LABELS))
for index, label in enumerate(LABELS):
    ax.bar(x[index], observed[index], width=0.60, color=COLORS[label], alpha=0.84)
    ax.text(x[index], observed[index] + 9,
            f"{int(observed[index])}\n({100*observed[index]/len(df):.1f}%)",
            ha="center", fontsize=7.0)
ax.set_xticks(x, ["RNA\nonly", "Structure\nonly", "Joint", "Other /\nunresolved"])
ax.set_ylim(0, observed.max() * 1.27)
ax.set_ylabel("Bins, n")
ax.set_title("D  Response-mode counts\nand proportions", loc="left", fontweight="bold", pad=7)
ax.tick_params(axis="x", pad=3)
fig.subplots_adjust(left=0.15, right=0.98, top=0.78, bottom=0.24)
save_panel(fig, "Response_Fig_02D_structure_020")

# ---------------- E: exact 207 x 139 pt ----------------
fig, ax = plt.subplots(figsize=(207 / 72, 139 / 72))
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
               edgecolors="none", zorder=style["zorder"],
               label=short_labels[key])
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
for cutoff in [-0.5, 0.5]:
    ax.axvline(cutoff, color="#BBBBBB", lw=0.45, ls="--")
ax.set_xlabel("Bin-median RNA interaction log$_2$FC", labelpad=2)
ax.set_ylabel("Insulation interaction (Δ log$_2$)", labelpad=2)
ax.set_title("E  RNA–structure responses", loc="left", fontweight="bold", pad=6)
ax.legend(frameon=True, framealpha=0.92, edgecolor="none", loc="upper left",
          bbox_to_anchor=(1.03, 1.0), ncol=1, fontsize=7.0,
          handletextpad=0.25, borderpad=0.25, labelspacing=0.25)
ax.margins(x=0.055, y=0.10)
ax.tick_params(pad=1.5)
fig.subplots_adjust(left=0.17, right=0.68, top=0.80, bottom=0.25)
save_panel(fig, "Response_Fig_02E_structure_020")

# ---------------- F: exact 595.276 x 149.63 pt ----------------
fig = plt.figure(figsize=(595.276 / 72, 149.63 / 72))
grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.26,
                        left=0.065, right=0.985, top=0.75, bottom=0.25)

ax = fig.add_subplot(grid[0, 0])
joint_null = null[:, 2]
ax.hist(joint_null, bins=36, color=COLORS["Other / unresolved"],
        edgecolor="white", linewidth=0.35)
ax.axvline(null_mean[2], color="#555555", ls="--", lw=1.0,
           label=f"null mean = {null_mean[2]:.1f}")
ax.axvline(observed[2], color=COLORS["Joint response"], lw=1.6,
           label=f"observed = {int(observed[2])}")
ax.text(0.97, 0.95,
        f"two-sided P = {tests.P_two_sided_uncorrected.iloc[2]:.4f} {tests.two_sided_star.iloc[2]}\n"
        f"higher P = {tests.P_higher_uncorrected.iloc[2]:.4f} {tests.enrichment_star_higher.iloc[2]}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
        color=COLORS["Joint response"], fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.2))
ax.set_xlabel("Joint-response bins under circular shifts", labelpad=2)
ax.set_ylabel("Randomizations", labelpad=2)
ax.set_title("F  Joint-count spatial null", loc="left", fontweight="bold", pad=6)
ax.legend(frameon=False, fontsize=7.0, loc="upper left")
ax.tick_params(pad=1.5)

ax = fig.add_subplot(grid[0, 1])
pathway_labels = LABELS[:3]
x = np.arange(len(pathway_labels))
enrichment_p = tests.P_higher_uncorrected.to_numpy()[:3]
ax.bar(x, -np.log10(enrichment_p), color=[COLORS[label] for label in pathway_labels],
       width=0.58, alpha=0.84)
ax.axhline(-np.log10(.05), color="black", ls="--", lw=0.7)
for index, label in enumerate(pathway_labels):
    ax.text(x[index], -np.log10(enrichment_p[index]) + 0.06,
            f"P={enrichment_p[index]:.3g}\n{tests.enrichment_star_higher.iloc[index]}",
            ha="center", fontsize=7.0)
ax.set_xticks(x, ["RNA only", "Structure only", "Joint"])
ax.set_ylabel("−log$_{10}$(higher P)", labelpad=2)
ax.set_ylim(0, max(2.7, -np.log10(enrichment_p.min()) + 0.65))
ax.set_title("Higher-than-null enrichment tests\n(uncorrected empirical P values)",
             loc="left", fontweight="bold", pad=6)
ax.tick_params(pad=1.5)
fig.text(0.065, 0.055,
         "Uncorrected empirical P values from 10,000 circular shifts of RNA labels; higher-than-null tests evaluate enrichment only.",
         fontsize=7.0, color="#555555")
save_panel(fig, "Response_Fig_02F_structure_020")
print("Saved final-size D/E/F panels with editable fonts >=7 pt")
