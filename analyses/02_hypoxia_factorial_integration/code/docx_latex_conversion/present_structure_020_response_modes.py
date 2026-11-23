#!/usr/bin/env python3
"""Present four response classes using the 0.20 structural-effect threshold.

D shows class counts and proportions. E shows RNA-structure joint distributions.
F shows the joint-count spatial null and uncorrected two-sided P value, plus
one-sided higher-than-null enrichment tests for all four classes. No multiple-
testing correction is applied, as requested; results are labelled exploratory.
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
INPUT = ROOT / "docx_latex_conversion/generated/tiered_structure_modes/Response_Fig2E_tiered_structure_bins.csv"
OUT = ROOT / "docx_latex_conversion/generated/structure_020_response_modes"
OUT.mkdir(parents=True, exist_ok=True)
N_SHIFT, SEED = 10000, 42
LABELS = ["RNA only", "Structure only", "Joint response", "Other / unresolved"]
THRESHOLD = 0.20

df = pd.read_csv(INPUT)
lo, hi = df.d_ins_interaction_lo, df.d_ins_interaction_hi
strong = (lo.gt(THRESHOLD) | hi.lt(-THRESHOLD)).to_numpy()
moderate = (~strong & (lo.gt(0) | hi.lt(0)) &
            df.d_ins_interaction.abs().ge(THRESHOLD).to_numpy())
structure = strong | moderate
rna = df.majority_RNA_responsive.to_numpy(dtype=bool)
df["structure_020_strong"] = strong
df["structure_020_moderate"] = moderate
df["structure_020_responsive"] = structure
df["structure_020_response_mode"] = np.select(
    [rna & ~structure, structure & ~rna, rna & structure],
    LABELS[:3], default=LABELS[3])
df.to_csv(OUT / "Response_Fig2E_structure_020_bins.csv", index=False)


def class_counts(rna_mask, structure_mask):
    return np.array([
        np.sum(rna_mask & ~structure_mask), np.sum(structure_mask & ~rna_mask),
        np.sum(rna_mask & structure_mask), np.sum(~rna_mask & ~structure_mask),
    ], dtype=float)


n = len(df)
observed = class_counts(rna, structure)
independent = np.array([
    rna.sum() * (n - structure.sum()) / n,
    structure.sum() * (n - rna.sum()) / n,
    rna.sum() * structure.sum() / n,
    (n - rna.sum()) * (n - structure.sum()) / n])
rng = np.random.default_rng(SEED)
null = np.empty((N_SHIFT, 4))
for index in range(N_SHIFT):
    shift = int(rng.integers(1, n))
    null[index] = class_counts(np.roll(rna, shift), structure)
null_mean = null.mean(axis=0)
null_lo, null_hi = np.percentile(null, [2.5, 97.5], axis=0)
p_high = (1 + (null >= observed).sum(axis=0)) / (N_SHIFT + 1)
p_low = (1 + (null <= observed).sum(axis=0)) / (N_SHIFT + 1)
p_two = (1 + (np.abs(null - null_mean) >= np.abs(observed - null_mean)).sum(axis=0)) / (N_SHIFT + 1)

def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

rows = []
for index, label in enumerate(LABELS):
    rows.append({
        "response_mode": label,
        "observed_n": int(observed[index]),
        "observed_percent": 100 * observed[index] / n,
        "independent_expected_n": independent[index],
        "null_mean_n": null_mean[index],
        "null_95_low_n": null_lo[index], "null_95_high_n": null_hi[index],
        "observed_to_null_ratio": observed[index] / null_mean[index],
        "P_higher_uncorrected": p_high[index],
        "P_lower_uncorrected": p_low[index],
        "P_two_sided_uncorrected": p_two[index],
        "enrichment_star_higher": star(p_high[index]),
        "two_sided_star": star(p_two[index]),
    })
results = pd.DataFrame(rows)
results.to_csv(OUT / "Response_Fig2F_structure_020_enrichment_tests.csv", index=False)
summary = {
    "structure_threshold": THRESHOLD,
    "n_strong_structure": int(strong.sum()),
    "n_moderate_structure": int(moderate.sum()),
    "n_structure_total": int(structure.sum()),
    "n_RNA": int(rna.sum()),
    "class_counts": {label: int(observed[index]) for index, label in enumerate(LABELS)},
    "joint_null_mean": float(null_mean[2]),
    "joint_ratio": float(observed[2] / null_mean[2]),
    "joint_P_two_sided_uncorrected": float(p_two[2]),
    "joint_P_higher_uncorrected": float(p_high[2]),
    "n_circular_shifts": N_SHIFT,
    "seed": SEED,
    "multiple_testing": "Not applied; uncorrected exploratory empirical P values",
    "interpretation_note": "Fixed RNA and structure margins make all four class deviations transformations of the joint count. Higher-than-null enrichment tests are shown separately for each class.",
}
(OUT / "Response_Fig2DEF_structure_020_summary.json").write_text(json.dumps(summary, indent=2))
print(results.to_string(index=False), flush=True)

# ---------------- D/E/F figure ----------------
colors = {"RNA only": "#2A9D8F", "Structure only": "#6F4E9C",
          "Joint response": "#D97925", "Other / unresolved": "#C8CDD1"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.labelsize": 7.4, "axes.titlesize": 9.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig = plt.figure(figsize=(15.0, 4.5))
grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.45, 1.35], wspace=0.30,
                        left=0.055, right=0.985, top=0.79, bottom=0.20)

# D: counts and proportions only.
ax = fig.add_subplot(grid[0, 0])
x = np.arange(len(LABELS))
for index, label in enumerate(LABELS):
    ax.bar(x[index], observed[index], width=0.60, color=colors[label], alpha=0.84)
    ax.text(x[index], observed[index] + 8,
            f"{int(observed[index])}\n({100*observed[index]/n:.1f}%)",
            ha="center", fontsize=6.8)
ax.set_xticks(x, ["RNA only", "Structure only", "Joint\nresponse", "Other /\nunresolved"])
ax.set_ylim(0, observed.max() * 1.24)
ax.set_ylabel("Bins, n")
ax.set_title("D  Response-mode counts and proportions", loc="left", fontweight="bold", pad=8)

# E: four-class joint distribution with all categories shown as filled points.
ax = fig.add_subplot(grid[0, 1])
point_styles = {
    "Other / unresolved": dict(size=7, alpha=0.42, zorder=1),
    "RNA only": dict(size=12, alpha=0.70, zorder=2),
    "Structure only": dict(size=15, alpha=0.70, zorder=3),
    "Joint response": dict(size=18, alpha=0.76, zorder=4),
}
for key in ["Other / unresolved", "RNA only", "Structure only", "Joint response"]:
    subset = df[df.structure_020_response_mode.eq(key)]
    style = point_styles[key]
    ax.scatter(subset.median_interaction_log2FC, subset.d_ins_interaction,
               s=style["size"], color=colors[key], alpha=style["alpha"],
               edgecolors="none", zorder=style["zorder"],
               label=f"{key} (n={len(subset)})")
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
for cutoff in [-0.5, 0.5]:
    ax.axvline(cutoff, color="#BBBBBB", lw=0.45, ls="--")
ax.set_xlabel("Bin-median RNA interaction log$_2$FC")
ax.set_ylabel("Insulation interaction (Δ log$_2$)")
ax.set_title("E  RNA–structure response modes", loc="left", fontweight="bold", pad=8)
ax.legend(frameon=True, framealpha=0.92, edgecolor="none", loc="lower left",
          fontsize=5.7, handletextpad=0.35, borderpad=0.35, labelspacing=0.32)
ax.margins(x=0.055, y=0.10)

# F: joint null and directional enrichment P values for every class.
fgrid = grid[0, 2].subgridspec(2, 1, height_ratios=[1.2, 0.8], hspace=0.65)
ax = fig.add_subplot(fgrid[0])
joint_null = null[:, 2]
ax.hist(joint_null, bins=36, color="#C8CDD1", edgecolor="white", linewidth=0.35)
ax.axvline(null_mean[2], color="#555555", ls="--", lw=1.0,
           label=f"null mean = {null_mean[2]:.1f}")
ax.axvline(observed[2], color=colors["Joint response"], lw=1.6,
           label=f"observed = {int(observed[2])}")
ax.text(0.97, 0.95,
        f"two-sided P = {p_two[2]:.4f} {star(p_two[2])}\n"
        f"higher P = {p_high[2]:.4f} {star(p_high[2])}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
        color=colors["Joint response"], fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.2))
ax.set_xlabel("Joint-response bins under circular shifts")
ax.set_ylabel("Randomizations")
ax.set_title("F  Joint-count spatial null", loc="left", fontweight="bold", pad=7)
ax.legend(frameon=False, fontsize=6.3, loc="upper left")

ax = fig.add_subplot(fgrid[1])
pathway_labels = LABELS[:3]
x = np.arange(len(pathway_labels))
enrichment_p = results.P_higher_uncorrected.to_numpy()[:3]
bar_colors = [colors[label] for label in pathway_labels]
ax.bar(x, -np.log10(enrichment_p), color=bar_colors, width=0.58, alpha=0.84)
ax.axhline(-np.log10(.05), color="black", ls="--", lw=0.7)
for index, label in enumerate(pathway_labels):
    ax.text(x[index], -np.log10(enrichment_p[index]) + 0.06,
            f"P={enrichment_p[index]:.3g}\n{results.enrichment_star_higher.iloc[index]}",
            ha="center", fontsize=6.2)
ax.set_xticks(x, ["RNA only", "Structure only", "Joint"])
ax.set_ylabel("−log$_{10}$(higher P)")
ax.set_ylim(0, max(2.7, -np.log10(enrichment_p.min()) + 0.65))
ax.set_title("Higher-than-null enrichment tests (uncorrected)",
             loc="left", fontweight="bold", pad=6)
fig.suptitle("Response classification using the 0.20 structural-effect threshold",
             fontsize=12, fontweight="bold", y=0.97)
fig.text(0.055, 0.035,
         "Uncorrected empirical P values from 10,000 circular shifts of RNA labels. Higher-than-null tests evaluate enrichment only; "
         "two-sided deviations are linked by fixed margins.",
         fontsize=6.5, color="#555555")
stem = "Response_Fig2DEF_structure_020_response_modes"
fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
plt.close(fig)
print("Saved structure-0.20 D/E/F outputs", flush=True)
