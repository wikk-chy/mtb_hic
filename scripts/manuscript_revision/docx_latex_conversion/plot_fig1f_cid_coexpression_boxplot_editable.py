#!/usr/bin/env python3
"""Editable-vector boxplot of CID versus same-size random gene-set co-expression.

Each point represents one gene set's mean pairwise Pearson r, not an individual
overlapping gene pair. The empirical P value compares 19-set global means over
1,000 randomizations and is intentionally unadjusted, as requested.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "hic_review"
OUT = ROOT / "docx_latex_conversion/generated"
ASSIGNMENTS = OUT / "Fig1f_CID_transcription_18_boundaries_gene_assignments.csv"
RNA = BASE / "review_data/rnaseq/AKR_RNA_seq_merged.csv"
REFERENCE_NULLS = OUT / "Fig1f_CID_gene_set_coordination_18_boundaries_nulls.csv.gz"
SEED, N_PERM = 42, 1000

# Reconstruct the exact gene-level correlation matrix used in the main test.
genes = pd.read_csv(ASSIGNMENTS).sort_values("start").reset_index(drop=True)
counts = pd.read_csv(RNA).set_index("gene").loc[genes.gene]
expression = np.log2(counts.div(counts.sum(axis=0), axis=1) * 1e6 + 1.0).to_numpy().T
valid = np.std(expression, axis=0) > 0
labels = genes.cid.to_numpy(dtype=int)[valid]
correlation = np.corrcoef(expression[:, valid], rowvar=False)
groups = [np.where(labels == cid)[0] for cid in sorted(set(labels[labels >= 0]))]
assert len(groups) == 19


def mean_pair_r(indices):
    n = len(indices)
    if n < 2:
        return np.nan
    sub = correlation[np.ix_(indices, indices)]
    return float((sub.sum() - np.trace(sub)) / (n * (n - 1)))


observed_by_cid = np.array([mean_pair_r(group) for group in groups])
observed_global = float(np.mean(observed_by_cid))
rng = np.random.default_rng(SEED)
random_rows = []
random_global = []
for iteration in range(1, N_PERM + 1):
    random_groups = [rng.choice(correlation.shape[0], len(group), replace=False)
                     for group in groups]
    values = [mean_pair_r(group) for group in random_groups]
    random_global.append(np.mean(values))
    for cid, value in enumerate(values, start=1):
        random_rows.append({"iteration": iteration, "cid": cid, "mean_pair_r": value})
random_by_set = np.array([row["mean_pair_r"] for row in random_rows])
random_global = np.array(random_global)

# Verify exact agreement with the already audited random-gene-set null.
reference = pd.read_csv(REFERENCE_NULLS).query(
    "level == 'gene' and null_model == 'random_sets' and metric == 'multi_sample_mean_r'"
).null_value.to_numpy()
np.testing.assert_allclose(random_global, reference, rtol=1e-12, atol=1e-12)
random_mean = float(random_global.mean())
p_two = (np.sum(np.abs(random_global - random_mean) >=
                  abs(observed_global - random_mean)) + 1) / (N_PERM + 1)
percent = (observed_global - random_mean) / abs(random_mean) * 100

observed_table = pd.DataFrame({
    "group": "CID", "iteration": 0, "cid": np.arange(1, 20),
    "mean_pair_r": observed_by_cid})
random_table = pd.DataFrame(random_rows)
random_table.insert(0, "group", "same_size_random")
boxplot_data = pd.concat([observed_table, random_table], ignore_index=True)
boxplot_data.to_csv(OUT / "Fig1f_CID_coexpression_boxplot_data.csv", index=False)
summary = {
    "observed_global_mean_r": observed_global,
    "random_global_mean_r": random_mean,
    "relative_increase_percent": percent,
    "two_sided_empirical_p_uncorrected": float(p_two),
    "observed_cid_set_median_r": float(np.median(observed_by_cid)),
    "random_gene_set_median_r": float(np.median(random_by_set)),
    "n_observed_cid_sets": len(observed_by_cid),
    "n_random_gene_sets": len(random_by_set),
    "n_randomizations": N_PERM,
    "p_value_note": "Empirical P compares 19-set global means, not a Mann-Whitney test of the boxplot values",
    "font_editability": "PDF uses TrueType fonts; SVG text remains text rather than paths",
}
with open(OUT / "Fig1f_CID_coexpression_boxplot_editable_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)

C_OBS, C_RANDOM = "#0072B2", "#B8C0C8"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.linewidth": 0.6,
})
fig, axis = plt.subplots(figsize=(4, 9))
boxplot = axis.boxplot(
    [observed_by_cid, random_by_set], widths=0.50, patch_artist=True,
    showfliers=False, medianprops=dict(color="black", lw=1.2),
    whiskerprops=dict(color="#555555", lw=0.8),
    capprops=dict(color="#555555", lw=0.8))
for patch, color in zip(boxplot["boxes"], [C_OBS, C_RANDOM]):
    patch.set_facecolor(color)
    patch.set_alpha(0.80)
# Show all 19 observed CID-level values; random points are omitted for legibility.
jitter = np.random.default_rng(SEED).uniform(-0.10, 0.10, len(observed_by_cid))
axis.scatter(np.ones(len(observed_by_cid)) + jitter, observed_by_cid,
             s=13, color="#154C6E", alpha=0.85, linewidth=0, zorder=3)
axis.set_xticks([1, 2])
axis.set_xticklabels(["CID gene sets\n(n = 19)",
                      "Same-size random\ngene sets\n(n = 19,000)"], fontsize=7)
axis.set_ylabel("Mean pairwise Pearson $r$ per gene set\n(10 samples)", fontsize=7.5)
axis.set_title("F  CID gene-set co-expression\nper-set distributions",
               loc="left", fontsize=9, fontweight="bold", pad=8)
axis.text(0.97, 0.97,
          f"Global mean = {observed_global:.3f} vs {random_mean:.3f}\n"
          f"+{percent:.2f}%; $P$ = {p_two:.3f} (uncorrected)",
          transform=axis.transAxes, ha="right", va="top", fontsize=7.2,
          color=C_OBS, fontweight="bold",
          bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.4))
axis.tick_params(axis="y", labelsize=7)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
fig.text(0.5, 0.04,
         "Empirical P compares 19-set means over 1,000 randomizations;\nno multiple-testing correction",
         ha="center", va="bottom", fontsize=6.2, color="#555555")
fig.subplots_adjust(left=0.23, right=0.96, top=0.87, bottom=0.18)
stem = "Fig1f_CID_coexpression_boxplot_editable"
for extension in ["pdf", "svg", "png"]:
    fig.savefig(OUT / f"{stem}.{extension}", dpi=400 if extension == "png" else None,
                bbox_inches="tight")
plt.close(fig)
print(json.dumps(summary, indent=2))
