#!/usr/bin/env python3
"""Recalculate manuscript Fig. 1f using all 18 DI-defined CID boundaries.

The expression metric, distance-matched cross-boundary controls, random seed,
and circular-shift homogeneity test reproduce the current Fig. 1f workflow.
A legacy-boundary run is used only as an algorithmic equivalence check. The
new figure and tables are standalone; manuscript and reviewer files are not
modified by this script.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "hic_review"
SOURCE_DIR = BASE / "review_results/R4_comment6/coexpression_wt"
OUT = ROOT / "docx_latex_conversion/generated"
OUT.mkdir(parents=True, exist_ok=True)
RNA = BASE / "review_data/rnaseq/AKR_RNA_seq_merged.csv"
GTF = BASE / "annotation.gtf"
BD_CSV = BASE / "review_results/R4_comment2_CID/tables/cid_boundaries_recalled.csv"
LEGACY_SUMMARY = SOURCE_DIR / "tables/summary.json"

N_BINS, BINSIZE, GENOME = 883, 5000, 4411532
SEED = 42
N_SHIFTS = 1000
C_CID, C_CROSS = "#8C2F39", "#B8C0C8"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.6,
})

# Identical expression inputs to the current main-figure analysis.
rna = pd.read_csv(RNA)
rna["WT"] = (rna["WT-rep1"] + rna["WT-rep2"]) / 2
gtf = pd.read_csv(
    GTF, sep="\t", comment="#", header=None,
    names=["chr", "src", "feat", "start", "end", "score", "strand", "frame", "attr"])
gtf = gtf[gtf.feat == "gene"].copy()
gtf["gene_id"] = gtf.attr.str.extract(r'gene_id "?([^";]+)"?')
genes = (gtf.merge(rna[["gene", "WT"]], left_on="gene_id", right_on="gene")
         .sort_values("start").reset_index(drop=True))
genes["expr_norm"] = genes.WT / (genes.end - genes.start)
genes = genes[genes.expr_norm > 0].reset_index(drop=True)
n_genes = len(genes)
starts = genes.start.to_numpy(dtype=int)
ends = genes.end.to_numpy(dtype=int)
centers = ((starts + ends) // 2).astype(int)
wt_expr = genes.expr_norm.to_numpy(dtype=float)
gene_bins = (centers / BINSIZE).astype(int)

# Lookup table equivalent to the original pos_to_gene(), including its rule
# for overlapping genes: the first start-sorted gene containing a position wins.
positions = np.arange(GENOME, dtype=int)
position_to_gene = np.minimum(
    np.searchsorted(starts, positions, side="right"), n_genes - 1).astype(np.int32)
for gene_index in range(n_genes - 1, -1, -1):
    position_to_gene[starts[gene_index]:ends[gene_index] + 1] = gene_index


def divergence(gene_i, gene_j):
    expr_i, expr_j = wt_expr[gene_i], wt_expr[gene_j]
    return max(expr_i, expr_j) / min(expr_i, expr_j) - 1


def analyze_cids(boundary_bins):
    """Run the original pairwise and homogeneity calculations for one boundary set."""
    boundary_bins = sorted(int(value) for value in boundary_bins)
    vertices = [0] + boundary_bins + [N_BINS - 1]
    cids = [(vertices[k] * BINSIZE, vertices[k + 1] * BINSIZE)
            for k in range(len(vertices) - 1)]

    def cid_of_bin(bin_index):
        for cid_index, (left, right) in enumerate(cids):
            if left <= bin_index * BINSIZE < right:
                return cid_index
        return -1

    gene_cid = np.array([cid_of_bin(value) for value in gene_bins])
    within, cross = [], []
    rng = np.random.default_rng(SEED)
    for cid_index in range(len(cids)):
        members = np.where(gene_cid == cid_index)[0]
        for left_index in range(len(members)):
            for right_index in range(left_index + 1, len(members)):
                gene_i, gene_j = members[left_index], members[right_index]
                distance_bp = abs(starts[gene_j] - starts[gene_i])
                within.append((divergence(gene_i, gene_j), distance_bp))
                # Preserve the original rule: at most three draws per within-
                # CID pair, retaining the first valid cross-boundary control.
                for _ in range(3):
                    position = int(rng.integers(0, GENOME - distance_bp - 1))
                    control_i = int(position_to_gene[position])
                    control_j = int(position_to_gene[position + distance_bp])
                    if (control_i != control_j and
                            gene_cid[control_i] != gene_cid[control_j]):
                        cross.append(divergence(control_i, control_j))
                        break
    within_div = np.array([value[0] for value in within])
    cross_div = np.array(cross)
    pairwise_p = stats.mannwhitneyu(
        within_div, cross_div, alternative="less").pvalue

    def mean_mad(partition_boundaries):
        partition_vertices = [0] + list(partition_boundaries) + [N_BINS - 1]
        mads = []
        for index in range(len(partition_vertices) - 1):
            members = np.where(
                (gene_bins >= partition_vertices[index]) &
                (gene_bins < partition_vertices[index + 1]))[0]
            if len(members) > 2:
                log_expression = np.log(wt_expr[members])
                mads.append(np.median(np.abs(
                    log_expression - np.median(log_expression))))
        return float(np.mean(mads))

    observed_mad = mean_mad(boundary_bins)
    null_mad = []
    for _ in range(N_SHIFTS):
        shift = int(rng.integers(0, N_BINS))
        shifted = sorted((value + shift) % N_BINS for value in boundary_bins)
        null_mad.append(mean_mad(shifted))
    null_mad = np.array(null_mad)
    homogeneity_p = (np.sum(null_mad <= observed_mad) + 1) / (N_SHIFTS + 1)
    return {
        "boundary_bins": boundary_bins,
        "n_cids": len(cids),
        "cids": cids,
        "gene_cid": gene_cid,
        "within_div": within_div,
        "cross_div": cross_div,
        "within_median": float(np.median(within_div)),
        "cross_median": float(np.median(cross_div)),
        "pairwise_p": float(pairwise_p),
        "observed_mad": observed_mad,
        "null_mad": null_mad,
        "null_mad_mean": float(null_mad.mean()),
        "null_mad_sd": float(null_mad.std()),
        "homogeneity_p": float(homogeneity_p),
    }


print("Checking the implementation against the existing Fig. 1f values ...", flush=True)
boundaries = pd.read_csv(BD_CSV)
legacy_bins = sorted(boundaries.loc[boundaries.confirmed, "bin"].astype(int).tolist())
legacy = analyze_cids(legacy_bins)
with open(LEGACY_SUMMARY) as handle:
    legacy_reference = json.load(handle)
assert len(legacy["within_div"]) == 793588
assert len(legacy["cross_div"]) == 540968
np.testing.assert_allclose(
    [legacy["within_median"], legacy["cross_median"], legacy["pairwise_p"],
     legacy["observed_mad"], legacy["null_mad_mean"], legacy["homogeneity_p"]],
    [legacy_reference["within_CID_median"], legacy_reference["cross_median"],
     legacy_reference["p_within_vs_cross"], legacy_reference["cid_homogeneity_obs"],
     legacy_reference["cid_homogeneity_null_mean"], legacy_reference["cid_homogeneity_p"]],
    rtol=1e-11, atol=1e-12)
print("Legacy algorithm check passed exactly within numerical tolerance.", flush=True)

all_bins = sorted(boundaries.loc[
    boundaries.transition_ok.astype(str).str.lower().eq("true"), "bin"].astype(int).tolist())
assert len(all_bins) == 18, f"Expected 18 boundaries, found {len(all_bins)}"
print(f"Recalculating Fig. 1f with all {len(all_bins)} CID boundaries ...", flush=True)
result = analyze_cids(all_bins)
relative_difference = ((result["cross_median"] - result["within_median"]) /
                       result["cross_median"] * 100)
print(f"Within-CID: median={result['within_median']:.6f}, n={len(result['within_div']):,}", flush=True)
print(f"Cross-boundary: median={result['cross_median']:.6f}, n={len(result['cross_div']):,}", flush=True)
print(f"One-sided Mann-Whitney P={result['pairwise_p']:.6e}", flush=True)
print(f"Median difference={relative_difference:.3f}% of the control median", flush=True)
print(f"Homogeneity MAD={result['observed_mad']:.6f}; null={result['null_mad_mean']:.6f}; "
      f"P={result['homogeneity_p']:.6f}", flush=True)

# Numerical deliverables.
stem = "Fig1f_CID_transcription_18_boundaries"
group_stats = pd.DataFrame({
    "group": ["within_CID", "distance_matched_cross_boundary"],
    "median_divergence": [result["within_median"], result["cross_median"]],
    "n_gene_pairs": [len(result["within_div"]), len(result["cross_div"])],
    "one_sided_mann_whitney_p": [result["pairwise_p"], np.nan],
})
group_stats.to_csv(OUT / f"{stem}_group_stats.csv", index=False)
assignments = genes[["gene", "start", "end"]].copy()
assignments["center"] = centers
assignments["bin"] = gene_bins
assignments["cid"] = result["gene_cid"]
assignments["expr_norm"] = wt_expr
assignments.to_csv(OUT / f"{stem}_gene_assignments.csv", index=False)
pd.DataFrame({"shift_index": np.arange(1, N_SHIFTS + 1),
              "mean_within_cid_mad": result["null_mad"]}).to_csv(
    OUT / f"{stem}_homogeneity_null.csv", index=False)
summary = {
    "figure": "Fig. 1f",
    "cid_boundaries": len(all_bins),
    "n_cids": result["n_cids"],
    "boundary_bins": all_bins,
    "boundary_positions_kb": [value * 5 for value in all_bins],
    "expression_metric": "mean WT counts of two replicates divided by gene length",
    "pairwise_divergence": "max(expr_i, expr_j) / min(expr_i, expr_j) - 1",
    "within_cid_median": result["within_median"],
    "within_cid_n_gene_pairs": len(result["within_div"]),
    "cross_boundary_median": result["cross_median"],
    "cross_boundary_n_gene_pairs": len(result["cross_div"]),
    "one_sided_mann_whitney_p": result["pairwise_p"],
    "median_reduction_percent_of_control": float(relative_difference),
    "pairwise_test_caveat": "Gene pairs overlap and are not independent observations.",
    "cross_boundary_control_rule": "Up to three distance-matched draws per within-CID pair; first valid cross-boundary pair retained.",
    "homogeneity_observed_mean_mad": result["observed_mad"],
    "homogeneity_null_mean_mad": result["null_mad_mean"],
    "homogeneity_null_sd_mad": result["null_mad_sd"],
    "homogeneity_empirical_p": result["homogeneity_p"],
    "homogeneity_test": "1,000 circular shifts of all 18 boundaries; one-sided test for lower observed MAD",
    "seed": SEED,
    "n_bins": N_BINS,
    "terminal_bin_exclusive": N_BINS - 1,
    "algorithm_check_against_existing_fig1f": "passed",
    "source_boundary_table": str(BD_CSV),
}
with open(OUT / f"{stem}_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)

# Standalone panel F in the same linear-scale style as the current main figure.
fig, axis = plt.subplots(figsize=(72 / 25.4, 80 / 25.4))
boxplot = axis.boxplot(
    [result["within_div"], result["cross_div"]], showfliers=False,
    widths=0.55, patch_artist=True, medianprops=dict(color="black", lw=1.2))
for patch, color in zip(boxplot["boxes"], [C_CID, C_CROSS]):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
axis.set_xticks([1, 2])
axis.set_xticklabels([
    f"within CID\n(n={len(result['within_div']):,})",
    f"cross-boundary\ndist.-matched\n(n={len(result['cross_div']):,})",
], fontsize=6.5)
q1, q3 = np.percentile(result["cross_div"], [25, 75])
whisker_top = q3 + 1.5 * (q3 - q1)
axis.set_ylim(0, whisker_top * 1.38)
bracket_y = whisker_top * 1.08
axis.plot([1, 1, 2, 2], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
          color="black", lw=0.7)
p_mantissa, p_exponent = f"{result['pairwise_p']:.2e}".split("e")
p_label = rf"$P = {p_mantissa} \times 10^{{{int(p_exponent)}}}$"
axis.text(1.5, whisker_top * 1.19, p_label,
          ha="center", va="center", fontsize=7.2)
axis.text(0.04, 0.975,
          f"median = {result['within_median']:.3f} vs {result['cross_median']:.3f}",
          transform=axis.transAxes, fontsize=6.5, va="top")
axis.set_ylabel("pairwise expression divergence\nmax/min − 1, length-normalized WT RNA",
                fontsize=7)
axis.set_title("F  Within-CID pairs vs cross-boundary\n"
               "distance-matched; 18 CID boundaries",
               loc="left", fontsize=8.5, fontweight="bold", pad=7)
axis.tick_params(axis="y", labelsize=6.5)
fig.subplots_adjust(left=0.22, right=0.97, top=0.77, bottom=0.23)
fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
plt.close(fig)
print(f"Saved standalone Fig. 1f and numerical results to {OUT}", flush=True)
