#!/usr/bin/env python3
"""R2 Comment 6 - binding_distribution, part 1.

Peak-level statistics of the curated 634 Rv0047c (NapM) ChIP-seq peaks
(Minch et al. 2015, Nat Commun; curated in R1_mechanism_reanalysis):
  - width / score / vpm distributions
  - inter-peak gap and nearest-neighbor distance distributions
  - nearest-neighbor distance vs random expectation (circular genome, seed=42)
  - genome coverage = merged (union) peak interval length / 4,411,532

Outputs: tables/peak_stats_summary.csv, tables/spatial_null_summary.csv,
         figures/fig1_peak_spatial_stats.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
PEAKS = BASE + "hic_review/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
OUT_T = BASE + "hic_review/review_results/R2_comments/binding_distribution/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/binding_distribution/figures/"
GENOME_LEN = 4_411_532
SEED = 42
N_SIM = 10_000

peaks = pd.read_csv(PEAKS, sep="\t")
assert len(peaks) == 634, f"expected 634 peaks, got {len(peaks)}"
peaks["width"] = peaks["end"] - peaks["start"]

# ---- merged coverage ------------------------------------------------------
iv = peaks[["start", "end"]].sort_values("start").values
merged = []
for s, e in iv:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
merged_len = sum(e - s for s, e in merged)
coverage = merged_len / GENOME_LEN
n_merged = len(merged)

# ---- gaps and nearest-neighbor distances (circular) -----------------------
centers = np.sort(peaks["center"].values)
gaps = np.diff(centers)
wrap_gap = centers[0] + GENOME_LEN - centers[-1]
gaps_circ = np.append(gaps, wrap_gap)          # 634 gaps on the circle
nn_dist = np.minimum(gaps_circ[:-1], gaps_circ[1:])  # NN distance of each peak

def mean_nn(cs):
    cs = np.sort(cs)
    g = np.diff(cs)
    g = np.append(g, cs[0] + GENOME_LEN - cs[-1])
    return np.minimum(g[:-1], g[1:]).mean()

obs_mean_nn = mean_nn(centers)
rng = np.random.default_rng(SEED)
null_mean_nn = np.empty(N_SIM)
for i in range(N_SIM):
    null_mean_nn[i] = mean_nn(rng.uniform(0, GENOME_LEN, len(centers)))
p_cluster = (null_mean_nn >= obs_mean_nn).mean()   # NN smaller than random => clustered
p_dispers = (null_mean_nn <= obs_mean_nn).mean()   # NN larger than random => dispersed

# ---- summary table --------------------------------------------------------
summary = pd.DataFrame([
    ("n_peaks", len(peaks)),
    ("width_median_bp", peaks.width.median()),
    ("width_mean_bp", round(peaks.width.mean(), 1)),
    ("width_q05_bp", peaks.width.quantile(0.05)),
    ("width_q95_bp", peaks.width.quantile(0.95)),
    ("width_min_bp", peaks.width.min()),
    ("width_max_bp", peaks.width.max()),
    ("score_median", round(peaks.score.median(), 4)),
    ("score_min", round(peaks.score.min(), 4)),
    ("score_max", round(peaks.score.max(), 4)),
    ("vpm_median", round(peaks.vpm.median(), 1)),
    ("vpm_min", peaks.vpm.min()),
    ("vpm_max", peaks.vpm.max()),
    ("n_merged_intervals", n_merged),
    ("merged_length_bp", merged_len),
    ("genome_coverage", round(coverage, 5)),
    ("gap_median_bp", np.median(gaps_circ)),
    ("gap_mean_bp", round(gaps_circ.mean(), 1)),
    ("gap_max_bp", gaps_circ.max()),
    ("gap_min_bp", gaps_circ.min()),
    ("nn_dist_median_bp", np.median(nn_dist)),
    ("nn_dist_mean_bp", round(nn_dist.mean(), 1)),
], columns=["metric", "value"])
summary.to_csv(OUT_T + "peak_stats_summary.csv", index=False)

spatial = pd.DataFrame([
    ("obs_mean_nn_bp", round(obs_mean_nn, 1)),
    ("null_mean_nn_mean_bp", round(null_mean_nn.mean(), 1)),
    ("null_mean_nn_q025_bp", round(np.quantile(null_mean_nn, 0.025), 1)),
    ("null_mean_nn_q975_bp", round(np.quantile(null_mean_nn, 0.975), 1)),
    ("p_two_sided_clustered", round(2 * min(p_cluster, p_dispers), 4)),
    ("n_sim", N_SIM),
    ("seed", SEED),
], columns=["metric", "value"])
spatial.to_csv(OUT_T + "spatial_null_summary.csv", index=False)

# ---- figure ----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))

ax = axes[0, 0]
ax.hist(peaks.width, bins=40, color="#6a51a3", edgecolor="white", lw=0.3)
ax.axvline(peaks.width.median(), color="black", ls="--", lw=0.8)
ax.set_xlabel("peak width (bp)"); ax.set_ylabel("count")
ax.set_title(f"A  width (median {peaks.width.median():.0f} bp)", loc="left", fontweight="bold")

ax = axes[0, 1]
ax.hist(np.log10(gaps_circ), bins=40, color="#6a51a3", edgecolor="white", lw=0.3)
ax.set_xlabel("log10 inter-peak gap (bp)"); ax.set_ylabel("count")
ax.set_title(f"B  gaps (median {np.median(gaps_circ):.0f} bp)", loc="left", fontweight="bold")

ax = axes[0, 2]
ax.hist(null_mean_nn, bins=50, color="lightgrey", edgecolor="white", lw=0.3, label="random null")
ax.axvline(obs_mean_nn, color="#6a51a3", lw=1.2, label="observed")
ax.set_xlabel("mean nearest-neighbor distance (bp)"); ax.set_ylabel("count (10k sims)")
ax.set_title("C  spatial clustering vs null", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7)

ax = axes[1, 0]
ax.hist(peaks.score, bins=40, color="#3182bd", edgecolor="white", lw=0.3)
ax.set_xlabel("peak score"); ax.set_ylabel("count")
ax.set_title(f"D  score (median {peaks.score.median():.3f})", loc="left", fontweight="bold")

ax = axes[1, 1]
ax.hist(np.log10(peaks.vpm), bins=40, color="#3182bd", edgecolor="white", lw=0.3)
ax.set_xlabel("log10 VPM"); ax.set_ylabel("count")
ax.set_title(f"E  VPM (median {peaks.vpm.median():.0f})", loc="left", fontweight="bold")

ax = axes[1, 2]
ax.axis("off")
txt = (f"634 peaks -> {n_merged} merged intervals\n"
       f"merged length = {merged_len:,} bp\n"
       f"coverage = {coverage*100:.2f}% of genome\n\n"
       f"mean NN = {obs_mean_nn:.0f} bp\n"
       f"null 95% CI [{np.quantile(null_mean_nn,0.025):.0f}, "
       f"{np.quantile(null_mean_nn,0.975):.0f}] bp\n"
       f"two-sided p = {2*min(p_cluster,p_dispers):.4f}")
ax.text(0.05, 0.95, txt, va="top", ha="left", fontsize=8, family="Arial")
ax.set_title("F  coverage & clustering", loc="left", fontweight="bold")

for a in axes.flat:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig1_peak_spatial_stats.png", dpi=300)
fig.savefig(OUT_F + "fig1_peak_spatial_stats.pdf")
print(summary.to_string(index=False))
print(spatial.to_string(index=False))
print("OK")
