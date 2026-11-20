from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
MATRIX_DIR = ROOT / "hic_review/review_data/rep_icedmatrix"
OUT = ROOT / "analysis/reproducibility_qc_20261105"
OUT.mkdir(exist_ok=True)
CONDITIONS = {
    "WT": "wt",
    "napM-KO": "napm_ko",
    "hypoxia": "hypoxia",
    "latent": "latent",
}

series_by_condition = {}
qc_by_condition = {}
for condition, prefix in CONDITIONS.items():
    samples = {}
    qc_rows = []
    for rep in range(1, 5):
        path = MATRIX_DIR / f"{prefix}_{rep}_5000_iced.matrix"
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=["bin1", "bin2", "value"])
        series = pd.Series(frame["value"].to_numpy(), index=pd.MultiIndex.from_arrays(
            [frame["bin1"], frame["bin2"]], names=["bin1", "bin2"]))
        samples[rep] = series
        distances = frame["bin2"] - frame["bin1"]
        p_by_distance = frame.groupby(distances)["value"].sum()
        qc_rows.append((rep, p_by_distance))
    series_by_condition[condition] = samples
    qc_by_condition[condition] = qc_rows

pair_rows = []
summary_rows = []
for condition, samples in series_by_condition.items():
    pair_metrics = []
    for a, b in combinations(sorted(samples), 2):
        aligned = pd.concat([samples[a], samples[b]], axis=1, keys=["a", "b"]).fillna(0)
        x = aligned["a"].to_numpy(float)
        y = aligned["b"].to_numpy(float)
        rho = stats.spearmanr(x, y).statistic
        pa = qc_by_condition[condition][a - 1][1]
        pb = qc_by_condition[condition][b - 1][1]
        p_curve = pd.concat([pa, pb], axis=1, keys=["a", "b"]).fillna(0)
        p_curve = p_curve / p_curve.sum(axis=0)
        p_curve = p_curve.loc[p_curve.index >= 20]
        p_curve = np.log10(p_curve + 1e-12)
        curve_r = stats.pearsonr(p_curve["a"], p_curve["b"]).statistic
        pair_rows.append({"condition": condition, "rep_a": a, "rep_b": b,
                          "spearman_matrix_rho": rho, "pearson_P_of_s_r": curve_r})
        pair_metrics.append((rho, curve_r))
    values = np.array(pair_metrics)
    summary_rows.append({
        "condition": condition,
        "matrix_spearman_mean": values[:, 0].mean(),
        "matrix_spearman_min": values[:, 0].min(),
        "matrix_spearman_max": values[:, 0].max(),
        "P_of_s_pearson_mean": values[:, 1].mean(),
        "P_of_s_pearson_min": values[:, 1].min(),
        "P_of_s_pearson_max": values[:, 1].max(),
    })

pairs = pd.DataFrame(pair_rows)
pairs.to_csv(OUT / "replicate_pairwise_metrics.csv", index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "reproducibility_summary.csv", index=False)

# Existing per-replicate QC metrics (independent of the old log1p correlation analysis).
metrics = pd.read_csv(ROOT / "hic_review/review_results/R4_comment4/tables/replicate_metrics.csv")
qc_rows = []
for condition, frame in metrics.groupby("condition", sort=False):
    qc_rows.append({
        "condition": condition,
        "n_replicates": len(frame),
        "bad_bins_min": int(frame["n_bad_bins"].min()),
        "bad_bins_max": int(frame["n_bad_bins"].max()),
        "bad_bin_percent": 100 * frame["n_bad_bins"].max() / 883,
        "total_contacts_mean": frame["total_contacts"].mean(),
        "total_contacts_min": frame["total_contacts"].min(),
        "total_contacts_max": frame["total_contacts"].max(),
        "short_fraction_mean": frame["frac_short_lt100kb"].mean(),
        "short_fraction_sd": frame["frac_short_lt100kb"].std(ddof=1),
        "long_fraction_mean": frame["frac_long_gt1Mb"].mean(),
        "long_fraction_sd": frame["frac_long_gt1Mb"].std(ddof=1),
        "compaction_slope_mean": frame["compaction_slope"].mean(),
        "compaction_slope_sd": frame["compaction_slope"].std(ddof=1),
    })
qc = pd.DataFrame(qc_rows)
qc.to_csv(OUT / "replicate_QC_summary.csv", index=False)

print("Reproducibility:")
print(summary.to_string(index=False))
print("\nQC:")
print(qc.to_string(index=False))
print(f"\nOutputs: {OUT}")
