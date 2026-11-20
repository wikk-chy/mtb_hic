#!/usr/bin/env python3
"""Reviewer 2 Comment 4: spatial distribution and ori-to-ter directionality.

Uses the validated 5-kb per-replicate insulation and long-range-contact
fractions from R1_mechanism_reanalysis. No manuscript files are modified.

Responses are stress minus WT for WT cells only. Regional heterogeneity is
tested across 20 consecutive genome intervals. Monotonic directionality is
tested separately on the two arms, each oriented from ori toward ter.
Circular-shift permutation preserves the response vector's serial structure;
10,000 shifts and seed 42 are used throughout.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
NSHIFT = 10_000
NREGION = 20
GENOME_SIZE = 4_411_532
ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
SOURCE = ROOT / "hic_review/review_results/R1_mechanism_reanalysis/03_structure_metrics/bin_level_v2.csv"
OUT = ROOT / "hic_review/review_results/R2_comments/R2C4_spatial_response"
TABLES = OUT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

b = pd.read_csv(SOURCE)
n = len(b)
assert n == 883
bin_pos = (b["start"].to_numpy() + b["end"].to_numpy()) / 2
region = np.minimum((np.arange(n) * NREGION // n), NREGION - 1)

wt_ins = b[[f"ins_WT_rep{i}" for i in range(1, 5)]].mean(axis=1)
wt_lr = b[[f"lrfrac_WT_rep{i}" for i in range(1, 5)]].mean(axis=1)
responses = {
    "hypoxia_insulation": b["d_ins_hyp_WT"].to_numpy(dtype=float),
    "latent_insulation": (b["ins_latent"] - wt_ins).to_numpy(dtype=float),
    "hypoxia_long_range_fraction": (
        b[[f"lrfrac_hypoxia_rep{i}" for i in range(1, 5)]].mean(axis=1) - wt_lr
    ).to_numpy(dtype=float),
    "latent_long_range_fraction": (
        b[[f"lrfrac_latent_rep{i}" for i in range(1, 5)]].mean(axis=1) - wt_lr
    ).to_numpy(dtype=float),
}
for key in responses:
    responses[key][~np.isfinite(responses[key])] = np.nan


def regional_means(y):
    sums = np.bincount(region, weights=np.nan_to_num(y), minlength=NREGION)
    counts = np.bincount(region, weights=np.isfinite(y), minlength=NREGION)
    return np.divide(sums, counts, out=np.full(NREGION, np.nan), where=counts > 0)


def heterogeneity(y):
    obs = np.nanvar(regional_means(y))
    null = np.empty(NSHIFT)
    for k in range(NSHIFT):
        shift = int(rng.integers(1, n))
        null[k] = np.nanvar(regional_means(np.roll(y, shift)))
    p = (1 + np.sum(null >= obs)) / (NSHIFT + 1)
    return obs, p


def arm_test(y, indices, x):
    yy = y[indices]
    xx = np.asarray(x, dtype=float)
    ok = np.isfinite(yy)
    yy, xx = yy[ok], xx[ok]
    slope = stats.linregress(xx, yy).slope
    rho = stats.spearmanr(xx, yy).statistic
    null = np.empty(NSHIFT)
    for k in range(NSHIFT):
        shift = int(rng.integers(1, len(yy)))
        null[k] = stats.spearmanr(xx, np.roll(yy, shift)).statistic
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (NSHIFT + 1)
    return slope, rho, p, len(yy)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1)
    return out


# Two equal 441-bin arms; the central terminus-proximal bin 441 is omitted.
arm1 = np.arange(0, 441)
arm2 = np.arange(442, 883)[::-1]
x1 = bin_pos[arm1] / 1e6
x2 = (GENOME_SIZE - bin_pos[arm2]) / 1e6

stat_rows = []
profile_rows = []
for name, y in responses.items():
    condition, metric = name.split("_", 1)
    means = regional_means(y)
    for r in range(NREGION):
        members = region == r
        profile_rows.append({
            "condition": condition,
            "metric": metric,
            "region": r + 1,
            "start_bp": int(b.loc[members, "start"].min()),
            "end_bp": int(b.loc[members, "end"].max()),
            "n_bins": int(members.sum()),
            "n_finite": int(np.isfinite(y[members]).sum()),
            "mean_response": means[r],
            "median_response": np.nanmedian(y[members]),
            "fraction_positive": np.nanmean(y[members] > 0),
        })
    obs, p = heterogeneity(y)
    stat_rows.append({
        "condition": condition,
        "metric": metric,
        "test": "heterogeneity_across_20_regions",
        "arm": "genome",
        "effect": obs,
        "effect_name": "variance_of_20_region_means",
        "spearman_rho": np.nan,
        "p_circular_shift": p,
        "n": int(np.isfinite(y).sum()),
    })
    for arm, indices, x in (("ori_to_ter_arm1", arm1, x1),
                            ("ori_to_ter_arm2", arm2, x2)):
        slope, rho, p, count = arm_test(y, indices, x)
        stat_rows.append({
            "condition": condition,
            "metric": metric,
            "test": "monotonic_ori_to_ter_trend",
            "arm": arm,
            "effect": slope,
            "effect_name": "linear_slope_response_per_Mb",
            "spearman_rho": rho,
            "p_circular_shift": p,
            "n": count,
        })

stats_df = pd.DataFrame(stat_rows)
stats_df["p_bh_within_test_family"] = np.nan
for test in stats_df["test"].unique():
    mask = stats_df["test"] == test
    stats_df.loc[mask, "p_bh_within_test_family"] = bh_adjust(stats_df.loc[mask, "p_circular_shift"])
stats_df.to_csv(TABLES / "spatial_directionality_stats.csv", index=False)
pd.DataFrame(profile_rows).to_csv(TABLES / "regional_response_profiles.csv", index=False)

summary = {
    "source": str(SOURCE.relative_to(ROOT)),
    "resolution_bp": 5000,
    "n_bins": n,
    "n_regions": NREGION,
    "n_circular_shifts": NSHIFT,
    "seed": SEED,
    "genome_size_bp": GENOME_SIZE,
    "arm_definition": "Two 441-bin arms oriented from ori toward ter; central bin 441 omitted",
    "metrics": list(responses),
    "interpretation_boundary": (
        "Endpoint population Hi-C metrics describe spatial contact/insulation changes, "
        "not temporal synchrony or direct physical compaction measurements."
    ),
}
with open(TABLES / "analysis_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)

print(stats_df.to_string(index=False))
print("\nSaved:", TABLES)
