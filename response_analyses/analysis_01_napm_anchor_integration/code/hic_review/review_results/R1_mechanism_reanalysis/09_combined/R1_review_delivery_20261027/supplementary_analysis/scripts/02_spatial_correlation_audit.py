#!/usr/bin/env python3
"""Spatially aware inference for baseline-KO versus WT-hypoxia association.

Primary data are the existing Fig4A per-bin estimates (pairwise-median contrasts).
P values use two-sided circular shifts of the complete genome vectors, preserving
spatial autocorrelation. Confidence intervals use paired circular moving-block
bootstrap (10, 20, and 40 bins = 50, 100, and 200 kb). An exhaustive enumeration
of all 883 rotations and a mean-contrast sensitivity are also reported.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
TAB = DEL / 'supplementary_analysis/tables'
TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
NSHIFT = 10000
NBOOT = 10000
N = 883
BLOCKS = [10, 20, 40]


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def random_shift_null(x, y, shifts):
    out = np.empty(len(shifts))
    for k, shift in enumerate(shifts):
        out[k] = spearman(x, np.roll(y, int(shift)))
    return out


def exhaustive_shift_null(x, y):
    return random_shift_null(x, y, np.arange(N))


def block_bootstrap(x, y, block, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(NBOOT)
    offsets = np.arange(block)
    nblocks = int(np.ceil(N / block))
    for k in range(NBOOT):
        starts = rng.integers(0, N, nblocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:N] % N
        out[k] = spearman(x[idx], y[idx])
    return out


b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
assert len(b) == N
x = b.d_ins_KO_WT.to_numpy(float)
y = b.d_ins_hyp_WT.to_numpy(float)
for a in (x, y):
    a[~np.isfinite(a)] = np.nan
observed = spearman(x, y)
ok = np.isfinite(x) & np.isfinite(y)
naive_p = float(stats.spearmanr(x[ok], y[ok]).pvalue)

rng = np.random.default_rng(SEED)
shifts = rng.integers(0, N, NSHIFT)
null = random_shift_null(x, y, shifts)
p_shift = float((1 + np.sum(np.abs(null) >= abs(observed))) / (NSHIFT + 1))
pd.DataFrame({'shift_bins': shifts, 'null_spearman_rho': null}).to_csv(
    TAB / 'fig4A_circular_shift_null_10000.csv', index=False)

all_null = exhaustive_shift_null(x, y)
p_exact = float(np.mean(np.abs(all_null) >= abs(observed) - 1e-12))
pd.DataFrame({'shift_bins': np.arange(N), 'spearman_rho': all_null}).to_csv(
    TAB / 'fig4A_exhaustive_883_rotations.csv', index=False)

boot_rows = []
ci_map = {}
for block in BLOCKS:
    boot = block_bootstrap(x, y, block, SEED + block)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    ci_map[block] = (float(lo), float(hi))
    pd.DataFrame({'block_bins': block, 'bootstrap_spearman_rho': boot}).to_csv(
        TAB / f'fig4A_circular_block_bootstrap_{block}bins.csv', index=False)
    boot_rows.append(dict(block_bins=block, block_kb=block * 5,
                          n_bootstrap=NBOOT, ci_lo=lo, ci_hi=hi,
                          bootstrap_median=float(np.nanmedian(boot)),
                          n_finite=int(np.isfinite(boot).sum())))
boot_summary = pd.DataFrame(boot_rows)
boot_summary.to_csv(TAB / 'fig4A_block_bootstrap_ci_summary.csv', index=False)

# Sensitivity using condition-mean contrasts for both axes.
def rep_matrix(prefixes):
    arr = np.column_stack([b[c].to_numpy(float) for c in prefixes])
    arr[~np.isfinite(arr)] = np.nan
    return arr
wt = rep_matrix([f'ins_WT_rep{i}' for i in range(1, 5)])
ko = rep_matrix([f'ins_KO_rep{i}' for i in range(1, 5)])
hyp = rep_matrix([f'ins_hypoxia_rep{i}' for i in range(1, 5)])
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    x_mean = np.nanmean(ko, axis=1) - np.nanmean(wt, axis=1)
    y_mean = np.nanmean(hyp, axis=1) - np.nanmean(wt, axis=1)
obs_mean = spearman(x_mean, y_mean)
null_mean = random_shift_null(x_mean, y_mean, shifts)
p_mean = float((1 + np.sum(np.abs(null_mean) >= abs(obs_mean))) / (NSHIFT + 1))
boot_mean = block_bootstrap(x_mean, y_mean, 20, SEED + 200)
ci_mean = np.nanpercentile(boot_mean, [2.5, 97.5])

summary = pd.DataFrame([
    dict(analysis='primary_pairwise_median_contrasts', estimator=observed,
         ci_lo=ci_map[20][0], ci_hi=ci_map[20][1], p=p_shift,
         n_bins=int(ok.sum()), n_permutations=NSHIFT,
         ci_method='paired circular moving-block bootstrap; 20 bins (100 kb)',
         p_method='two-sided random circular shift; 10000 shifts; seed 42'),
    dict(analysis='primary_exhaustive_rotation_sensitivity', estimator=observed,
         ci_lo=ci_map[20][0], ci_hi=ci_map[20][1], p=p_exact,
         n_bins=int(ok.sum()), n_permutations=N,
         ci_method='paired circular moving-block bootstrap; 20 bins (100 kb)',
         p_method='two-sided exhaustive enumeration of all 883 rotations including identity'),
    dict(analysis='mean_contrast_sensitivity', estimator=obs_mean,
         ci_lo=ci_mean[0], ci_hi=ci_mean[1], p=p_mean,
         n_bins=int((np.isfinite(x_mean) & np.isfinite(y_mean)).sum()),
         n_permutations=NSHIFT,
         ci_method='paired circular moving-block bootstrap; 20 bins (100 kb)',
         p_method='two-sided random circular shift; 10000 shifts; seed 42'),
    dict(analysis='legacy_naive_bin_independence_reference_only', estimator=observed,
         ci_lo=np.nan, ci_hi=np.nan, p=naive_p, n_bins=int(ok.sum()),
         n_permutations=np.nan, ci_method='not used',
         p_method='ordinary Spearman asymptotic p; invalid under spatial autocorrelation'),
])
summary.to_csv(TAB / 'fig4A_spatial_correlation_summary.csv', index=False)

metadata = dict(seed=SEED, n_shift=NSHIFT, n_bootstrap=NBOOT,
                block_sizes_bins=BLOCKS, block_sizes_kb=[v * 5 for v in BLOCKS],
                primary_block_bins=20,
                input=str(RA / '03_structure_metrics/bin_level_v2.csv'),
                primary_x='d_ins_KO_WT', primary_y='d_ins_hyp_WT')
(TAB / 'fig4A_spatial_analysis_metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n')

print(f'Primary rho={observed:.9f}, naive p={naive_p:.3e}, n={ok.sum()}')
print(f'Random circular-shift p={p_shift:.9f}; exhaustive-rotation p={p_exact:.9f}')
print(boot_summary.to_string(index=False))
print(f'Mean-contrast sensitivity rho={obs_mean:.9f}, CI={ci_mean}, shift p={p_mean:.9f}')
print('Wrote outputs to', TAB)
