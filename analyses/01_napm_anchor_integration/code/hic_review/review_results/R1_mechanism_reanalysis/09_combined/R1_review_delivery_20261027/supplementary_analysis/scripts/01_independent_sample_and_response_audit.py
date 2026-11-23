#!/usr/bin/env python3
"""Independent-sample 2x2 audit and unified per-bin hypoxia responses.

Primary exact randomization test: permute oxygen-condition labels within each
genotype, preserving WT/HYP n=4/4 and KO/KO-HYP n=4/3. There are
C(8,4)*C(7,3)=2450 assignments including the observed assignment. The test
statistic is the genotype contrast of genome-mean hypoxia responses:
(KO-HYP - KO) - (HYP - WT). Its sharp randomization null is no oxygen-condition
effect within either genotype; it is not an interaction-only test conditional
on arbitrary main effects. A genotype-label permutation within oxygen condition
is reported separately as a sensitivity analysis.
"""
from pathlib import Path
from itertools import combinations
import json
import warnings
import numpy as np
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
TAB = DEL / 'supplementary_analysis/tables'
TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
NBOOT = 10000
rng = np.random.default_rng(SEED)

b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
assert len(b) == 883
conditions = {
    'WT': [f'ins_WT_rep{i}' for i in range(1, 5)],
    'KO': [f'ins_KO_rep{i}' for i in range(1, 5)],
    'HYP': [f'ins_hypoxia_rep{i}' for i in range(1, 5)],
    'KO-HYP': [f'ins_KO_hypoxia_rep{i}' for i in range(1, 4)],
}
genotype = {'WT': 'WT', 'HYP': 'WT', 'KO': 'KO', 'KO-HYP': 'KO'}
oxygen = {'WT': 'normoxia', 'KO': 'normoxia', 'HYP': 'hypoxia', 'KO-HYP': 'hypoxia'}
rows = []
for condition, cols in conditions.items():
    for rep, col in enumerate(cols, 1):
        values = pd.to_numeric(b[col], errors='coerce').to_numpy(float)
        values[~np.isfinite(values)] = np.nan
        rows.append(dict(sample=f'{condition}_{rep}', condition=condition,
                         genotype=genotype[condition], oxygen=oxygen[condition],
                         replicate=rep, source_column=col,
                         n_finite_bins=int(np.isfinite(values).sum()),
                         genome_mean_insulation=float(np.nanmean(values))))
samples = pd.DataFrame(rows)
assert len(samples) == 15
samples.to_csv(TAB / 'independent_sample_genome_mean_insulation.csv', index=False)

g = {c: samples.loc[samples.condition == c, 'genome_mean_insulation'].to_numpy(float)
     for c in conditions}
wt_response = g['HYP'].mean() - g['WT'].mean()
ko_response = g['KO-HYP'].mean() - g['KO'].mean()
observed = ko_response - wt_response

# Nonparametric sample-level bootstrap, resampling biological samples within cells.
boot = np.empty(NBOOT)
for k in range(NBOOT):
    bs = {c: rng.choice(g[c], size=len(g[c]), replace=True) for c in conditions}
    boot[k] = ((bs['KO-HYP'].mean() - bs['KO'].mean())
               - (bs['HYP'].mean() - bs['WT'].mean()))
ci = np.percentile(boot, [2.5, 97.5])
pd.DataFrame({'bootstrap_interaction': boot}).to_csv(
    TAB / 'independent_sample_bootstrap_distribution.csv', index=False)

# Exact oxygen-label permutations within genotype.
wt_pool = np.concatenate([g['WT'], g['HYP']])
ko_pool = np.concatenate([g['KO'], g['KO-HYP']])
wt_stats = []
wt_assignments = []
for base_idx in combinations(range(8), 4):
    mask = np.zeros(8, dtype=bool)
    mask[list(base_idx)] = True
    base = wt_pool[mask].mean()
    hyp = wt_pool[~mask].mean()
    wt_stats.append(hyp - base)
    wt_assignments.append(';'.join(map(str, base_idx)))
ko_stats = []
ko_assignments = []
for hyp_idx in combinations(range(7), 3):
    mask = np.zeros(7, dtype=bool)
    mask[list(hyp_idx)] = True
    base = ko_pool[~mask].mean()
    hyp = ko_pool[mask].mean()
    ko_stats.append(hyp - base)
    ko_assignments.append(';'.join(map(str, hyp_idx)))
wt_stats = np.asarray(wt_stats)
ko_stats = np.asarray(ko_stats)
assert len(wt_stats) == 70 and len(ko_stats) == 35
perm = (ko_stats[:, None] - wt_stats[None, :]).ravel()
assert len(perm) == 2450
p_primary = float(np.mean(np.abs(perm) >= abs(observed) - 1e-12))
pd.DataFrame({
    'ko_assignment_index': np.repeat(np.arange(35), 70),
    'wt_assignment_index': np.tile(np.arange(70), 35),
    'interaction_statistic': perm,
}).to_csv(TAB / 'exact_oxygen_label_permutation_2450.csv', index=False)
pd.DataFrame({
    'wt_assignment_index': np.arange(70),
    'wt_normoxia_pool_indices': wt_assignments,
    'wt_hypoxia_response': wt_stats,
}).to_csv(TAB / 'exact_oxygen_label_WT_assignments.csv', index=False)
pd.DataFrame({
    'ko_assignment_index': np.arange(35),
    'ko_hypoxia_pool_indices': ko_assignments,
    'ko_hypoxia_response': ko_stats,
}).to_csv(TAB / 'exact_oxygen_label_KO_assignments.csv', index=False)

# Sensitivity: permute genotype labels within each oxygen condition.
norm_pool = np.concatenate([g['WT'], g['KO']])
hyp_pool = np.concatenate([g['HYP'], g['KO-HYP']])
wt_norm = []
ko_norm = []
for wt_idx in combinations(range(8), 4):
    mask = np.zeros(8, dtype=bool)
    mask[list(wt_idx)] = True
    wt_norm.append(norm_pool[mask].mean())
    ko_norm.append(norm_pool[~mask].mean())
wt_hyp = []
ko_hyp = []
for koh_idx in combinations(range(7), 3):
    mask = np.zeros(7, dtype=bool)
    mask[list(koh_idx)] = True
    ko_hyp.append(hyp_pool[mask].mean())
    wt_hyp.append(hyp_pool[~mask].mean())
wt_norm, ko_norm = np.asarray(wt_norm), np.asarray(ko_norm)
wt_hyp, ko_hyp = np.asarray(wt_hyp), np.asarray(ko_hyp)
perm_g = ((ko_hyp[:, None] - ko_norm[None, :])
          - (wt_hyp[:, None] - wt_norm[None, :])).ravel()
assert len(perm_g) == 2450
p_genotype = float(np.mean(np.abs(perm_g) >= abs(observed) - 1e-12))
pd.DataFrame({'interaction_statistic': perm_g}).to_csv(
    TAB / 'exact_genotype_label_permutation_2450_sensitivity.csv', index=False)

# Legacy all-combination genome-mean contrast, retained only for reconciliation.
wt_reps = np.column_stack([b[c].to_numpy(float) for c in conditions['WT']])
ko_reps = np.column_stack([b[c].to_numpy(float) for c in conditions['KO']])
hyp_reps = np.column_stack([b[c].to_numpy(float) for c in conditions['HYP']])
koh_reps = np.column_stack([b[c].to_numpy(float) for c in conditions['KO-HYP']])
for arr in (wt_reps, ko_reps, hyp_reps, koh_reps):
    arr[~np.isfinite(arr)] = np.nan
r_wt = np.stack([hyp_reps[:, h] - wt_reps[:, r] for r in range(4) for h in range(4)])
r_ko = np.stack([koh_reps[:, h] - ko_reps[:, r] for r in range(4) for h in range(3)])
legacy_wt = np.nanmean(np.nanmean(r_wt, axis=1))
legacy_ko = np.nanmean(np.nanmean(r_ko, axis=1))
legacy_effect = legacy_ko - legacy_wt

summary = pd.DataFrame([
    dict(test='WT genome-mean hypoxia response', effect=wt_response,
         ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n_samples=8,
         method='mean(HYP 4) - mean(WT 4)',
         null='descriptive'),
    dict(test='KO genome-mean hypoxia response', effect=ko_response,
         ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n_samples=7,
         method='mean(KO-HYP 3) - mean(KO 4)',
         null='descriptive'),
    dict(test='genotype contrast of genome-mean hypoxia responses',
         effect=observed, ci_lo=ci[0], ci_hi=ci[1], p=p_primary,
         n_samples=15,
         method='exact oxygen-label permutation within genotype; 2450 assignments; two-sided',
         null='sharp null: no oxygen-condition effect within either genotype'),
    dict(test='genotype-label permutation sensitivity',
         effect=observed, ci_lo=ci[0], ci_hi=ci[1], p=p_genotype,
         n_samples=15,
         method='exact genotype-label permutation within oxygen condition; 2450 assignments; two-sided',
         null='sharp null: no genotype effect within either oxygen condition'),
    dict(test='legacy replicate-combination contrast (not inferential)',
         effect=legacy_effect, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
         n_samples=28,
         method='16 WT and 12 KO pairwise replicate combinations; shared samples',
         null='not a valid independent-sample null'),
])
summary.to_csv(TAB / 'independent_sample_factorial_summary.csv', index=False)

# Unified per-bin responses: both genotypes use a difference of condition means.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    wt_bin = np.nanmean(hyp_reps, axis=1) - np.nanmean(wt_reps, axis=1)
    ko_bin = np.nanmean(koh_reps, axis=1) - np.nanmean(ko_reps, axis=1)
interaction = ko_bin - wt_bin
ok = np.isfinite(wt_bin) & np.isfinite(ko_bin)
same = np.sign(wt_bin) == np.sign(ko_bin)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.abs(ko_bin) / np.where(np.abs(wt_bin) > 0, np.abs(wt_bin), np.nan)
quad = np.full(len(b), 'unclassified', dtype=object)
quad[ok & ~same] = 'sign_reversed'
quad[ok & same & (ratio >= 2.0)] = 'enhanced'
quad[ok & same & (ratio < 0.5)] = 'attenuated'
quad[ok & same & (ratio >= 0.5) & (ratio < 2.0)] = 'concordant'
quad[ok & same & (np.abs(wt_bin) == 0) & (np.abs(ko_bin) == 0)] = 'concordant'
quad[ok & same & (np.abs(wt_bin) == 0) & (np.abs(ko_bin) > 0)] = 'enhanced'
perbin = pd.DataFrame({
    'bin': b.bin, 'start': b.start, 'end': b.end,
    'WT_hypoxia_response_mean_difference': wt_bin,
    'KO_hypoxia_response_mean_difference': ko_bin,
    'genotype_by_hypoxia_interaction': interaction,
    'quadrant_class': quad,
    'stored_WT_pairwise_median_response': b.d_ins_hyp_WT,
})
perbin.to_csv(TAB / 'unified_mean_response_perbin.csv', index=False)
qorder = ['concordant', 'attenuated', 'enhanced', 'sign_reversed']
qcounts = pd.DataFrame({
    'quadrant': qorder,
    'n_bins': [int(np.sum(quad[ok] == q)) for q in qorder],
})
qcounts['fraction'] = qcounts.n_bins / int(ok.sum())
qcounts['n_bins_total'] = int(ok.sum())
qcounts.to_csv(TAB / 'unified_mean_response_quadrant_counts.csv', index=False)
old_q, old_ok = b.d_ins_hyp_WT.to_numpy(float), ok
comparison = pd.DataFrame({
    'quantity': ['n_finite', 'WT mean absolute estimator difference',
                 'WT maximum absolute estimator difference',
                 'KO maximum absolute estimator difference'],
    'value': [int(ok.sum()),
              np.nanmean(np.abs(b.d_ins_hyp_WT.to_numpy(float) - wt_bin)),
              np.nanmax(np.abs(b.d_ins_hyp_WT.to_numpy(float) - wt_bin)),
              np.nanmax(np.abs(b.d_ins_hyp_KO.to_numpy(float) - ko_bin))],
})
comparison.to_csv(TAB / 'response_estimator_reconciliation.csv', index=False)

metadata = dict(seed=SEED, n_bootstrap=NBOOT, n_exact_assignments=2450,
                independent_samples=15,
                cell_sizes={c: len(v) for c, v in conditions.items()},
                primary_permutation='oxygen labels within genotype',
                input=str(RA / '03_structure_metrics/bin_level_v2.csv'))
(TAB / 'independent_sample_analysis_metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n')

print(samples.to_string(index=False))
print('\nCondition means:', {c: float(v.mean()) for c, v in g.items()})
print(f'WT response={wt_response:+.9f}; KO response={ko_response:+.9f}')
print(f'DID={observed:+.9f}; sample bootstrap 95% CI [{ci[0]:+.9f},{ci[1]:+.9f}]')
print(f'Exact oxygen-label permutation p={p_primary:.9f}; genotype-label sensitivity p={p_genotype:.9f}')
print(f'Legacy DID={legacy_effect:+.9f}; legacy WT={legacy_wt:+.9f}; KO={legacy_ko:+.9f}')
print('\nUnified quadrants:\n', qcounts.to_string(index=False))
print('\nWrote outputs to', TAB)
