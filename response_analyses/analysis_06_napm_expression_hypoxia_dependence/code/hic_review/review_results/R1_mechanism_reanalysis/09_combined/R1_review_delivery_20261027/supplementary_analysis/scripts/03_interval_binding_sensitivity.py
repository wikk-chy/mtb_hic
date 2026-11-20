#!/usr/bin/env python3
"""Sensitivity of key NapM associations to interval-overlap binding (454 bins).

Peak-centred metaprofiles intentionally remain on 433 unique peak-centre bins.
Here, all other association tests are compared between the legacy centre-bin
mask (433) and the interval-overlap mask (454). Matching uses the original four
covariates and greedy one-to-one algorithm. Because 454 interval-bound bins exceed
the 429 unbound bins, at most 429 matched pairs are possible; unmatched bound bins
are explicitly exported rather than silently discarded.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
TAB = DEL / 'supplementary_analysis/tables'
TAB.mkdir(parents=True, exist_ok=True)
N = 883
NPERM = 10000
NBOOT = 10000
SEED = 42
CALIPER = 1.0
FEATS = ['weight_WT', 'ins_WT', 'expr_wt_mean', 'n_genes']

b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
assert len(b) == N and np.array_equal(b.bin.to_numpy(), np.arange(N))
centre_mask = b.chip_bound.to_numpy(bool)
interval_mask = np.zeros(N, dtype=bool)
interval_bins = pd.read_csv(RA / '01_chip_curated/NapM_bound_bins_interval.csv')['bin'].astype(int)
interval_mask[interval_bins.to_numpy()] = True
assert centre_mask.sum() == 433 and interval_mask.sum() == 454


def greedy_match(mask):
    use = b.dropna(subset=FEATS).copy()
    bound_df = use[mask[use.bin.to_numpy(int)]].copy()
    pool_df = use[~mask[use.bin.to_numpy(int)]].copy()
    combined = pd.concat([bound_df[FEATS], pool_df[FEATS]])
    mu = combined.mean()
    sd = combined.std(ddof=1)
    bv = ((bound_df[FEATS] - mu) / sd).to_numpy(float)
    pv = ((pool_df[FEATS] - mu) / sd).to_numpy(float)
    avail = np.ones(len(pool_df), dtype=bool)
    remaining = set(range(len(bound_df)))
    rows = []
    while remaining and avail.any():
        candidates = []
        for bi in remaining:
            distances = np.sqrt(((pv[avail] - bv[bi]) ** 2).sum(axis=1))
            local = int(np.argmin(distances))
            pj = np.flatnonzero(avail)[local]
            candidates.append((float(distances[local]), bi, int(pj)))
        distance, bi, pj = min(candidates)
        rows.append(dict(bound_bin=int(bound_df.bin.iloc[bi]),
                         matched_bin=int(pool_df.bin.iloc[pj]),
                         match_distance=distance,
                         caliper_exceeded=bool(distance > CALIPER)))
        remaining.remove(bi)
        avail[pj] = False
    for bi in sorted(remaining):
        rows.append(dict(bound_bin=int(bound_df.bin.iloc[bi]), matched_bin=np.nan,
                         match_distance=np.nan, caliper_exceeded=True))
    out = pd.DataFrame(rows).sort_values('bound_bin').reset_index(drop=True)
    qc = dict(n_bound_feature_complete=int(len(bound_df)),
              n_pool_feature_complete=int(len(pool_df)),
              n_pairs=int(out.matched_bin.notna().sum()),
              n_unmatched_bound=int(out.matched_bin.isna().sum()),
              n_caliper_exceeded=int(out.caliper_exceeded.sum()),
              match_distance_median=float(out.match_distance.median()),
              match_distance_max=float(out.match_distance.max()),
              caliper=CALIPER, features=FEATS)
    return out, qc


pairs_centre, qc_centre = greedy_match(centre_mask)
pairs_interval, qc_interval = greedy_match(interval_mask)
legacy_pairs = pd.read_csv(RA / '05_matched_controls/matched_pairs.csv')
legacy_compare = pairs_centre.dropna(subset=['matched_bin']).copy()
legacy_compare['matched_bin'] = legacy_compare.matched_bin.astype(int)
centre_identical = legacy_compare[['bound_bin', 'matched_bin']].sort_values('bound_bin').reset_index(drop=True).equals(
    legacy_pairs[['bound_bin', 'matched_bin']].astype(int).sort_values('bound_bin').reset_index(drop=True))
assert centre_identical, 'Centre-bin matching failed to reproduce the legacy pairs.'
pairs_centre.insert(0, 'binding_definition', 'centre_433')
pairs_interval.insert(0, 'binding_definition', 'interval_454')
pd.concat([pairs_centre, pairs_interval], ignore_index=True).to_csv(
    TAB / 'binding_definition_matched_pairs.csv', index=False)
pd.DataFrame([dict(binding_definition='centre_433', **qc_centre),
              dict(binding_definition='interval_454', **qc_interval)]).to_csv(
    TAB / 'binding_definition_matching_qc.csv', index=False)
pairs_interval[pairs_interval.matched_bin.isna()].to_csv(
    TAB / 'interval_binding_unmatched_bound_bins.csv', index=False)

# Covariate balance for matched pairs.
balance_rows = []
for definition, pairs in [('centre_433', pairs_centre), ('interval_454', pairs_interval)]:
    pp = pairs.dropna(subset=['matched_bin'])
    bi = pp.bound_bin.to_numpy(int)
    mi = pp.matched_bin.to_numpy(int)
    for feature in FEATS:
        bv = b[feature].to_numpy(float)[bi]
        mv = b[feature].to_numpy(float)[mi]
        pooled_sd = np.sqrt((np.var(bv, ddof=1) + np.var(mv, ddof=1)) / 2)
        balance_rows.append(dict(binding_definition=definition, feature=feature,
                                 n_pairs=len(pp), bound_mean=np.mean(bv),
                                 matched_mean=np.mean(mv),
                                 standardized_mean_difference=(np.mean(bv)-np.mean(mv))/pooled_sd))
pd.DataFrame(balance_rows).to_csv(TAB / 'binding_definition_matching_balance.csv', index=False)

shifts = np.random.default_rng(SEED).integers(0, N, NPERM)
rng = np.random.default_rng(SEED + 1)
# Source the interaction explicitly from the unified condition-mean responses.
# The legacy stored interaction agrees with this definition to numerical
# precision; the assertion prevents future estimator drift.
unified = pd.read_csv(TAB / 'unified_mean_response_perbin.csv')
assert len(unified) == N and np.array_equal(unified.bin.to_numpy(), np.arange(N))
interaction_mean = unified.genotype_by_hypoxia_interaction.to_numpy(float)
assert np.allclose(interaction_mean, b.d_ins_interaction.to_numpy(float),
                   equal_nan=True)
metrics = {
    'baseline_abs_dins': np.abs(b.d_ins_KO_WT.to_numpy(float)),
    'baseline_signed_dins': b.d_ins_KO_WT.to_numpy(float),
    'local_contact_signed': b.d_localcont_KO_WT.to_numpy(float),
    'local_contact_abs': np.abs(b.d_localcont_KO_WT.to_numpy(float)),
    'hypoxia_interaction_signed': interaction_mean,
    'hypoxia_interaction_abs': np.abs(interaction_mean),
    'long_range_fraction_interaction': b.d_lrfrac_interaction.to_numpy(float),
}
result_rows = []
null_store = {}
for definition, pairs in [('centre_433', pairs_centre), ('interval_454', pairs_interval)]:
    pp = pairs.dropna(subset=['matched_bin'])
    B = pp.bound_bin.to_numpy(int)
    M = pp.matched_bin.to_numpy(int)
    for metric, values in metrics.items():
        values = values.copy()
        values[~np.isfinite(values)] = np.nan
        ok = np.isfinite(values[B]) & np.isfinite(values[M])
        B2, M2 = B[ok], M[ok]
        diffs = values[B2] - values[M2]
        observed = float(np.mean(diffs))
        boot = rng.choice(diffs, size=(NBOOT, len(diffs)), replace=True).mean(axis=1)
        ci = np.percentile(boot, [2.5, 97.5])
        null = np.empty(NPERM)
        for k, shift in enumerate(shifts):
            null[k] = np.nanmean(values[(B2 + shift) % N] - values[(M2 + shift) % N])
        p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (NPERM + 1))
        wp = float(stats.wilcoxon(diffs).pvalue)
        result_rows.append(dict(binding_definition=definition, metric=metric,
                                n_pairs=len(diffs), effect=observed,
                                ci_lo=ci[0], ci_hi=ci[1], circular_shift_p=p,
                                naive_wilcoxon_p=wp,
                                null_mean=float(null.mean()),
                                null_sd=float(null.std(ddof=1)),
                                ci_method='ordinary matched-pair bootstrap; descriptive',
                                p_method='two-sided circular shift of matched-pair offsets'))
        null_store[f'{definition}__{metric}'] = null
results = pd.DataFrame(result_rows)
results.to_csv(TAB / 'interval_binding_matched_sensitivity.csv', index=False)
np.savez_compressed(TAB / 'interval_binding_matched_shift_nulls.npz', **null_store)

# Structure-detected overlap uses every bin, including bound bins lost in matching.
structure = b.structure_detected.to_numpy(bool)
overlaps = []
for definition, mask in [('centre_433', centre_mask), ('interval_454', interval_mask)]:
    a = int(np.sum(mask & structure))
    bb = int(np.sum(mask & ~structure))
    c = int(np.sum(~mask & structure))
    d = int(np.sum(~mask & ~structure))
    odds, fisher_p = stats.fisher_exact([[a, bb], [c, d]])
    se = np.sqrt(1/a + 1/bb + 1/c + 1/d)
    ci = np.exp(np.log(odds) + np.array([-1.96, 1.96]) * se)
    indices = np.flatnonzero(mask)
    null = np.empty(NPERM)
    for k, shift in enumerate(shifts):
        shifted = np.zeros(N, dtype=bool)
        shifted[(indices + shift) % N] = True
        null[k] = np.sum(shifted & structure)
    p_shift = float((1 + np.sum(null >= a)) / (NPERM + 1))
    overlaps.append(dict(binding_definition=definition, table_bound_detected=a,
                         table_bound_not_detected=bb, table_unbound_detected=c,
                         table_unbound_not_detected=d, odds_ratio=odds,
                         ci_lo=ci[0], ci_hi=ci[1], fisher_p=fisher_p,
                         circular_shift_p_one_sided=p_shift,
                         null_mean_overlap=float(null.mean()),
                         n_bound_bins=int(mask.sum())))
    null_store[f'{definition}__structure_overlap'] = null
pd.DataFrame(overlaps).to_csv(TAB / 'interval_binding_structure_overlap.csv', index=False)
np.savez_compressed(TAB / 'interval_binding_all_shift_nulls.npz', **null_store)

metadata = dict(seed=SEED, n_permutations=NPERM, n_bootstrap=NBOOT,
                caliper=CALIPER, features=FEATS,
                centre_bins=int(centre_mask.sum()), interval_bins=int(interval_mask.sum()),
                centre_matching_reproduced=centre_identical,
                note='Caliper is diagnostic and was not used to exclude pairs, matching legacy analysis.')
(TAB / 'interval_binding_analysis_metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n')

print('Matching QC:', json.dumps({'centre': qc_centre, 'interval': qc_interval}, indent=2))
print(results[['binding_definition', 'metric', 'n_pairs', 'effect', 'ci_lo', 'ci_hi',
               'circular_shift_p', 'naive_wilcoxon_p']].to_string(index=False))
print('\nStructure overlap:\n', pd.DataFrame(overlaps).to_string(index=False))
print('\nWrote outputs to', TAB)
