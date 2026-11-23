#!/usr/bin/env python3
"""Independent KO Chromosight recall sensitivity for WT-called loops.

Primary question: are WT-called loops with NapM-occupied anchors less likely
to be recovered by an independent Chromosight call on the napM-KO map?

Important: non-recovery is a caller/threshold outcome, not proof that a loop
has physically disappeared. We therefore report recovery alongside continuous
same-coordinate KO/WT O/E changes from Response Figure 1C.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
OUT = DEL / 'mechanism_reorganization_20261028/loop_recall_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NPERM = 10000
PRIMARY_TOL = 1

wt_calls = pd.read_csv(ROOT / 'hic_review/review_results/R4_comment3_loops/chromosight_out/wt.tsv', sep='\t')
ko_original = pd.read_csv(ROOT / 'hic_review/review_results/R4_comment3_loops/chromosight_out/napm_ko.tsv', sep='\t')
ko_calls = pd.read_csv(OUT / 'chromosight_rerun/napm_ko.tsv', sep='\t')
pd.testing.assert_frame_equal(ko_calls, ko_original, check_exact=True)
assert len(wt_calls) == 41 and len(ko_calls) == 100

anchors = pd.read_csv(DEL / 'supplementary_analysis/tables/corrected_figure_sources/Fig1_final_A_loop_anchors.csv')
quant = pd.read_csv(DEL / 'mechanism_reorganization_20261028/source_data/Fig1C_loop_dose_per_loop.csv')
loops = anchors.merge(
    wt_calls[['bin1', 'bin2', 'score', 'qvalue']],
    on=['bin1', 'bin2'], how='left', validate='one_to_one')
assert len(loops) == 41 and loops.score.notna().all()
loops['n_bound_anchors'] = loops.anchor1_bound.astype(int) + loops.anchor2_bound.astype(int)
loops['distance_kb'] = (loops.center2 - loops.center1) / 1000
loops['span_tercile'] = pd.qcut(loops.distance_kb, 3, labels=False).astype(int)
loops = loops.merge(
    quant[['loop_id', 'WT_mean', 'KO_mean', 'd_log2OE', 'qc_include']],
    on='loop_id', how='left', validate='one_to_one')

ko_calls = ko_calls.copy()
ko_calls.insert(0, 'ko_call_id', [f'K{i:03d}' for i in range(len(ko_calls))])


def match_calls(tolerance):
    records = []
    for _, row in loops.iterrows():
        delta1 = (ko_calls.bin1 - int(row.bin1)).abs()
        delta2 = (ko_calls.bin2 - int(row.bin2)).abs()
        candidates = ko_calls[(delta1 <= tolerance) & (delta2 <= tolerance)].copy()
        if candidates.empty:
            records.append(dict(recovered=False, ko_call_id='', ko_bin1=np.nan,
                                ko_bin2=np.nan, shift1=np.nan, shift2=np.nan,
                                ko_score=np.nan, ko_qvalue=np.nan))
        else:
            candidates['shift1'] = (candidates.bin1 - int(row.bin1)).abs()
            candidates['shift2'] = (candidates.bin2 - int(row.bin2)).abs()
            candidates = candidates.sort_values(
                ['shift1', 'shift2', 'score'], ascending=[True, True, False])
            best = candidates.iloc[0]
            records.append(dict(recovered=True, ko_call_id=best.ko_call_id,
                                ko_bin1=int(best.bin1), ko_bin2=int(best.bin2),
                                shift1=int(best.shift1), shift2=int(best.shift2),
                                ko_score=float(best.score), ko_qvalue=float(best.qvalue)))
    return pd.DataFrame(records, index=loops.index)


for tolerance in [0, 1, 2]:
    matched = match_calls(tolerance).add_suffix(f'_tol{tolerance}')
    loops = pd.concat([loops, matched], axis=1)

loops['recovered_primary'] = loops[f'recovered_tol{PRIMARY_TOL}']
loops['not_recovered_primary'] = ~loops.recovered_primary
loops['any_bound'] = loops.n_bound_anchors >= 1
loops['both_bound_primary'] = loops.n_bound_anchors == 2
loops.to_csv(OUT / 'wt_loop_ko_recall_per_loop.csv', index=False)

summary_rows = []
for tolerance in [0, 1, 2]:
    recovered = loops[f'recovered_tol{tolerance}']
    for group, indices in [('all', np.ones(len(loops), dtype=bool))] + [
            (str(n), loops.n_bound_anchors.to_numpy() == n) for n in [0, 1, 2]]:
        subset = loops.loc[indices & loops.d_log2OE.notna()]
        summary_rows.append(dict(
            tolerance_bins=tolerance, anchor_group=group, n_loops=int(indices.sum()),
            n_recovered=int(recovered[indices].sum()),
            n_not_recovered=int((~recovered[indices]).sum()),
            recovery_rate=float(recovered[indices].mean()),
            nonrecovery_rate=float((~recovered[indices]).mean()),
            n_with_continuous_OE=len(subset),
            median_d_log2OE=float(subset.d_log2OE.median()) if len(subset) else np.nan,
            median_KO_OE=float(subset.KO_mean.median()) if len(subset) else np.nan))
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / 'ko_recall_group_summary.csv', index=False)


def span_stratified_permutation(labels, nonrecovered, strata, observed, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, bool)
    nonrecovered = np.asarray(nonrecovered, float)
    strata = np.asarray(strata, int)
    stratum_indices = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(NPERM)
    for k in range(NPERM):
        permuted = labels.copy()
        for idx in stratum_indices:
            permuted[idx] = rng.permutation(labels[idx])
        null[k] = nonrecovered[permuted].mean() - nonrecovered[~permuted].mean()
    p = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (NPERM + 1))
    return p, null


test_rows = []
permutation_frames = []
for tolerance in [0, 1, 2]:
    nonrecovered = (~loops[f'recovered_tol{tolerance}']).to_numpy(float)
    for comparison, labels in [('both_vs_0_or_1', loops.n_bound_anchors.to_numpy() == 2),
                               ('any_vs_none', loops.n_bound_anchors.to_numpy() >= 1)]:
        table = np.array([[int(((~labels) & (nonrecovered == 0)).sum()),
                           int(((~labels) & (nonrecovered == 1)).sum())],
                          [int((labels & (nonrecovered == 0)).sum()),
                           int((labels & (nonrecovered == 1)).sum())]])
        odds_ratio, fisher_p = stats.fisher_exact(table, alternative='two-sided')
        observed = float(nonrecovered[labels].mean() - nonrecovered[~labels].mean())
        perm_p, null = span_stratified_permutation(
            labels, nonrecovered, loops.span_tercile.to_numpy(), observed,
            SEED + 100 * tolerance + (comparison == 'any_vs_none'))
        test_rows.append(dict(
            tolerance_bins=tolerance, comparison=comparison,
            n_target=int(labels.sum()), n_reference=int((~labels).sum()),
            target_not_recovered=int(nonrecovered[labels].sum()),
            reference_not_recovered=int(nonrecovered[~labels].sum()),
            target_nonrecovery_rate=float(nonrecovered[labels].mean()),
            reference_nonrecovery_rate=float(nonrecovered[~labels].mean()),
            nonrecovery_difference=observed, fisher_or=odds_ratio,
            fisher_two_sided_p=fisher_p,
            span_stratified_permutation_two_sided_p=perm_p,
            fisher_table_reference_recovered_notrecovered=str(table[0].tolist()),
            fisher_table_target_recovered_notrecovered=str(table[1].tolist())))
        permutation_frames.append(pd.DataFrame(dict(
            tolerance_bins=tolerance, comparison=comparison,
            permutation_iteration=np.arange(1, NPERM + 1),
            null_nonrecovery_difference=null)))
tests = pd.DataFrame(test_rows)
tests.to_csv(OUT / 'ko_recall_binding_tests.csv', index=False)
pd.concat(permutation_frames, ignore_index=True).to_csv(
    OUT / 'ko_recall_binding_permutation.csv', index=False)

# Continuous-signal context for the binary caller outcome.
continuous_rows = []
for tolerance in [0, 1, 2]:
    subset = loops.loc[loops.d_log2OE.notna()].copy()
    recalled = subset[f'recovered_tol{tolerance}']
    a = subset.loc[recalled, 'd_log2OE'].to_numpy(float)
    b = subset.loc[~recalled, 'd_log2OE'].to_numpy(float)
    if len(a) and len(b):
        u_stat, mw_p = stats.mannwhitneyu(a, b, alternative='two-sided', method='auto')
    else:
        u_stat, mw_p = np.nan, np.nan
    continuous_rows.append(dict(
        tolerance_bins=tolerance, n_recovered=len(a), n_not_recovered=len(b),
        median_d_log2OE_recovered=float(np.median(a)) if len(a) else np.nan,
        median_d_log2OE_not_recovered=float(np.median(b)) if len(b) else np.nan,
        median_KO_OE_recovered=float(subset.loc[recalled, 'KO_mean'].median()),
        median_KO_OE_not_recovered=float(subset.loc[~recalled, 'KO_mean'].median()),
        mannwhitney_u=u_stat, mannwhitney_two_sided_p=mw_p,
        n_not_recovered_with_d_log2OE_ge_0=int((subset.loc[~recalled, 'd_log2OE'] >= 0).sum()),
        n_not_recovered_with_KO_OE_ge_1=int((subset.loc[~recalled, 'KO_mean'] >= 1).sum())))
continuous = pd.DataFrame(continuous_rows)
continuous.to_csv(OUT / 'ko_recall_continuous_signal_context.csv', index=False)

metadata = dict(
    seed=SEED, n_permutation=NPERM, primary_tolerance_bins=PRIMARY_TOL,
    n_WT_called_loops=len(wt_calls), n_KO_called_loops=len(ko_calls),
    ko_rerun_exactly_matches_existing_output=True,
    chromosight_parameters=dict(pattern='loops', pearson=0.3, min_dist=30000,
                                max_dist=1000000, resolution=5000),
    recovery_definition='A WT loop is recovered when both anchors match an independent KO call within the stated per-anchor bin tolerance.',
    primary_binding_comparison='Two NapM-occupied anchors versus zero or one occupied anchor.',
    interpretation_boundary='Non-recovery is a binary caller outcome and must not be equated with physical loop disappearance. Continuous same-coordinate O/E remains primary.',
    shared_anchor_caveat='Loops are genomic features and some share anchors; Fisher and loop-label permutation tests are descriptive sensitivity analyses, not independent biological replication.')
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

print('KO rerun exactly matches the existing 100-call output.')
print('\nOverall recovery:')
print(summary[summary.anchor_group.eq('all')].to_string(index=False))
print('\nRecovery by NapM-occupied anchor count:')
print(summary[(summary.tolerance_bins.eq(PRIMARY_TOL)) & (~summary.anchor_group.eq('all'))].to_string(index=False))
print('\nBinding association tests:')
print(tests.to_string(index=False))
print('\nContinuous-signal context:')
print(continuous.to_string(index=False))
print('\nNon-recovered loops at primary tolerance:')
cols = ['loop_id', 'bin1', 'bin2', 'n_bound_anchors', 'WT_mean', 'KO_mean',
        'd_log2OE', 'recovered_tol0', 'recovered_tol1', 'recovered_tol2']
print(loops.loc[~loops.recovered_primary, cols].to_string(index=False))
print('\nWrote outputs to', OUT)
