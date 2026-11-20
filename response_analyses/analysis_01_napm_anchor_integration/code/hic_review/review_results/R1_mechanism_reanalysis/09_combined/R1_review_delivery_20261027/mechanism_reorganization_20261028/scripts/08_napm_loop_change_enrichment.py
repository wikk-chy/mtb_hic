#!/usr/bin/env python3
"""Test NapM-associated loop-change magnitude, enrichment and direction.

Q1: Do loops with two NapM-occupied anchors change more than other loops?
    Signed d_log2OE tests weakening; absolute d_log2OE tests change of either
    direction. Continuous outcomes are primary.
Q2: Are NapM-occupied loops enriched among strongly changing loops, and do
    they preferentially weaken rather than strengthen? Threshold analyses are
    sensitivity analyses, not independent confirmations of Q1.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
OUT = DEL / 'mechanism_reorganization_20261028/loop_change_enrichment'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
NBOOT = 10000
NPERM = 10000

quant = pd.read_csv(DEL / 'mechanism_reorganization_20261028/source_data/Fig1C_loop_dose_per_loop.csv')
raw = pd.read_csv(RA / '06_permutation_tests/loop_oe_wt_ko_check.csv')
m = quant.merge(raw[['loop_id'] + [f'WT{i}' for i in range(1, 5)] + [f'KO{i}' for i in range(1, 5)]],
                on='loop_id', validate='one_to_one')
assert len(m) == 39
assert np.allclose(np.log2(m[[f'KO{i}' for i in range(1, 5)]].mean(axis=1) /
                           m[[f'WT{i}' for i in range(1, 5)]].mean(axis=1)),
                   m.d_log2OE, atol=1e-10)
m['double'] = m.n_bound_anchors.eq(2)
m['any'] = m.n_bound_anchors.ge(1)
m['abs_d_log2OE'] = m.d_log2OE.abs()


def adjusted_coefficient(labels, y, log_span, log_wt):
    labels = np.asarray(labels, float)
    y = np.asarray(y, float)
    x = np.column_stack([np.ones(len(y)), labels, log_span, log_wt])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    adjusted = residual / np.maximum(1 - leverage, 1e-12)
    covariance = xtx_inv @ (x.T @ (x * (adjusted ** 2)[:, None])) @ xtx_inv
    se = float(np.sqrt(covariance[1, 1]))
    df = len(y) - x.shape[1]
    critical = float(stats.t.ppf(0.975, df))
    return float(beta[1]), se, float(beta[1] - critical * se), float(beta[1] + critical * se)


def permutation_p(labels, y, strata, log_span, log_wt, observed, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, bool)
    strata = np.asarray(strata, int)
    stratum_indices = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(NPERM)
    for k in range(NPERM):
        permuted = labels.copy()
        for idx in stratum_indices:
            permuted[idx] = rng.permutation(labels[idx])
        null[k] = adjusted_coefficient(permuted, y, log_span, log_wt)[0]
    p = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (NPERM + 1))
    return p, null


def bootstrap_contrasts(frame, seed):
    rng = np.random.default_rng(seed)
    n = len(frame)
    labels = {name: frame[name].to_numpy(bool) for name in ['double', 'any']}
    strata = frame.span_tercile.to_numpy(int)
    wt_all = frame[[f'WT{i}' for i in range(1, 5)]].to_numpy(float)
    ko_all = frame[[f'KO{i}' for i in range(1, 5)]].to_numpy(float)
    stratum_indices = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = {f'{label}_{metric}': np.empty(NBOOT)
           for label in ['double', 'any'] for metric in ['signed', 'absolute']}
    for k in range(NBOOT):
        selected = np.concatenate([rng.choice(idx, size=len(idx), replace=True)
                                   for idx in stratum_indices])
        delta = np.empty(n)
        for j, loop_index in enumerate(selected):
            wt = wt_all[loop_index]
            ko = ko_all[loop_index]
            wt = wt[np.isfinite(wt)]
            ko = ko[np.isfinite(ko)]
            wt_mean = rng.choice(wt, size=4, replace=True).mean()
            ko_mean = rng.choice(ko, size=4, replace=True).mean()
            delta[j] = np.log2(ko_mean / wt_mean)
        for label_name, label_values in labels.items():
            target = label_values[selected]
            for metric, values in [('signed', delta), ('absolute', np.abs(delta))]:
                if target.any() and (~target).any():
                    out[f'{label_name}_{metric}'][k] = values[target].mean() - values[~target].mean()
                else:
                    out[f'{label_name}_{metric}'][k] = np.nan
    return out


contrast_rows = []
bootstrap_frames = []
permutation_frames = []
analysis_sets = [('all_39', m.copy()), ('QC_38_excluding_L11', m[m.loop_id.ne('L11')].copy())]
for analysis_index, (analysis_set, frame) in enumerate(analysis_sets):
    boot = bootstrap_contrasts(frame, SEED + 1000 * analysis_index)
    boot_df = pd.DataFrame(boot)
    boot_df.insert(0, 'analysis_set', analysis_set)
    boot_df.insert(1, 'bootstrap_iteration', np.arange(1, NBOOT + 1))
    bootstrap_frames.append(boot_df)
    for comparison_index, (comparison, label_name) in enumerate([
            ('double_vs_0_or_1', 'double'), ('any_vs_none', 'any')]):
        labels = frame[label_name].to_numpy(bool)
        for metric_index, (metric, values) in enumerate([
                ('signed_d_log2OE', frame.d_log2OE.to_numpy(float)),
                ('absolute_d_log2OE', frame.abs_d_log2OE.to_numpy(float))]):
            target = values[labels]
            reference = values[~labels]
            observed = float(target.mean() - reference.mean())
            lo, hi = np.nanpercentile(boot[f'{label_name}_{"signed" if metric_index == 0 else "absolute"}'],
                                      [2.5, 97.5])
            log_span = np.log2(frame.distance_kb.to_numpy(float))
            log_wt = np.log2(frame.WT_mean.to_numpy(float))
            beta, se, hc3_lo, hc3_hi = adjusted_coefficient(labels, values, log_span, log_wt)
            perm_p, null = permutation_p(labels, values, frame.span_tercile.to_numpy(),
                                         log_span, log_wt, beta,
                                         SEED + 1000 * analysis_index + 10 * comparison_index + metric_index)
            permutation_frames.append(pd.DataFrame(dict(
                analysis_set=analysis_set, comparison=comparison, metric=metric,
                permutation_iteration=np.arange(1, NPERM + 1), null_adjusted_effect=null)))
            contrast_rows.append(dict(
                analysis_set=analysis_set, comparison=comparison, metric=metric,
                n_target=int(labels.sum()), n_reference=int((~labels).sum()),
                target_mean=float(target.mean()), reference_mean=float(reference.mean()),
                target_median=float(np.median(target)), reference_median=float(np.median(reference)),
                mean_difference_target_minus_reference=observed,
                median_difference_target_minus_reference=float(np.median(target) - np.median(reference)),
                bootstrap_ci_lo=lo, bootstrap_ci_hi=hi,
                adjusted_effect=beta, hc3_se=se, hc3_ci_lo=hc3_lo, hc3_ci_hi=hc3_hi,
                span_stratified_permutation_two_sided_p=perm_p,
                adjustment='log2(loop span) + log2(WT mean O/E)'))
contrasts = pd.DataFrame(contrast_rows)
contrasts.to_csv(OUT / 'continuous_binding_change_tests.csv', index=False)
pd.concat(bootstrap_frames, ignore_index=True).to_csv(OUT / 'continuous_binding_change_bootstrap.csv', index=False)
pd.concat(permutation_frames, ignore_index=True).to_csv(OUT / 'continuous_binding_change_permutation.csv', index=False)

# Threshold-based reciprocal summaries: P(change|binding) and P(binding|change)
# are two descriptions of the same 2x2 table, not independent confirmations.
enrichment_rows = []
for comparison, label_name in [('double_vs_0_or_1', 'double'), ('any_vs_none', 'any')]:
    labels = m[label_name].to_numpy(bool)
    for threshold in [0.25, 0.5, 1.0]:
        definitions = [(f'absolute_change_ge_{threshold}', m.abs_d_log2OE.to_numpy() >= threshold)]
        if threshold == 0.5:
            definitions += [('weakening_le_minus_0.5', m.d_log2OE.to_numpy() <= -0.5),
                            ('strengthening_ge_0.5', m.d_log2OE.to_numpy() >= 0.5)]
        for definition, event in definitions:
            table = np.array([[int(((~labels) & event).sum()), int(((~labels) & ~event).sum())],
                              [int((labels & event).sum()), int((labels & ~event).sum())]])
            odds_ratio, fisher_p = stats.fisher_exact(table[::-1], alternative='two-sided')
            target_n = int(labels.sum())
            reference_n = int((~labels).sum())
            target_event = int((labels & event).sum())
            reference_event = int(((~labels) & event).sum())
            event_n = int(event.sum())
            nonevent_n = int((~event).sum())
            enrichment_rows.append(dict(
                comparison=comparison, definition=definition,
                target_n=target_n, target_event=target_event,
                reference_n=reference_n, reference_event=reference_event,
                target_event_rate=target_event / target_n,
                reference_event_rate=reference_event / reference_n,
                event_rate_difference=target_event / target_n - reference_event / reference_n,
                target_fraction_among_event=target_event / event_n if event_n else np.nan,
                target_fraction_among_nonevent=(target_n - target_event) / nonevent_n if nonevent_n else np.nan,
                event_n=event_n, nonevent_n=nonevent_n,
                fisher_odds_ratio=odds_ratio, fisher_two_sided_p=fisher_p,
                table_reference_event_nonevent=str(table[0].tolist()),
                table_target_event_nonevent=str(table[1].tolist())))
enrichment = pd.DataFrame(enrichment_rows)
enrichment.to_csv(OUT / 'strong_change_binding_enrichment.csv', index=False)

# Among changing loops, is the direction preferentially weakening?
direction_rows = []
for threshold in [0.25, 0.5, 1.0]:
    changing = m.abs_d_log2OE.to_numpy() >= threshold
    weakening = m.d_log2OE.to_numpy() < 0
    for comparison, label_name in [('double_vs_0_or_1', 'double'), ('any_vs_none', 'any')]:
        labels = m[label_name].to_numpy(bool)
        target = changing & labels
        reference = changing & ~labels
        table = np.array([[int((reference & weakening).sum()), int((reference & ~weakening).sum())],
                          [int((target & weakening).sum()), int((target & ~weakening).sum())]])
        odds_ratio, fisher_p = stats.fisher_exact(table[::-1], alternative='two-sided')
        target_n = int(target.sum())
        reference_n = int(reference.sum())
        target_weakening = int((target & weakening).sum())
        reference_weakening = int((reference & weakening).sum())
        direction_rows.append(dict(
            comparison=comparison, change_threshold=threshold,
            target_changing_n=target_n, target_weakening_n=target_weakening,
            reference_changing_n=reference_n, reference_weakening_n=reference_weakening,
            target_weakening_fraction=target_weakening / target_n if target_n else np.nan,
            reference_weakening_fraction=reference_weakening / reference_n if reference_n else np.nan,
            fisher_odds_ratio=odds_ratio, fisher_two_sided_p=fisher_p,
            table_reference_weakening_strengthening=str(table[0].tolist()),
            table_target_weakening_strengthening=str(table[1].tolist())))
direction = pd.DataFrame(direction_rows)
direction.to_csv(OUT / 'changing_loop_direction_preference.csv', index=False)

m.to_csv(OUT / 'per_loop_change_and_binding.csv', index=False)
metadata = dict(
    seed=SEED, n_bootstrap=NBOOT, n_permutation=NPERM,
    primary_comparison='Two NapM-occupied anchors versus zero or one occupied anchor.',
    signed_effect='log2(KO mean O/E / WT mean O/E); negative target-minus-reference contrasts indicate greater weakening.',
    absolute_effect='Absolute log2(KO mean O/E / WT mean O/E); positive target-minus-reference contrasts indicate greater change of either direction.',
    primary_change_threshold='Absolute d_log2OE >= 0.5 (at least approximately 1.41-fold change); other thresholds are sensitivity analyses.',
    direction_definition='Among loops passing the absolute-change threshold, d_log2OE < 0 is weakening and d_log2OE > 0 is strengthening.',
    non_independence='Threshold enrichment and change-rate comparisons are reciprocal summaries of the same 2x2 table, not independent evidence.',
    caveats=['Loops are genomic features, not independent biological replicates.',
             'Some loops share anchors; Fisher and loop-label permutation analyses remain exploratory.',
             'A threshold for strong change is descriptive; continuous outcomes are primary.'])
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

print('Continuous contrasts:')
print(contrasts[['analysis_set', 'comparison', 'metric', 'n_target', 'n_reference',
                 'target_mean', 'reference_mean', 'mean_difference_target_minus_reference',
                 'bootstrap_ci_lo', 'bootstrap_ci_hi', 'adjusted_effect',
                 'span_stratified_permutation_two_sided_p']].to_string(index=False))
print('\nStrong-change enrichment:')
print(enrichment[['comparison', 'definition', 'target_n', 'target_event', 'reference_n',
                  'reference_event', 'target_event_rate', 'reference_event_rate',
                  'target_fraction_among_event', 'target_fraction_among_nonevent',
                  'fisher_two_sided_p']].to_string(index=False))
print('\nDirection among changing loops:')
print(direction.to_string(index=False))
print('\nWrote outputs to', OUT)
