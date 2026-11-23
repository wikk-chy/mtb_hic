#!/usr/bin/env python3
"""Complete-case sensitivity for the Fig. 3B nested-model block CV.

The legacy nested CV used 858 bins, median-imputed baseline expression, and
zero-imputed missing RNA log2 fold change; its baseline model also omitted
n_genes. Here all models use the same 848 complete bins as the multiple
regression, with no RNA or expression imputation. M0 contains the same baseline
covariates as that regression (weight_WT, ins_WT, expr_wt_mean and n_genes),
M1 adds RNA log2 fold change, and M2 adds NapM occupancy.

Cross-validation uses 10 contiguous genomic blocks in bin order. Block
bootstrap (1,000 rounds, seed 42) supplies descriptive 95% intervals for the
incremental out-of-sample R2 differences. This is predictive sensitivity, not
causal inference or a correction of the ordinary regression P value.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
TAB = DEL / 'supplementary_analysis/tables'
TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BLOCKS = 10
N_BOOT = 1000
RNG = np.random.default_rng(SEED)

b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
cols = ['bin', 'd_ins_KO_WT', 'weight_WT', 'ins_WT', 'expr_wt_mean',
        'n_genes', 'rna_mean_log2FC_KOvsWT', 'chip_bound']
m = b[cols].copy()
numeric = [c for c in cols if c != 'chip_bound']
complete = np.isfinite(m[numeric].to_numpy(float)).all(axis=1)
m = m.loc[complete].reset_index(drop=True)
assert len(m) == 848, f'Unexpected complete-case n: {len(m)}'

# Preserve genomic order and the mapping back to the 5-kb genome bins.
block_id = np.arange(len(m)) * N_BLOCKS // len(m)
assign = m[['bin']].copy()
assign['complete_case_row'] = np.arange(len(m))
assign['cv_block'] = block_id
assign.to_csv(TAB / 'nested_cv_complete_case_block_assignments.csv', index=False)

y = m.d_ins_KO_WT.to_numpy(float)
X0 = m[['weight_WT', 'ins_WT', 'expr_wt_mean', 'n_genes']].to_numpy(float)
X1 = np.column_stack([X0, m.rna_mean_log2FC_KOvsWT.to_numpy(float)])
X2 = np.column_stack([X1, m.chip_bound.to_numpy(float)])
models = [('M0_baseline', X0), ('M1_plus_RNA', X1), ('M2_plus_occupancy', X2)]


def cv_r2(X, yv, blocks):
    pred = np.full(len(yv), np.nan)
    for blk in np.unique(blocks):
        train = blocks != blk
        test = blocks == blk
        Xt = np.column_stack([np.ones(train.sum()), X[train]])
        beta = np.linalg.lstsq(Xt, yv[train], rcond=None)[0]
        Xe = np.column_stack([np.ones(test.sum()), X[test]])
        pred[test] = Xe @ beta
    return 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)


observed = {name: cv_r2(X, y, block_id) for name, X in models}
model_table = pd.DataFrame([
    dict(model=name, cv_R2=observed[name], n_bins=len(m),
         n_predictors=X.shape[1], imputation='none; complete cases only',
         baseline_covariates='weight_WT + ins_WT + expr_wt_mean + n_genes')
    for name, X in models
])
model_table.to_csv(TAB / 'nested_cv_complete_case_models.csv', index=False)

boot = np.empty((N_BOOT, 2))
blocks = np.arange(N_BLOCKS)
for k in range(N_BOOT):
    selected = RNG.choice(blocks, size=N_BLOCKS, replace=True)
    idx = np.concatenate([np.flatnonzero(block_id == blk) for blk in selected])
    new_blocks = np.concatenate([
        np.full(np.sum(block_id == blk), j) for j, blk in enumerate(selected)
    ])
    r0 = cv_r2(X0[idx], y[idx], new_blocks)
    r1 = cv_r2(X1[idx], y[idx], new_blocks)
    r2 = cv_r2(X2[idx], y[idx], new_blocks)
    boot[k] = (r1 - r0, r2 - r1)

boot_df = pd.DataFrame(boot, columns=['dR2_M1_vs_M0', 'dR2_M2_vs_M1'])
boot_df.to_csv(TAB / 'nested_cv_complete_case_bootstrap.csv', index=False)
ci = np.percentile(boot, [2.5, 97.5], axis=0)
comparison = pd.DataFrame([
    dict(comparison='M1_plus_RNA - M0_baseline',
         dR2=observed['M1_plus_RNA'] - observed['M0_baseline'],
         ci_lo=ci[0, 0], ci_hi=ci[1, 0], n_bins=len(m),
         n_bootstrap=N_BOOT, method='10-block CV; block bootstrap 95% CI'),
    dict(comparison='M2_plus_occupancy - M1_plus_RNA',
         dR2=observed['M2_plus_occupancy'] - observed['M1_plus_RNA'],
         ci_lo=ci[0, 1], ci_hi=ci[1, 1], n_bins=len(m),
         n_bootstrap=N_BOOT, method='10-block CV; block bootstrap 95% CI'),
])
comparison.to_csv(TAB / 'nested_cv_complete_case_comparison.csv', index=False)

legacy_models = pd.read_csv(
    RA.parent / 'R1_mechanism/R1_mechanism_zhenghe/tables/stats_fig2_nested_models.csv')
legacy_cmp = pd.read_csv(
    RA.parent / 'R1_mechanism/R1_mechanism_zhenghe/tables/stats_fig2_model_comparison.csv')
legacy = pd.DataFrame([
    dict(item='legacy_n_bins', value=int(legacy_models.n_bins.iloc[0])),
    dict(item='legacy_M0_M1_M2_CV_R2',
         value=';'.join(f'{v:.12f}' for v in legacy_models.cv_R2)),
    dict(item='legacy_dR2_M1_vs_M0', value=legacy_cmp.dR2_M1_vs_M0.iloc[0]),
    dict(item='legacy_dR2_M2_vs_M1', value=legacy_cmp.dR2_M2_vs_M1.iloc[0]),
    dict(item='legacy_missing_RNA_rule', value='filled with 0'),
    dict(item='legacy_missing_expression_rule', value='filled with median'),
    dict(item='legacy_baseline_covariates', value='weight_WT + ins_WT + expr_wt_mean'),
])
legacy.to_csv(TAB / 'nested_cv_legacy_reconciliation.csv', index=False)

metadata = dict(seed=SEED, n_boot=N_BOOT, n_blocks=N_BLOCKS, n_complete=len(m),
                input=str(RA / '03_structure_metrics/bin_level_v2.csv'),
                complete_case_rule='all model covariates, RNA and outcome finite; no imputation',
                models=[name for name, _ in models],
                M0_predictors=['weight_WT', 'ins_WT', 'expr_wt_mean', 'n_genes'])
(TAB / 'nested_cv_complete_case_metadata.json').write_text(
    json.dumps(metadata, indent=2) + '\n')

print(model_table.to_string(index=False))
print(comparison.to_string(index=False))
print('Legacy reconciliation:')
print(legacy.to_string(index=False))
print('Wrote outputs to', TAB)
