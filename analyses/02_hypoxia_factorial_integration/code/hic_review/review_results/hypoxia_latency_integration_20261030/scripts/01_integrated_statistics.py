#!/usr/bin/env python3
"""Integrated hypoxia and latency analyses from the project's curated tables.

Primary Hi-C effect sizes retain the established robust pairwise-median/90%
interval definitions. Welch tests are supplied as complementary per-bin
inference. RNA hypoxia contrasts are recalculated from raw counts using an
eight-sample TMM + limma-voom + eBayes factorial model; baseline and latent
two-group contrasts come from the existing limma result tables. No GO/KEGG
enrichment is recomputed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
TAB, SRC = OUT / 'tables', OUT / 'source_data'
for p in [TAB, SRC, OUT / 'figures', OUT / 'logs', OUT / 'scripts']:
    p.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261030)
N, BINSIZE, GENOME = 883, 5000, 4411532
DET = {'detected_increase', 'detected_decrease'}
N_PERM, N_BOOT, BLOCK = 10000, 2000, 20


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1)
    return out


def classify(est, lo, hi, bound=.25):
    est, lo, hi = np.asarray(est), np.asarray(lo), np.asarray(hi)
    return np.select([lo > bound, hi < -bound, (lo >= -bound) & (hi <= bound)],
                     ['detected_increase', 'detected_decrease', 'supports_small_change'],
                     default='cannot_determine')


def rna_class(effect, q):
    effect, q = np.asarray(effect, float), np.asarray(q, float)
    return np.select([np.isfinite(effect) & np.isfinite(q) & (q < .05) & (np.abs(effect) >= 1) & (effect > 0),
                      np.isfinite(effect) & np.isfinite(q) & (q < .05) & (np.abs(effect) >= 1) & (effect < 0),
                      np.isfinite(effect) & np.isfinite(q) & (q >= .05) & (np.abs(effect) < .5)],
                     ['detected_increase', 'detected_decrease', 'supports_small_change'],
                     default='cannot_determine')


def pair_stats(a, b):
    """b-a across all replicate pairs; robust point and 5-95% interval."""
    ds = (np.asarray(b)[:, None, :] - np.asarray(a)[None, :, :]).reshape(-1, np.asarray(a).shape[1])
    return (np.nanmedian(ds, axis=0), np.nanpercentile(ds, 5, axis=0),
            np.nanpercentile(ds, 95, axis=0))


def welch(groups, signs):
    est = np.zeros(N); var = np.zeros(N); num = np.zeros(N); den = np.zeros(N)
    for arr, sign in zip(groups, signs):
        arr = np.asarray(arr, float)
        n = np.isfinite(arr).sum(axis=0)
        mean = np.nanmean(arr, axis=0)
        v = np.nanvar(arr, axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            q = v / n
        est += sign * mean
        var += q
        num += q
        den += np.where(n > 1, q * q / (n - 1), 0)
    se = np.sqrt(var)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = est / se
        df = np.where(den > 0, num * num / den, np.nan)
        p = 2 * stats.t.sf(np.abs(t), df)
    p[~np.isfinite(est) | ~np.isfinite(se) | (se == 0)] = np.nan
    return est, se, p


def circ_spearman(x, y, n_perm=N_PERM):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    obs = stats.spearmanr(x[ok], y[ok]).statistic
    null = np.full(n_perm, np.nan)
    for k in range(n_perm):
        xr = np.roll(x, int(RNG.integers(1, N)))
        good = np.isfinite(xr) & np.isfinite(y)
        null[k] = stats.spearmanr(xr[good], y[good]).statistic
    p = (np.nansum(np.abs(null) >= abs(obs)) + 1) / (np.isfinite(null).sum() + 1)
    return obs, p, null


def block_boot_indices(n, n_boot=N_BOOT, block=BLOCK):
    out = np.empty((n_boot, n), int)
    nblocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    for b in range(n_boot):
        starts = RNG.integers(0, N, nblocks)
        idx = np.concatenate([(s + offsets) % N for s in starts])[:n]
        out[b] = idx
    return out


def ci_boot(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) else (np.nan, np.nan)


b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
pairs = pd.read_csv(RA / '05_matched_controls/matched_pairs.csv')
assert len(b) == N and len(g) == 4109 and b['bin'].is_unique and g.gene_id.is_unique
assert np.array_equal(b['bin'].to_numpy(), np.arange(N))
assert (g.bin_tss.to_numpy() == ((g.tss.to_numpy() - 1) // BINSIZE)).all()
b['NapM_bound'] = b.chip_bound.astype(bool)
g['NapM_occupied'] = g.binding_class.isin(['promoter_bound', 'gene_body_bound'])

# Replace the superseded unmoderated OLS fields with the recalculated
# eight-sample TMM + limma-voom + eBayes factorial results.
limma = pd.read_csv(TAB / 'Hypoxia_limma_voom_for_integration.csv').set_index('gene_id')
limma_cols = ['ko_effect_base', 'p_ko', 'padj_ko', 'hyp_response_wt', 'p_hyp', 'padj_hyp',
              'hyp_response_ko', 'p_hyp_ko', 'padj_hyp_ko', 'interaction', 'p_int', 'padj_int']
assert len(limma) == 3814 and limma.index.is_unique
for col in limma_cols:
    g[col] = g.gene_id.map(limma[col])
assert g.hyp_response_ko.notna().sum() == 3814

# Replicate arrays: rows are biological replicates, columns are 5-kb bins.
rep = {}
for cond, prefix, nr in [('WT', 'WT', 4), ('KO', 'KO', 4), ('HYP', 'hypoxia', 4),
                         ('KOHYP', 'KO_hypoxia', 3), ('LAT', 'latent', 4)]:
    rep[cond] = b[[f'ins_{prefix}_rep{i}' for i in range(1, nr + 1)]].to_numpy(float).T
    rep[cond] = np.where(np.isfinite(rep[cond]), rep[cond], np.nan)

# WT-latent robust contrast, analogous to the established baseline definition.
lat_pt, lat_lo, lat_hi = pair_stats(rep['WT'], rep['LAT'])
b['d_ins_latent_WT'] = lat_pt
b['d_ins_latent_WT_lo'] = lat_lo
b['d_ins_latent_WT_hi'] = lat_hi
b['latent_ins_class'] = classify(lat_pt, lat_lo, lat_hi)
b['WT_hypoxia_ins_class'] = classify(b.d_ins_hyp_WT, b.d_ins_hyp_WT_lo, b.d_ins_hyp_WT_hi)
b['KO_hypoxia_ins_class'] = classify(b.d_ins_hyp_KO, b.d_ins_hyp_KO_lo, b.d_ins_hyp_KO_hi)
b['baseline_ins_class'] = b.ins_change_class
b['interaction_ins_class'] = b.interaction_class

# Complementary Welch tests for all primary comparisons.
contrasts = {
    'baseline_KO_WT': (['KO', 'WT'], [1, -1]),
    'WT_hypoxia': (['HYP', 'WT'], [1, -1]),
    'KO_hypoxia': (['KOHYP', 'KO'], [1, -1]),
    'hypoxia_interaction': (['KOHYP', 'KO', 'HYP', 'WT'], [1, -1, -1, 1]),
    'WT_latent': (['LAT', 'WT'], [1, -1]),
}
for name, (conds, signs) in contrasts.items():
    est, se, p = welch([rep[c] for c in conds], signs)
    b[f'{name}_welch_effect'] = est
    b[f'{name}_welch_se'] = se
    b[f'{name}_welch_p'] = p
    b[f'{name}_welch_FDR'] = bh(p)

# Replicate concordance for robust pairwise-median contrasts.
for name, a, z, eff in [
    ('baseline', 'WT', 'KO', 'd_ins_KO_WT'),
    ('WT_hypoxia', 'WT', 'HYP', 'd_ins_hyp_WT'),
    ('WT_latent', 'WT', 'LAT', 'd_ins_latent_WT')]:
    ds = (rep[z][:, None, :] - rep[a][None, :, :]).reshape(-1, N)
    b[f'{name}_pair_same_direction_fraction'] = (np.sign(ds) == np.sign(b[eff].to_numpy())[None, :]).mean(axis=0)
# Unbalanced KO response and four-group interaction: per-replicate residual signs around the fitted contrast.
ko_ds = (rep['KOHYP'][:, None, :] - rep['KO'][None, :, :]).reshape(-1, N)
b['KO_hypoxia_pair_same_direction_fraction'] = (np.sign(ko_ds) == np.sign(b.d_ins_hyp_KO.to_numpy())[None, :]).mean(axis=0)

# RNA model classes and descriptive replicate-index contrasts.
g['WT_hypoxia_RNA_class'] = rna_class(g.hyp_response_wt, g.padj_hyp)
g['KO_hypoxia_RNA_class'] = rna_class(g.hyp_response_ko, g.padj_hyp_ko)
g['hypoxia_interaction_RNA_class'] = rna_class(g.interaction, g.padj_int)
g['factorial_baseline_RNA_class'] = rna_class(g.ko_effect_base, g.padj_ko)
g['baseline_RNA_class'] = g.class_KOvsWT
g['latent_RNA_class'] = g.class_latentvsWT

# Raw-count log2CPM replicate summaries for direction/consistency audits.
raw = pd.read_csv(ROOT / 'hic_review/review_data/rnaseq/AKR_RNA_seq_merged.csv')
emap = pd.read_csv(RA.parent / 'R1_mechanism/tables/ebg_to_rv_mapping.csv') if False else pd.read_csv(ROOT / 'hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/ebg_to_rv_mapping.csv')
rv2ebg = dict(zip(emap.gene_id, emap.merged_ebg_id))
count_cols = ['WT-rep1', 'WT-rep2', 'NapM-KO-rep1', 'NapM-KO-rep2', 'Hypoxia-rep1', 'Hypoxia-rep2',
              'NapM_Ko-Hypoxia-rep1', 'NapM_Ko-Hypoxia-rep2', 'Latent-rep1', 'Latent-rep2']
lib = raw[count_cols].sum(axis=0)
logcpm = np.log2(raw[count_cols] / lib * 1e6 + .5)
logcpm['gene'] = raw.gene
lc = logcpm.set_index('gene')
for _, row in g.iterrows():
    gid = row.gene_id
    key = gid if gid in lc.index else rv2ebg.get(gid)
    if key not in lc.index:
        continue
    v = lc.loc[key, count_cols].to_numpy(float)
    ds = {
        'baseline_RNA_rep_delta': [v[2] - v[0], v[3] - v[1]],
        'WT_hypoxia_RNA_rep_delta': [v[4] - v[0], v[5] - v[1]],
        'KO_hypoxia_RNA_rep_delta': [v[6] - v[2], v[7] - v[3]],
        'WT_latent_RNA_rep_delta': [v[8] - v[0], v[9] - v[1]],
    }
    idx = g.index[g.gene_id.eq(gid)][0]
    for name, vals in ds.items():
        g.loc[idx, name + '1'] = vals[0]
        g.loc[idx, name + '2'] = vals[1]

# Gene-level output tables with explicit contrast names.
common_g = ['gene_id', 'gene_name', 'start', 'end', 'strand', 'tss', 'bin_tss', 'binding_class',
            'NapM_occupied', 'gene_peak_score_max', 'gene_peak_vpm_max', 'expr_wt_gene', 'operon_ids' if 'operon_ids' in g else 'gene_length']
hyp_g_cols = common_g + ['ko_effect_base', 'padj_ko', 'factorial_baseline_RNA_class',
    'hyp_response_wt', 'padj_hyp', 'WT_hypoxia_RNA_class', 'hyp_response_ko', 'padj_hyp_ko',
    'KO_hypoxia_RNA_class', 'interaction', 'padj_int', 'hypoxia_interaction_RNA_class',
    'WT_hypoxia_RNA_rep_delta1', 'WT_hypoxia_RNA_rep_delta2',
    'KO_hypoxia_RNA_rep_delta1', 'KO_hypoxia_RNA_rep_delta2']
lat_g_cols = common_g + ['log2FC_KOvsWT', 'padj_KOvsWT', 'baseline_RNA_class',
    'log2FC_latentvsWT', 'padj_latentvsWT', 'latent_RNA_class',
    'baseline_RNA_rep_delta1', 'baseline_RNA_rep_delta2',
    'WT_latent_RNA_rep_delta1', 'WT_latent_RNA_rep_delta2']
g[hyp_g_cols].to_csv(SRC / 'Hypoxia_gene_level.csv', index=False)
g[lat_g_cols].to_csv(SRC / 'Latency_gene_level.csv', index=False)

# Add bin-level RNA summaries; statistical unit remains the unique TSS bin.
def add_rna_bin(effect, q, cls, tag):
    tmp = pd.DataFrame({'bin': g.bin_tss, 'effect': effect, 'q': q, 'class': cls})
    tmp['tested'] = np.isfinite(tmp.effect) & np.isfinite(tmp.q)
    tmp['up'] = tmp['class'].eq('detected_increase')
    tmp['down'] = tmp['class'].eq('detected_decrease')
    agg = tmp.groupby('bin').agg(**{
        f'RNA_{tag}_mean_log2FC': ('effect', 'mean'),
        f'RNA_{tag}_mean_abs_log2FC': ('effect', lambda x: x.abs().mean()),
        f'RNA_{tag}_n_tested_genes': ('tested', 'sum'),
        f'RNA_{tag}_n_DEG_up': ('up', 'sum'),
        f'RNA_{tag}_n_DEG_down': ('down', 'sum')})
    agg[f'RNA_{tag}_any_DEG'] = (agg[f'RNA_{tag}_n_DEG_up'] + agg[f'RNA_{tag}_n_DEG_down']) > 0
    return agg

rna_specs = [
    (g.hyp_response_wt, g.padj_hyp, g.WT_hypoxia_RNA_class, 'WT_hypoxia'),
    (g.hyp_response_ko, g.padj_hyp_ko, g.KO_hypoxia_RNA_class, 'KO_hypoxia'),
    (g.interaction, g.padj_int, g.hypoxia_interaction_RNA_class, 'interaction'),
    (g.log2FC_KOvsWT, g.padj_KOvsWT, g.baseline_RNA_class, 'baseline'),
    (g.log2FC_latentvsWT, g.padj_latentvsWT, g.latent_RNA_class, 'latent')]
for spec in rna_specs:
    b = b.merge(add_rna_bin(*spec), left_on='bin', right_index=True, how='left', validate='1:1')

# Directional gene-level intersection for latency (genes remain gene-level evidence).
base_up = g.baseline_RNA_class.eq('detected_increase'); base_dn = g.baseline_RNA_class.eq('detected_decrease')
lat_up = g.latent_RNA_class.eq('detected_increase'); lat_dn = g.latent_RNA_class.eq('detected_decrease')
g['latency_dual_DEG'] = (base_up | base_dn) & (lat_up | lat_dn)
g['latency_RNA_direction_relation'] = np.select([
    (base_up & lat_dn) | (base_dn & lat_up), (base_up & lat_up) | (base_dn & lat_dn)],
    ['opposite_antagonistic_association', 'same_partial_similar_state'], default='not_dual_DEG')
g['latency_RNA_replicate_concordant'] = (
    np.sign(g.baseline_RNA_rep_delta1) == np.sign(g.log2FC_KOvsWT)) & (
    np.sign(g.baseline_RNA_rep_delta2) == np.sign(g.log2FC_KOvsWT)) & (
    np.sign(g.WT_latent_RNA_rep_delta1) == np.sign(g.log2FC_latentvsWT)) & (
    np.sign(g.WT_latent_RNA_rep_delta2) == np.sign(g.log2FC_latentvsWT))
g[lat_g_cols + ['latency_dual_DEG', 'latency_RNA_direction_relation', 'latency_RNA_replicate_concordant']].to_csv(
    SRC / 'Latency_gene_level.csv', index=False)
for tag, mask in [('dual_DEG', g.latency_dual_DEG),
                  ('dual_DEG_opposite', g.latency_RNA_direction_relation.eq('opposite_antagonistic_association')),
                  ('dual_DEG_same', g.latency_RNA_direction_relation.eq('same_partial_similar_state')),
                  ('NapM_dual_DEG', g.latency_dual_DEG & g.NapM_occupied)]:
    z = g[mask].groupby('bin_tss').gene_id.apply(lambda x: ';'.join(x)).rename(f'latency_{tag}_genes')
    b = b.merge(z, left_on='bin', right_index=True, how='left')
    b[f'latency_{tag}_genes'] = b[f'latency_{tag}_genes'].fillna('')

# Response DEG counts and direction agreement.
count_rows = []
for label, cls, eff, q in [
    ('WT hypoxia response', g.WT_hypoxia_RNA_class, g.hyp_response_wt, g.padj_hyp),
    ('KO hypoxia response', g.KO_hypoxia_RNA_class, g.hyp_response_ko, g.padj_hyp_ko),
    ('Hypoxia genotype interaction', g.hypoxia_interaction_RNA_class, g.interaction, g.padj_int),
    ('Baseline KO vs WT', g.baseline_RNA_class, g.log2FC_KOvsWT, g.padj_KOvsWT),
    ('WT latent response', g.latent_RNA_class, g.log2FC_latentvsWT, g.padj_latentvsWT)]:
    count_rows.append(dict(contrast=label, n_tested=int(np.isfinite(eff).sum()),
        n_DEG=int(pd.Series(cls).isin(DET).sum()), n_up=int(pd.Series(cls).eq('detected_increase').sum()),
        n_down=int(pd.Series(cls).eq('detected_decrease').sum()),
        n_FDR_0_05=int(pd.Series(q).lt(.05).sum()), median_log2FC=float(pd.Series(eff).median()),
        median_abs_log2FC=float(pd.Series(eff).abs().median())))
pd.DataFrame(count_rows).to_csv(TAB / 'RNA_response_counts.csv', index=False)

# Hypoxia: NapM-bound versus baseline-matched unbound bins under circular shifts.
def matched_test(effect, tag, use_mask=None):
    pp = pairs if use_mask is None else pairs[use_mask].copy()
    x = np.asarray(effect, float)
    bi, mi = pp.bound_bin.to_numpy(int), pp.matched_bin.to_numpy(int)
    good = np.isfinite(x[bi]) & np.isfinite(x[mi])
    pp = pp[good]; bi, mi = bi[good], mi[good]
    delta = x[bi] - x[mi]
    obs = float(np.median(delta))
    boot = np.median(delta[RNG.integers(0, len(delta), (N_BOOT, len(delta)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    null = np.empty(N_PERM)
    bound_mask = np.zeros(N, bool); matched_mask = np.zeros(N, bool)
    bound_mask[bi] = True; matched_mask[mi] = True
    for k in range(N_PERM):
        s = int(RNG.integers(1, N))
        null[k] = np.nanmedian(x[np.roll(bound_mask, s)]) - np.nanmedian(x[np.roll(matched_mask, s)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return dict(comparison=tag, n_pairs=len(pp), observed_median_paired_difference=obs,
                bootstrap95_lo=lo, bootstrap95_hi=hi, circular_shift_p=p,
                null_mean=float(np.mean(null)), null_sd=float(np.std(null, ddof=1))), null

match_rows, null_rows = [], []
for col, tag in [('d_ins_hyp_WT', 'WT hypoxia response'), ('d_ins_hyp_KO', 'KO hypoxia response'),
                 ('d_ins_interaction', 'Hypoxia genotype interaction')]:
    for transform, suffix in [(lambda x: x, 'signed'), (np.abs, 'absolute')]:
        res, null = matched_test(transform(b[col].to_numpy(float)), f'{tag} ({suffix})')
        match_rows.append(res)
        null_rows.append(pd.DataFrame({'comparison': res['comparison'], 'iteration': np.arange(N_PERM), 'null_statistic': null}))
        sens, _ = matched_test(transform(b[col].to_numpy(float)), f'{tag} ({suffix}; caliper sensitivity)',
                               ~pairs.caliper_exceeded)
        match_rows.append(sens)
pd.DataFrame(match_rows).to_csv(TAB / 'Hypoxia_NapM_matched_tests.csv', index=False)
pd.concat(null_rows, ignore_index=True).to_csv(SRC / 'Hypoxia_NapM_circular_shift_nulls.csv.gz', index=False, compression='gzip')

# Structure-transcription correlations at unique-bin level, with spatially structured nulls.
cor_rows = []
for xcol, ycol, label in [
    ('RNA_WT_hypoxia_mean_log2FC', 'd_ins_hyp_WT', 'WT hypoxia'),
    ('RNA_KO_hypoxia_mean_log2FC', 'd_ins_hyp_KO', 'KO hypoxia'),
    ('RNA_interaction_mean_log2FC', 'd_ins_interaction', 'Hypoxia interaction'),
    ('RNA_baseline_mean_log2FC', 'd_ins_KO_WT', 'Baseline KO vs WT'),
    ('RNA_latent_mean_log2FC', 'd_ins_latent_WT', 'WT latent')]:
    x, y = b[xcol].to_numpy(float), b[ycol].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    rho, p, _ = circ_spearman(x, y)
    idx = block_boot_indices(N)
    boot_values = []
    for ii in idx:
        good = np.isfinite(x[ii]) & np.isfinite(y[ii])
        boot_values.append(stats.spearmanr(x[ii][good], y[ii][good]).statistic)
    boots = np.array(boot_values)
    lo, hi = ci_boot(boots)
    cor_rows.append(dict(contrast=label, n_unique_TSS_bins=int(ok.sum()), spearman_rho=rho,
                         block_bootstrap95_lo=lo, block_bootstrap95_hi=hi, circular_shift_p=p,
                         block_size_bins=BLOCK, n_permutations=N_PERM))
    # Absolute-effect coupling addresses magnitude independently of direction.
    rhoa, pa, _ = circ_spearman(np.abs(x), np.abs(y))
    cor_rows.append(dict(contrast=label + ' (absolute effects)', n_unique_TSS_bins=int(ok.sum()), spearman_rho=rhoa,
                         block_bootstrap95_lo=np.nan, block_bootstrap95_hi=np.nan, circular_shift_p=pa,
                         block_size_bins=BLOCK, n_permutations=N_PERM))
pd.DataFrame(cor_rows).to_csv(TAB / 'Structure_transcription_correlations.csv', index=False)

# Parsimonious exploratory regressions: four or five predictors, no mediation claim.
def ols_fit(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    pred = X1 @ beta
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / sst
    return beta, r2


def regression(outcome, rna, baseline_ins, tag, include_baseline_delta=False):
    cols = [outcome, rna, baseline_ins, 'expr_wt_mean', 'NapM_bound'] + (['d_ins_KO_WT'] if include_baseline_delta else [])
    z = b[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    # expr_wt_mean is missing only in bins without a mapped expressed gene; complete cases are explicit.
    y = z[outcome].to_numpy(float)
    cont = [rna, baseline_ins, 'expr_wt_mean'] + (['d_ins_KO_WT'] if include_baseline_delta else [])
    parts = []
    for c in cont:
        v = z[c].to_numpy(float)
        parts.append((v - v.mean()) / v.std(ddof=1))
    parts.append(z.NapM_bound.to_numpy(float))
    X = np.column_stack(parts)
    rna_j = 0
    keep = np.ones(X.shape[1], bool); keep[rna_j] = False
    beta, r2 = ols_fit(X, y)
    _, r2_null = ols_fit(X[:, keep], y)
    delta_r2 = r2 - r2_null
    # Map complete-case rows back to genomic bins for circular moving-block resampling.
    original_idx = z.index.to_numpy(int)
    boots_beta, boots_dr2 = [], []
    for _ in range(N_BOOT):
        starts = RNG.integers(0, N, int(np.ceil(len(z) / BLOCK)))
        block_bins = np.concatenate([(s + np.arange(BLOCK)) % N for s in starts])
        lookup = pd.Series(np.arange(len(z)), index=original_idx)
        ii = lookup.reindex(block_bins).dropna().to_numpy(int)
        if len(ii) < X.shape[1] + 5:
            continue
        bb, rr = ols_fit(X[ii], y[ii])
        _, rn = ols_fit(X[ii][:, keep], y[ii])
        boots_beta.append(bb[1])
        boots_dr2.append(rr - rn)
    blo, bhi = ci_boot(boots_beta); dlo, dhi = ci_boot(boots_dr2)
    return dict(model=tag, n_bins=len(z), n_predictors=X.shape[1], RNA_beta_per_1SD=beta[1],
                RNA_beta_block95_lo=blo, RNA_beta_block95_hi=bhi, full_R2=r2,
                covariate_only_R2=r2_null, incremental_R2_RNA=delta_r2,
                incremental_R2_block95_lo=dlo, incremental_R2_block95_hi=dhi,
                covariates='standardized RNA response, baseline insulation, baseline expression, NapM-bound' +
                           (', baseline KO-WT insulation' if include_baseline_delta else ''),
                interpretation='Exploratory association only; no transcriptional mediation or causal inference.')

reg_rows = [
    regression('d_ins_hyp_WT', 'RNA_WT_hypoxia_mean_log2FC', 'ins_WT', 'WT hypoxia'),
    regression('d_ins_hyp_KO', 'RNA_KO_hypoxia_mean_log2FC', 'ins_KO', 'KO hypoxia'),
    regression('d_ins_interaction', 'RNA_interaction_mean_log2FC', 'ins_WT', 'Hypoxia interaction'),
    regression('d_ins_latent_WT', 'RNA_latent_mean_log2FC', 'ins_WT', 'WT latent')]
pd.DataFrame(reg_rows).to_csv(TAB / 'Structure_transcription_regressions.csv', index=False)

# Operational hypoxia response classes at the bin level. These are evidence categories, not mechanisms.
wt_sd = b.WT_hypoxia_ins_class.isin(DET); ko_sd = b.KO_hypoxia_ins_class.isin(DET); int_sd = b.interaction_ins_class.isin(DET)
wt_rd = b.RNA_WT_hypoxia_any_DEG.fillna(False); ko_rd = b.RNA_KO_hypoxia_any_DEG.fillna(False); int_rd = b.RNA_interaction_any_DEG.fillna(False)
wt_same = (np.sign(b.d_ins_hyp_WT) == np.sign(b.d_ins_hyp_KO)) & (np.sign(b.RNA_WT_hypoxia_mean_log2FC) == np.sign(b.RNA_KO_hypoxia_mean_log2FC))
local_rna_tested = b[['RNA_WT_hypoxia_n_tested_genes', 'RNA_KO_hypoxia_n_tested_genes',
                      'RNA_interaction_n_tested_genes']].min(axis=1).fillna(0).gt(0)
local_rna_small = b[['RNA_WT_hypoxia_mean_abs_log2FC', 'RNA_KO_hypoxia_mean_abs_log2FC',
                     'RNA_interaction_mean_abs_log2FC']].max(axis=1).lt(.5)
no_local_rna_response = local_rna_tested & local_rna_small & ~wt_rd & ~ko_rd & ~int_rd
hyp_class = np.select([
    (wt_sd | ko_sd) & wt_rd & ko_rd & wt_same,
    b.NapM_bound & int_sd & no_local_rna_response,
    b.NapM_bound & int_sd & (wt_rd | ko_rd | int_rd),
    int_sd | wt_sd | ko_sd],
    ['Transcription-associated_consistent_pattern',
     'NapM-associated_candidate_no_detected_local_RNA_response',
     'Mixed_or_unresolved_NapM_structure_RNA',
     'Mixed_or_unresolved'], default='No_detected_structural_response')
b['hypoxia_response_evidence_class'] = hyp_class

# Latency candidate bins: pre-specified evidence completeness, not outcome cherry-picking.
base_sd = b.baseline_ins_class.isin(DET); lat_sd = b.latent_ins_class.isin(DET)
b['latency_dual_structure'] = base_sd & lat_sd
b['latency_structure_direction_relation'] = np.select([
    b.latency_dual_structure & (np.sign(b.d_ins_KO_WT) != np.sign(b.d_ins_latent_WT)),
    b.latency_dual_structure & (np.sign(b.d_ins_KO_WT) == np.sign(b.d_ins_latent_WT))],
    ['opposite_antagonistic_association', 'same_partial_similar_state'], default='not_dual_structure')
b['latency_RNA_dual_in_bin'] = b.latency_dual_DEG_genes.ne('')
b['latency_NapM_dual_DEG_in_bin'] = b.latency_NapM_dual_DEG_genes.ne('')
b['latency_baseline_abnormal'] = base_sd | b.RNA_baseline_any_DEG.fillna(False)
b['latency_WT_response'] = lat_sd | b.RNA_latent_any_DEG.fillna(False)
b['latency_candidate_eligible'] = b.NapM_bound & b.latency_baseline_abnormal & b.latency_WT_response & (
    b.latency_dual_structure | b.latency_RNA_dual_in_bin)
b['latency_evidence_score'] = (3 * b.latency_dual_structure.astype(int) + 3 * b.latency_RNA_dual_in_bin.astype(int)
    + 2 * b.latency_NapM_dual_DEG_in_bin.astype(int) + b.NapM_bound.astype(int)
    + np.minimum(2, np.log1p(b.chip_bin_npeaks.fillna(0)))
    + np.minimum(2, b.d_ins_latent_WT.abs().fillna(0))
    + np.minimum(2, b.RNA_latent_mean_abs_log2FC.fillna(0) / 2))

# Connected candidate regions prevent adjacent bins from becoming independent candidates.
cand_bins = set(b.loc[b.latency_candidate_eligible, 'bin'].astype(int))
regions, seen = [], set()
for start in sorted(cand_bins):
    if start in seen:
        continue
    stack, comp = [start], []; seen.add(start)
    while stack:
        x = stack.pop(); comp.append(x)
        for y in [(x - 1) % N, (x + 1) % N]:
            if y in cand_bins and y not in seen:
                seen.add(y); stack.append(y)
    regions.append(sorted(comp))
bin_region = {x: f'LATR{i+1:02d}' for i, comp in enumerate(regions) for x in comp}
b['latency_candidate_region'] = b['bin'].map(bin_region).fillna('')
reg_rows = []
for rid, z in b[b.latency_candidate_region.ne('')].groupby('latency_candidate_region'):
    genes = sorted(set(';'.join(z.latency_dual_DEG_genes).split(';')) - {''})
    napm_genes = sorted(set(';'.join(z.latency_NapM_dual_DEG_genes).split(';')) - {''})
    reg_rows.append(dict(region_id=rid, bins=';'.join(map(str, z['bin'])), n_bins=len(z),
        start=int(z.start.min()), end=int(z.end.max()), genes=';'.join(genes), NapM_dual_DEG_genes=';'.join(napm_genes),
        n_dual_structure_bins=int(z.latency_dual_structure.sum()), n_dual_RNA_bins=int(z.latency_RNA_dual_in_bin.sum()),
        n_opposite_structure_bins=int(z.latency_structure_direction_relation.eq('opposite_antagonistic_association').sum()),
        n_same_structure_bins=int(z.latency_structure_direction_relation.eq('same_partial_similar_state').sum()),
        max_peak_score=float(z.chip_bin_score_max.max()), evidence_score=float(z.latency_evidence_score.max()),
        max_abs_baseline_ins=float(z.d_ins_KO_WT.abs().max()), max_abs_latent_ins=float(z.d_ins_latent_WT.abs().max()),
        evidence_note='Candidate association only; no KO-latent contrast exists. Same-direction patterns are not latency defects.'))
pd.DataFrame(reg_rows).sort_values('evidence_score', ascending=False).to_csv(TAB / 'Latency_candidate_regions.csv', index=False)

# Complete source tables and long-form replicate insulation data.
hyp_b_cols = ['bin', 'start', 'end', 'NapM_bound', 'chip_bin_npeaks', 'chip_bin_score_max', 'chip_bin_vpm_max',
    'ins_WT', 'ins_KO', 'd_ins_KO_WT', 'd_ins_KO_WT_lo', 'd_ins_KO_WT_hi', 'baseline_ins_class',
    'd_ins_hyp_WT', 'd_ins_hyp_WT_lo', 'd_ins_hyp_WT_hi', 'WT_hypoxia_ins_class',
    'd_ins_hyp_KO', 'd_ins_hyp_KO_lo', 'd_ins_hyp_KO_hi', 'KO_hypoxia_ins_class',
    'd_ins_interaction', 'd_ins_interaction_lo', 'd_ins_interaction_hi', 'interaction_ins_class',
    'interaction_p', 'interaction_padj', 'hypoxia_response_evidence_class', 'expr_wt_mean']
hyp_b_cols += [c for c in b if c.startswith('RNA_') and any(x in c for x in ['WT_hypoxia', 'KO_hypoxia', 'interaction'])]
hyp_b_cols += [c for c in b if c.endswith('_welch_FDR')]
b[hyp_b_cols].to_csv(SRC / 'Hypoxia_bin_level.csv', index=False)
lat_b_cols = ['bin', 'start', 'end', 'NapM_bound', 'chip_bin_npeaks', 'chip_bin_score_max', 'chip_bin_vpm_max',
    'ins_WT', 'ins_KO', 'ins_latent', 'd_ins_KO_WT', 'd_ins_KO_WT_lo', 'd_ins_KO_WT_hi', 'baseline_ins_class',
    'd_ins_latent_WT', 'd_ins_latent_WT_lo', 'd_ins_latent_WT_hi', 'latent_ins_class',
    'latency_dual_structure', 'latency_structure_direction_relation', 'latency_dual_DEG_genes',
    'latency_dual_DEG_opposite_genes', 'latency_dual_DEG_same_genes', 'latency_NapM_dual_DEG_genes',
    'latency_candidate_eligible', 'latency_candidate_region', 'latency_evidence_score', 'expr_wt_mean']
lat_b_cols += [c for c in b if c.startswith('RNA_') and any(x in c for x in ['baseline', 'latent'])]
lat_b_cols += [c for c in b if c in ['baseline_KO_WT_welch_FDR', 'WT_latent_welch_FDR',
                                      'baseline_pair_same_direction_fraction', 'WT_latent_pair_same_direction_fraction']]
b[lat_b_cols].to_csv(SRC / 'Latency_bin_level.csv', index=False)
long_rows = []
for cond, arr in rep.items():
    for r in range(arr.shape[0]):
        for i in range(N):
            long_rows.append(dict(condition=cond, replicate=r + 1, bin=i, start=int(b.loc[i, 'start']),
                                  end=int(b.loc[i, 'end']), insulation=arr[r, i]))
pd.DataFrame(long_rows).to_csv(SRC / 'HiC_insulation_per_replicate.csv.gz', index=False, compression='gzip')

# Design/QC summary.
summary = {
    'genome': {'name': 'H37Rv ASM19595v2', 'length_bp': GENOME, 'bins': N, 'bin_size_bp': BINSIZE},
    'HiC_replicates': {'WT': 4, 'KO': 4, 'WT_hypoxia': 4, 'KO_hypoxia': 3, 'WT_latent': 4, 'KO_latent': 0},
    'RNA_replicates': {'WT': 2, 'KO': 2, 'WT_hypoxia': 2, 'KO_hypoxia': 2, 'WT_latent': 2, 'KO_latent': 0},
    'ChIP': {'n_peaks': 634, 'condition': 'baseline WT only', 'condition_specific_occupancy_tested': False},
    'HiC_class_rule': 'Robust all-replicate-pair median and 5th-95th percentile interval; detected if interval excludes +/-0.25 log2 insulation.',
    'RNA_hypoxia_model': 'Recalculated 8-sample factorial model: edgeR TMM + limma-voom + eBayes (limma 3.62.2); residual df=4; BH across 3814 validated genes; DEG=FDR<0.05 and |log2FC|>=1.',
    'RNA_baseline_latent': 'Existing two-group limma result tables; BH across genes; DEG=FDR<0.05 and |log2FC|>=1.',
    'spatial_tests': {'matched_bin_pairs': len(pairs), 'circular_shift_iterations': N_PERM,
                      'moving_block_bootstrap_iterations': N_BOOT, 'block_bins': BLOCK, 'seed': 20261030},
    'counts': {'latency_candidate_bins': int(b.latency_candidate_eligible.sum()),
               'latency_candidate_regions': len(reg_rows),
               'latency_dual_DEG_genes': int(g.latency_dual_DEG.sum()),
               'latency_NapM_dual_DEG_genes': int((g.latency_dual_DEG & g.NapM_occupied).sum())},
    'limitations': [
        'Baseline ChIP-seq cannot establish condition-specific NapM occupancy.',
        'Structure-RNA coupling and regression are associations, not mediation or causality.',
        'Adjacent bins and genes sharing a TSS bin are not independent structural evidence.',
        'RNA factorial model has only two replicates per group; empirical-Bayes moderation does not eliminate limited-replicate uncertainty.',
        'No KO-latent data exist; candidate patterns cannot establish altered latency establishment or phenotype.',
        'Same-direction KO baseline and WT-latent changes denote partial molecular similarity, not a latency defect.'
    ]
}
with open(TAB / 'analysis_metadata.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary['counts'], indent=2))
print(pd.DataFrame(count_rows).to_string(index=False))
print('\nHypoxia evidence classes:', b.hypoxia_response_evidence_class.value_counts().to_dict())
print('Outputs:', OUT)
