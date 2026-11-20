#!/usr/bin/env python3
"""Prepare statistics and in-situ source data for the full mechanism figure."""
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC, TAB = OUT / 'source_data', OUT / 'tables'
N, BIN, GENOME = 883, 5000, 4411532
DET = {'detected_increase', 'detected_decrease'}

hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
hyp_b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
orig_b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
lat_regions = pd.read_csv(TAB / 'Latency_candidate_regions.csv')
peaks = pd.read_csv(RA / '01_chip_curated/NapM_binding_curated.tsv', sep='\t')
bg = pd.read_csv(ROOT / 'hic_review/review_data/chip_seq_data/Rv0047c.bedgraph', sep=r'\s+',
                 header=None, names=['chrom', 'start', 'end', 'signal'])
logcpm = pd.read_csv(SRC / 'Hypoxia_limma_voom_normalized_logCPM.csv')
op = pd.read_csv(ROOT / 'hic_review/review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv')
opmap = {}
for row in op.itertuples():
    for gene in ast.literal_eval(row.gene_list):
        opmap.setdefault(gene, set()).add(str(row.operon_id))

# A/B: direct interaction DEGs and descriptive k-means response-shape clusters.
interaction = hyp_g[hyp_g.hypoxia_interaction_RNA_class.isin(DET)].copy()
assert len(interaction) == 730
km = KMeans(n_clusters=4, n_init=100, random_state=20261030)
interaction['cluster_raw'] = km.fit_predict(interaction[['hyp_response_wt', 'hyp_response_ko']])
centers = interaction.groupby('cluster_raw').agg(
    WT_center=('hyp_response_wt', 'mean'), KO_center=('hyp_response_ko', 'mean'),
    interaction_center=('interaction', 'mean'), n_genes=('gene_id', 'size')).reset_index()
centers = centers.sort_values('interaction_center', ascending=False).reset_index(drop=True)
centers['cluster'] = [f'C{i+1}' for i in range(len(centers))]
cluster_map = dict(zip(centers.cluster_raw, centers.cluster))
interaction['cluster'] = interaction.cluster_raw.map(cluster_map)
interaction['abs_interaction'] = interaction.interaction.abs()
interaction['interaction_rank_FDR'] = interaction.padj_int.rank(method='min').astype(int)
interaction['interaction_rank_abs_effect'] = interaction.abs_interaction.rank(method='min', ascending=False).astype(int)
interaction = interaction.sort_values(['cluster', 'interaction'], ascending=[True, False])
interaction.to_csv(SRC / 'Mechanism_RNA_interaction_clusters.csv', index=False)
centers[['cluster', 'n_genes', 'WT_center', 'KO_center', 'interaction_center']].to_csv(
    TAB / 'Mechanism_RNA_cluster_summary.csv', index=False)

# C: all-bin evidence classification and the complete stringent structural pool.
bin_cols = ['bin', 'start', 'end', 'NapM_bound', 'chip_bin_score_max', 'd_ins_hyp_WT',
            'd_ins_hyp_KO', 'd_ins_interaction', 'interaction_padj', 'interaction_ins_class',
            'RNA_WT_hypoxia_mean_abs_log2FC', 'RNA_KO_hypoxia_mean_abs_log2FC',
            'RNA_interaction_mean_log2FC', 'RNA_interaction_mean_abs_log2FC',
            'RNA_WT_hypoxia_any_DEG', 'RNA_KO_hypoxia_any_DEG', 'RNA_interaction_any_DEG',
            'hypoxia_response_evidence_class']
hyp_b[bin_cols].to_csv(SRC / 'Mechanism_bin_evidence_classes.csv', index=False)
napm_class = 'NapM-associated_candidate_no_detected_local_RNA_response'
candidates = hyp_b[hyp_b.hypoxia_response_evidence_class.eq(napm_class)].copy()
assert len(candidates) == 17
candidates['abs_interaction'] = candidates.d_ins_interaction.abs()
candidates['rank_abs_interaction'] = candidates.abs_interaction.rank(method='min', ascending=False).astype(int)
candidates['rank_FDR'] = candidates.interaction_padj.rank(method='min').astype(int)
candidates = candidates.sort_values('abs_interaction', ascending=False)
candidates['genes'] = candidates['bin'].map(
    hyp_g.groupby('bin_tss').gene_name.apply(lambda x: ';'.join(sorted(set(str(v) for v in x if isinstance(v, str) and v)))))
candidates[['bin', 'genes', 'chip_bin_score_max', 'd_ins_hyp_WT', 'd_ins_hyp_KO',
            'd_ins_interaction', 'abs_interaction', 'interaction_padj', 'rank_abs_interaction',
            'rank_FDR']].to_csv(TAB / 'Mechanism_structural_candidate_ranking.csv', index=False)

# Selection is explicit and is evaluated against the complete candidate sets.
assert int(candidates.iloc[0]['bin']) == 874, 'ESX-2 should have the largest absolute interaction in the stringent pool'
assert 'LATR01' in set(lat_regions.region_id)
assert int(hyp_b.loc[hyp_b['bin'].eq(0), 'interaction_padj'].iloc[0] > 0)
selected_genes = hyp_g.set_index('gene_name').loc[['devR', 'devS', 'sigE']]
selection = pd.DataFrame([
    {'locus': 'DosR/DosS', 'center_bin': 699, 'selection_class': 'RNA interaction / stress regulator',
     'reason': 'devR and devS have attenuated hypoxia induction in KO; both are shown with all 730 interaction DEGs and sample-level normalized expression.'},
    {'locus': 'SigE', 'center_bin': 272, 'selection_class': 'RNA interaction / transcription-structure dissociation',
     'reason': 'Large negative RNA interaction but no detected local insulation interaction; displayed as a dissociation example, not a DosR/DosS downstream claim.'},
    {'locus': 'oriC/dnaA', 'center_bin': 0, 'selection_class': napm_class,
     'reason': 'Significant positive insulation interaction without detected local RNA response; also overlaps the independently defined LATR01 latency-candidate region.'},
    {'locus': 'ESX-2', 'center_bin': 874, 'selection_class': napm_class,
     'reason': 'Largest absolute insulation interaction among all 17 stringent NapM-associated candidates with no detected local RNA response.'},
])
selection.to_csv(TAB / 'Mechanism_locus_selection.csv', index=False)

loci = [
    dict(locus='DosR_DosS', title='DosR/DosS (devR/devS)', center=699, half=6, focal=[699]),
    dict(locus='SigE', title='SigE', center=272, half=6, focal=[272]),
    dict(locus='oriC', title='oriC/dnaA', center=0, half=6, focal=[0]),
    dict(locus='ESX2', title='ESX-2', center=874, half=6, focal=[874]),
]
for loc in loci:
    center = loc['center']
    loc['bins'] = np.array([(center + i) % N for i in range(-loc['half'], loc['half'] + 1)], int)
    center_bp = center * BIN + BIN / 2
    raw_pos = loc['bins'] * BIN + BIN / 2
    loc['pos_kb'] = ((raw_pos - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000

# Matrix readers follow the validated 1-based sparse, symmetrized format.
def load_matrix(path):
    d = pd.read_csv(path, sep=r'\s+', header=None, names=['b1', 'b2', 'v'])
    m = np.zeros((N, N))
    i, j = d.b1.to_numpy(int) - 1, d.b2.to_numpy(int) - 1
    m[i, j] = d.v.to_numpy(float)
    m[j, i] = d.v.to_numpy(float)
    return m


def circular_expected(m):
    idx = np.arange(N)
    expected = np.full(N // 2 + 1, np.nan)
    for off in range(N // 2 + 1):
        vals = m[idx, (idx + off) % N]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        expected[off] = vals.mean() if len(vals) else np.nan
    return expected


def local_oe(m, expected, bins):
    i, j = np.meshgrid(bins, bins, indexing='ij')
    delta = np.abs(j - i)
    dist = np.minimum(delta, N - delta)
    vals = m[np.ix_(bins, bins)]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where((vals > 0) & np.isfinite(vals) & (expected[dist] > 0), vals / expected[dist], np.nan)


cond_files = {'WT': ('wt', 4), 'KO': ('napm_ko', 4),
              'WT_hypoxia': ('hypoxia', 4), 'KO_hypoxia': ('ko-hypoxia', 3)}
map_store = {c: {r: {} for r in range(nr)} for c, (p, nr) in cond_files.items()}
for cond, (prefix, nr) in cond_files.items():
    for r in range(nr):
        path = ROOT / f'hic_review/review_data/rep_icedmatrix/{prefix}_{r+1}_5000_iced.matrix'
        m = load_matrix(path)
        expected = circular_expected(m)
        for loc in loci:
            map_store[cond][r][loc['locus']] = local_oe(m, expected, loc['bins'])

mean_maps, response_maps, rep_response_maps = [], [], []
for loc in loci:
    means = {}
    for cond, (_, nr) in cond_files.items():
        mean_map = np.nanmean(np.stack([map_store[cond][r][loc['locus']] for r in range(nr)]), axis=0)
        means[cond] = mean_map
        for a, b1 in enumerate(loc['bins']):
            for c, b2 in enumerate(loc['bins']):
                mean_maps.append(dict(locus=loc['locus'], condition=cond, bin1=int(b1), bin2=int(b2),
                    position1_kb=loc['pos_kb'][a], position2_kb=loc['pos_kb'][c], mean_OE=mean_map[a, c]))
    for genotype, hyp, norm in [('WT', 'WT_hypoxia', 'WT'), ('KO', 'KO_hypoxia', 'KO')]:
        with np.errstate(invalid='ignore', divide='ignore'):
            response = np.log2(means[hyp] / means[norm])
        for a, b1 in enumerate(loc['bins']):
            for c, b2 in enumerate(loc['bins']):
                response_maps.append(dict(locus=loc['locus'], genotype=genotype, bin1=int(b1), bin2=int(b2),
                    position1_kb=loc['pos_kb'][a], position2_kb=loc['pos_kb'][c], log2_hyp_norm_OE=response[a, c]))
        for r in range(cond_files[hyp][1]):
            # Descriptive same-index replicate differences; primary inference remains in bin-level tables.
            with np.errstate(invalid='ignore', divide='ignore'):
                rr = np.log2(map_store[hyp][r][loc['locus']] / map_store[norm][min(r, cond_files[norm][1]-1)][loc['locus']])
            for a, b1 in enumerate(loc['bins']):
                for c, b2 in enumerate(loc['bins']):
                    rep_response_maps.append(dict(locus=loc['locus'], genotype=genotype, replicate=r+1,
                        bin1=int(b1), bin2=int(b2), log2_hyp_norm_OE=rr[a, c]))

pd.DataFrame(mean_maps).to_csv(SRC / 'Mechanism_locus_condition_maps.csv', index=False)
pd.DataFrame(response_maps).to_csv(SRC / 'Mechanism_locus_response_maps.csv', index=False)
pd.DataFrame(rep_response_maps).to_csv(SRC / 'Mechanism_locus_response_maps_per_replicate.csv.gz',
                                      index=False, compression='gzip')

# Aligned insulation, RNA, ChIP and gene-model tracks.
def relative_kb(x, center):
    x = np.asarray(x, float)
    center_bp = center * BIN + BIN / 2
    return ((x - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000


ins_rows, rna_rows, gene_rows, peak_rows, bg_rows = [], [], [], [], []
orig_idx = orig_b.set_index('bin')
for loc in loci:
    for i, pos in zip(loc['bins'], loc['pos_kb']):
        row = hyp_b.loc[hyp_b['bin'].eq(i)].iloc[0]
        original = orig_idx.loc[i]
        ins_rows.append(dict(locus=loc['locus'], bin=int(i), position_kb=pos,
            is_focal=bool(i in loc['focal']), NapM_bound=bool(row.NapM_bound),
            WT=original.ins_WT, KO=original.ins_KO, WT_hypoxia=original.ins_hypoxia,
            KO_hypoxia=original.ins_KO_hypoxia, WT_response=row.d_ins_hyp_WT,
            KO_response=row.d_ins_hyp_KO, interaction=row.d_ins_interaction,
            interaction_FDR=row.interaction_padj, evidence_class=row.hypoxia_response_evidence_class))
    zgenes = hyp_g[hyp_g.bin_tss.isin(loc['bins'])].copy()
    for row in zgenes.itertuples():
        gene_rows.append(dict(locus=loc['locus'], gene_id=row.gene_id, gene_name=row.gene_name,
            start=row.start, end=row.end, strand=row.strand, tss=row.tss,
            start_kb=float(relative_kb(row.start, loc['center'])),
            end_kb=float(relative_kb(row.end, loc['center'])),
            tss_kb=float(relative_kb(row.tss, loc['center'])), bin_tss=row.bin_tss,
            is_focal_bin=bool(row.bin_tss in loc['focal']), binding_class=row.binding_class,
            operon_ids=';'.join(sorted(opmap.get(row.gene_id, set())))))
        rna_rows.append(dict(locus=loc['locus'], gene_id=row.gene_id, gene_name=row.gene_name,
            tss=row.tss, tss_kb=float(relative_kb(row.tss, loc['center'])), bin_tss=row.bin_tss,
            binding_class=row.binding_class, is_focal_bin=bool(row.bin_tss in loc['focal']),
            WT_hypoxia_log2FC=row.hyp_response_wt, WT_hypoxia_FDR=row.padj_hyp,
            KO_hypoxia_log2FC=row.hyp_response_ko, KO_hypoxia_FDR=row.padj_hyp_ko,
            interaction_log2FC=row.interaction, interaction_FDR=row.padj_int))
    span = loc['half'] * BIN + BIN
    center_bp = loc['center'] * BIN + BIN / 2
    for row in peaks.itertuples():
        d = (row.center - center_bp + GENOME / 2) % GENOME - GENOME / 2
        if abs(d) <= span:
            peak_rows.append(dict(locus=loc['locus'], peak_start=row.start, peak_end=row.end,
                peak_center=row.center, start_kb=float(relative_kb(row.start, loc['center'])),
                end_kb=float(relative_kb(row.end, loc['center'])), center_kb=float(d/1000),
                score=row.score, vpm=row.vpm))
    for row in bg.itertuples():
        d = ((row.start + row.end) / 2 - center_bp + GENOME / 2) % GENOME - GENOME / 2
        if abs(d) <= span:
            bg_rows.append(dict(locus=loc['locus'], start=row.start, end=row.end,
                start_kb=float(relative_kb(row.start, loc['center'])),
                end_kb=float(relative_kb(row.end, loc['center'])), signal=row.signal,
                usage='Display signal only; curated peaks define occupancy.'))

pd.DataFrame(ins_rows).to_csv(SRC / 'Mechanism_locus_insulation_tracks.csv', index=False)
pd.DataFrame(rna_rows).to_csv(SRC / 'Mechanism_locus_RNA_tracks.csv', index=False)
pd.DataFrame(gene_rows).to_csv(SRC / 'Mechanism_locus_gene_models.csv', index=False)
pd.DataFrame(peak_rows).to_csv(SRC / 'Mechanism_locus_NapM_peaks.csv', index=False)
pd.DataFrame(bg_rows).to_csv(SRC / 'Mechanism_locus_NapM_signal.csv', index=False)

# Sample-level expression for the three highlighted transcriptional regulators.
expr = logcpm[logcpm.gene_id.isin(['Rv3133c', 'Rv3132c', 'Rv1221'])].copy()
expr.to_csv(SRC / 'Mechanism_highlighted_gene_normalized_logCPM.csv', index=False)
assert len(expr) == 3

metadata = {
    'RNA_interaction': '(KO-hypoxia - KO) - (WT-hypoxia - WT); TMM + limma-voom + eBayes; FDR < 0.05 and |log2FC| >= 1.',
    'RNA_clustering': 'Descriptive k-means (k=4) on unstandardized WT and KO hypoxia log2FC for the 730 interaction DEGs; clusters are response shapes, not pathways.',
    'structural_classification': 'Existing operational bin-level evidence classes: detected insulation interaction, NapM occupancy, local RNA testing and effect thresholds; not unsupervised mechanism discovery.',
    'structural_candidate_pool': '17 NapM-bound bins with detected structural interaction, tested local genes, no WT/KO/interaction local DEG, and maximum mean absolute RNA effect < 0.5 across the three contrasts.',
    'ESX2_selection': 'Rank 1 by absolute insulation interaction among all 17 stringent candidates.',
    'oriC_selection': 'Significant positive structural interaction in the same stringent pool and overlap with independently defined LATR01.',
    'map_display': 'Response maps are log2(replicate-mean hypoxia O/E / replicate-mean normoxia O/E). Any display interpolation is visual only at 5-kb resolution.',
    'causal_boundary': 'Neither RNA clusters nor bin evidence classes establish direct effects, indirect effects, mediation or independent pathways.'
}
with open(TAB / 'Mechanism_full_figure_metadata.json', 'w') as fh:
    json.dump(metadata, fh, indent=2)

print('\nRNA cluster summary:')
print(centers[['cluster', 'n_genes', 'WT_center', 'KO_center', 'interaction_center']].to_string(index=False))
print('\nHighlighted genes:')
print(interaction[interaction.gene_name.isin(['devR', 'devS', 'sigE'])][
    ['gene_name', 'cluster', 'hyp_response_wt', 'hyp_response_ko', 'interaction', 'padj_int',
     'interaction_rank_FDR', 'interaction_rank_abs_effect']].to_string(index=False))
print('\nAll 17 structural candidates:')
print(candidates[['bin', 'genes', 'd_ins_interaction', 'interaction_padj', 'rank_abs_interaction', 'rank_FDR']].to_string(index=False))
print('\nWrote full mechanism-figure source data.')
