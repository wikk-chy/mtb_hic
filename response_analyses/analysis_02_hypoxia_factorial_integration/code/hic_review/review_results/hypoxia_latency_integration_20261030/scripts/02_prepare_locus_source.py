#!/usr/bin/env python3
"""Freeze representative-locus selection and generate aligned browser source data."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
TAB, SRC = OUT / 'tables', OUT / 'source_data'
N, BIN, GENOME = 883, 5000, 4411532
hyp_b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
lat_b = pd.read_csv(SRC / 'Latency_bin_level.csv')
lat_g = pd.read_csv(SRC / 'Latency_gene_level.csv')
peaks = pd.read_csv(RA / '01_chip_curated/NapM_binding_curated.tsv', sep='\t')
op = pd.read_csv(ROOT / 'hic_review/review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv')
opmap = {}
for row in op.itertuples():
    for gene in ast.literal_eval(row.gene_list):
        opmap.setdefault(gene, set()).add(str(row.operon_id))

# Selection rules are frozen before locus-level visualization.
napm_class = 'NapM-associated_candidate_no_detected_local_RNA_response'
trans_class = 'Transcription-associated_consistent_pattern'
napm_pool = hyp_b[hyp_b.hypoxia_response_evidence_class.eq(napm_class) & hyp_b.ins_WT.notna() & hyp_b.ins_KO.notna()]
trans_pool = hyp_b[hyp_b.hypoxia_response_evidence_class.eq(trans_class) & hyp_b.ins_WT.notna() & hyp_b.ins_KO.notna()]
h119 = napm_pool.sort_values(['interaction_padj', 'chip_bin_score_max'], ascending=[True, False]).iloc[0]
h862 = trans_pool.sort_values(['interaction_padj', 'd_ins_interaction'], ascending=[True, True]).iloc[0]
assert int(h119['bin']) == 119 and int(h862['bin']) == 862
lat_regions = pd.read_csv(TAB / 'Latency_candidate_regions.csv').sort_values('evidence_score', ascending=False)
assert lat_regions.iloc[:3].region_id.tolist() == ['LATR01', 'LATR39', 'LATR18']

loci = [
    dict(model='Hypoxia', locus='H1_NapM_candidate', center=119, half=6, focal_bins='119',
         title='Rv0504c–Rv0507 region',
         selection='Lowest interaction FDR among NapM-associated candidates with no detected local RNA response and valid pooled baseline insulation.',
         evidence_class=napm_class),
    dict(model='Hypoxia', locus='H2_transcription_associated', center=862, half=6, focal_bins='862',
         title='Rv3836–Rv3841 region',
         selection='Strongest significant interaction among transcription-associated consistent patterns with valid pooled baseline insulation; unbound comparison locus.',
         evidence_class=trans_class),
    dict(model='Latency', locus='L1_oriC', center=0, half=7, focal_bins='0;880;881;882',
         title='oriC-proximal dnaA region', selection='Highest latency candidate-region evidence score (LATR01).',
         evidence_class='Same-direction structure/RNA patterns indicate partial molecular similarity, not a latency defect.'),
    dict(model='Latency', locus='L2_PE13_PPE18', center=267, half=6, focal_bins='267',
         title='PE13–PPE18 region', selection='Second-highest latency candidate-region evidence score (LATR39).',
         evidence_class='Opposite insulation directions but same-direction RNA decreases: mixed pattern.'),
    dict(model='Latency', locus='L3_nrdZ_pncB2', center=133, half=7, focal_bins='132;133',
         title='nrdZ–pncB2 region', selection='Third-highest latency candidate-region evidence score (LATR18).',
         evidence_class='Opposite insulation directions with gene-specific same/opposite RNA patterns: mixed pattern.'),
]
for loc in loci:
    center = loc['center']
    loc['bins'] = np.array([(center + i) % N for i in range(-loc['half'], loc['half'] + 1)], int)
    center_bp = center * BIN + BIN / 2
    raw_pos = loc['bins'] * BIN + BIN / 2
    loc['pos_kb'] = ((raw_pos - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000
    loc['focal'] = [int(x) for x in loc['focal_bins'].split(';')]
pd.DataFrame([{k: v for k, v in loc.items() if k not in ['bins', 'pos_kb', 'focal']} for loc in loci]).to_csv(
    TAB / 'Representative_locus_selection.csv', index=False)

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

cond_files = {
    'WT': ('wt', 4), 'KO': ('napm_ko', 4), 'WT_hypoxia': ('hypoxia', 4),
    'KO_hypoxia': ('ko-hypoxia', 3), 'WT_latent': ('latent', 4)}
map_store = {c: {r: {} for r in range(nr)} for c, (p, nr) in cond_files.items()}
contact_rows = []
for cond, (prefix, nr) in cond_files.items():
    needed = [loc for loc in loci if (loc['model'] == 'Hypoxia' and cond in ['WT', 'KO', 'WT_hypoxia', 'KO_hypoxia'])
              or (loc['model'] == 'Latency' and cond in ['WT', 'KO', 'WT_latent'])]
    if not needed:
        continue
    for r in range(nr):
        path = ROOT / f'hic_review/review_data/rep_icedmatrix/{prefix}_{r+1}_5000_iced.matrix'
        m = load_matrix(path); expected = circular_expected(m)
        for loc in needed:
            oe = local_oe(m, expected, loc['bins'])
            map_store[cond][r][loc['locus']] = oe
            for a, b1 in enumerate(loc['bins']):
                for c, b2 in enumerate(loc['bins']):
                    contact_rows.append(dict(model=loc['model'], locus=loc['locus'], condition=cond,
                        replicate=r+1, bin1=int(b1), bin2=int(b2), position1_kb=loc['pos_kb'][a],
                        position2_kb=loc['pos_kb'][c], OE=oe[a, c]))
pd.DataFrame(contact_rows).to_csv(SRC / 'Representative_locus_contact_maps_per_replicate.csv.gz', index=False, compression='gzip')

# Mean maps, insulation, ChIP peaks/signal, RNA lollipops and gene models.
mean_map_rows, ins_rows, peak_rows, rna_rows, gene_rows, focal_rows = [], [], [], [], [], []
bg = pd.read_csv(ROOT / 'hic_review/review_data/chip_seq_data/Rv0047c.bedgraph', sep=r'\s+',
                 header=None, names=['chrom', 'start', 'end', 'signal'])
bg_rows = []

def relative_kb(x, center):
    x = np.asarray(x, float)
    center_bp = center * BIN + BIN / 2
    return ((x - center_bp + GENOME / 2) % GENOME - GENOME / 2) / 1000

for loc in loci:
    conditions = ['WT', 'KO', 'WT_hypoxia', 'KO_hypoxia'] if loc['model'] == 'Hypoxia' else ['WT', 'KO', 'WT_latent']
    for cond in conditions:
        nr = cond_files[cond][1]
        mean_map = np.nanmean(np.stack([map_store[cond][r][loc['locus']] for r in range(nr)]), axis=0)
        for a, b1 in enumerate(loc['bins']):
            for c, b2 in enumerate(loc['bins']):
                mean_map_rows.append(dict(model=loc['model'], locus=loc['locus'], condition=cond,
                    bin1=int(b1), bin2=int(b2), position1_kb=loc['pos_kb'][a], position2_kb=loc['pos_kb'][c],
                    mean_OE=mean_map[a, c]))
    source_b = hyp_b if loc['model'] == 'Hypoxia' else lat_b
    source_g = hyp_g if loc['model'] == 'Hypoxia' else lat_g
    for i, pos in zip(loc['bins'], loc['pos_kb']):
        row = source_b.loc[source_b['bin'].eq(i)].iloc[0]
        out = dict(model=loc['model'], locus=loc['locus'], bin=int(i), position_kb=pos,
                   is_focal=bool(i in loc['focal']), NapM_bound=bool(row.NapM_bound),
                   WT_insulation=row.ins_WT, KO_insulation=row.ins_KO)
        if loc['model'] == 'Hypoxia':
            out.update(WT_hypoxia_delta_insulation=row.d_ins_hyp_WT, KO_hypoxia_delta_insulation=row.d_ins_hyp_KO,
                       interaction_delta_insulation=row.d_ins_interaction, interaction_FDR=row.interaction_padj,
                       evidence_class=row.hypoxia_response_evidence_class)
        else:
            out.update(WT_latent_delta_insulation=row.d_ins_latent_WT, baseline_delta_insulation=row.d_ins_KO_WT,
                       latent_insulation_class=row.latent_ins_class, baseline_insulation_class=row.baseline_ins_class)
        ins_rows.append(out)
        if i in loc['focal']:
            focal_rows.append(out)
    zgenes = source_g[source_g.bin_tss.isin(loc['bins'])].copy()
    for row in zgenes.itertuples():
        operons = ';'.join(sorted(opmap.get(row.gene_id, set())))
        gene_rows.append(dict(model=loc['model'], locus=loc['locus'], gene_id=row.gene_id, gene_name=row.gene_name,
            start=row.start, end=row.end, strand=row.strand, tss=row.tss,
            start_kb=float(relative_kb(row.start, loc['center'])), end_kb=float(relative_kb(row.end, loc['center'])),
            tss_kb=float(relative_kb(row.tss, loc['center'])), bin_tss=row.bin_tss,
            is_focal_bin=bool(row.bin_tss in loc['focal']), binding_class=row.binding_class, operon_ids=operons))
        rout = dict(model=loc['model'], locus=loc['locus'], gene_id=row.gene_id, gene_name=row.gene_name,
                    tss=row.tss, tss_kb=float(relative_kb(row.tss, loc['center'])), bin_tss=row.bin_tss,
                    binding_class=row.binding_class, is_focal_bin=bool(row.bin_tss in loc['focal']))
        if loc['model'] == 'Hypoxia':
            rout.update(WT_hypoxia_log2FC=row.hyp_response_wt, WT_hypoxia_FDR=row.padj_hyp,
                        KO_hypoxia_log2FC=row.hyp_response_ko, KO_hypoxia_FDR=row.padj_hyp_ko,
                        interaction_log2FC=row.interaction, interaction_FDR=row.padj_int)
        else:
            rout.update(baseline_KO_WT_log2FC=row.log2FC_KOvsWT, baseline_FDR=row.padj_KOvsWT,
                        WT_latent_log2FC=row.log2FC_latentvsWT, latent_FDR=row.padj_latentvsWT,
                        direction_relation=row.latency_RNA_direction_relation,
                        replicate_concordant=row.latency_RNA_replicate_concordant)
        rna_rows.append(rout)
    span = loc['half'] * BIN + BIN
    center_bp = loc['center'] * BIN + BIN / 2
    for row in peaks.itertuples():
        d = (row.center - center_bp + GENOME / 2) % GENOME - GENOME / 2
        if abs(d) <= span:
            peak_rows.append(dict(model=loc['model'], locus=loc['locus'], peak_start=row.start, peak_end=row.end,
                peak_center=row.center, start_kb=float(relative_kb(row.start, loc['center'])),
                end_kb=float(relative_kb(row.end, loc['center'])), center_kb=float(d / 1000), score=row.score, vpm=row.vpm))
    # Signal is display-only; curated peaks above define occupancy.
    for row in bg.itertuples():
        d = ((row.start + row.end) / 2 - center_bp + GENOME / 2) % GENOME - GENOME / 2
        if abs(d) <= span:
            bg_rows.append(dict(model=loc['model'], locus=loc['locus'], start=row.start, end=row.end,
                start_kb=float(relative_kb(row.start, loc['center'])), end_kb=float(relative_kb(row.end, loc['center'])),
                signal=row.signal, usage='Display signal only; occupancy is defined by curated Minch2015 peaks.'))

pd.DataFrame(mean_map_rows).to_csv(SRC / 'Representative_locus_mean_contact_maps.csv', index=False)
pd.DataFrame(ins_rows).to_csv(SRC / 'Representative_locus_insulation_tracks.csv', index=False)
pd.DataFrame(peak_rows).to_csv(SRC / 'Representative_locus_NapM_peaks.csv', index=False)
pd.DataFrame(bg_rows).to_csv(SRC / 'Representative_locus_NapM_signal.csv', index=False)
pd.DataFrame(rna_rows).to_csv(SRC / 'Representative_locus_RNA_tracks.csv', index=False)
pd.DataFrame(gene_rows).to_csv(SRC / 'Representative_locus_gene_models.csv', index=False)
pd.DataFrame(focal_rows).to_csv(TAB / 'Representative_locus_focal_bin_summary.csv', index=False)

# Panel-specific tables for traceability.
pd.read_csv(TAB / 'RNA_response_counts.csv').to_csv(SRC / 'FigHypA_RNA_response_counts.csv', index=False)
hyp_g.to_csv(SRC / 'FigHypA_gene_responses.csv', index=False)
matched = pd.read_csv(RA / '05_matched_controls/matched_pairs.csv')
for c in ['d_ins_hyp_WT', 'd_ins_hyp_KO', 'd_ins_interaction']:
    matched['bound_' + c] = matched.bound_bin.map(hyp_b.set_index('bin')[c])
    matched['unbound_' + c] = matched.matched_bin.map(hyp_b.set_index('bin')[c])
matched.to_csv(SRC / 'FigHypB_matched_bin_pairs.csv', index=False)
hyp_b.to_csv(SRC / 'FigHypC_bin_cross_analysis.csv', index=False)
lat_b.to_csv(SRC / 'FigLatABC_bin_integration.csv', index=False)
lat_g.to_csv(SRC / 'FigLatABC_gene_integration.csv', index=False)
metadata = {
    'selection_order': 'Genome-wide statistics were completed before representative-locus visualization.',
    'hypoxia_loci': {'NapM_candidate_bin': 119, 'transcription_associated_bin': 862},
    'latency_loci': {'LATR01': 'oriC/dnaA', 'LATR39': 'PE13-PPE18', 'LATR18': 'nrdZ-pncB2'},
    'coordinate_system': 'All browser tracks use kb relative to the focal center bin; circular coordinates are unwrapped.',
    'contact_map': 'Per-replicate circular-distance O/E; plotted maps are replicate means; raw per-replicate values supplied.',
    'ChIP_signal_note': 'Legacy bedgraph is display signal only. Occupancy and peak statistics use 634 curated Minch2015 Rv0047c peaks.',
    'independence_note': 'Each locus is one local regulatory/structural unit; multiple genes and bins within it are not independent mechanisms.'
}
with open(TAB / 'Representative_locus_metadata.json', 'w') as fh:
    json.dump(metadata, fh, indent=2)
print(json.dumps(metadata, indent=2))
