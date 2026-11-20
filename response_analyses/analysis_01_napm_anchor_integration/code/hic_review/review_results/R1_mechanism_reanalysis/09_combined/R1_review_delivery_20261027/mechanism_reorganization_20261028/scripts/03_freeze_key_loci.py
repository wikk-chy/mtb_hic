#!/usr/bin/env python3
"""Freeze key-locus values and their evidence tiers for integrated response figures."""
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
TAB = OUT / 'tables'
TAB.mkdir(parents=True, exist_ok=True)

g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
p = pd.read_csv(RA / '01_chip_curated/NapM_binding_curated.tsv', sep='\t')
l = pd.read_csv(RA / '02_loop_curated/loops_41_curated.tsv', sep='\t')
ids = ['Rv0001', 'Rv2381c', 'Rv2382c', 'Rv2383c', 'Rv2384',
       'Rv0967', 'Rv0969', 'Rv1992c']
notes = {
    'Rv0001': ('oriC–dnaA', 'published direct molecular axis; current multi-omics convergence',
               'NapM occupies the dnaA gene body and two loop anchors; published NapM–DnaA biochemistry supplies a plausible direct axis. This study does not test DnaA activity or replication initiation.', '[1]'),
    'Rv2381c': ('mbt-1', 'local NapM occupancy plus operon-level association',
                'NapM occupies the mbtD gene body. Coordinated mbt-1 repression is compatible with direct local action, operon propagation, or established IdeR/HupB control; these alternatives are not separated.', '[2]'),
    'Rv2382c': ('mbt-1', 'indirect or operon-propagated association',
                'No local NapM promoter/gene-body occupancy; mbtC repression cannot be labelled a direct NapM effect.', '[2]'),
    'Rv2383c': ('mbt-1', 'indirect or operon-propagated association',
                'No local NapM promoter/gene-body occupancy; mbtB repression cannot be labelled a direct NapM effect.', '[2]'),
    'Rv2384': ('mbt-1', 'indirect or operon-propagated association',
                'No local NapM promoter/gene-body occupancy; mbtA repression cannot be labelled a direct NapM effect.', '[2]'),
    'Rv0967': ('csoR–ctpV', 'locally unbound indirect physiological programme',
               'The copper-responsive CsoR/cso mechanism is published, but no local NapM occupancy is detected and hypoxia sensing by CsoR is not tested.', '[3]'),
    'Rv0969': ('csoR–ctpV', 'locally unbound indirect physiological programme',
               'The copper-responsive CsoR/cso mechanism is published, but no local NapM occupancy is detected and copper flux is not measured here.', '[3]'),
    'Rv1992c': ('ctpG', 'locally unbound indirect physiological programme',
               'Copper-responsive expression and possible compensation for CtpV are published; the present data do not establish a NapM-to-copper signalling pathway.', '[3]'),
}
cols = ['gene_id', 'gene_name', 'start', 'end', 'strand', 'tss', 'bin_tss',
        'binding_class', 'n_peaks_promoter_up300', 'n_peaks_gene_body',
        'gene_peak_score_max', 'bin_d_ins_KO_WT', 'log2FC_KOvsWT', 'padj_KOvsWT',
        'log2FC_hypoxiavsWT', 'padj_hypoxiavsWT']
out = g[g.gene_id.isin(ids)][cols].copy()
out['locus'] = out.gene_id.map(lambda x: notes[x][0])
out['evidence_tier'] = out.gene_id.map(lambda x: notes[x][1])
out['mechanistic_interpretation'] = out.gene_id.map(lambda x: notes[x][2])
out['reference'] = out.gene_id.map(lambda x: notes[x][3])
out.to_csv(TAB / 'key_loci_evidence.csv', index=False)

peaks = p[(p.center <= 130000) | p.center.between(2640000, 2700000)].copy()
peaks['locus_window'] = peaks.center.map(lambda x: 'oriC–dnaA:0–130kb' if x <= 130000 else 'mbt-1:2.64–2.70Mb')
peaks.to_csv(TAB / 'key_loci_napm_peaks.csv', index=False)
loops = l[((l.bin1 <= 26) & (l.bin2 <= 26)) |
          ((l.bin1.between(528, 540)) & (l.bin2.between(528, 540)))].copy()
loops.to_csv(TAB / 'key_loci_loops.csv', index=False)

# Fail loudly if a source update invalidates the evidence narrative.
r = out.set_index('gene_id')
assert r.loc['Rv0001', 'binding_class'] == 'gene_body_bound'
assert int(r.loc['Rv0001', 'n_peaks_gene_body']) == 2
assert r.loc['Rv2381c', 'binding_class'] == 'gene_body_bound'
assert all(r.loc[x, 'binding_class'] == 'unbound' for x in ['Rv2382c', 'Rv2383c', 'Rv2384', 'Rv0967', 'Rv0969', 'Rv1992c'])
assert {'L00', 'L01'}.issubset(set(l.loc[l.both_bound, 'loop_id']))
print(out[['gene_id', 'gene_name', 'binding_class', 'bin_d_ins_KO_WT', 'log2FC_KOvsWT', 'evidence_tier']].to_string(index=False))
print('\nLoops in oriC window:', loops[['loop_id', 'bin1', 'bin2', 'both_bound']].to_dict('records'))
print('Wrote key-locus evidence, peak and loop tables to', TAB)
