#!/usr/bin/env python3
"""Validate the complete mechanism figure, source data and reviewer response."""
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC, TAB, FIG = OUT / 'source_data', OUT / 'tables', OUT / 'figures'
response = (OUT / 'NapM_hypoxia_full_mechanism_reviewer_response.md').read_text()
checks = []


def check(name, condition, detail=''):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})


hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
clusters = pd.read_csv(SRC / 'Mechanism_RNA_interaction_clusters.csv')
summary = pd.read_csv(TAB / 'Mechanism_RNA_cluster_summary.csv').set_index('cluster')
candidates = pd.read_csv(TAB / 'Mechanism_structural_candidate_ranking.csv')
selection = pd.read_csv(TAB / 'Mechanism_locus_selection.csv')
lat = pd.read_csv(TAB / 'Latency_candidate_regions.csv')

check('exactly 730 interaction DEGs', len(clusters) == 730)
expected_sizes = {'C1': 197, 'C2': 188, 'C3': 254, 'C4': 91}
check('cluster sizes match figure and response', clusters.cluster.value_counts().to_dict() == expected_sizes,
      str(clusters.cluster.value_counts().to_dict()))
highlight = clusters[clusters.gene_name.isin(['devR', 'devS', 'sigE'])]
check('three highlighted regulators are all in C3', len(highlight) == 3 and set(highlight.cluster) == {'C3'})
check('C3 centers match response', abs(summary.loc['C3', 'WT_center'] - 2.256539) < 1e-5 and
      abs(summary.loc['C3', 'KO_center'] - .700615) < 1e-5)
check('all 17 stringent candidates shown', len(candidates) == 17)
check('ESX-2 ranks first by absolute interaction',
      int(candidates.sort_values('abs_interaction', ascending=False).iloc[0]['bin']) == 874 and
      int(candidates.loc[candidates['bin'].eq(874), 'rank_abs_interaction'].iloc[0]) == 1)
check('oriC has independent LATR01 evidence', 'LATR01' in set(lat.region_id) and
      any(str(x) == '0' for region in lat[lat.region_id.eq('LATR01')]['bins'] for x in region.split(';')))
check('four intended loci selected', set(selection.locus) == {'DosR/DosS', 'SigE', 'oriC/dnaA', 'ESX-2'})

# Verify every displayed response-map pixel against the condition-mean maps.
condition = pd.read_csv(SRC / 'Mechanism_locus_condition_maps.csv')
response_maps = pd.read_csv(SRC / 'Mechanism_locus_response_maps.csv')
lookup = condition.set_index(['locus', 'condition', 'bin1', 'bin2']).mean_OE.to_dict()
errors = []
for row in response_maps.itertuples():
    hyp = lookup[(row.locus, 'WT_hypoxia' if row.genotype == 'WT' else 'KO_hypoxia', row.bin1, row.bin2)]
    norm = lookup[(row.locus, 'WT' if row.genotype == 'WT' else 'KO', row.bin1, row.bin2)]
    expected = np.log2(hyp / norm) if np.isfinite(hyp) and np.isfinite(norm) and hyp > 0 and norm > 0 else np.nan
    if np.isfinite(expected) and np.isfinite(row.log2_hyp_norm_OE):
        errors.append(abs(expected - row.log2_hyp_norm_OE))
    else:
        errors.append(0.0 if np.isnan(expected) and np.isnan(row.log2_hyp_norm_OE) else np.inf)
check('all response-map values match condition-mean O/E ratios', max(errors) < 1e-12, f'max_error={max(errors):.3g}')
check('response maps contain four loci and two genotypes',
      set(response_maps.locus) == {'DosR_DosS', 'SigE', 'oriC', 'ESX2'} and set(response_maps.genotype) == {'WT', 'KO'})

expr = pd.read_csv(SRC / 'Mechanism_highlighted_gene_normalized_logCPM.csv')
check('three highlighted genes have eight normalized samples', len(expr) == 3 and expr.shape[1] == 9)

# Figure implementation and delivery constraints.
script = (OUT / 'scripts/08_build_full_mechanism_figure.py').read_text()
check('Hi-C response maps use bilinear display interpolation', "interpolation='bilinear'" in script)
check('figure has no schematic flowchart boxes', 'FancyArrowPatch' not in script and 'Working model:' not in script)
check('no rescue/complementation proposal', all(term not in response.lower() for term in ['rescue', 'complementation', 'complementing']))
for phrase in ['not necessarily the highest-ranking genes by effect size',
               'not because it had the largest hypoxia effect',
               'not proof of two independent pathways',
               'Failure to detect local DEGs does not exclude transcriptional involvement',
               'interpolation is used only for display']:
    check(f'response safeguard: {phrase}', phrase in response)
for token in ['730 RNA-interaction DEGs', '212 positive and 518 negative',
              '+2.257 and +0.701', 'Seventeen bins', '−1.041', '+0.301', '−0.005']:
    check(f'response contains {token}', token in response)

pdf = FIG / 'NapM_hypoxia_full_mechanism_evidence.pdf'
png = FIG / 'NapM_hypoxia_full_mechanism_evidence.png'
doc = pdfium.PdfDocument(str(pdf))
check('complete figure is one PDF page', len(doc) == 1, f'pages={len(doc)}')
doc.close()
with Image.open(png) as im:
    check('high-resolution PNG has expected dimensions', im.size == (5400, 6600), str(im.size))
check('figure deliverables are nonempty', pdf.stat().st_size > 0 and png.stat().st_size > 0)

result = pd.DataFrame(checks)
result.to_csv(TAB / 'Full_mechanism_validation.csv', index=False)
print(result.to_string(index=False))
print(f'\n{int(result.passed.sum())}/{len(result)} checks passed')
if not result.passed.all():
    raise SystemExit('Full mechanism validation failed')
