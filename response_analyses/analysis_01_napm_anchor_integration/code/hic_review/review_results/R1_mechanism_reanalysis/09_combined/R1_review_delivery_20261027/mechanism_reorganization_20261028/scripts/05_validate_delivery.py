#!/usr/bin/env python3
"""Validate integrated Reviewer 1 figure, source-data and writing deliverables."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
TAB = OUT / 'tables'
SD = OUT / 'source_data'
FIG = OUT / 'figures'
response = (OUT / 'R1_response_draft.md').read_text()
legends = (OUT / 'figure_legends.md').read_text()
refs = (OUT / 'references.md').read_text()
checks = []


def check(name, condition, detail=''):
    checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    if not condition:
        raise AssertionError(f'{name}: {detail}')


def close(a, b, tol=5e-5):
    return np.isclose(float(a), float(b), atol=tol, rtol=0)


# Deliverable existence and basic file integrity.
for n in range(1, 5):
    for ext in ['pdf', 'png']:
        p = FIG / f'R1_Response_Figure{n}.{ext}'
        check(f'Figure {n} {ext} exists', p.exists() and p.stat().st_size > 10000, p)
    check(f'Figure {n} is one-page PDF', len(PdfReader(FIG / f'R1_Response_Figure{n}.pdf').pages) == 1)
    with Image.open(FIG / f'R1_Response_Figure{n}.png') as im:
        check(f'Figure {n} PNG resolution', im.width >= 3000 and im.height >= 2000, im.size)
for f in ['R1_response_draft.md', 'figure_legends.md', 'references.md',
          'tables/key_loci_evidence.csv', 'source_data/source_manifest.csv',
          'integrated_figures.log']:
    check(f'Deliverable {f}', (OUT / f).exists() and (OUT / f).stat().st_size > 0)

manifest = pd.read_csv(SD / 'source_manifest.csv')
check('Panel source files all exist', all(Path(x).exists() for x in manifest.source_file), len(manifest))
check('Source manifest has no duplicate output names', not manifest.source_file.duplicated().any())
check('Source manifest includes all figures', set(f'Fig{i}' for i in range(1, 5)) == set(x[:4] for x in manifest.panel))

# Key numerical assertions used in the response.
a = pd.read_csv(SD / 'Fig1A_anchor_enrichment.csv').iloc[0]
check('Anchor observed 17/41', int(a.observed) == 17 and int(a.null_count if 'null_count' in a else 41) == 41)
check('Anchor null mean and fold', close(a.null_mean, 10.897, 0.001) and close(a.fold_enrichment, 1.56, 0.01))
check('Anchor empirical P', close(a.p_empirical, 0.0328, 1e-6))
rob = pd.read_csv(SD / 'Fig1B_robustness.csv')
check('16/18 robustness intervals cross zero', len(rob) == 18 and int(((rob.ci_lo <= 0) & (rob.ci_hi >= 0)).sum()) == 16)
dose = pd.read_csv(SD / 'Fig1C_loop_dose_summary.csv').set_index('metric')
s = dose.loc['adjusted_slope_per_anchor']
check('Loop-dose beta/CI/P', close(s.effect, -0.056623, 1e-5) and close(s.ci_lo, -0.158744, 1e-5) and close(s.ci_hi, 0.028597, 1e-5) and close(s.p_span_stratified_permutation, 0.238876, 1e-5))

ind = pd.read_csv(SD / 'Fig2A_independent_samples.csv')
check('15 independent samples', len(ind) == 15 and ind.groupby('condition').size().to_dict() == {'HYP': 4, 'KO': 4, 'KO-HYP': 3, 'WT': 4})
fact = pd.read_csv(SD / 'Fig2A_factorial_contrast.csv')
did = fact[fact.test == 'genotype contrast of genome-mean hypoxia responses'].iloc[0]
check('Difference-in-differences and exact P', close(did.effect, -0.1079895, 1e-6) and close(did.p, 0.000408163, 1e-8))
fp15 = pd.read_csv(SD / 'Fig2B_fingerprint_15_test_samples.csv')
check('Heatmap exactly 15 samples/no latent', len(fp15) == 15 and not fp15.condition.isin(['latent']).any())
rv = pd.read_csv(SD / 'Fig2B_response_vectors.csv')
check('Nine fingerprint metrics and responses', len(rv) == 9 and rv[['R_WT', 'R_KO']].notna().all().all())
sp = pd.read_csv(SD / 'Fig2C_spatial_summary.csv').iloc[0]
check('Spatial rho and CI', close(sp.estimator, 0.526526, 1e-5) and close(sp.ci_lo, 0.438132, 1e-5) and close(sp.ci_hi, 0.609226, 1e-5))
splits = pd.read_csv(SD / 'Fig2D_disjoint_reference.csv')
check('18 disjoint references positive', len(splits) == 18 and (splits.ci_lo > 0).all() and close(splits.rho.min(), 0.478262, 1e-5) and close(splits.rho.max(), 0.564127, 1e-5))

sets = pd.read_csv(SD / 'Fig3A_gene_sets.csv').set_index('gene_set')
for gene_set, n, direct in [('Baseline_structure_RNA', 269, 45),
                            ('Baseline_structure_RNA_unbound', 224, 0),
                            ('Hypoxia_structure_RNA', 1143, 241),
                            ('Interaction_structure_hypoxia_RNA', 550, 111),
                            ('Baseline_double_anchor_RNA', 10, 5)]:
    check(f'Gene-set size {gene_set}', int(sets.loc[gene_set, 'n_genes']) == n and int(sets.loc[gene_set, 'n_direct']) == direct)
for db in ['GO_BP', 'KEGG']:
    d = pd.read_csv(TAB / f'{db}_conditional_spatial_enrichment.csv')
    for col in ['q_conditional_within_gene_set', 'q_spatial_within_gene_set',
                'q_conditional_all_gene_sets', 'q_spatial_all_gene_sets']:
        check(f'No sensitivity FDR <0.10 {db} {col}', not (d[col] < 0.10).any())

loci = pd.read_csv(TAB / 'key_loci_evidence.csv').set_index('gene_id')
check('dnaA occupancy/effect', loci.loc['Rv0001', 'binding_class'] == 'gene_body_bound' and int(loci.loc['Rv0001', 'n_peaks_gene_body']) == 2 and close(loci.loc['Rv0001', 'bin_d_ins_KO_WT'], -0.382968, 1e-5) and close(loci.loc['Rv0001', 'log2FC_KOvsWT'], -1.232386, 1e-5))
check('mbt direct/unbound distinction', loci.loc['Rv2381c', 'binding_class'] == 'gene_body_bound' and all(loci.loc[x, 'binding_class'] == 'unbound' for x in ['Rv2382c', 'Rv2383c', 'Rv2384']))
check('Copper genes locally unbound', all(loci.loc[x, 'binding_class'] == 'unbound' for x in ['Rv0967', 'Rv0969', 'Rv1992c']))
peaks = pd.read_csv(TAB / 'key_loci_napm_peaks.csv')
check('oriC peak centres', {721.0, 1097.0}.issubset(set(peaks.center)))
check('mbtD peak centres', {2668488.0, 2669680.0}.issubset(set(peaks.center)))
loops = pd.read_csv(TAB / 'key_loci_loops.csv').set_index('loop_id')
check('L00/L01 both occupied', all(loops.loc[x, 'both_bound'] for x in ['L00', 'L01']))

# Writing contract: exact panel citations, references and honest manuscript status.
for text_name, text in [('response', response), ('legends', legends)]:
    for token in ['Response Fig. 1A', 'Response Fig. 1B', 'Response Fig. 1C'] if text_name == 'response' else []:
        check(f'{text_name} cites {token}', token in text)
for panel in ['1A', '1B', '1C', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '3E', '4A', '4B', '4C']:
    pattern = rf'Reponse Fig\. {panel}|Response Fig\. {panel}|Response Figs\. {panel}'
    check(f'Response cites panel {panel}', re.search(pattern, response) is not None)
for token in ['[1]', '[2]', '[3]', '10.1038/s42003-019-0314-9', '10.1128/JB.00359-16', '10.1016/j.tube.2011.12.006']:
    check(f'Response reference token {token}', token in response)
for doi in ['10.1038/s42003-019-0314-9', '10.1128/JB.00359-16', '10.1016/j.tube.2011.12.006']:
    check(f'Reference file DOI {doi}', doi in refs and doi in legends)
check('Manuscript edit status explicit', 'The main manuscript source has **not** been edited' in response and 'not yet applied' in response)
check('No false completed-manuscript claim', 'we have revised the manuscript' not in response.lower() and 'we have edited the manuscript' not in response.lower())
check('No SVD inference in response', 'SVD' not in response)
check('Enrichment described as exploratory', 'hypothesis-generating' in response and 'No pathway in any gene set retained FDR < 0.10' in response)
check('Original reviewer wording present', 'falls short of a molecular explanation' in response and 'direct or indirect role' in response)

report = dict(status='PASS', n_checks=len(checks), checks=checks)
(OUT / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['Integrated Reviewer 1 delivery validation: PASS', f'Checks passed: {len(checks)}', '']
lines += [f'PASS | {x["check"]} | {x["detail"]}' for x in checks]
(OUT / 'validation_report.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines[:3]))
print('Figures: 4 PDF + 4 PNG; panel source tables:', len(manifest))
print('Response words:', len(response.split()), '; legend words:', len(legends.split()))
