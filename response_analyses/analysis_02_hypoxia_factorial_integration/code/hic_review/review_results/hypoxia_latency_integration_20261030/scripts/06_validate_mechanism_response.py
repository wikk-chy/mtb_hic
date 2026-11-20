#!/usr/bin/env python3
"""Validate mechanism-figure source values, response text and delivery constraints."""
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
from PIL import Image

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC, FIG, TAB = OUT / 'source_data', OUT / 'figures', OUT / 'tables'
response = (OUT / 'NapM_hypoxia_reviewer_response.md').read_text()
hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
hyp_b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
locus = pd.read_csv(SRC / 'Mechanism_model_locus_source.csv')
global_src = pd.read_csv(SRC / 'Mechanism_model_global_source.csv')
checks = []


def check(name, condition, detail=''):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})


# Exact source-data preservation.
for row in locus[locus.panel.eq('A')].itertuples():
    original = hyp_g.loc[hyp_g.gene_id.eq(row.gene_id)].iloc[0]
    check(f'{row.gene_id} WT/KO/interaction source values',
          abs(original.hyp_response_wt - row.WT_response) < 1e-12 and
          abs(original.hyp_response_ko - row.KO_response) < 1e-12 and
          abs(original.interaction - row.interaction) < 1e-12 and
          abs(original.padj_int - row.interaction_FDR) < 1e-20)
for row in locus[locus.panel.eq('B')].itertuples():
    original = hyp_b.loc[hyp_b['bin'].eq(row.bin)].iloc[0]
    check(f'bin {row.bin} WT/KO/interaction source values',
          abs(original.d_ins_hyp_WT - row.WT_response) < 1e-12 and
          abs(original.d_ins_hyp_KO - row.KO_response) < 1e-12 and
          abs(original.d_ins_interaction - row.interaction) < 1e-12 and
          abs(original.interaction_padj - row.interaction_FDR) < 1e-20)

# Numerical statements in the reviewer response.
for token in ['730 genes', '+2.297', '+0.409', '−1.887', '3.60 × 10⁻⁸',
              '+2.068', '+1.046', '−1.022', '5.61 × 10⁻⁷',
              '+2.448', '−0.349', '−2.798', '3.05 × 10⁻⁹',
              '−0.005', 'FDR = 0.877', '+0.301', '3.20 × 10⁻⁷',
              '−1.041', '9.42 × 10⁻⁷', 'ρ = 0.028', 'P = 0.419',
              '0.000253', 'P = 0.708']:
    check(f'response contains {token}', token in response)

# Local-DEG statements must match all three RNA contrasts.
for bin_id in [0, 874]:
    row = hyp_b.loc[hyp_b['bin'].eq(bin_id)].iloc[0]
    check(f'bin {bin_id} has no detected local DEG in WT/KO/interaction',
          not bool(row.RNA_WT_hypoxia_any_DEG) and
          not bool(row.RNA_KO_hypoxia_any_DEG) and
          not bool(row.RNA_interaction_any_DEG))
sigE_bin = hyp_b.loc[hyp_b['bin'].eq(272)].iloc[0]
check('sigE bin has no detected structural response',
      sigE_bin.hypoxia_response_evidence_class == 'No_detected_structural_response')

# Logical safeguards and user constraint.
for phrase in ['two distinguishable response patterns',
               'not evidence that transcription is uninvolved',
               'do not demonstrate that DosR/DosS and SigE constitute a single regulatory pathway',
               'not claimed to be independent pathways',
               'Baseline WT ChIP-seq does not establish NapM occupancy during hypoxia']:
    check(f'caution retained: {phrase}', phrase in response)
check('no rescue/complementation proposal',
      all(term not in response.lower() for term in ['rescue', 'complementation', 'complementing']))

pdf = FIG / 'NapM_hypoxia_working_model.pdf'
png = FIG / 'NapM_hypoxia_working_model.png'
doc = pdfium.PdfDocument(str(pdf))
check('mechanism PDF is exactly one page', len(doc) == 1, f'pages={len(doc)}')
doc.close()
with Image.open(png) as im:
    check('mechanism PNG is A4-landscape ratio', abs(im.width / im.height - 1.4142) < .01,
          f'{im.width}x{im.height}')
check('figure source files exist', all(p.exists() and p.stat().st_size > 0 for p in [pdf, png]))

result = pd.DataFrame(checks)
result.to_csv(TAB / 'Mechanism_response_validation.csv', index=False)
print(result.to_string(index=False))
print(f'\n{int(result.passed.sum())}/{len(result)} checks passed')
if not result.passed.all():
    raise SystemExit('Mechanism response validation failed')
