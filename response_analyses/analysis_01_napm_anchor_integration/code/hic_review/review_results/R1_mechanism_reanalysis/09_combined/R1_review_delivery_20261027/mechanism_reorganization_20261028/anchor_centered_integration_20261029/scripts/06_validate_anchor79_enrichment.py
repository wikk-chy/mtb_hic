#!/usr/bin/env python3
"""Validate all-79 anchor-DEG GO/KEGG enrichment and delivered figure."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from PIL import Image

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
res = pd.read_csv(TAB / 'anchor79_GO_KEGG_enrichment.csv')
query = pd.read_csv(TAB / 'anchor79_query_genes.csv')
checks = []


def ok(name, detail):
    checks.append((name, detail))


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    adj = np.minimum.accumulate((ranked*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


assert len(query) == 79 and query.gene_id.nunique() == 79
assert int(query.up.sum()) == 48 and int(query.down.sum()) == 31
ok('Query identity', '79 unique changed-anchor DEGs; 48 up; 31 down')

assert set(res.gene_set) == {'anchor79_vs_RNA_tested', 'anchor79_vs_eligible_anchor_RNA_tested',
                             'anchor79_vs_eligible_anchor_DEGs'}
assert set(res.database) == {'GO_BP', 'GO_CC', 'GO_MF', 'KEGG'}
assert res.groupby(['gene_set', 'database']).ngroups == 12
for key, z in res.groupby(['gene_set', 'database']):
    assert np.allclose(bh(z.p_value), z.q_BH_all_tested_terms, rtol=1e-11, atol=1e-13), key
    assert z.n_terms_tested.eq(len(z)).all()
ok('Test and BH structure', '3 backgrounds x 4 databases; all 12 BH vectors independently recomputed')

primary = res[res.gene_set.eq('anchor79_vs_RNA_tested')]
assert not primary.q_BH_all_tested_terms.lt(.05).any()
assert primary.groupby('database').query_size.first().to_dict() == {'GO_BP': 43, 'GO_CC': 48, 'GO_MF': 44, 'KEGG': 25}
ok('Primary standard-background result', 'No GO-BP/CC/MF or KEGG term passes FDR<0.05')

expected_tests = [
    ('anchor79_vs_RNA_tested', 'GO:0015562', 2525, 11, 44, 3),
    ('anchor79_vs_eligible_anchor_RNA_tested', 'GO:0015562', 724, 3, 44, 3),
    ('anchor79_vs_eligible_anchor_DEGs', 'GO:0015562', 145, 3, 44, 3),
    ('anchor79_vs_RNA_tested', 'mtu02010', 1122, 84, 25, 6),
    ('anchor79_vs_RNA_tested', 'mtu01200', 1122, 96, 25, 7),
]
for gene_set, term, M, n, N, k in expected_tests:
    row = res[(res.gene_set.eq(gene_set)) & (res.term.eq(term))].iloc[0]
    assert (row.background_size, row.term_size, row.query_size, row.overlap) == (M, n, N, k)
    p = hypergeom.sf(k-1, M, n, N)
    assert np.isclose(row.p_value, p, rtol=1e-12, atol=1e-15), (gene_set, term, p, row.p_value)
ok('Key hypergeometric tests', 'Efflux in three backgrounds plus standard-background ABC transporters and carbon metabolism')

sig = res[res.q_BH_all_tested_terms.lt(.05)]
assert len(sig) == 1
r = sig.iloc[0]
assert r.gene_set == 'anchor79_vs_eligible_anchor_RNA_tested' and r.term == 'GO:0015562'
assert np.isclose(r.q_BH_all_tested_terms, .02943637975584328, rtol=1e-10)
assert set(r.genes.split(',')) == {'Rv2686c', 'Rv2687c', 'Rv2688c'}
drivers = query[query.gene_id.isin(r.genes.split(','))]
assert len(drivers) == 3 and drivers.changed_anchor_ids.eq('L29_left').all()
assert drivers.NapM_occupied.eq(False).all() and drivers.insulation_changed.eq(False).all()
ok('Sole FDR hit and locus dependence',
   f'{r["name"]}; q={r.q_BH_all_tested_terms:.12g}; all three drivers at L29_left; no NapM occupancy or insulation change')

urease = primary[(primary.term.isin(['GO:0009039', 'GO:0035550', 'GO:0006995']))]
assert len(urease) == 3
assert all(set(row.genes.split(',')) == {'Rv1848', 'Rv1850'} for row in urease.itertuples())
ok('Urease signal redundancy', 'GO-MF urease activity, GO-CC urease complex and BP nitrogen-starvation response all use only Rv1848/Rv1850')

pdf = FIG / 'R1_anchor79_GO_KEGG_enrichment.pdf'
png = FIG / 'R1_anchor79_GO_KEGG_enrichment.png'
assert pdf.read_bytes().startswith(b'%PDF')
assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
with Image.open(png) as im:
    im.verify()
with Image.open(png) as im:
    assert im.width >= 4000 and im.height >= 2800
    dims = f'{im.width}x{im.height}'
ok('Figure integrity', f'Valid PDF and PNG signatures; PNG {dims}')

lines = ['ALL-79 ANCHOR-DEG ENRICHMENT VALIDATION', '========================================', '']
for name, detail in checks:
    lines.extend([f'[PASS] {name}', f'  {detail}'])
lines += ['', f'ALL {len(checks)} CHECK GROUPS PASSED.',
          'Primary interpretation: no standard-background FDR-significant GO/KEGG term. The sole anchor-matched GO-MF hit is background-sensitive and driven by one three-gene locus.']
report = '\n'.join(lines)+'\n'
(OUT / 'anchor79_validation_report.txt').write_text(report)
print(report)
