#!/usr/bin/env python3
"""Independent numerical and file-integrity checks for anchor-centred delivery."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from PIL import Image

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
BASE = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
OUT = BASE / 'anchor_centered_integration_20261029'
TAB = OUT / 'tables'
SRC = OUT / 'source_data'
FIG = OUT / 'figures'
DET = {'detected_increase', 'detected_decrease'}
GENOME = 4415000
checks = []


def ok(name, detail=''):
    checks.append((name, 'PASS', detail))


def independent_bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


# Rebuild all memberships from source tables, not from the integration script.
loops = pd.read_csv(BASE / 'loop_change_enrichment/per_loop_change_and_binding.csv')
changed = loops[loops.abs_d_log2OE.ge(.5)].copy()
expected_loops = ['L00', 'L12', 'L14', 'L15', 'L16', 'L17', 'L19', 'L23', 'L28', 'L29', 'L32']
assert changed.loop_id.tolist() == expected_loops
left = changed[['loop_id', 'center1']].rename(columns={'center1': 'center'})
right = changed[['loop_id', 'center2']].rename(columns={'center2': 'center'})
anchors = pd.concat([left, right], ignore_index=True)
assert len(anchors) == 22 and anchors.center.nunique() == 21
ok('Changed loops and anchors', '11 loops; 22 anchor instances; 21 unique centers')

g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
g['RNA_tested'] = g.log2FC_KOvsWT.notna() & g.padj_KOvsWT.notna()
g['DEG'] = g.class_KOvsWT.isin(DET)
g['up'] = g.class_KOvsWT.eq('detected_increase')
g['down'] = g.class_KOvsWT.eq('detected_decrease')
g['NapM'] = g.binding_class.isin(['promoter_bound', 'gene_body_bound'])
g['Ichange'] = g.bin_ins_change_class.isin(DET)
assert g.RNA_tested.sum() == 4060 and g.DEG.sum() == 820


def near_mask(centers, width):
    x = g.tss.to_numpy(float)
    mask = np.zeros(len(g), bool)
    for center in centers:
        d = np.abs(x-float(center))
        mask |= np.minimum(d, GENOME-d) <= width
    return mask


expected_windows = {
    5000: (197, 33, 23, 10, 590, 117),
    10000: (406, 79, 48, 31, 1118, 226),
    25000: (862, 168, 106, 62, 2170, 447),
}
recomputed = []
for width, expected in expected_windows.items():
    near = near_mask(changed.center1.tolist()+changed.center2.tolist(), width)
    eligible = near_mask(loops.center1.tolist()+loops.center2.tolist(), width)
    got = (int((near & g.RNA_tested).sum()), int((near & g.DEG).sum()),
           int((near & g.up).sum()), int((near & g.down).sum()),
           int((eligible & g.RNA_tested).sum()), int((eligible & g.DEG).sum()))
    assert got == expected, (width, got, expected)
    recomputed.append((width, *got))
    if width == 10000:
        primary_near = near
reported = pd.read_csv(TAB / 'anchor_window_sensitivity.csv')
assert len(reported) == 3
for row in recomputed:
    r = reported[reported.window_bp.eq(row[0])].iloc[0]
    assert tuple(int(r[c]) for c in reported.columns[1:]) == row[1:]
ok('Independent circular-distance window recalculation',
   '5 kb 197/33; 10 kb 406/79 (48 up, 31 down); 25 kb 862/168')

# Check the exported membership vector against independently rebuilt membership.
member = pd.read_csv(SRC / 'anchor_and_three_way_gene_membership.csv')
assert member.gene_id.tolist() == g.gene_id.tolist()
assert np.array_equal(member.in_changed_anchor.to_numpy(bool), primary_near)
anchor_only = int((primary_near & g.RNA_tested & ~g.DEG).sum())
both = int((primary_near & g.DEG).sum())
deg_only = int((~primary_near & g.DEG).sum())
neither = int((~primary_near & g.RNA_tested & ~g.DEG).sum())
assert (anchor_only, both, deg_only, neither) == (327, 79, 741, 2913)
ok('Two-set Venn partition', 'Anchor-only 327; overlap 79; DEG-only 741; neither 2913')

# Exact three-way combinations among the 4,060 RNA-tested genes.
rna = g[g.RNA_tested].copy()
rna['code'] = [''.join(['N' if n else '-', 'I' if i else '-', 'D' if d else '-'])
               for n, i, d in zip(rna.NapM, rna.Ichange, rna.DEG)]
expected_combos = {'---': 1801, '--D': 442, '-I-': 768, '-ID': 224,
                   'N--': 489, 'N-D': 109, 'NI-': 182, 'NID': 45}
got_combos = rna.groupby('code').size().to_dict()
assert got_combos == expected_combos
assert sum(got_combos.values()) == 4060
assert (int(rna.NapM.sum()), int(rna.Ichange.sum()), int(rna.DEG.sum())) == (825, 1219, 820)
reported_combos = pd.read_csv(TAB / 'three_way_exact_combinations.csv')
assert reported_combos.set_index('combination_code').n_genes.to_dict() == expected_combos
assert len(pd.read_csv(TAB / 'NapM_insulation_DEG_triple_genes.csv')) == 45
ok('Three-way UpSet and set sizes', 'NapM 825; insulation 1219; DEG 820; NID 45; total 4060')

# Independently verify the sole significant hypergeometric test and all BH columns.
enrich = pd.read_csv(TAB / 'anchor_and_three_way_GO_KEGG_enrichment.csv')
abc = enrich[(enrich.gene_set.eq('changed_anchor_up_vs_eligible_anchor_DEGs')) &
             (enrich.term.eq('mtu02010'))].iloc[0]
p = hypergeom.sf(5, 51, 8, 11)
assert np.isclose(p, abc.p_value, rtol=1e-12, atol=1e-15)
assert int(abc.overlap) == 6 and int(abc.term_size) == 8
assert int(abc.query_size) == 11 and int(abc.background_size) == 51
assert np.isclose(abc.fold_enrichment, (6/11)/(8/51), rtol=1e-12)
assert set(abc.genes.split(',')) == {'Rv1970', 'Rv1971', 'Rv2325c', 'Rv2686c', 'Rv2687c', 'Rv2688c'}
for (gene_set, database), z in enrich.groupby(['gene_set', 'database']):
    q = independent_bh(z.p_value.to_numpy())
    assert np.allclose(q, z.q_BH_all_tested_terms.to_numpy(), rtol=1e-11, atol=1e-13), (gene_set, database)
sig = enrich[enrich.q_BH_all_tested_terms.lt(.05)]
assert len(sig) == 1 and sig.iloc[0].term == 'mtu02010'
assert np.isclose(sig.iloc[0].q_BH_all_tested_terms, .010564447795769557, rtol=1e-11)
assert len(pd.read_csv(TAB / 'anchor_and_three_way_FDR_significant.csv')) == 1
assert (enrich[enrich.gene_set.str.startswith('triple_vs_')].q_BH_all_tested_terms >= .05).all()
ok('ABC transporters hypergeometric test',
   f'M=51, n=8, N=11, k=6; P={p:.12g}; fold={abc.fold_enrichment:.9f}; q={abc.q_BH_all_tested_terms:.12g}')
ok('FDR scope and null results',
   'Exactly one FDR<0.05 row across all gene sets; no triple-intersection GO/KEGG result passes FDR')

# Check representative-locus source tracks and candidates.
browser = pd.read_csv(SRC / 'representative_browser_source.csv')
for locus, candidate in [('L00', 'Rv0001'), ('L32', 'Rv3196A')]:
    z = browser[browser.locus.eq(locus)]
    assert {'loop', 'anchor', 'anchor_window', 'NapM_peak', 'insulation_bin', 'RNA_gene'} <= set(z.track)
    assert len(z[z.track.eq('anchor')]) == 2
    assert len(z[z.track.eq('anchor_window')]) == 2
    r = z[(z.track.eq('RNA_gene')) & z.id.eq(candidate)].iloc[0]
    assert bool(r.DEG) and bool(r.NapM_occupied) and bool(r.insulation_changed)
    assert bool(r.triple_intersection) and bool(r.in_changed_anchor)
summary = pd.read_csv(TAB / 'representative_locus_summary.csv')
assert summary.locus.tolist() == ['L00', 'L32']
assert summary.n_anchor_DEGs_in_view.tolist() == [5, 5]
ok('Representative browser source',
   'Both loci contain loop, 2 anchors/windows, NapM, insulation and RNA tracks; dnaA and Rv3196A flags verified')

# File signatures and decodable images.
for stem in ['R1_anchor_centered_integration', 'R1_representative_anchor_loci']:
    pdf = FIG / f'{stem}.pdf'
    png = FIG / f'{stem}.png'
    assert pdf.read_bytes().startswith(b'%PDF')
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with Image.open(png) as im:
        im.verify()
    with Image.open(png) as im:
        assert im.width >= 4000 and im.height >= 3000
        dims = f'{im.width}x{im.height}'
    ok(f'Figure files: {stem}', f'PDF signature valid; PNG valid; {dims}')

lines = ['ANCHOR-CENTRED INTEGRATION VALIDATION', '=======================================', '']
for name, status, detail in checks:
    lines.append(f'[{status}] {name}')
    if detail:
        lines.append(f'  {detail}')
lines += ['', f'ALL {len(checks)} CHECK GROUPS PASSED.',
          'Interpretation boundary: overlaps, enrichment and browser co-localization do not establish direct NapM regulation, differential NapM binding, contact-to-RNA causality, chromosomal plasticity, or dormancy establishment.']
report = '\n'.join(lines)+'\n'
(OUT / 'validation_report.txt').write_text(report)
print(report)
