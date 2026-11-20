#!/usr/bin/env python3
"""Exploratory GO/KEGG enrichment for multi-omics joint-change gene sets.

Gene-level hypergeometric tests are a screen, not final causal evidence. Direct,
unbound and loop-anchor sets use their corresponding eligible gene universes.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
R2 = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
TAB = OUT / 'tables'
TAB.mkdir(parents=True, exist_ok=True)
DET = {'detected_increase', 'detected_decrease'}


def bh(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv').set_index('bin')
a = pd.read_csv(RA / '08_extended/fig2_loops/tables/panelG_anchor_genes.csv')
g['rna_ko_detected'] = g.class_KOvsWT.isin(DET)
g['rna_hypoxia_detected'] = g.class_hypoxiavsWT.isin(DET)
g['structure_ko_detected'] = g.bin_ins_change_class.isin(DET)
lo = g.bin_tss.map(b.d_ins_hyp_WT_lo).to_numpy(float)
hi = g.bin_tss.map(b.d_ins_hyp_WT_hi).to_numpy(float)
g['structure_hypoxia_detected'] = np.isfinite(lo) & np.isfinite(hi) & (lo * hi > 0)
g['structure_interaction_detected'] = g.bin_tss.map(b.interaction_class).isin(DET)
g['direct_napm'] = g.binding_class.isin(['promoter_bound', 'gene_body_bound'])
g['double_anchor_gene'] = g.gene_id.isin(set(a.loc[a.is_anchor_double, 'gene_id']))
g['eligible_ko'] = g.log2FC_KOvsWT.notna() & g.bin_d_ins_KO_WT.notna()
g['eligible_hypoxia'] = g.log2FC_hypoxiavsWT.notna() & g.bin_tss.map(b.d_ins_hyp_WT).notna()

baseline = g.structure_ko_detected & g.rna_ko_detected
hypoxia = g.structure_hypoxia_detected & g.rna_hypoxia_detected
interaction = g.structure_interaction_detected & g.rna_hypoxia_detected
specs = [
    ('Baseline_structure_RNA', baseline, g.eligible_ko, 'KO vs WT'),
    ('Baseline_structure_RNA_direct', baseline & g.direct_napm,
     g.eligible_ko & g.direct_napm, 'KO vs WT; direct NapM-bound background'),
    ('Baseline_structure_RNA_unbound', baseline & ~g.direct_napm,
     g.eligible_ko & ~g.direct_napm, 'KO vs WT; locally unbound background'),
    ('Baseline_double_anchor_RNA', g.double_anchor_gene & g.rna_ko_detected & g.structure_ko_detected,
     g.eligible_ko & g.double_anchor_gene, 'KO vs WT; double-anchor-gene background'),
    ('Hypoxia_structure_RNA', hypoxia, g.eligible_hypoxia, 'Hypoxia vs WT'),
    ('Hypoxia_structure_RNA_direct', hypoxia & g.direct_napm,
     g.eligible_hypoxia & g.direct_napm, 'Hypoxia vs WT; direct NapM-bound background'),
    ('Interaction_structure_hypoxia_RNA', interaction, g.eligible_hypoxia,
     'Structural genotype-by-hypoxia interaction plus hypoxia RNA'),
]

set_frames = []
summary = []
for name, mask, universe, description in specs:
    selected = g.loc[mask & universe].copy()
    selected['gene_set'] = name
    set_frames.append(selected)
    summary.append(dict(gene_set=name, description=description, n_genes=len(selected),
                        n_bins=selected.bin_tss.nunique(), n_universe=int(universe.sum()),
                        n_direct=int(selected.direct_napm.sum()),
                        n_rna_up=int((selected.log2FC_hypoxiavsWT > 0).sum()) if name.startswith(('Hypoxia', 'Interaction'))
                        else int((selected.log2FC_KOvsWT > 0).sum())))
pd.concat(set_frames, ignore_index=True).to_csv(TAB / 'multiomics_gene_sets.csv', index=False)
pd.DataFrame(summary).to_csv(TAB / 'multiomics_gene_set_summary.csv', index=False)

# M. tuberculosis H37Rv KEGG mappings already cached in the project.
k2p = pd.read_csv(R2 / 'kegg_gene2pathway.tsv', sep='\t', header=None, names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pnames = pd.read_csv(R2 / 'kegg_pathway_names.tsv', sep='\t', header=None, names=['term', 'name'])
pnames['name'] = pnames.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_map = dict(zip(pnames.term, pnames.name))

# GO biological-process mappings from the project-cached GAF and OBO files.
rv_pat = re.compile(r'Rv\d{3,4}[A-Za-z]?(?![A-Za-z0-9])')
rows = []
with open(R2 / '30.M_tuberculosis_ATCC_25618.goa') as handle:
    for line in handle:
        if line.startswith('!'):
            continue
        fields = line.rstrip('\n').split('\t')
        if fields[8] != 'P':
            continue
        for gene in set(rv_pat.findall(fields[10])):
            rows.append((gene, fields[4]))
gaf = pd.DataFrame(rows, columns=['gene', 'term']).drop_duplicates()
go_name = {}
current_id = current_name = None
with open(R2 / 'go-basic.obo') as handle:
    for line in handle:
        line = line.rstrip('\n')
        if line == '[Term]':
            if current_id and current_name:
                go_name[current_id] = current_name
            current_id = current_name = None
        elif line.startswith('id: '):
            current_id = line[4:]
        elif line.startswith('name: '):
            current_name = line[6:]
if current_id and current_name:
    go_name[current_id] = current_name

databases = {
    'GO_BP': (gaf, go_name),
    'KEGG': (k2p[['gene', 'term']], kegg_map),
}

for db, (mapping, names) in databases.items():
    term2genes = mapping.groupby('term').gene.apply(set).to_dict()
    annotated = set(mapping.gene)
    all_results = []
    for name, mask, universe, description in specs:
        universe_genes = set(g.loc[universe, 'gene_id']) & annotated
        query = set(g.loc[mask & universe, 'gene_id']) & universe_genes
        M, N = len(universe_genes), len(query)
        rows = []
        for term, genes in term2genes.items():
            target = genes & universe_genes
            n = len(target)
            if n < 3:
                continue
            overlap = query & target
            k = len(overlap)
            if k < 2:
                continue
            p = hypergeom.sf(k - 1, M, n, N)
            rows.append(dict(gene_set=name, database=db, term=term,
                             name=names.get(term, term), overlap=k, term_size=n,
                             query_size=N, background_size=M,
                             fold_enrichment=(k / N) / (n / M), p_value=p,
                             genes=','.join(sorted(overlap))))
        result = pd.DataFrame(rows)
        if len(result):
            result['q_within_gene_set'] = bh(result.p_value)
            all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if len(combined):
        combined['q_all_gene_sets'] = bh(combined.p_value)
        combined = combined.sort_values(['gene_set', 'p_value'])
    combined.to_csv(TAB / f'{db}_enrichment_screen.csv', index=False)
    print(f'\n{db}:')
    for name, *_ in specs:
        x = combined[(combined.gene_set == name) & (combined.q_within_gene_set < 0.10)] if len(combined) else pd.DataFrame()
        print(f'  {name}: {len(x)} terms with within-set q < 0.10')
        if len(x):
            print(x[['name', 'overlap', 'term_size', 'fold_enrichment', 'p_value', 'q_within_gene_set', 'genes']].head(8).to_string(index=False))
print('\nWrote exploratory gene-set and enrichment tables to', TAB)
