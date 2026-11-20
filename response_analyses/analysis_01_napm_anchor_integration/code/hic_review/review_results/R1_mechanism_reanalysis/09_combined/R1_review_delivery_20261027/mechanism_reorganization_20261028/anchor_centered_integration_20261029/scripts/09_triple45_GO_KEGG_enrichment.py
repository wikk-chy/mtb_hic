#!/usr/bin/env python3
"""GO BP/CC/MF and KEGG enrichment for the 45 three-way intersection DEGs."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
CACHE = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'

# Use the same query definition and RNA-tested background as the 79-gene analysis.
g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
g['RNA_tested'] = g.log2FC_KOvsWT.notna() & g.padj_KOvsWT.notna()
member = pd.read_csv(OUT / 'source_data/anchor_and_three_way_gene_membership.csv')
assert member.gene_id.tolist() == g.gene_id.tolist()
triple = pd.read_csv(TAB / 'NapM_insulation_DEG_triple_genes.csv')
query = set(triple.gene_id)
assert len(query) == len(triple) == 45
assert triple[['NapM_occupied', 'insulation_changed', 'DEG']].all().all()
assert set(triple.loc[triple.triple_intersection, 'gene_id']) == query
assert query == set(member.loc[member.triple_intersection, 'gene_id'])
universe = set(g.loc[g.RNA_tested, 'gene_id'])
assert len(universe) == 4060 and query <= universe
triple.to_csv(TAB / 'triple45_query_genes.csv', index=False)

# Direct KEGG gene-pathway mappings.
k2p = pd.read_csv(CACHE / 'kegg_gene2pathway.tsv', sep='\t', header=None,
                  names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pn = pd.read_csv(CACHE / 'kegg_pathway_names.tsv', sep='\t', header=None,
                 names=['term', 'name'])
pn['name'] = pn.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_names = dict(zip(pn.term, pn.name))

# Direct GO annotations, with all three namespaces as in the 79-gene analysis.
rv_pat = re.compile(r'Rv\d{3,4}[A-Za-z]?(?![A-Za-z0-9])')
go_rows = []
with open(CACHE / '30.M_tuberculosis_ATCC_25618.goa') as handle:
    for line in handle:
        if line.startswith('!'):
            continue
        fields = line.rstrip('\n').split('\t')
        for gene in set(rv_pat.findall(fields[10])):
            go_rows.append((gene, fields[4]))
gaf = pd.DataFrame(go_rows, columns=['gene', 'term']).drop_duplicates()
go_names, go_namespace, obsolete = {}, {}, set()
cid = cname = cns = None
with open(CACHE / 'go-basic.obo') as handle:
    for line in handle:
        line = line.rstrip('\n')
        if line == '[Term]':
            if cid and cname:
                go_names[cid] = cname
                go_namespace[cid] = cns
            cid = cname = cns = None
        elif line.startswith('id: '):
            cid = line[4:]
        elif line.startswith('name: '):
            cname = line[6:]
        elif line.startswith('namespace: '):
            cns = line[11:]
        elif line.startswith('is_obsolete: true') and cid:
            obsolete.add(cid)
if cid and cname:
    go_names[cid] = cname
    go_namespace[cid] = cns
aspect_to_db = {'biological_process': 'GO_BP', 'cellular_component': 'GO_CC',
                'molecular_function': 'GO_MF'}
dbs = {'KEGG': (k2p[['gene', 'term']].drop_duplicates(), kegg_names)}
for aspect, db in aspect_to_db.items():
    terms = {term for term, ns in go_namespace.items()
             if ns == aspect and term not in obsolete}
    dbs[db] = (gaf[gaf.term.isin(terms)][['gene', 'term']].drop_duplicates(), go_names)


def bh(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def enrichment(mapping, names, database):
    annotated_universe = universe & set(mapping.gene)
    q = query & annotated_universe
    M, N = len(annotated_universe), len(q)
    assert M > N > 0
    rows = []
    for term, genes in mapping.groupby('term').gene.apply(set).items():
        term_genes = genes & annotated_universe
        n = len(term_genes)
        if n < 3:
            continue
        overlap = q & term_genes
        k = len(overlap)
        rows.append(dict(
            gene_set='triple45_vs_RNA_tested', database=database, term=term,
            name=names.get(term, term), overlap=k, term_size=n, query_size=N,
            background_size=M, fold_enrichment=(k / N) / (n / M),
            p_value=hypergeom.sf(k - 1, M, n, N),
            genes=','.join(sorted(overlap))))
    result = pd.DataFrame(rows)
    result['q_BH_all_tested_terms'] = bh(result.p_value)
    result['n_terms_tested'] = len(result)
    return result


res = pd.concat([enrichment(mapping, names, database)
                 for database, (mapping, names) in dbs.items()], ignore_index=True)
res = res.sort_values(['database', 'p_value', 'term']).reset_index(drop=True)
res.to_csv(TAB / 'triple45_GO_KEGG_enrichment.csv', index=False)
res[res.q_BH_all_tested_terms.lt(.05)].to_csv(
    TAB / 'triple45_FDR_significant.csv', index=False)
res[res.p_value.lt(.05)].to_csv(TAB / 'triple45_nominal_P_significant.csv', index=False)
summary = res.groupby('database', as_index=False).agg(
    query_annotated=('query_size', 'first'),
    background_annotated=('background_size', 'first'),
    n_terms_tested=('term', 'size'),
    nominal_P_0_05=('p_value', lambda x: int((x < .05).sum())),
    FDR_0_05=('q_BH_all_tested_terms', lambda x: int((x < .05).sum())),
    minimum_P=('p_value', 'min'), minimum_q=('q_BH_all_tested_terms', 'min'))
summary.to_csv(TAB / 'triple45_enrichment_summary.csv', index=False)
metadata = {
    'query': '45 unique RNA-tested DEGs with baseline NapM occupancy and detected insulation change',
    'gene_set': 'triple45_vs_RNA_tested',
    'background': 'All 4,060 RNA-tested genes',
    'databases': ['GO_BP', 'GO_CC', 'GO_MF', 'KEGG'],
    'annotation': 'Direct GAF GO and KEGG gene-term mappings; terms with >=3 genes in each annotated background',
    'test': 'One-sided hypergeometric enrichment; BH across all tested terms including zero-overlap terms, separately per database',
    'display_filter': 'Nominal P < 0.05, matching the 79-gene minimal GO/KEGG figures',
    'interpretation': 'Overlap and enrichment do not establish direct NapM regulation or contact-to-RNA causality.',
}
(TAB / 'triple45_enrichment_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(summary.to_string(index=False))
print('\nNOMINAL P < 0.05')
print(res[res.p_value.lt(.05)][['database', 'term', 'name', 'overlap',
      'fold_enrichment', 'p_value', 'q_BH_all_tested_terms', 'genes']].to_string(index=False))
