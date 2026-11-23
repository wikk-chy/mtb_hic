#!/usr/bin/env python3
"""GO (BP/CC/MF) and KEGG enrichment for all 79 changed-anchor DEGs."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
BASE = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
OUT = BASE / 'anchor_centered_integration_20261029'
TAB = OUT / 'tables'
SRC = OUT / 'source_data'
CACHE = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
DET = {'detected_increase', 'detected_decrease'}

# Query and three explicitly defined backgrounds.
g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
g['RNA_tested'] = g.log2FC_KOvsWT.notna() & g.padj_KOvsWT.notna()
g['DEG'] = g.class_KOvsWT.isin(DET)
member = pd.read_csv(SRC / 'anchor_and_three_way_gene_membership.csv')
assert member.gene_id.tolist() == g.gene_id.tolist()
query = set(member.loc[member.in_changed_anchor & member.DEG, 'gene_id'])
assert len(query) == 79
rna_universe = set(g.loc[g.RNA_tested, 'gene_id'])
eligible_rna = set(member.loc[member.in_eligible_anchor & member.RNA_tested, 'gene_id'])
eligible_deg = set(member.loc[member.in_eligible_anchor & member.DEG, 'gene_id'])
assert (len(rna_universe), len(eligible_rna), len(eligible_deg)) == (4060, 1118, 226)
assert query <= eligible_deg <= eligible_rna <= rna_universe

backgrounds = [
    dict(gene_set='anchor79_vs_RNA_tested', universe=rna_universe,
         role='primary_standard', background_label='All 4,060 RNA-tested genes'),
    dict(gene_set='anchor79_vs_eligible_anchor_RNA_tested', universe=eligible_rna,
         role='sensitivity_anchor_matched', background_label='1,118 RNA-tested genes near all 39-loop anchors'),
    dict(gene_set='anchor79_vs_eligible_anchor_DEGs', universe=eligible_deg,
         role='sensitivity_DEG_conditioned', background_label='226 DEGs near all 39-loop anchors'),
]
query_table = member[member.gene_id.isin(query)].copy()
query_table.to_csv(TAB / 'anchor79_query_genes.csv', index=False)
background_membership = member[['gene_id', 'gene_name', 'in_changed_anchor', 'DEG',
                                'in_eligible_anchor', 'RNA_tested']].copy()
background_membership['query_anchor79'] = background_membership.gene_id.isin(query)
background_membership['background_RNA_tested'] = background_membership.gene_id.isin(rna_universe)
background_membership['background_eligible_anchor_RNA'] = background_membership.gene_id.isin(eligible_rna)
background_membership['background_eligible_anchor_DEGs'] = background_membership.gene_id.isin(eligible_deg)
background_membership.to_csv(SRC / 'anchor79_background_gene_membership.csv', index=False)

# KEGG direct gene-pathway mappings.
k2p = pd.read_csv(CACHE / 'kegg_gene2pathway.tsv', sep='\t', header=None, names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pn = pd.read_csv(CACHE / 'kegg_pathway_names.tsv', sep='\t', header=None, names=['term', 'name'])
pn['name'] = pn.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_names = dict(zip(pn.term, pn.name))

# GO direct annotations; namespaces and names come from go-basic.obo.
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
go_names = {}
go_namespace = {}
obsolete = set()
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
    terms = {term for term, ns in go_namespace.items() if ns == aspect and term not in obsolete}
    dbs[db] = (gaf[gaf.term.isin(terms)][['gene', 'term']].drop_duplicates(), go_names)


def bh(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def enrichment(query_genes, universe_genes, mapping, names, gene_set, database, role, background_label):
    universe = set(universe_genes) & set(mapping.gene)
    q = set(query_genes) & universe
    M, N = len(universe), len(q)
    rows = []
    assert M > N > 0
    for term, genes in mapping.groupby('term').gene.apply(set).items():
        term_genes = genes & universe
        n = len(term_genes)
        if n < 3:
            continue
        overlap = q & term_genes
        k = len(overlap)
        rows.append(dict(
            gene_set=gene_set, role=role, background_label=background_label,
            database=database, term=term, name=names.get(term, term),
            overlap=k, term_size=n, query_size=N, background_size=M,
            fold_enrichment=(k/N)/(n/M),
            p_value=hypergeom.sf(k-1, M, n, N),
            genes=','.join(sorted(overlap)),
        ))
    result = pd.DataFrame(rows)
    result['q_BH_all_tested_terms'] = bh(result.p_value)
    result['n_terms_tested'] = len(result)
    return result


results = []
for spec in backgrounds:
    for database, (mapping, names) in dbs.items():
        results.append(enrichment(query, spec['universe'], mapping, names,
                                  spec['gene_set'], database, spec['role'],
                                  spec['background_label']))
res = pd.concat(results, ignore_index=True)
res = res.sort_values(['gene_set', 'database', 'p_value', 'term']).reset_index(drop=True)
res.to_csv(TAB / 'anchor79_GO_KEGG_enrichment.csv', index=False)
res[res.q_BH_all_tested_terms.lt(.05)].to_csv(TAB / 'anchor79_FDR_significant.csv', index=False)
res[res.p_value.lt(.05)].to_csv(TAB / 'anchor79_nominal_P_significant.csv', index=False)
top = res.sort_values(['gene_set', 'database', 'p_value', 'term']).groupby(
    ['gene_set', 'database'], group_keys=False).head(10)
top.to_csv(TAB / 'anchor79_top10_per_database_and_background.csv', index=False)

summary = res.groupby(['gene_set', 'role', 'background_label', 'database'], as_index=False).agg(
    query_annotated=('query_size', 'first'), background_annotated=('background_size', 'first'),
    n_terms_tested=('term', 'size'), nominal_P_0_05=('p_value', lambda x: int((x < .05).sum())),
    FDR_0_05=('q_BH_all_tested_terms', lambda x: int((x < .05).sum())),
    minimum_P=('p_value', 'min'), minimum_q=('q_BH_all_tested_terms', 'min'))
summary.to_csv(TAB / 'anchor79_enrichment_summary.csv', index=False)
metadata = dict(
    query='79 unique DEGs whose TSS lies within +/-10 kb of any changed-loop anchor',
    query_up_down={'up': 48, 'down': 31},
    databases=['GO_BP', 'GO_CC', 'GO_MF', 'KEGG'],
    annotation='Direct GAF GO and KEGG gene-term mappings; terms with >=3 genes in each annotated background',
    test='One-sided hypergeometric enrichment; BH across all tested terms including zero-overlap terms, separately per gene set and database',
    backgrounds={s['gene_set']: {'n_before_annotation_filter': len(s['universe']),
                                 'role': s['role'], 'label': s['background_label']} for s in backgrounds},
    interpretation='Standard RNA-tested enrichment is primary for the requested all-79 analysis. Anchor-matched and DEG-conditioned analyses assess sensitivity to background choice. No result by itself establishes direct NapM regulation or contact-to-RNA causality.',
)
(TAB / 'anchor79_enrichment_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(summary.to_string(index=False))
print('\nFDR SIGNIFICANT TERMS')
cols = ['gene_set', 'database', 'term', 'name', 'overlap', 'term_size', 'query_size',
        'background_size', 'fold_enrichment', 'p_value', 'q_BH_all_tested_terms', 'genes']
print(res[res.q_BH_all_tested_terms.lt(.05)][cols].to_string(index=False))
print('\nPRIMARY STANDARD: TOP TERMS')
print(res[(res.gene_set.eq('anchor79_vs_RNA_tested')) & (res.overlap.gt(0))].groupby(
    'database', group_keys=False).head(8)[cols].to_string(index=False))
