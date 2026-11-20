#!/usr/bin/env python3
"""Audit RNA+-conditional enrichment for the NapM-bound structural-interaction set.

This is an exploratory sensitivity analysis, not a true RNA-interaction test:
structure requires a genotype-by-hypoxia interaction, whereas RNA requires a
WT hypoxia response. The direct-bound eligible background matches the query.
The analysis mirrors 02_conditional_spatial_enrichment.py but writes only the
new direct-interaction result and leaves all previously frozen tables intact.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy import sparse

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
R2 = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
TAB = OUT / 'tables'
DET = {'detected_increase', 'detected_decrease'}
GENE_SET = 'Interaction_structure_hypoxia_RNA_direct'


def bh(values):
    p = np.asarray(values, dtype=float)
    if not len(p):
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
op = pd.read_csv(RA / '08_extended/fig6_operon/tables/operon_level.csv')
gene2op = {}
for row in op.itertuples():
    for gene in str(row.gene_list).split(';'):
        if gene:
            gene2op[gene] = row.operon_id
g['operon_id'] = [gene2op.get(x, x) for x in g.gene_id]
g['region_100kb'] = (g.tss // 100000).astype(int)
g['rna_hypoxia_detected'] = g.class_hypoxiavsWT.isin(DET)
g['structure_interaction_detected'] = g.bin_tss.map(b.interaction_class).isin(DET)
g['direct_napm'] = g.binding_class.isin(['promoter_bound', 'gene_body_bound'])
g['eligible_hypoxia'] = g.log2FC_hypoxiavsWT.notna() & g.bin_tss.map(b.d_ins_hyp_WT).notna()

eligible = g.eligible_hypoxia.to_numpy(bool) & g.direct_napm.to_numpy(bool)
rna = g.rna_hypoxia_detected.to_numpy(bool)
structure = g.structure_interaction_detected.to_numpy(bool)
idx = np.flatnonzero(eligible)
idx = idx[np.argsort(g.tss.to_numpy()[idx], kind='stable')]
genes = g.gene_id.to_numpy(str)[idx]
operons = g.operon_id.to_numpy(str)[idx]
regions = g.region_100kb.to_numpy(int)[idx]
rna = rna[idx]
structure = structure[idx]
gene_index = {gene: i for i, gene in enumerate(genes)}

k2p = pd.read_csv(R2 / 'kegg_gene2pathway.tsv', sep='\t', header=None, names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pnames = pd.read_csv(R2 / 'kegg_pathway_names.tsv', sep='\t', header=None, names=['term', 'name'])
pnames['name'] = pnames.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_names = dict(zip(pnames.term, pnames.name))

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
go_names = {}
current_id = current_name = None
with open(R2 / 'go-basic.obo') as handle:
    for line in handle:
        line = line.rstrip('\n')
        if line == '[Term]':
            if current_id and current_name:
                go_names[current_id] = current_name
            current_id = current_name = None
        elif line.startswith('id: '):
            current_id = line[4:]
        elif line.startswith('name: '):
            current_name = line[6:]
if current_id and current_name:
    go_names[current_id] = current_name

results = []
for db, mapping, names in [('GO_BP', gaf, go_names), ('KEGG', k2p[['gene', 'term']], kegg_names)]:
    annotated = set(mapping.gene)
    keep = np.array([x in annotated for x in genes])
    universe_mask = rna & keep
    query_mask = rna & structure & keep
    universe_genes = set(genes[universe_mask])
    query_genes = set(genes[query_mask])
    M, N = len(universe_genes), len(query_genes)
    submap = mapping[mapping.gene.isin(universe_genes)].copy()
    term2genes = submap.groupby('term').gene.apply(set).to_dict()
    terms = sorted(term2genes)
    term_index = {term: j for j, term in enumerate(terms)}
    rr, cc = [], []
    for term in terms:
        for gene in term2genes[term]:
            if gene in gene_index:
                rr.append(gene_index[gene])
                cc.append(term_index[term])
    ann = sparse.csr_matrix((np.ones(len(rr), dtype=np.int16), (rr, cc)),
                            shape=(len(genes), len(terms)))
    observed = np.asarray(ann[np.flatnonzero(query_mask)].sum(axis=0)).ravel().astype(int)
    null = np.empty((len(genes) - 1, len(terms)), dtype=np.int16)
    for j, shift in enumerate(range(1, len(genes))):
        shifted_query = rna & np.roll(structure, shift) & keep
        null[j] = np.asarray(ann[np.flatnonzero(shifted_query)].sum(axis=0)).ravel()
    rows_out = []
    for j, term in enumerate(terms):
        target = term2genes[term] & universe_genes
        overlap = query_genes & target
        n, k = len(target), len(overlap)
        if n < 3 or k < 2:
            continue
        p_cond = hypergeom.sf(k - 1, M, n, N)
        p_spatial = (1 + int(np.sum(null[:, j] >= k))) / len(genes)
        overlap_ops = [operons[gene_index[x]] for x in overlap]
        op_counts = pd.Series(overlap_ops).value_counts()
        top_op = op_counts.index[0]
        drop = operons == top_op
        universe_drop = universe_mask & ~drop
        query_drop = query_mask & ~drop
        target_drop = np.array([x in target for x in genes]) & universe_drop
        M2, N2 = int(universe_drop.sum()), int(query_drop.sum())
        n2 = int(target_drop.sum())
        k2 = int((query_drop & target_drop).sum())
        p_drop = hypergeom.sf(k2 - 1, M2, n2, N2) if k2 >= 1 else 1.0
        overlap_regions = [regions[gene_index[x]] for x in overlap]
        region_counts = pd.Series(overlap_regions).value_counts()
        rows_out.append(dict(
            gene_set=GENE_SET, database=db, term=term, name=names.get(term, term),
            overlap=k, term_size=n, query_size=N, background_size=M,
            fold_enrichment=(k / N) / (n / M), p_conditional_RNAplus=p_cond,
            p_gene_order_circular_shift=p_spatial, n_circular_shifts=len(genes) - 1,
            n_operons_overlap=len(op_counts), top_operon=top_op,
            top_operon_genes=int(op_counts.iloc[0]),
            top_operon_fraction=float(op_counts.iloc[0] / k),
            n_100kb_regions_overlap=len(region_counts),
            top_100kb_region=int(region_counts.index[0]),
            top_100kb_fraction=float(region_counts.iloc[0] / k),
            p_drop_top_operon=p_drop, genes=','.join(sorted(overlap)),
            definition='Structural genotype-by-hypoxia interaction plus WT-hypoxia RNA; NapM-bound eligible background',
        ))
    result = pd.DataFrame(rows_out)
    if len(result):
        result['q_conditional_within_gene_set'] = bh(result.p_conditional_RNAplus)
        result['q_spatial_within_gene_set'] = bh(result.p_gene_order_circular_shift)
        results.append(result)
    print(db, 'eligible', len(genes), 'RNA+ annotated', M, 'joint', N,
          'terms', len(result), 'shifts', len(genes) - 1, flush=True)

combined = pd.concat(results, ignore_index=True)
combined = combined.sort_values(['database', 'p_conditional_RNAplus', 'p_gene_order_circular_shift'])
combined.to_csv(TAB / 'Interaction_direct_conditional_spatial_enrichment_audit.csv', index=False)
print(combined[['database', 'name', 'overlap', 'fold_enrichment',
                'p_conditional_RNAplus', 'q_conditional_within_gene_set',
                'p_gene_order_circular_shift', 'q_spatial_within_gene_set', 'genes']].head(20).to_string(index=False))
print('FDR-significant conditional terms:', int((combined.q_conditional_within_gene_set < 0.10).sum()))
print('FDR-significant spatial terms:', int((combined.q_spatial_within_gene_set < 0.10).sum()))
print('Wrote', TAB / 'Interaction_direct_conditional_spatial_enrichment_audit.csv')
