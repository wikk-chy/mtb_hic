#!/usr/bin/env python3
"""Conditional RNA+ and gene-order circular-shift enrichment sensitivity.

For each contrast, the universe is restricted to genes with a detected RNA
response. The query additionally requires the relevant structural response.
This asks whether structural change marks functional enrichment beyond the
RNA-responsive background. Gene-order circular shifts of the structural label
preserve its exact run-length pattern while randomizing its position relative
to RNA response and pathway annotations. Operon-dependence is descriptive:
the largest contributing operon is removed and the conditional test repeated.
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
TAB.mkdir(parents=True, exist_ok=True)
DET = {'detected_increase', 'detected_decrease'}


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
a = pd.read_csv(RA / '08_extended/fig2_loops/tables/panelG_anchor_genes.csv')
op = pd.read_csv(RA / '08_extended/fig6_operon/tables/operon_level.csv')

# One operon assignment per gene; unassigned genes remain singleton loci.
gene2op = {}
for row in op.itertuples():
    for gene in str(row.gene_list).split(';'):
        if gene:
            gene2op[gene] = row.operon_id
g['operon_id'] = [gene2op.get(x, x) for x in g.gene_id]
g['region_100kb'] = (g.tss // 100000).astype(int)

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

baseline_eligible = g.eligible_ko.to_numpy(bool)
hypoxia_eligible = g.eligible_hypoxia.to_numpy(bool)
baseline_rna = g.rna_ko_detected.to_numpy(bool)
hypoxia_rna = g.rna_hypoxia_detected.to_numpy(bool)
baseline_structure = g.structure_ko_detected.to_numpy(bool)
hypoxia_structure = g.structure_hypoxia_detected.to_numpy(bool)
interaction_structure = g.structure_interaction_detected.to_numpy(bool)
direct = g.direct_napm.to_numpy(bool)
double_anchor = g.double_anchor_gene.to_numpy(bool)

specs = [
    dict(gene_set='Baseline_structure_RNA', eligible=baseline_eligible,
         rna=baseline_rna, structure=baseline_structure,
         description='KO vs WT; RNA+ background'),
    dict(gene_set='Baseline_structure_RNA_direct', eligible=baseline_eligible & direct,
         rna=baseline_rna, structure=baseline_structure,
         description='KO vs WT; NapM-bound RNA+ background'),
    dict(gene_set='Baseline_structure_RNA_unbound', eligible=baseline_eligible & ~direct,
         rna=baseline_rna, structure=baseline_structure,
         description='KO vs WT; locally unbound RNA+ background'),
    dict(gene_set='Baseline_double_anchor_RNA', eligible=baseline_eligible & double_anchor,
         rna=baseline_rna, structure=baseline_structure,
         description='KO vs WT; double-anchor-gene RNA+ background'),
    dict(gene_set='Hypoxia_structure_RNA', eligible=hypoxia_eligible,
         rna=hypoxia_rna, structure=hypoxia_structure,
         description='Hypoxia vs WT; RNA+ background'),
    dict(gene_set='Hypoxia_structure_RNA_direct', eligible=hypoxia_eligible & direct,
         rna=hypoxia_rna, structure=hypoxia_structure,
         description='Hypoxia vs WT; NapM-bound RNA+ background'),
    dict(gene_set='Interaction_structure_hypoxia_RNA', eligible=hypoxia_eligible,
         rna=hypoxia_rna, structure=interaction_structure,
         description='Hypoxia RNA+ background; structural genotype-by-hypoxia interaction'),
]

# Project-cached M. tuberculosis H37Rv annotations.
k2p = pd.read_csv(R2 / 'kegg_gene2pathway.tsv', sep='\t', header=None, names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pnames = pd.read_csv(R2 / 'kegg_pathway_names.tsv', sep='\t', header=None, names=['term', 'name'])
pnames['name'] = pnames.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_map = dict(zip(pnames.term, pnames.name))

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

databases = {'GO_BP': (gaf, go_name), 'KEGG': (k2p[['gene', 'term']], kegg_map)}
all_gene_ids = g.gene_id.to_numpy(str)
all_operons = g.operon_id.to_numpy(str)
all_regions = g.region_100kb.to_numpy(int)

for db, (mapping, names) in databases.items():
    combined_sets = []
    for spec in specs:
        eligible_global = spec['eligible']
        rna_global = spec['rna']
        structure_global = spec['structure']
        idx = np.flatnonzero(eligible_global)
        # Genomic order is essential: rolling elsewhere would destroy spatial runs.
        idx = idx[np.argsort(g.tss.to_numpy()[idx], kind='stable')]
        genes = all_gene_ids[idx]
        operons = all_operons[idx]
        regions = all_regions[idx]
        rna = rna_global[idx]
        structure = structure_global[idx]
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
        gene_index = {gene: i for i, gene in enumerate(genes)}
        rr, cc = [], []
        for term in terms:
            for gene in term2genes[term]:
                if gene in gene_index:
                    rr.append(gene_index[gene])
                    cc.append(term_index[term])
        ann = sparse.csr_matrix((np.ones(len(rr), dtype=np.int16), (rr, cc)),
                                shape=(len(genes), len(terms)))
        observed = np.asarray(ann[np.flatnonzero(query_mask)].sum(axis=0)).ravel().astype(int)

        # Every non-zero circular displacement is used exactly once.
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
            M2 = int(universe_drop.sum())
            N2 = int(query_drop.sum())
            n2 = int(target_drop.sum())
            k2 = int((query_drop & target_drop).sum())
            p_drop = hypergeom.sf(k2 - 1, M2, n2, N2) if k2 >= 1 else 1.0
            overlap_regions = [regions[gene_index[x]] for x in overlap]
            region_counts = pd.Series(overlap_regions).value_counts()
            rows_out.append(dict(
                gene_set=spec['gene_set'], database=db, term=term,
                name=names.get(term, term), overlap=k, term_size=n,
                query_size=N, background_size=M,
                fold_enrichment=(k / N) / (n / M),
                p_conditional_RNAplus=p_cond,
                p_gene_order_circular_shift=p_spatial,
                n_circular_shifts=len(genes) - 1,
                n_operons_overlap=len(op_counts), top_operon=top_op,
                top_operon_genes=int(op_counts.iloc[0]),
                top_operon_fraction=float(op_counts.iloc[0] / k),
                n_100kb_regions_overlap=len(region_counts),
                top_100kb_region=int(region_counts.index[0]),
                top_100kb_fraction=float(region_counts.iloc[0] / k),
                p_drop_top_operon=p_drop,
                genes=','.join(sorted(overlap)),
            ))
        result = pd.DataFrame(rows_out)
        if len(result):
            result['q_conditional_within_gene_set'] = bh(result.p_conditional_RNAplus)
            result['q_spatial_within_gene_set'] = bh(result.p_gene_order_circular_shift)
            combined_sets.append(result)
        print(db, spec['gene_set'], 'eligible', len(genes), 'RNA+ annotated', M,
              'joint', N, 'terms', len(result), 'shifts', len(genes) - 1, flush=True)

    combined = pd.concat(combined_sets, ignore_index=True)
    combined['q_conditional_all_gene_sets'] = bh(combined.p_conditional_RNAplus)
    combined['q_spatial_all_gene_sets'] = bh(combined.p_gene_order_circular_shift)
    combined = combined.sort_values(['gene_set', 'p_gene_order_circular_shift',
                                     'p_conditional_RNAplus'])
    combined.to_csv(TAB / f'{db}_conditional_spatial_enrichment.csv', index=False)

# A compact, predeclared set of pathways for figure planning and auditing.
keys = [
    ('Baseline_structure_RNA', 'KEGG', 'mtu01053'),
    ('Baseline_structure_RNA', 'GO_BP', 'GO:0046688'),
    ('Hypoxia_structure_RNA', 'GO_BP', 'GO:0046688'),
    ('Hypoxia_structure_RNA', 'GO_BP', 'GO:0006412'),
    ('Hypoxia_structure_RNA', 'KEGG', 'mtu03010'),
    ('Interaction_structure_hypoxia_RNA', 'GO_BP', 'GO:0042773'),
]
frames = []
for db in databases:
    x = pd.read_csv(TAB / f'{db}_conditional_spatial_enrichment.csv')
    frames.append(x)
all_results = pd.concat(frames, ignore_index=True)
key_table = pd.concat([
    all_results[(all_results.gene_set == gene_set) &
                (all_results.database == db) & (all_results.term == term)]
    for gene_set, db, term in keys
], ignore_index=True)
key_table.to_csv(TAB / 'key_pathway_conditional_spatial_sensitivity.csv', index=False)
print('\nKey pathway sensitivity:')
print(key_table[['gene_set', 'name', 'overlap', 'fold_enrichment',
                 'p_conditional_RNAplus', 'q_conditional_within_gene_set',
                 'p_gene_order_circular_shift', 'q_spatial_within_gene_set',
                 'top_operon', 'top_operon_fraction', 'p_drop_top_operon']].to_string(index=False))
print('\nWrote conditional and spatial sensitivity tables to', TAB)
