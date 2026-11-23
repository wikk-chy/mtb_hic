#!/usr/bin/env python3
"""Directional KO-recall test and pathway enrichment for changing-loop genes.

Primary pathway question: which functions are overrepresented among strict KO-vs-WT
DEGs whose TSS lies inside one of the 11 strongly changing WT-called loops?
The primary background is all RNA-tested genes. Two sensitivity analyses use
loop-region backgrounds. Overlapping loop intervals are deduplicated at gene level.
"""
from pathlib import Path
import json
import re
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
R2 = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
BASE = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
OUT = BASE / 'changed_loop_pathway_enrichment'
OUT.mkdir(parents=True, exist_ok=True)
DET = {'detected_increase', 'detected_decrease'}
CHANGE_THRESHOLD = 0.5


def bh(p):
    """Benjamini-Hochberg correction over every tested term."""
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


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


# ---------------------------------------------------------------------
# Recall: quantify the observed direction without forcing significance.
recall = pd.read_csv(BASE / 'loop_recall_sensitivity/wt_loop_ko_recall_per_loop.csv')
target = recall.n_bound_anchors.eq(2)
nonrecovery = ~recall.recovered_primary
table = np.array([
    [int((~target & ~nonrecovery).sum()), int((~target & nonrecovery).sum())],
    [int((target & ~nonrecovery).sum()), int((target & nonrecovery).sum())],
])
assert table.tolist() == [[14, 10], [6, 11]]
ref_rec, ref_non = table[0]
tgt_rec, tgt_non = table[1]
n_ref, n_tgt = table[0].sum(), table[1].sum()
p_ref, p_tgt = ref_non / n_ref, tgt_non / n_tgt
l_ref, u_ref = wilson(ref_non, n_ref)
l_tgt, u_tgt = wilson(tgt_non, n_tgt)
diff = p_tgt - p_ref
# Newcombe hybrid-score interval for the difference in non-recovery rates.
diff_lo = diff - np.sqrt((p_tgt - l_tgt) ** 2 + (u_ref - p_ref) ** 2)
diff_hi = diff + np.sqrt((u_tgt - p_tgt) ** 2 + (p_ref - l_ref) ** 2)
fisher_two = stats.fisher_exact(table, alternative='two-sided')
fisher_directional = stats.fisher_exact(table, alternative='greater')
try:
    conditional_or = stats.contingency.odds_ratio(table, kind='conditional')
    or_ci = conditional_or.confidence_interval(0.95)
    or_est, or_lo, or_hi = conditional_or.statistic, or_ci.low, or_ci.high
except Exception:
    or_est, or_lo, or_hi = np.nan, np.nan, np.nan
recall_test = pd.DataFrame([dict(
    tolerance_bins=1,
    target='two_NapM_occupied_anchors', reference='zero_or_one_occupied_anchor',
    target_recovered=int(tgt_rec), target_not_recovered=int(tgt_non), target_n=int(n_tgt),
    reference_recovered=int(ref_rec), reference_not_recovered=int(ref_non), reference_n=int(n_ref),
    target_recovery_rate=tgt_rec / n_tgt, reference_recovery_rate=ref_rec / n_ref,
    target_nonrecovery_rate=p_tgt, reference_nonrecovery_rate=p_ref,
    nonrecovery_difference=diff, newcombe95_ci_lo=diff_lo, newcombe95_ci_hi=diff_hi,
    sample_nonrecovery_odds_ratio=fisher_two.statistic,
    conditional_nonrecovery_odds_ratio=or_est,
    conditional_or95_ci_lo=or_lo, conditional_or95_ci_hi=or_hi,
    fisher_two_sided_p=fisher_two.pvalue,
    fisher_directional_greater_nonrecovery_p=fisher_directional.pvalue,
    interpretation='Lower observed recovery in double-bound loops, but not statistically significant; non-recovery is not proof of physical disappearance.'
)])
recall_test.to_csv(OUT / 'recall_directional_test.csv', index=False)

# ---------------------------------------------------------------------
# Gene assignment by TSS to loop intervals. The 11 loops are the same set
# used in the strong-change sensitivity analysis (39 loops with continuous O/E).
loops = pd.read_csv(BASE / 'loop_change_enrichment/per_loop_change_and_binding.csv')
assert len(loops) == 39 and loops.d_log2OE.notna().all()
changed = loops[loops.abs_d_log2OE >= CHANGE_THRESHOLD].copy()
assert len(changed) == 11
changed.to_csv(OUT / 'changed_loops.csv', index=False)

g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
assert g.gene_id.is_unique
# Existing class definition: RNA BH FDR < 0.05 and |log2FC| >= 1.
g['is_KO_DEG'] = g.class_KOvsWT.isin(DET)
for cls in DET:
    check = g.class_KOvsWT.eq(cls)
    assert (g.loc[check, 'padj_KOvsWT'] < 0.05).all()
    assert (g.loc[check, 'log2FC_KOvsWT'].abs() >= 1).all()
g['RNA_tested'] = g.log2FC_KOvsWT.notna() & g.padj_KOvsWT.notna()
g['direct_NapM'] = g.binding_class.isin(['promoter_bound', 'gene_body_bound'])


def assign_regions(frame, prefix):
    memberships = [[] for _ in range(len(g))]
    tss = g.tss.to_numpy()
    for row in frame.itertuples():
        inside = (tss >= row.center1) & (tss <= row.center2)
        for idx in np.flatnonzero(inside):
            memberships[idx].append(row.loop_id)
    g[f'in_{prefix}_loop'] = [bool(x) for x in memberships]
    g[f'{prefix}_loop_ids'] = [';'.join(x) for x in memberships]


assign_regions(loops, 'eligible')
assign_regions(changed, 'changed')
member_cols = [
    'gene_id', 'gene_name', 'start', 'end', 'strand', 'tss',
    'in_changed_loop', 'changed_loop_ids', 'in_eligible_loop', 'eligible_loop_ids',
    'RNA_tested', 'is_KO_DEG', 'log2FC_KOvsWT', 'padj_KOvsWT', 'class_KOvsWT',
    'direct_NapM', 'binding_class'
]
g[member_cols].to_csv(OUT / 'changed_loop_gene_membership.csv', index=False)

# Explicitly distinct questions/backgrounds, including directional follow-up.
changed_all = g.in_changed_loop
eligible_all = g.in_eligible_loop
changed_deg = g.in_changed_loop & g.is_KO_DEG & g.RNA_tested
specs = [
    dict(gene_set='changed_region_DEGs_genome_background',
         description='Primary: strict KO-vs-WT DEGs in 11 changing-loop regions vs all RNA-tested genes',
         query=changed_deg, universe=g.RNA_tested),
    dict(gene_set='changed_region_DEGs_up_genome_background',
         description='Directional follow-up: upregulated strict DEGs in changing-loop regions vs all RNA-tested genes',
         query=changed_deg & g.log2FC_KOvsWT.gt(0), universe=g.RNA_tested),
    dict(gene_set='changed_region_DEGs_down_genome_background',
         description='Directional follow-up: downregulated strict DEGs in changing-loop regions vs all RNA-tested genes',
         query=changed_deg & g.log2FC_KOvsWT.lt(0), universe=g.RNA_tested),
    dict(gene_set='changed_region_DEGs_loop_background',
         description='Sensitivity: strict KO-vs-WT DEGs in changing-loop regions vs RNA-tested genes in all 39 tested-loop regions',
         query=changed_deg, universe=eligible_all & g.RNA_tested),
    dict(gene_set='changed_region_all_genes_loop_background',
         description='Exploratory: all genes in changing-loop regions vs genes in all 39 tested-loop regions',
         query=changed_all, universe=eligible_all),
]
summary = []
for spec in specs:
    qmask = spec['query'] & spec['universe']
    umask = spec['universe']
    summary.append(dict(
        gene_set=spec['gene_set'], description=spec['description'],
        n_query_genes=int(qmask.sum()), n_universe_genes=int(umask.sum()),
        n_query_up=int((qmask & g.log2FC_KOvsWT.gt(0)).sum()),
        n_query_down=int((qmask & g.log2FC_KOvsWT.lt(0)).sum()),
        n_query_direct_NapM=int((qmask & g.direct_NapM).sum()),
        n_changing_loops=len(changed), change_threshold=CHANGE_THRESHOLD,
        region_assignment='TSS between loop anchor centers; genes deduplicated across overlapping loops',
        DEG_definition='Existing detected class: RNA BH FDR < 0.05 and |log2FC| >= 1'
    ))
pd.DataFrame(summary).to_csv(OUT / 'changed_loop_gene_set_summary.csv', index=False)

# ---------------------------------------------------------------------
# Annotation mappings cached in the project.
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

databases = {'GO_BP': (gaf, go_names), 'KEGG': (k2p[['gene', 'term']], kegg_names)}
for db, (mapping, names) in databases.items():
    term2genes = mapping.groupby('term').gene.apply(set).to_dict()
    annotated = set(mapping.gene)
    all_results = []
    for spec in specs:
        universe_genes = set(g.loc[spec['universe'], 'gene_id']) & annotated
        query_genes = set(g.loc[spec['query'], 'gene_id']) & universe_genes
        M, N = len(universe_genes), len(query_genes)
        assert N > 0 and M > N
        result_rows = []
        for term, genes in term2genes.items():
            target_genes = genes & universe_genes
            n = len(target_genes)
            if n < 3:
                continue
            overlap_genes = query_genes & target_genes
            k = len(overlap_genes)
            p = stats.hypergeom.sf(k - 1, M, n, N)
            result_rows.append(dict(
                gene_set=spec['gene_set'], database=db, term=term,
                name=names.get(term, term), overlap=k, term_size=n,
                query_size=N, background_size=M,
                gene_ratio=k / N, fold_enrichment=(k / N) / (n / M),
                p_value=p, genes=','.join(sorted(overlap_genes))))
        result = pd.DataFrame(result_rows)
        result['q_BH_all_tested_terms'] = bh(result.p_value)
        all_results.append(result)
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(['gene_set', 'p_value', 'term'])
    combined.to_csv(OUT / f'{db}_enrichment.csv', index=False)

# ---------------------------------------------------------------------
# Primary-result horizontal bars. If no term survives FDR, the figure says so
# explicitly rather than presenting nominal pathways as established enrichment.
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
primary = 'changed_region_DEGs_genome_background'
primary_summary = next(x for x in summary if x['gene_set'] == primary)
fig, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
fig.subplots_adjust(left=0.25, right=0.965, top=0.79, bottom=0.14, wspace=0.72)
colors = {'GO_BP': '#3C7F9E', 'KEGG': '#B26A3B'}
any_fdr = False
for ax, db in zip(axes, ['GO_BP', 'KEGG']):
    data = pd.read_csv(OUT / f'{db}_enrichment.csv')
    data = data[(data.gene_set == primary) & (data.overlap >= 2)].copy()
    any_fdr |= bool((data.q_BH_all_tested_terms < 0.05).any())
    data = data.sort_values(['p_value', 'term']).head(10)
    data = data.sort_values('p_value', ascending=False)
    if data.empty:
        ax.text(0.5, 0.5, 'No terms with at least two genes', transform=ax.transAxes,
                ha='center', va='center', color='#666666')
        ax.set_axis_off()
        continue
    y = np.arange(len(data))
    bars = ax.barh(y, data.fold_enrichment, color=colors[db], alpha=0.88, height=0.66)
    labels = [textwrap.fill(x, 38) for x in data.name]
    ax.set_yticks(y, labels)
    ax.set_xlabel('Fold enrichment')
    ax.axvline(1, color='#777777', linestyle=(0, (3, 3)), linewidth=0.8)
    ax.grid(axis='x', color='#E5E8EA', linewidth=0.7, zorder=0)
    ax.set_title(f"{db.replace('_', ' ')} enrichment", loc='left', fontweight='bold')
    xmax = max(1.2, float(data.fold_enrichment.max()) * 1.42)
    ax.set_xlim(0, xmax)
    for bar, row in zip(bars, data.itertuples()):
        ax.text(bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{int(row.overlap)} genes; q={row.q_BH_all_tested_terms:.3g}",
                va='center', fontsize=8.1, color='#333333')

n_deg = primary_summary['n_query_genes']
fig.suptitle('Functions of DEGs within 11 strongly changing loop regions',
             x=0.055, y=0.955, ha='left', fontsize=15, fontweight='bold')
fig.text(0.055, 0.905,
         f"{n_deg} strict KO-vs-WT DEGs; all RNA-tested genes as background; TSS-based region assignment",
         ha='left', fontsize=10, color='#4C5258')
fig.text(0.055, 0.865,
         'Top nominal terms with ≥2 genes; bars show fold enrichment and labels show gene count and BH-adjusted q.',
         ha='left', fontsize=9, color='#5D646B')
if not any_fdr:
    fig.text(0.055, 0.825, 'No GO-BP or KEGG term passes BH FDR < 0.05; interpret all displayed pathways as exploratory.',
             ha='left', fontsize=9.5, color='#9A3F36', fontweight='bold')
fig.text(0.055, 0.055,
         'DEG: RNA BH FDR < 0.05 and |log$_2$FC| ≥ 1. Strong loop change: |log$_2$(KO/WT O/E)| ≥ 0.5. '
         'Pathway enrichment does not by itself establish direct NapM regulation.',
         ha='left', fontsize=8.7, color='#4C5258')
fig.savefig(OUT / 'changed_loop_DEG_GO_KEGG_enrichment.pdf', bbox_inches='tight')
fig.savefig(OUT / 'changed_loop_DEG_GO_KEGG_enrichment.png', dpi=400, bbox_inches='tight')

# Compact response-ready figure: FDR-significant terms only, with RNA directions.
gene_fc = g.set_index('gene_id').log2FC_KOvsWT.to_dict()
fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9))
fig.subplots_adjust(left=0.24, right=0.965, top=0.74, bottom=0.20, wspace=0.67)
fig_records = []
for ax, db in zip(axes, ['GO_BP', 'KEGG']):
    data = pd.read_csv(OUT / f'{db}_enrichment.csv')
    data = data[(data.gene_set == primary) & (data.overlap >= 2) &
                (data.q_BH_all_tested_terms < 0.05)].copy()
    data = data.sort_values(['p_value', 'term'], ascending=[False, True])
    if data.empty:
        ax.text(0.5, 0.5, 'No FDR-significant terms', transform=ax.transAxes,
                ha='center', va='center', color='#666666')
        ax.set_axis_off()
        continue
    y = np.arange(len(data))
    bars = ax.barh(y, data.fold_enrichment, color=colors[db], alpha=0.90, height=0.62)
    ax.set_yticks(y, [textwrap.fill(x, 38) for x in data.name])
    ax.set_xlabel('Fold enrichment')
    ax.axvline(1, color='#777777', linestyle=(0, (3, 3)), linewidth=0.8)
    ax.grid(axis='x', color='#E5E8EA', linewidth=0.7, zorder=0)
    ax.set_title(f"{db.replace('_', ' ')}: BH FDR < 0.05", loc='left', fontweight='bold')
    xmax = max(1.2, float(data.fold_enrichment.max()) * 1.62)
    ax.set_xlim(0, xmax)
    for bar, row in zip(bars, data.itertuples()):
        pathway_genes = row.genes.split(',')
        n_up = sum(gene_fc[gene] > 0 for gene in pathway_genes)
        n_down = sum(gene_fc[gene] < 0 for gene in pathway_genes)
        ax.text(bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{int(row.overlap)} genes ({n_up} up/{n_down} down); q={row.q_BH_all_tested_terms:.3g}",
                va='center', fontsize=8.5, color='#333333')
        fig_records.append(dict(database=db, term=row.term, name=row.name,
                                overlap=int(row.overlap), n_up=n_up, n_down=n_down,
                                fold_enrichment=row.fold_enrichment, p_value=row.p_value,
                                q_BH_all_tested_terms=row.q_BH_all_tested_terms, genes=row.genes))
pd.DataFrame(fig_records).to_csv(OUT / 'FDR_significant_pathways_with_direction.csv', index=False)
fig.suptitle('FDR-significant pathways among DEGs in changing-loop regions',
             x=0.055, y=0.955, ha='left', fontsize=15, fontweight='bold')
fig.text(0.055, 0.885,
         f"{n_deg} strict KO-vs-WT DEGs in 11 loop regions; all RNA-tested genes as background",
         ha='left', fontsize=10, color='#4C5258')
fig.text(0.055, 0.835,
         'Bars show fold enrichment; labels report pathway-gene RNA directions and BH-adjusted q.',
         ha='left', fontsize=9, color='#5D646B')
fig.text(0.055, 0.065,
         'DEG: RNA BH FDR < 0.05 and |log$_2$FC| ≥ 1. Strong loop change: |log$_2$(KO/WT O/E)| ≥ 0.5. '
         'Regional enrichment does not establish direct NapM regulation.',
         ha='left', fontsize=8.7, color='#4C5258')
fig.savefig(OUT / 'changed_loop_DEG_GO_KEGG_FDR_significant.pdf', bbox_inches='tight')
fig.savefig(OUT / 'changed_loop_DEG_GO_KEGG_FDR_significant.png', dpi=400, bbox_inches='tight')
plt.close(fig)

metadata = dict(
    n_eligible_loops=len(loops), n_changing_loops=len(changed),
    changing_loop_ids=changed.loop_id.tolist(), change_threshold=CHANGE_THRESHOLD,
    region_assignment='Gene TSS between loop anchor centers; one gene counted once even if loops overlap',
    DEG_definition='Existing detected_increase/detected_decrease: RNA BH FDR < 0.05 and |log2FC| >= 1',
    primary_gene_set=primary,
    primary_background='All RNA-tested genes, restricted to genes annotated in each database',
    multiple_testing='BH within each gene set and database over every term with >=3 background genes, including zero-overlap terms',
    display_rule='Exploratory figure: top 10 nominal terms with >=2 genes; response figure: only BH FDR <0.05 terms with RNA-direction counts',
    directional_follow_up='Separate upregulated and downregulated DEG enrichment tables use all RNA-tested genes as background',
    recall_primary_tolerance_bins=1,
    recall_caveat='Binary caller non-recovery is not equivalent to physical loop disappearance; loops can share anchors'
)
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

print('Recall directional test:')
print(recall_test.to_string(index=False))
print('\nChanged loops:', ', '.join(changed.loop_id))
print('\nGene-set summary:')
print(pd.DataFrame(summary).to_string(index=False))
for db in ['GO_BP', 'KEGG']:
    data = pd.read_csv(OUT / f'{db}_enrichment.csv')
    print(f'\n{db} primary results (overlap >= 2):')
    x = data[(data.gene_set == primary) & (data.overlap >= 2)].sort_values('p_value')
    print(x[['name', 'overlap', 'term_size', 'fold_enrichment', 'p_value',
             'q_BH_all_tested_terms', 'genes']].head(15).to_string(index=False))
    print('FDR < 0.05 terms:', int((x.q_BH_all_tested_terms < 0.05).sum()))
print('\nWrote outputs to', OUT)
