#!/usr/bin/env python3
"""GO/KEGG enrichment for genes in the 104 unresolved NapM structure/RNA bins.

The primary background is deliberately locus-matched: RNA-complete genes in
NapM-bound bins with detected genotype x hypoxia insulation interactions and
local RNA tests. Direct GO/KEGG annotations are used without GO propagation.
"""
from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
BASE = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC = BASE / 'source_data'
OUT = BASE / 'pathway_enrichment_unresolved104'
CACHE = ROOT / 'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
OPERON = ROOT / 'hic_review/review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv'
OUT.mkdir(parents=True, exist_ok=True)
DET = {'detected_increase', 'detected_decrease'}
TARGET_CLASS = 'Mixed_or_unresolved_NapM_structure_RNA'
RNA_COLS = ['hyp_response_wt', 'padj_hyp', 'hyp_response_ko', 'padj_hyp_ko',
            'interaction', 'padj_int']

# ----------------------------- gene sets ---------------------------------
b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
assert not g.gene_id.duplicated().any()
local_rna_tested = b[['RNA_WT_hypoxia_n_tested_genes', 'RNA_KO_hypoxia_n_tested_genes',
                      'RNA_interaction_n_tested_genes']].min(axis=1).fillna(0).gt(0)
target_bins = set(b.loc[b.hypoxia_response_evidence_class.eq(TARGET_CLASS), 'bin'].astype(int))
background_bins = set(b.loc[b.NapM_bound & b.interaction_ins_class.isin(DET) &
                            local_rna_tested, 'bin'].astype(int))
assert len(target_bins) == 104
assert target_bins <= background_bins
rna_complete = np.isfinite(g[RNA_COLS]).all(axis=1)
member = g.loc[rna_complete & g.bin_tss.isin(background_bins)].copy()
member['bin_tss'] = member.bin_tss.astype(int)
member['is_target'] = member.bin_tss.isin(target_bins)
member['is_background'] = True
query = set(member.loc[member.is_target, 'gene_id'])
universe = set(member.gene_id)
assert len(member) == len(universe)
assert query <= universe
assert member.loc[member.is_target, 'bin_tss'].nunique() == 104

# Operon annotations; unannotated genes are treated as singleton units only in
# leave-one-operon/singleton-out sensitivity analysis.
op = pd.read_csv(OPERON)
gene_to_operons = {}
operon_to_genes = {}
for row in op.itertuples(index=False):
    genes = set(ast.literal_eval(row.gene_list))
    operon_to_genes[str(row.operon_id)] = genes
    for gene in genes:
        gene_to_operons.setdefault(gene, set()).add(str(row.operon_id))
member['operon_ids'] = member.gene_id.map(
    lambda x: ';'.join(sorted(gene_to_operons.get(x, set()))))
member['n_operon_annotations'] = member.gene_id.map(
    lambda x: len(gene_to_operons.get(x, set())))
member['sensitivity_unit'] = member.apply(
    lambda r: r.operon_ids if r.operon_ids else 'unassigned_singleton:' + r.gene_id, axis=1)
member_cols = ['gene_id', 'gene_name', 'start', 'end', 'strand', 'bin_tss',
               'is_target', 'is_background', 'operon_ids', 'n_operon_annotations',
               'sensitivity_unit'] + RNA_COLS
member[member_cols].sort_values(['bin_tss', 'start', 'gene_id']).to_csv(
    OUT / 'gene_membership.csv', index=False)
gene_info = member.set_index('gene_id')

# ----------------------------- annotations -------------------------------
k2p = pd.read_csv(CACHE / 'kegg_gene2pathway.tsv', sep='\t', header=None,
                  names=['gene', 'term'])
k2p['gene'] = k2p.gene.str.replace('mtu:', '', regex=False)
k2p['term'] = k2p.term.str.replace('path:', '', regex=False)
pn = pd.read_csv(CACHE / 'kegg_pathway_names.tsv', sep='\t', header=None,
                 names=['term', 'name'])
pn['name'] = pn.name.str.replace(' - Mycobacterium tuberculosis H37Rv', '', regex=False)
kegg_names = dict(zip(pn.term, pn.name))

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
in_term = False

def commit_go_term():
    if in_term and cid and cname:
        go_names[cid] = cname
        go_namespace[cid] = cns

with open(CACHE / 'go-basic.obo') as handle:
    for line in handle:
        line = line.rstrip('\n')
        if line in {'[Term]', '[Typedef]'}:
            commit_go_term()
            in_term = line == '[Term]'
            cid = cname = cns = None
        elif not in_term:
            continue
        elif line.startswith('id: '):
            cid = line[4:]
        elif line.startswith('name: '):
            cname = line[6:]
        elif line.startswith('namespace: '):
            cns = line[11:]
        elif line.startswith('is_obsolete: true') and cid:
            obsolete.add(cid)
commit_go_term()

aspect_to_db = {'biological_process': 'GO_BP', 'cellular_component': 'GO_CC',
                'molecular_function': 'GO_MF'}
dbs = {'KEGG': (k2p[['gene', 'term']].drop_duplicates(), kegg_names)}
for aspect, db in aspect_to_db.items():
    terms = {term for term, ns in go_namespace.items()
             if ns == aspect and term not in obsolete}
    dbs[db] = (gaf[gaf.term.isin(terms)][['gene', 'term']].drop_duplicates(), go_names)

# ----------------------------- statistics --------------------------------
def bh(p_values):
    p = np.asarray(p_values, float)
    if len(p) == 0:
        return p
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def enrichment(query_genes, universe_genes, mapping, names, database):
    """One-sided hypergeometric test; BH includes zero-overlap terms."""
    annotated_universe = set(universe_genes) & set(mapping.gene)
    q = set(query_genes) & annotated_universe
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
        overlap_bins = set(gene_info.loc[sorted(overlap), 'bin_tss']) if k else set()
        overlap_operons = set()
        for gene in overlap:
            overlap_operons.update(gene_to_operons.get(gene, set()))
        rows.append(dict(
            database=database, term=term, name=names.get(term, term),
            overlap=k, term_size=n, query_size=N, background_size=M,
            expected_overlap=N * n / M,
            fold_enrichment=(k / N) / (n / M),
            p_value=hypergeom.sf(k - 1, M, n, N),
            genes=','.join(sorted(overlap)),
            gene_names=','.join(gene_info.loc[sorted(overlap), 'gene_name'].fillna('')) if k else '',
            n_target_bins=len(overlap_bins),
            target_bins=','.join(map(str, sorted(overlap_bins))),
            n_operons=len(overlap_operons),
            operons=','.join(sorted(overlap_operons)),
            n_unassigned_overlap=sum(not gene_to_operons.get(x) for x in overlap),
        ))
    result = pd.DataFrame(rows)
    result['q_BH_all_tested_terms'] = bh(result.p_value)
    result['n_terms_tested'] = len(result)
    return result, annotated_universe, q


results = []
annotated_sets = {}
for database, (mapping, names) in dbs.items():
    result, annotated_universe, annotated_query = enrichment(
        query, universe, mapping, names, database)
    annotated_sets[database] = (annotated_universe, annotated_query)
    results.append(result)
res = pd.concat(results, ignore_index=True)
res = res.sort_values(['database', 'p_value', 'term']).reset_index(drop=True)

# Independent numerical and membership checks.
for row in res.itertuples(index=False):
    overlap = set(filter(None, str(row.genes).split(',')))
    mapping, _ = dbs[row.database]
    annotated_universe, annotated_query = annotated_sets[row.database]
    assert overlap <= annotated_query <= query
    assert row.background_size == len(annotated_universe)
    assert row.query_size == len(annotated_query)
    term_genes = set(mapping.loc[mapping.term.eq(row.term), 'gene']) & annotated_universe
    assert row.term_size == len(term_genes)
    assert row.overlap == len(overlap)
    expected_p = hypergeom.sf(row.overlap - 1, row.background_size,
                              row.term_size, row.query_size)
    assert np.isclose(row.p_value, expected_p, rtol=1e-12, atol=1e-15)
assert np.isfinite(res[['p_value', 'q_BH_all_tested_terms']]).all().all()
assert res[['p_value', 'q_BH_all_tested_terms']].apply(
    lambda x: x.between(0, 1).all()).all()

res.to_csv(OUT / 'GO_KEGG_enrichment.csv', index=False)
res[res.q_BH_all_tested_terms.lt(.05)].to_csv(OUT / 'FDR_significant.csv', index=False)
res[res.p_value.lt(.05)].to_csv(OUT / 'nominal_P_significant.csv', index=False)

summary = res.groupby('database', as_index=False).agg(
    query_annotated=('query_size', 'first'),
    background_annotated=('background_size', 'first'),
    n_terms_tested=('term', 'size'),
    nominal_P_0_05=('p_value', lambda x: int((x < .05).sum())),
    FDR_0_05=('q_BH_all_tested_terms', lambda x: int((x < .05).sum())),
    minimum_P=('p_value', 'min'), minimum_q=('q_BH_all_tested_terms', 'min'))
summary.insert(0, 'query_genes_before_annotation', len(query))
summary.insert(1, 'background_genes_before_annotation', len(universe))
summary.insert(2, 'target_bins', len(target_bins))
summary.insert(3, 'background_bins', len(background_bins))
summary.to_csv(OUT / 'summary.csv', index=False)

# ----------------------------- sensitivity -------------------------------
# All FDR-significant terms plus the five leading overlapping terms per database.
selected = []
for database, group in res.groupby('database'):
    sig = group[group.q_BH_all_tested_terms.lt(.05)]
    top = group[group.overlap.gt(0)].sort_values(['p_value', 'term']).head(5)
    selected.append(pd.concat([sig, top]).drop_duplicates(['database', 'term']))
selected = pd.concat(selected, ignore_index=True)
selected_keys = set(zip(selected.database, selected.term))

# Removal units contain only target genes; background stays fixed.
bin_units = {f'bin:{x}': set(member.loc[member.bin_tss.eq(x) & member.is_target, 'gene_id'])
             for x in sorted(target_bins)}
operon_units = {}
for operon_id, genes in operon_to_genes.items():
    removed = genes & query
    if removed:
        operon_units[f'operon:{operon_id}'] = removed
for gene in sorted(query):
    if not gene_to_operons.get(gene):
        operon_units[f'unassigned_singleton:{gene}'] = {gene}

sensitivity_rows = []
for database, (mapping, names) in dbs.items():
    db_selected = sorted(term for db, term in selected_keys if db == database)
    if not db_selected:
        continue
    for unit_type, units in [('leave_one_bin_out', bin_units),
                             ('leave_one_operon_or_singleton_out', operon_units)]:
        for unit, removed_genes in units.items():
            reduced_query = query - removed_genes
            reduced, _, reduced_annotated_query = enrichment(
                reduced_query, universe, mapping, names, database)
            reduced = reduced.set_index('term')
            for term in db_selected:
                original = res[(res.database.eq(database)) & (res.term.eq(term))].iloc[0]
                row = reduced.loc[term]
                original_overlap = set(filter(None, str(original.genes).split(',')))
                sensitivity_rows.append(dict(
                    database=database, term=term, name=original['name'],
                    sensitivity=unit_type, removed_unit=unit,
                    removed_target_genes=','.join(sorted(removed_genes)),
                    n_removed_target_genes=len(removed_genes),
                    removed_overlap_genes=','.join(sorted(removed_genes & original_overlap)),
                    n_removed_overlap_genes=len(removed_genes & original_overlap),
                    original_overlap=original.overlap,
                    remaining_overlap=row.overlap,
                    original_query_size=original.query_size,
                    remaining_query_size=len(reduced_annotated_query),
                    original_p=original.p_value, remaining_p=row.p_value,
                    original_q=original.q_BH_all_tested_terms,
                    remaining_q=row.q_BH_all_tested_terms,
                    retains_nominal_P=row.p_value < .05,
                    retains_FDR=row.q_BH_all_tested_terms < .05,
                ))

sensitivity = pd.DataFrame(sensitivity_rows)
sensitivity.to_csv(OUT / 'sensitivity.csv', index=False)

# Add compact sensitivity summaries to the selected-term table.
selected = selected.copy()
selected['selection_reason'] = pd.Series(index=selected.index, dtype='object')
for col in ['n_bin_removals', 'n_operon_or_singleton_removals',
            'n_bin_removals_losing_nominal_P', 'n_operon_removals_losing_nominal_P',
            'n_bin_removals_losing_FDR', 'n_operon_removals_losing_FDR',
            'worst_leave_bin_q', 'worst_leave_operon_q',
            'robust_FDR_to_all_single_bin_removals',
            'robust_FDR_to_all_single_operon_or_singleton_removals']:
    selected[col] = np.nan
for idx, row in selected.iterrows():
    s = sensitivity[(sensitivity.database.eq(row.database)) & (sensitivity.term.eq(row.term))]
    sb = s[s.sensitivity.eq('leave_one_bin_out')]
    so = s[s.sensitivity.eq('leave_one_operon_or_singleton_out')]
    selected.at[idx, 'selection_reason'] = 'FDR_significant' if row.q_BH_all_tested_terms < .05 else 'top5_exploratory'
    selected.at[idx, 'n_bin_removals'] = len(sb)
    selected.at[idx, 'n_operon_or_singleton_removals'] = len(so)
    selected.at[idx, 'n_bin_removals_losing_nominal_P'] = int((~sb.retains_nominal_P).sum())
    selected.at[idx, 'n_operon_removals_losing_nominal_P'] = int((~so.retains_nominal_P).sum())
    selected.at[idx, 'n_bin_removals_losing_FDR'] = int((~sb.retains_FDR).sum())
    selected.at[idx, 'n_operon_removals_losing_FDR'] = int((~so.retains_FDR).sum())
    selected.at[idx, 'worst_leave_bin_q'] = sb.remaining_q.max()
    selected.at[idx, 'worst_leave_operon_q'] = so.remaining_q.max()
    if row.q_BH_all_tested_terms < .05:
        selected.at[idx, 'robust_FDR_to_all_single_bin_removals'] = bool(sb.retains_FDR.all())
        selected.at[idx, 'robust_FDR_to_all_single_operon_or_singleton_removals'] = bool(so.retains_FDR.all())
selected = selected.sort_values(['database', 'p_value', 'term'])
selected.to_csv(OUT / 'selected_terms_with_sensitivity_summary.csv', index=False)

# ----------------------------- report ------------------------------------
def fmt_num(x):
    return f'{x:.2e}'


def markdown_table(df, columns, headers=None):
    if df.empty:
        return '（无）\n'
    headers = headers or columns
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if col in {'p_value', 'q_BH_all_tested_terms', 'original_q', 'worst_leave_bin_q',
                       'worst_leave_operon_q'} and pd.notna(value):
                value = fmt_num(value)
            elif col == 'fold_enrichment':
                value = f'{value:.2f}'
            value = str(value).replace('|', '\\|').replace('\n', ' ')
            values.append(value)
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


sig = res[res.q_BH_all_tested_terms.lt(.05)].sort_values(['database', 'p_value'])
top = res[res.overlap.gt(0)].groupby('database', group_keys=False).head(5)
report = [
    '# 104 个 NapM structure/RNA unresolved 区域的 GO/KEGG 富集分析\n',
    '## 分析设计\n',
    f'- 目标集：{len(target_bins)} 个 `{TARGET_CLASS}` bin 内，WT 缺氧、KO 缺氧和 genotype × hypoxia 交互三组 RNA 检验均完整的 **{len(query)} 个基因**。',
    f'- 匹配背景：NapM 结合、存在可检出结构交互且局部有 RNA 检验的 {len(background_bins)} 个 bin 内，满足相同基因级完整性要求的 **{len(universe)} 个基因**。目标集是背景集的子集。',
    '- 注释：GO 使用 GAF 直接注释并排除 obsolete term，不向祖先节点传播；KEGG 使用直接基因—通路映射。',
    '- 统计：单侧超几何富集；仅检验背景中至少 3 个基因的条目；GO BP、GO CC、GO MF、KEGG 分别对全部受试条目（包括零重叠条目）做 BH 校正。',
    '- 敏感性：对 FDR 显著条目及每个数据库前 5 个有重叠条目，逐一剔除目标 bin、操纵子；无操纵子注释的基因按单基因单元剔除。背景保持不变，每次重新计算全数据库 BH。\n',
    '## 各数据库检验规模\n',
    markdown_table(summary, ['database', 'query_annotated', 'background_annotated',
                             'n_terms_tested', 'nominal_P_0_05', 'FDR_0_05',
                             'minimum_P', 'minimum_q'],
                   ['数据库', '目标注释基因', '背景注释基因', '检验条目', 'P < 0.05',
                    'FDR < 0.05', '最小 P', '最小 Q']),
    '\n## 富集结果\n'
]
if sig.empty:
    report.append('**四个数据库均未检出 BH 校正后显著富集（FDR < 0.05）。** 下表仅列出每个数据库排名靠前的探索性条目，不能据此作确定性通路或机制结论。\n')
else:
    report.append(f'共检出 **{len(sig)} 个 FDR < 0.05 条目**。这些条目表示目标区域基因的功能组成相对匹配背景存在富集，不等同于 NapM 直接调控。\n')
    report.append(markdown_table(sig, ['database', 'term', 'name', 'overlap', 'term_size',
                                       'fold_enrichment', 'p_value', 'q_BH_all_tested_terms',
                                       'n_target_bins', 'n_operons', 'genes'],
                                 ['数据库', '条目', '名称', '重叠', '条目大小', '富集倍数',
                                  'P', 'BH Q', '目标 bin 数', '操纵子数', '重叠基因']))
report.append('\n### 每个数据库排名靠前的有重叠条目\n')
report.append(markdown_table(top, ['database', 'term', 'name', 'overlap', 'term_size',
                                   'fold_enrichment', 'p_value', 'q_BH_all_tested_terms',
                                   'n_target_bins', 'n_operons', 'genes'],
                             ['数据库', '条目', '名称', '重叠', '条目大小', '富集倍数',
                              'P', 'BH Q', '目标 bin 数', '操纵子数', '重叠基因']))
report.append('\n## 区域与操纵子敏感性\n')
if sig.empty:
    report.append('由于没有 FDR 显著条目，不存在可称为“校正后稳健”的通路。排名靠前条目的逐区域／逐操纵子结果已完整输出；nominal P 仅用于探索，不能替代 FDR 显著性。\n')
    nominal_selected = selected[selected.p_value.lt(.05)]
    if not nominal_selected.empty:
        report.append('名义显著条目的敏感性如下；失去 nominal P < 0.05 即记为不保留。\n')
        report.append(markdown_table(nominal_selected,
                                     ['database', 'term', 'name', 'p_value',
                                      'q_BH_all_tested_terms', 'n_bin_removals_losing_nominal_P',
                                      'n_operon_removals_losing_nominal_P'],
                                     ['数据库', '条目', '名称', '原始 P', '原始 BH Q',
                                      '失去名义显著的 bin 剔除数',
                                      '失去名义显著的操纵子／单基因剔除数']))
else:
    sens_summary = selected[selected.q_BH_all_tested_terms.lt(.05)]
    report.append(markdown_table(sens_summary, ['database', 'term', 'name',
                                                'n_bin_removals_losing_FDR',
                                                'n_operon_removals_losing_FDR',
                                                'worst_leave_bin_q', 'worst_leave_operon_q',
                                                'robust_FDR_to_all_single_bin_removals',
                                                'robust_FDR_to_all_single_operon_or_singleton_removals'],
                                 ['数据库', '条目', '名称', '失去 FDR 的 bin 剔除数',
                                  '失去 FDR 的操纵子剔除数', '最差 bin Q', '最差操纵子 Q',
                                  '所有单 bin 剔除后仍显著', '所有单操纵子剔除后仍显著']))
report.extend([
    '\n## 机制解释边界\n',
    '- 该分析检验的是 104 个 unresolved 区域的局部基因是否具有共同功能主题；它不能把这些区域自动归入“直接结构作用”或“间接转录作用”。',
    '- 若存在稳健富集，可作为这些区域可能共享应激／转录程序的支持性线索；若无富集，则说明在当前基因集、注释和匹配背景下未检出共同通路，不能反推不存在生物学联系。',
    '- 富集本身不能区分 NapM 的直接作用、转录介导、操纵子共调控或偶然共变。尤其当结果由少数 bin 或单个操纵子驱动时，应避免上升为普遍机制。',
    '- `sensitivity.csv` 保留所有逐单元剔除结果；`selected_terms_with_sensitivity_summary.csv` 汇总重点条目的脆弱性。\n'
])
(OUT / '中文富集分析报告.md').write_text('\n'.join(report), encoding='utf-8')

metadata = dict(
    target_class=TARGET_CLASS, target_bins=len(target_bins), target_genes=len(query),
    background_bins=len(background_bins), background_genes=len(universe),
    target_bin_ids=sorted(target_bins), background_bin_ids=sorted(background_bins),
    gene_level_requirement='Finite hyp_response_wt/padj_hyp, hyp_response_ko/padj_hyp_ko and interaction/padj_int',
    background_rule='NapM_bound and detected_increase/decrease interaction_ins_class and min local RNA tested count > 0',
    go_annotation='Direct GAF annotations; obsolete terms excluded; no ancestor propagation',
    go_gaf_date='2026-07-28', kegg_annotation='Direct KEGG gene-pathway mappings',
    test='One-sided hypergeometric; background terms with >=3 genes; BH per database including zero-overlap terms',
    sensitivity='Leave one target bin or one operon out; unannotated genes as singleton units; background fixed; full per-database BH recomputed',
    interpretation='Functional enrichment is supportive evidence only and cannot establish direct NapM action, transcriptional mediation, or causality.'
)
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

print(summary.to_string(index=False))
print('\nFDR SIGNIFICANT TERMS')
cols = ['database', 'term', 'name', 'overlap', 'term_size', 'query_size', 'background_size',
        'fold_enrichment', 'p_value', 'q_BH_all_tested_terms', 'n_target_bins', 'n_operons', 'genes']
print(sig[cols].to_string(index=False) if not sig.empty else 'None')
print('\nTOP OVERLAPPING TERMS PER DATABASE')
print(top[cols].to_string(index=False))
print(f'\nOutputs: {OUT}')
