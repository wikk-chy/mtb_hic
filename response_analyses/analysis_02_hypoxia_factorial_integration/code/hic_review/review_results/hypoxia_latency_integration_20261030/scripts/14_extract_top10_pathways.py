#!/usr/bin/env python3
"""Extract the ten leading GO/KEGG terms per database for the 508-gene set."""
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030/pathway_enrichment_unresolved104'
res = pd.read_csv(OUT / 'GO_KEGG_enrichment.csv')
assert len(res) == 218
assert set(res.database) == {'GO_BP', 'GO_CC', 'GO_MF', 'KEGG'}
assert res.query_size.gt(0).all() and res.background_size.gt(0).all()

columns = ['rank', 'database', 'term', 'name', 'overlap', 'term_size', 'query_size',
           'background_size', 'fold_enrichment', 'p_value', 'q_BH_all_tested_terms',
           'n_target_bins', 'n_operons', 'genes']
top = (res[res.overlap.gt(0)].sort_values(['database', 'p_value', 'term'])
       .groupby('database', group_keys=False).head(10).copy())
top['rank'] = top.groupby('database').cumcount() + 1
top = top[columns].sort_values(['database', 'rank'])
top.to_csv(OUT / 'Top10_pathways_by_database.csv', index=False)
assert len(top) == 40
assert top.groupby('database').size().eq(10).all()
assert top.groupby('database')['rank'].apply(lambda x: x.tolist() == list(range(1, 11))).all()

overall = (res[res.overlap.gt(0)].sort_values(['p_value', 'database', 'term'])
           .head(10).copy())
overall['rank'] = range(1, 11)
overall = overall[columns]
overall.to_csv(OUT / 'Top10_pathways_overall.csv', index=False)


def table(df):
    lines = ['| Rank | Term | Name | Overlap/term size | Fold | P | BH Q | Genes |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in df.itertuples(index=False):
        lines.append(f'| {r.rank} | {r.term} | {r.name} | {r.overlap}/{r.term_size} | '
                     f'{r.fold_enrichment:.2f} | {r.p_value:.3e} | '
                     f'{r.q_BH_all_tested_terms:.3e} | {r.genes} |')
    return '\n'.join(lines) + '\n'


report = ['# 104 个 unresolved 区域 508 个基因：GO/KEGG Top 10\n',
          '按各数据库单侧超几何检验 P 值升序提取，仅纳入重叠基因数大于 0 的条目。'
          'Top 10 是排序结果，不代表 FDR 显著；本次四个数据库均无 BH Q < 0.05 条目。'
          '背景为同类 NapM 结合、可检出结构交互且局部 RNA 可检验区域的 677 个基因。\n']
for db in ['GO_BP', 'GO_CC', 'GO_MF', 'KEGG']:
    report.extend([f'## {db}\n', table(top[top.database.eq(db)])])
report.extend(['## 四个数据库合并 P 值 Top 10\n', table(overall)])
(OUT / 'Top10_pathways_report.md').write_text('\n'.join(report), encoding='utf-8')

print(top[['database', 'rank', 'term', 'name', 'overlap', 'term_size',
           'fold_enrichment', 'p_value', 'q_BH_all_tested_terms']].to_string(index=False))
print(f'\nOutputs: {OUT}')
