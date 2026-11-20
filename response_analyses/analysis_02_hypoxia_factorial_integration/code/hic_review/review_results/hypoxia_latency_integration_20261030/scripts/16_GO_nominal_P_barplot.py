#!/usr/bin/env python3
"""Categorical bar plot for GO terms with nominal P < 0.05."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030/pathway_enrichment_unresolved104'
res = pd.read_csv(OUT / 'GO_KEGG_enrichment.csv')
go = res[res.database.str.startswith('GO_') & res.p_value.lt(.05)].copy()
go['minus_log10_P'] = -np.log10(go.p_value)
go = go.sort_values(['database', 'p_value', 'term'])
assert len(go) == 1 and go.database.iloc[0] == 'GO_MF'
assert go.term.iloc[0] == 'GO:0022857'
go.to_csv(OUT / 'GO_P005_category_barplot_data.csv', index=False)

plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
colors = {'GO_BP': '#2878B5', 'GO_CC': '#D97706', 'GO_MF': '#4C9A2A'}
labels = {'GO_BP': 'Biological process', 'GO_CC': 'Cellular component',
          'GO_MF': 'Molecular function'}
fig, ax = plt.subplots(figsize=(6.8, 2.25))
y = np.arange(len(go))
ax.barh(y, go.minus_log10_P, color=[colors[x] for x in go.database], height=.48)
ax.set_yticks(y, [f"{r['name']}  ({labels[r.database]})" for _, r in go.iterrows()])
ax.axvline(-np.log10(.05), color='#7B858A', ls='--', lw=.8)
ax.set_xlim(0, max(1.65, go.minus_log10_P.max() * 1.22))
ax.set_xlabel('−log10(nominal P)')
ax.set_title('GO terms with nominal P < 0.05', loc='left', fontweight='bold')
for yi, (_, r) in zip(y, go.iterrows()):
    ax.text(r.minus_log10_P + .035, yi,
            f"P = {r.p_value:.3f}; {r.overlap} genes; BH Q = {r.q_BH_all_tested_terms:.3f}",
            va='center', fontsize=8)
ax.text(0, -.62, 'Nominal-P filter only; no GO term reaches BH FDR < 0.05',
        transform=ax.transAxes, fontsize=8, color='#A33F38')
fig.subplots_adjust(left=.40, right=.97, top=.79, bottom=.28)
for ext in ['pdf', 'svg']:
    fig.savefig(OUT / f'GO_P005_category_barplot.{ext}', bbox_inches='tight')
fig.savefig(OUT / 'GO_P005_category_barplot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(go[['database', 'term', 'name', 'overlap', 'p_value', 'q_BH_all_tested_terms']].to_string(index=False))
