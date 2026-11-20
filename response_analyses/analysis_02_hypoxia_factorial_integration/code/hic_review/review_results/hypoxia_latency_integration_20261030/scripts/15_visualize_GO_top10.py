#!/usr/bin/env python3
"""Bubble plot of the top ten GO terms in each namespace."""
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030/pathway_enrichment_unresolved104'
df = pd.read_csv(OUT / 'Top10_pathways_by_database.csv')
go = df[df.database.isin(['GO_BP', 'GO_CC', 'GO_MF'])].copy()
assert len(go) == 30 and go.groupby('database').size().eq(10).all()
go['gene_ratio_percent'] = 100 * go.overlap / go.query_size
go['minus_log10_P'] = -np.log10(go.p_value)
go.to_csv(OUT / 'GO_Top10_bubble_plot_data.csv', index=False)

plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
panels = [('GO_BP', 'A  Biological process'), ('GO_CC', 'B  Cellular component'),
          ('GO_MF', 'C  Molecular function')]
norm = plt.Normalize(go.minus_log10_P.min(), go.minus_log10_P.max())
cmap = plt.cm.viridis
for ax, (db, title) in zip(axes, panels):
    d = go[go.database.eq(db)].sort_values('rank', ascending=False)
    y = np.arange(len(d))
    labels = [f"{r['rank']}. " + textwrap.fill(r['name'], 48)
              for _, r in d.iterrows()]
    sizes = 38 + 7 * d.overlap.to_numpy()
    ax.scatter(d.gene_ratio_percent, y, s=sizes, c=d.minus_log10_P,
               cmap=cmap, norm=norm, alpha=.88, edgecolor='#263238', linewidth=.35)
    ax.set_yticks(y, labels)
    ax.set_xlabel('Genes in term / annotated query genes (%)')
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11)
    ax.grid(axis='x', color='#D9DEE2', lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, d.gene_ratio_percent.max() * 1.22)
    ax.set_ylim(-.8, 9.8)
    # Mark nominal P < 0.05 without implying FDR significance.
    for yi, (_, row) in zip(y, d.iterrows()):
        if row.p_value < .05:
            ax.text(row.gene_ratio_percent + d.gene_ratio_percent.max() * .055, yi,
                    '*', va='center', ha='center', fontsize=12, color='#B4443C')

fig.suptitle('GO enrichment in 508 genes from 104 unresolved bins',
             x=.32, y=.985, ha='left', fontsize=14, fontweight='bold')
fig.text(.32, .958, 'Top 10 P-ranked terms per namespace; no term reaches BH FDR < 0.05',
         ha='left', fontsize=9, color='#4E5A60')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cax = fig.add_axes([.935, .37, .016, .28])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('−log10(P)')
size_values = [5, 20, 50]
handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#7A9EBF',
                  markeredgecolor='#263238', markeredgewidth=.35,
                  markersize=np.sqrt(38 + 7 * v), label=str(v)) for v in size_values]
fig.legend(handles=handles, title='Overlapping genes', loc='lower center',
           bbox_to_anchor=(.56, .012), ncol=3, frameon=False,
           columnspacing=1.8, handletextpad=.4)
fig.text(.32, .012, '* Nominal P < 0.05 only', color='#B4443C', fontsize=8)
fig.subplots_adjust(left=.32, right=.90, top=.92, bottom=.09, hspace=.42)
for ext in ['pdf', 'svg']:
    fig.savefig(OUT / f'GO_Top10_bubble_plot.{ext}', bbox_inches='tight')
fig.savefig(OUT / 'GO_Top10_bubble_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(OUT / 'GO_Top10_bubble_plot.pdf')
