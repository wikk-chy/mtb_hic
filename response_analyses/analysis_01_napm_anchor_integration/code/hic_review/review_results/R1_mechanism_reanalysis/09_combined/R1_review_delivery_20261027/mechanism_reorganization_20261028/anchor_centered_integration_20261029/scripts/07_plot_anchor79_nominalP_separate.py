#!/usr/bin/env python3
"""Separate concise GO and KEGG nominal-P enrichment plots for 79 anchor DEGs."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)

res = pd.read_csv(TAB / 'anchor79_GO_KEGG_enrichment.csv')
primary = res[(res.gene_set.eq('anchor79_vs_RNA_tested')) & res.p_value.lt(.05)].copy()
assert len(primary) == 12
assert not primary.q_BH_all_tested_terms.lt(.05).any()

GO_COLORS = {'GO_BP': '#287D8E', 'GO_CC': '#B47846', 'GO_MF': '#6B4C9A'}
GO_LABELS = {'GO_BP': 'Biological process', 'GO_CC': 'Cellular component', 'GO_MF': 'Molecular function'}
KEGG_COLOR = '#3D6E9E'
GREY = '#606B73'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10.5, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def plot_enrichment(data, databases, title, output_stem, colors, figsize, legend=False):
    z = data[data.database.isin(databases)].sort_values(['p_value', 'term'], ascending=[False, True]).copy()
    expected = 8 if databases != ['KEGG'] else 4
    assert len(z) == expected
    # Keep a compact, auditable source table containing the values actually plotted.
    source_cols = ['database', 'term', 'name', 'overlap', 'term_size', 'query_size',
                   'background_size', 'fold_enrichment', 'p_value', 'genes']
    z[source_cols].sort_values(['database', 'p_value', 'term']).to_csv(
        TAB / f'{output_stem}_plot_source.csv', index=False)

    fig, ax = plt.subplots(figsize=figsize)
    compact = figsize[1] < 6
    fig.subplots_adjust(left=.40, right=.965, top=.70 if compact else .77, bottom=.20)
    y = np.arange(len(z))
    values = -np.log10(z.p_value.to_numpy())
    bar_colors = [colors[d] for d in z.database]
    ax.barh(y, values, height=.60, color=bar_colors, alpha=.92)
    ax.axvline(-np.log10(.05), color='#77828A', lw=.9, ls=(0, (3, 3)))
    ax.text(-np.log10(.05)+.035, len(z)-.25, 'P = 0.05', fontsize=8.2,
            color=GREY, va='top', ha='left')
    ax.set_yticks(y, [textwrap.fill(name, 42) for name in z.name], fontsize=9.2)
    xmax = max(3.45, float(values.max())+.52)
    ax.set_xlim(0, xmax)
    for yi, value, p in zip(y, values, z.p_value):
        ax.text(value+.045, yi, f'P = {p:.2g}', va='center', ha='left',
                fontsize=8.8, color='#354049')
    ax.set_xlabel('−log₁₀(nominal P)')
    ax.set_xticks(np.arange(0, np.floor(xmax)+.1, .5))
    ax.tick_params(axis='x', labelsize=8.5)
    ax.grid(axis='x', color='#E7ECEF', lw=.65, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(handles=[Patch(facecolor=colors[d], label=GO_LABELS[d]) for d in databases],
                  frameon=False, loc='lower right', fontsize=8.4, ncol=1,
                  handlelength=1.2, borderaxespad=.2)
    fig.suptitle(title, x=.055, y=.965, ha='left', fontsize=15, fontweight='bold')
    fig.text(.055, .865 if compact else .900, '79 changed-anchor DEGs  |  Background: 4,060 RNA-tested genes',
             ha='left', fontsize=9.5, color=GREY)
    fig.text(.055, .805 if compact else .855, 'Hypergeometric enrichment; only nominal P < 0.05 terms are displayed',
             ha='left', fontsize=9.0, color=GREY)
    fig.text(.055, .065,
             'Nominal P values are exploratory. None of the displayed terms passes BH FDR < 0.05.',
             ha='left', fontsize=8.7, color=GREY)
    fig.savefig(FIG / f'{output_stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{output_stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\n{output_stem}: {len(z)} terms')
    print(z[['database', 'term', 'name', 'overlap', 'p_value']].sort_values('p_value').to_string(index=False))


plot_enrichment(primary, ['GO_BP', 'GO_CC', 'GO_MF'],
                'GO enrichment of changed-anchor DEGs',
                'R1_anchor79_GO_enrichment_nominalP', GO_COLORS, (9.8, 6.7), legend=True)
plot_enrichment(primary, ['KEGG'],
                'KEGG enrichment of changed-anchor DEGs',
                'R1_anchor79_KEGG_enrichment_nominalP', {'KEGG': KEGG_COLOR}, (9.8, 4.6))
