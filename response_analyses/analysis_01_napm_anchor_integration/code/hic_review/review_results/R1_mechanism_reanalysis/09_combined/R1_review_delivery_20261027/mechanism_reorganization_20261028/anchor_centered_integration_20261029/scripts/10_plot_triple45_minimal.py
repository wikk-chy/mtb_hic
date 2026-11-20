#!/usr/bin/env python3
"""Minimal GO and KEGG plots for 45 three-way intersection DEGs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
res = pd.read_csv(TAB / 'triple45_GO_KEGG_enrichment.csv')
res = res[res.gene_set.eq('triple45_vs_RNA_tested') & res.p_value.lt(.05)].copy()

INK = '#20262B'
MUTED = '#737B81'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11, 'axes.edgecolor': INK,
    'axes.linewidth': 1.0, 'xtick.color': INK,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def minimal_plot(data, title, color, stem, figsize):
    # Ascending fold produces the largest bar at the top of a horizontal bar plot.
    z = data.sort_values(['fold_enrichment', 'p_value'], ascending=[True, False]).copy()
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.28 if len(z) <= 4 else .20)
    y = np.arange(len(z))
    ax.barh(y, z.fold_enrichment, height=.56, color=color, edgecolor='none')
    ax.set_ylim(-.75, len(z)-.15)
    ax.set_yticks([])
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.tick_params(axis='x', labelsize=9, length=4, direction='out', pad=5)
    ax.set_xlabel('Fold enrichment', labelpad=10, color=INK)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK)
    ax.spines[['left', 'bottom']].set_linewidth(1.05)
    ax.grid(False)
    xmax = float(z.fold_enrichment.max())*1.22
    ax.set_xlim(0, xmax)
    # Reference-style on-bar labels and a quiet, aligned P-value column.
    for yi, row in zip(y, z.itertuples()):
        ax.text(.012, yi, row.name, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=10.2, color=INK)
        ax.text(.988, yi, f'P = {row.p_value:.2g}', transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=9.4, color=MUTED)
    fig.suptitle(title, x=.075, y=.945, ha='left', fontsize=14,
                 fontweight='normal', color=INK)
    fig.text(.075, .078,
             '45 NapM-bound, insulation-changed DEGs  ·  4,060 RNA-tested genes  ·  nominal P < 0.05',
             ha='left', fontsize=8.8, color=MUTED)
    fdr_text = ('No displayed term passes BH FDR < 0.05.'
                if not z.q_BH_all_tested_terms.lt(.05).any()
                else 'Filled bars: nominal P < 0.05; FDR values are reported in the source table.')
    fig.text(.075, .038, fdr_text, ha='left', fontsize=8.8, color=MUTED)
    fig.savefig(FIG / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    z.to_csv(TAB / f'{stem}_plot_source.csv', index=False)
    print(f'\n{stem}: {len(z)} terms')
    print(z[['database', 'term', 'name', 'fold_enrichment', 'p_value']].sort_values(
        'fold_enrichment', ascending=False).to_string(index=False))


minimal_plot(res[res.database.str.startswith('GO_')],
             'GO enrichment: nominal P < 0.05', '#7D99AB',
             'R1_triple45_GO_enrichment_minimal', (10.2, 6.2))
minimal_plot(res[res.database.eq('KEGG')],
             'KEGG enrichment: nominal P < 0.05', '#B89474',
             'R1_triple45_KEGG_enrichment_minimal', (10.2, 4.1))
