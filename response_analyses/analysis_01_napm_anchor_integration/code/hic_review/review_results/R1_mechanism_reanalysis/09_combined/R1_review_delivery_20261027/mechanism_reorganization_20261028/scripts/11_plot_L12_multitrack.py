#!/usr/bin/env python3
"""Genome-browser-style multitrack view of L12 and its his/arg driver loci."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrow
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
BASE = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
OUT = BASE / 'L12_multitrack'
OUT.mkdir(parents=True, exist_ok=True)

DET = {'detected_increase', 'detected_decrease'}
UP = '#C4514B'
DOWN = '#4779A8'
OTHER = '#C9CED2'
WT = '#3C7F9E'
KO = '#B26A3B'
NAPM = '#73529B'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ---------------------------------------------------------------------
# Source data and numerical checks.
loops = pd.read_csv(BASE / 'loop_change_enrichment/per_loop_change_and_binding.csv')
l12 = loops.loc[loops.loop_id.eq('L12')].iloc[0]
start, end = float(l12.center1), float(l12.center2)
assert (start, end) == (1692500.0, 2617500.0)
assert np.isclose(l12.WT_mean, 5.392821925179323)
assert np.isclose(l12.KO_mean, 2.315583127350743)
assert np.isclose(l12.d_log2OE, -1.2196648463921549)

g = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
g['is_strict_DEG'] = g.class_KOvsWT.isin(DET)
region = g[(g.tss >= start) & (g.tss <= end)].copy().sort_values('tss')
assert len(region) == 876 and int(region.is_strict_DEG.sum()) == 170
region.to_csv(OUT / 'L12_region_gene_track.csv', index=False)

peaks = pd.read_csv(RA / '09_combined/tables/Fig1_final_A_peaks.csv')
region_peaks = peaks[(peaks.center >= start) & (peaks.center <= end)].copy().sort_values('center')
assert len(region_peaks) == 132
region_peaks.to_csv(OUT / 'L12_NapM_peak_track.csv', index=False)

his_ids = [f'Rv{i}' for i in range(1599, 1607)]
arg_ids = [f'Rv{i}' for i in range(1652, 1658)]
highlight = g[g.gene_id.isin(his_ids + arg_ids)].copy().sort_values('start')
highlight['cluster'] = np.where(highlight.gene_id.isin(his_ids), 'his', 'arg')
assert len(highlight) == 14
assert highlight.is_strict_DEG.all() and highlight.log2FC_KOvsWT.gt(0).all()
assert highlight.binding_class.eq('unbound').all()
highlight.to_csv(OUT / 'L12_highlight_gene_clusters.csv', index=False)

loop_summary = pd.DataFrame([dict(
    loop_id='L12', start=start, end=end, span_kb=(end - start) / 1000,
    WT_mean_OE=l12.WT_mean, KO_mean_OE=l12.KO_mean, d_log2OE=l12.d_log2OE,
    WT1=l12.WT1, WT2=l12.WT2, WT3=l12.WT3, WT4=l12.WT4,
    KO1=l12.KO1, KO2=l12.KO2, KO3=l12.KO3, KO4=l12.KO4,
    recovered_in_independent_KO_call=True,
)])
loop_summary.to_csv(OUT / 'L12_loop_summary.csv', index=False)

# ---------------------------------------------------------------------
# Figure layout: three full-length tracks plus two cluster-level zooms.
fig = plt.figure(figsize=(16.0, 11.6))
gs = fig.add_gridspec(
    5, 2, height_ratios=[1.25, 1.05, 2.25, 1.65, 0.90],
    left=0.075, right=0.975, top=0.865, bottom=0.105,
    hspace=0.58, wspace=0.20,
)
ax_loop = fig.add_subplot(gs[0, :])
ax_chip = fig.add_subplot(gs[1, :], sharex=ax_loop)
ax_rna = fig.add_subplot(gs[2, :], sharex=ax_loop)
ax_his = fig.add_subplot(gs[3, 0])
ax_arg = fig.add_subplot(gs[3, 1])
ax_his_genes = fig.add_subplot(gs[4, 0], sharex=ax_his)
ax_arg_genes = fig.add_subplot(gs[4, 1], sharex=ax_arg)

full_xlim = (start - 18000, end + 18000)
ax_loop.set_xlim(*full_xlim)
ax_loop.set_ylim(-0.18, 1.37)
ax_loop.axis('off')
ax_loop.set_title('A  L12 loop contact: reduced strength in napM-KO', loc='left', fontweight='bold')
mid = (start + end) / 2
for condition, mean_oe, color in [('WT', l12.WT_mean, WT), ('napM-KO', l12.KO_mean, KO)]:
    height = mean_oe / l12.WT_mean
    path = MplPath([(start, 0.10), (mid, 0.10 + height), (end, 0.10)],
                   [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax_loop.add_patch(PathPatch(path, facecolor='none', edgecolor=color,
                                linewidth=3.2, capstyle='round'))
    ax_loop.scatter([start, end], [0.10, 0.10], s=24, color=color, zorder=3)
    ax_loop.text(mid, 0.16 + height, f'{condition}: mean O/E = {mean_oe:.2f}',
                 ha='center', va='bottom', fontsize=10.5, color=color, fontweight='bold')
ax_loop.text(end + 12000, 1.23,
             f'925 kb\nlog$_2$(KO/WT) = {l12.d_log2OE:.2f}',
             ha='right', va='top', fontsize=10, color='#333333',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#F4F5F6', edgecolor='#D9DDE0'))
ax_loop.text(start, -0.04, 'L12 left anchor', ha='center', va='top', fontsize=8.5, color='#555555')
ax_loop.text(end, -0.04, 'L12 right anchor', ha='center', va='top', fontsize=8.5, color='#555555')

his = highlight[highlight.cluster.eq('his')]
arg = highlight[highlight.cluster.eq('arg')]
his_span = (his.start.min(), his.end.max())
arg_span = (arg.start.min(), arg.end.max())
for ax in [ax_chip, ax_rna]:
    for span in [his_span, arg_span]:
        ax.axvspan(span[0], span[1], color='#E8C85D', alpha=0.24, lw=0, zorder=0)

# NapM peak-score track.
ax_chip.set_ylim(0, 1.08)
ax_chip.vlines(region_peaks.center, 0, region_peaks.score, color=NAPM,
               linewidth=0.85, alpha=0.74)
ax_chip.set_yticks([0, 0.5, 1.0])
ax_chip.set_ylabel('NapM peak score')
ax_chip.set_title('B  Curated NapM ChIP peaks across L12', loc='left', fontweight='bold')
ax_chip.grid(axis='y', color='#E7E9EB', linewidth=0.7, zorder=0)
ax_chip.tick_params(axis='x', labelbottom=False)
ax_chip.text(0.995, 0.88, f'{len(region_peaks)} peaks', transform=ax_chip.transAxes,
             ha='right', va='top', fontsize=9, color=NAPM)

# Full-region RNA track.
for values, mask, color, width in [
    (region.log2FC_KOvsWT, ~region.is_strict_DEG, OTHER, 0.72),
    (region.log2FC_KOvsWT, region.is_strict_DEG & region.log2FC_KOvsWT.gt(0), UP, 0.90),
    (region.log2FC_KOvsWT, region.is_strict_DEG & region.log2FC_KOvsWT.lt(0), DOWN, 0.90),
]:
    ax_rna.vlines(region.loc[mask, 'tss'], 0, values[mask], color=color,
                  linewidth=width, alpha=0.92 if color != OTHER else 0.72)
ax_rna.axhline(0, color='#444444', linewidth=0.85)
ax_rna.axhline(1, color='#BBBBBB', linewidth=0.65, linestyle=(0, (2, 3)))
ax_rna.axhline(-1, color='#BBBBBB', linewidth=0.65, linestyle=(0, (2, 3)))
ax_rna.set_ylim(-3.35, 4.20)
ax_rna.set_ylabel('RNA log$_2$(KO/WT)')
ax_rna.set_title('C  Gene-level transcription across L12', loc='left', fontweight='bold')
ax_rna.grid(axis='y', color='#ECEEEF', linewidth=0.6, zorder=0)
ax_rna.legend(handles=[
    Line2D([0], [0], color=UP, lw=3, label='Strict DEG: up'),
    Line2D([0], [0], color=DOWN, lw=3, label='Strict DEG: down'),
    Line2D([0], [0], color=OTHER, lw=3, label='Other genes'),
], loc='upper right', frameon=False, ncol=3, fontsize=8.8, handlelength=1.6)
for label, span in [('his cluster', his_span), ('arg cluster', arg_span)]:
    center = np.mean(span)
    ax_rna.annotate(label, xy=(center, 3.20), xytext=(center, 3.92),
                    ha='center', va='center', fontsize=9, color='#735F18', fontweight='bold',
                    arrowprops=dict(arrowstyle='-|>', color='#A88927', lw=0.9))
ax_rna.set_xlabel('Genomic position in M. tuberculosis H37Rv (Mb)')
ax_rna.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}'))
ax_rna.set_xticks(np.arange(1700000, 2600001, 100000))


def plot_cluster(rna_ax, gene_ax, genes, title, panel_letter, flank=4500):
    """RNA bars and strand-aware gene models for one highlighted operon."""
    genes = genes.sort_values('start').copy()
    xmin = max(0, int(genes.start.min()) - flank)
    xmax = int(genes.end.max()) + flank
    rna_ax.set_xlim(xmin, xmax)
    rna_ax.set_ylim(0, 3.05)
    centers = (genes.start + genes.end) / 2
    widths = genes.end - genes.start
    bars = rna_ax.bar(centers, genes.log2FC_KOvsWT, width=widths,
                      color=UP, edgecolor='white', linewidth=0.6, alpha=0.88)
    for bar, value in zip(bars, genes.log2FC_KOvsWT):
        rna_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.09, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=8, color='#71332F')
    rna_ax.set_ylabel('RNA log$_2$(KO/WT)')
    rna_ax.set_title(f'{panel_letter}  {title}', loc='left', fontweight='bold')
    rna_ax.grid(axis='y', color='#E7E9EB', linewidth=0.7, zorder=0)
    rna_ax.tick_params(axis='x', labelbottom=False)
    rna_ax.text(0.99, 0.94, 'All genes shown are strict upregulated DEGs',
                transform=rna_ax.transAxes, ha='right', va='top', fontsize=8.3, color='#6D7378')

    gene_ax.set_ylim(-0.55, 0.88)
    gene_ax.set_yticks([])
    gene_ax.spines[['left', 'top', 'right']].set_visible(False)
    for gene_index, row in enumerate(genes.itertuples()):
        color = UP
        if row.strand == '+':
            x0, dx = row.start, row.end - row.start
        else:
            x0, dx = row.end, row.start - row.end
        arrow = FancyArrow(x0, 0.02, dx, 0, width=0.15, head_width=0.30,
                           head_length=min(360, abs(dx) * 0.32),
                           length_includes_head=True, facecolor=color,
                           edgecolor='white', linewidth=0.5, alpha=0.90)
        gene_ax.add_patch(arrow)
        label = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
        label_y = 0.36 if gene_index % 2 == 0 else 0.61
        gene_ax.text((row.start + row.end) / 2, label_y, label,
                     ha='center', va='bottom', fontsize=8.2, color='#333333')
    gene_ax.text(0.99, 0.95, 'NapM-unbound\n(promoter/gene body)', transform=gene_ax.transAxes,
                 ha='right', va='top', fontsize=8.2, color=NAPM)
    gene_ax.set_xlabel('Genomic position (Mb)')
    gene_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.3f}'))
    gene_ax.tick_params(axis='x', labelsize=8.5)


plot_cluster(ax_his, ax_his_genes, his, 'his locus (Rv1599–Rv1606)', 'D')
plot_cluster(ax_arg, ax_arg_genes, arg, 'arg locus (Rv1652–Rv1657)', 'E')

fig.suptitle('L12 loop weakening and coordinated transcriptional changes at his/arg loci',
             x=0.055, y=0.970, ha='left', fontsize=16, fontweight='bold')
fig.text(0.055, 0.928,
         'H37Rv 1.6925–2.6175 Mb  |  L12 mean O/E: 5.39 (WT) → 2.32 (KO)  |  '
         '876 genes, including 170 strict DEGs',
         ha='left', fontsize=10.5, color='#4D555B')
fig.text(0.075, 0.040,
         'Loop-arc heights are scaled to mean O/E. Strict DEG: RNA BH FDR < 0.05 and |log$_2$FC| ≥ 1. '
         'The displayed his/arg genes lack promoter or gene-body NapM peaks;\n'
         'the multitrack view supports regional structure–transcription coupling but does not by itself establish direct or causal regulation.',
         ha='left', va='bottom', fontsize=8.8, color='#50585E')

fig.savefig(OUT / 'R1_L12_multitrack.pdf', bbox_inches='tight')
fig.savefig(OUT / 'R1_L12_multitrack.png', dpi=400, bbox_inches='tight')
plt.close(fig)

print('L12 loop summary:')
print(loop_summary.to_string(index=False))
print('\nRegion genes:', len(region), 'strict DEGs:', int(region.is_strict_DEG.sum()))
print('NapM peaks:', len(region_peaks))
print('\nHighlighted clusters:')
print(highlight[['cluster', 'gene_id', 'gene_name', 'start', 'end', 'strand',
                 'log2FC_KOvsWT', 'padj_KOvsWT', 'binding_class']].to_string(index=False))
print('\nWrote L12 multitrack figure to', OUT)
