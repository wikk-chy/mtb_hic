#!/usr/bin/env python3
"""Redraw mechanism-figure panels F–I without changing source values or statistics.

Input files are the validated figure-source CSVs produced by
07_prepare_full_mechanism_figure.py. This script only crops the display window
and changes visual encoding/layout. Each key region is one 5-kb bin; the display
contains that bin plus 25 kb on each side (-27.5 to +27.5 kb).
"""
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC = OUT / 'source_data'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# The focal source bin spans -2.5 to +2.5 kb. Adding 25 kb on each side
# gives the common display window below. Matrix cells are selected by centers.
KEY_HALF_KB = 2.5
FLANK_KB = 25.0
WINDOW_KB = KEY_HALF_KB + FLANK_KB
MATRIX_CENTER_LIMIT_KB = WINDOW_KB - KEY_HALF_KB
MAP_VMAX = 1.25  # Same symmetric scale as the source figure; no rescaling.

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.55,
    'axes.edgecolor': '#4D4D4D',
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# Colorblind-safe blue/white/red response scale and blue/orange condition code.
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    'blue_white_red', ['#2166AC', '#F7F7F7', '#B2182B']
)
CMAP.set_bad('#E5E5E5')  # Missing source cells remain visually distinct from zero.
NORM = mpl.colors.TwoSlopeNorm(vmin=-MAP_VMAX, vcenter=0, vmax=MAP_VMAX)
WT_BLUE = '#2166AC'
KO_ORANGE = '#D6604D'
NEG_BLUE = '#2166AC'
POS_RED = '#B2182B'
KEY_GREY = '#6E6E6E'
GENE_GREY = '#C7CBCE'
DARK = '#292929'

maps = pd.read_csv(SRC / 'Mechanism_locus_response_maps.csv')
ins = pd.read_csv(SRC / 'Mechanism_locus_insulation_tracks.csv')
rna = pd.read_csv(SRC / 'Mechanism_locus_RNA_tracks.csv')
genes = pd.read_csv(SRC / 'Mechanism_locus_gene_models.csv')

loci = [
    ('F', 'DosR_DosS', 'DosR/DosS region', {'Rv3133c', 'Rv3132c'}),
    ('G', 'SigE', 'SigE region', {'Rv1221'}),
    ('H', 'oriC', 'oriC/dnaA region', {'Rv0001'}),
    ('I', 'ESX2', 'ESX-2 region', {'Rv3886c', 'Rv3890c'}),
]

# Coordinate checks: do not infer a focal point when a source region has width.
for _, loc, _, _ in loci:
    focal = ins[(ins.locus == loc) & ins.is_focal]
    assert len(focal) == 1, f'{loc}: expected one source focal bin'
    assert np.isclose(focal.iloc[0].position_kb, 0), f'{loc}: focal center is not zero'


def clean_axis(ax, horizontal_guides=False):
    """Retain only quiet axes and optional light horizontal references."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    if horizontal_guides:
        ax.grid(axis='y', color='#E9E9E9', lw=0.45)
    ax.grid(axis='x', visible=False)


def mark_key_region(ax, heatmap=False):
    """Mark the full 5-kb key region, never its flanks."""
    if heatmap:
        ax.add_patch(Rectangle(
            (-KEY_HALF_KB, -KEY_HALF_KB), 2 * KEY_HALF_KB, 2 * KEY_HALF_KB,
            fill=False, edgecolor='#666666', lw=0.65, ls=(0, (2, 1.5)), zorder=4,
        ))
    else:
        ax.axvspan(-KEY_HALF_KB, KEY_HALF_KB, color=KEY_GREY, alpha=0.10,
                   lw=0, zorder=0)
        for boundary in (-KEY_HALF_KB, KEY_HALF_KB):
            ax.axvline(boundary, color='#A0A0A0', lw=0.45, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-WINDOW_KB, WINDOW_KB)


def response_matrix(loc, genotype):
    """Crop source matrix values by bin-center coordinates; no interpolation."""
    z = maps[(maps.locus == loc) & (maps.genotype == genotype)]
    z = z[
        z.position1_kb.between(-MATRIX_CENTER_LIMIT_KB, MATRIX_CENTER_LIMIT_KB)
        & z.position2_kb.between(-MATRIX_CENTER_LIMIT_KB, MATRIX_CENTER_LIMIT_KB)
    ]
    positions = sorted(set(z.position1_kb) | set(z.position2_kb))
    assert positions == sorted(z.position1_kb.unique()) == sorted(z.position2_kb.unique())
    lookup = {
        (r.position1_kb, r.position2_kb): r.log2_hyp_norm_OE
        for r in z.itertuples()
    }
    mat = np.full((len(positions), len(positions)), np.nan)
    for i, p1 in enumerate(positions):
        for j, p2 in enumerate(positions):
            mat[i, j] = lookup[(p1, p2)]
    return mat


def plot_response_map(ax, loc, genotype, column_header=False, bottom_row=False):
    mat = response_matrix(loc, genotype)
    # Flat pcolormesh cells are the vector equivalent of nearest/no interpolation:
    # one displayed quadrilateral per source matrix element, with no smoothing.
    edges = np.linspace(-WINDOW_KB, WINDOW_KB, mat.shape[0] + 1)
    mesh = ax.pcolormesh(
        edges, edges, np.ma.masked_invalid(mat), cmap=CMAP, norm=NORM,
        shading='flat', edgecolors='none', rasterized=False,
    )
    mark_key_region(ax, heatmap=True)
    ax.set_ylim(-WINDOW_KB, WINDOW_KB)
    ax.set_aspect('equal', adjustable='box')
    ticks = [-25, 0, 25]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if not bottom_row:
        ax.tick_params(axis='x', labelbottom=False)
    else:
        ax.set_xlabel('Position (kb)', labelpad=3)
    if genotype == 'WT':
        ax.set_ylabel('Position (kb)', labelpad=3)
    else:
        ax.tick_params(axis='y', labelleft=False)
    if column_header:
        ax.set_title('WT' if genotype == 'WT' else 'napM-KO',
                     fontsize=9, fontweight='bold', pad=5)
    ax.spines[:].set_linewidth(0.5)
    return mesh


def plot_insulation(ax, loc, ylim, first_row=False):
    z = ins[
        (ins.locus == loc)
        & ins.position_kb.between(-MATRIX_CENTER_LIMIT_KB, MATRIX_CENTER_LIMIT_KB)
    ].sort_values('position_kb')
    mark_key_region(ax)
    for col, color, ls in [
        ('WT', WT_BLUE, '-'),
        ('WT_hypoxia', WT_BLUE, '--'),
        ('KO', KO_ORANGE, '-'),
        ('KO_hypoxia', KO_ORANGE, '--'),
    ]:
        ax.plot(z.position_kb, z[col], color=color, ls=ls, lw=1.05,
                solid_capstyle='round', dash_capstyle='round', zorder=3)
    ax.axhline(0, color='#C8C8C8', lw=0.5, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel('Insulation\nscore (log2)', labelpad=5)
    ax.tick_params(axis='x', labelbottom=False, length=0)
    clean_axis(ax, horizontal_guides=True)
    if first_row:
        ax.text(0, 0.97, '5-kb key region', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.2, color='#777777')


def plot_rna_interaction(ax, loc, ylim):
    z = rna[(rna.locus == loc) & rna.tss_kb.between(-WINDOW_KB, WINDOW_KB)].copy()
    mark_key_region(ax)
    ax.axhline(0, color='#8F8F8F', lw=0.55, zorder=1)
    for row in z.itertuples():
        value = row.interaction_log2FC
        if not np.isfinite(value):
            continue
        # Preserve the original significance rule exactly.
        significant = (
            np.isfinite(row.interaction_FDR)
            and row.interaction_FDR < 0.05
            and abs(value) >= 1
        )
        color = POS_RED if value >= 0 else NEG_BLUE
        alpha = 1.0 if significant else 0.38
        ax.vlines(row.tss_kb, 0, value, color=color,
                  lw=0.85 if significant else 0.55, alpha=alpha, zorder=2)
        ax.scatter(row.tss_kb, value, s=11 if significant else 4.5,
                   color=color, alpha=alpha, edgecolor='white', linewidth=0.2,
                   zorder=3)
    ax.set_ylim(*ylim)
    ax.set_yticks([-5, 0, 5])
    ax.set_ylabel('Genotype × hypoxia\ninteraction log2FC', labelpad=5)
    ax.tick_params(axis='x', labelbottom=False, length=0)
    clean_axis(ax, horizontal_guides=True)


def gene_display_name(row):
    return row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id


def place_key_gene_labels(key_rows):
    """Greedy horizontal-collision avoidance on up to three label levels."""
    placements = []
    occupied = [[], [], []]
    levels = [1.00, 0.77, 0.55]
    for row in sorted(key_rows, key=lambda r: (r.start_kb + r.end_kb) / 2):
        center = (row.start_kb + row.end_kb) / 2
        name = gene_display_name(row)
        half_width = max(1.7, len(name) * 0.48)
        interval = (center - half_width, center + half_width)
        chosen = None
        for level, used in enumerate(occupied):
            if all(interval[1] < left or interval[0] > right for left, right in used):
                chosen = level
                break
        if chosen is None:
            # Deterministic fallback: use the level with the largest gap.
            chosen = int(np.argmin([len(used) for used in occupied]))
        occupied[chosen].append(interval)
        placements.append((row, name, center, levels[chosen]))
    return placements


def plot_genes(ax, loc, key_gene_ids):
    z = genes[(genes.locus == loc) & genes.tss_kb.between(-WINDOW_KB, WINDOW_KB)].copy()
    mark_key_region(ax)
    key_rows = []
    for row in z.itertuples():
        start, end = sorted([row.start_kb, row.end_kb])
        shown_start = max(start, -WINDOW_KB)
        shown_end = min(end, WINDOW_KB)
        if shown_end <= shown_start:
            continue
        key = row.gene_id in key_gene_ids
        y = 0.32 if row.strand == '+' else 0.08
        length = shown_end - shown_start
        x = shown_start if row.strand == '+' else shown_end
        dx = length if row.strand == '+' else -length
        ax.add_patch(FancyArrow(
            x, y, dx, 0,
            width=0.030 if key else 0.020,
            head_width=0.105 if key else 0.075,
            head_length=min(0.55, length * 0.40),
            length_includes_head=True,
            facecolor=DARK if key else GENE_GREY,
            edgecolor=DARK if key else GENE_GREY,
            lw=0.25 if key else 0.15,
            alpha=1.0 if key else 0.78,
            zorder=3 if key else 2,
        ))
        if key:
            key_rows.append(row)
    for row, name, center, label_y in place_key_gene_labels(key_rows):
        arrow_y = 0.32 if row.strand == '+' else 0.08
        ax.plot([center, center], [arrow_y + 0.07, label_y - 0.035],
                color='#777777', lw=0.4, zorder=2)
        ax.text(center, label_y, name, ha='center', va='bottom', fontsize=6.7,
                color=DARK, fontweight='semibold', fontstyle='italic', zorder=4)
    ax.set_ylim(-0.05, 1.16)
    ax.set_yticks([])
    ax.set_ylabel('Gene\nannotation', labelpad=5)
    ax.set_xticks([-25, -12.5, 0, 12.5, 25])
    ax.set_xlabel('Position relative to key-region center (kb)', labelpad=3)
    clean_axis(ax)


# Common insulation range across all four cropped loci.
ins_display = ins[ins.position_kb.between(-MATRIX_CENTER_LIMIT_KB, MATRIX_CENTER_LIMIT_KB)]
ins_values = ins_display[['WT', 'WT_hypoxia', 'KO', 'KO_hypoxia']].to_numpy(float)
ins_finite = ins_values[np.isfinite(ins_values)]
ins_ylim = (np.floor((ins_finite.min() - 0.10) * 10) / 10,
            np.ceil((ins_finite.max() + 0.10) * 10) / 10)

# Zero-centered common RNA-interaction range across all cropped loci.
rna_display = rna[rna.tss_kb.between(-WINDOW_KB, WINDOW_KB)]
rna_abs_max = np.nanmax(np.abs(rna_display.interaction_log2FC.to_numpy(float)))
rna_ylim = (-np.ceil(rna_abs_max * 2) / 2, np.ceil(rna_abs_max * 2) / 2)

fig = plt.figure(figsize=(11.8, 10.4))
outer = fig.add_gridspec(
    4, 3, left=0.080, right=0.985, bottom=0.105, top=0.855,
    width_ratios=[1, 1, 3.0], wspace=0.12, hspace=0.48,
)

for i, (label, loc, title, key_gene_ids) in enumerate(loci):
    ax_wt = fig.add_subplot(outer[i, 0])
    ax_ko = fig.add_subplot(outer[i, 1])
    plot_response_map(ax_wt, loc, 'WT', column_header=(i == 0), bottom_row=(i == 3))
    plot_response_map(ax_ko, loc, 'KO', column_header=(i == 0), bottom_row=(i == 3))
    if i == 0:
        ax_wt.set_title('WT', fontsize=9, fontweight='bold', pad=28)
        ax_ko.set_title('napM-KO', fontsize=9, fontweight='bold', pad=28)

    # Panel letter and region name share one left-aligned title baseline.
    ax_wt.text(-0.43, 1.18, label, transform=ax_wt.transAxes, fontsize=14,
               fontweight='bold', va='top', ha='left', color=DARK)
    ax_wt.text(-0.25, 1.18, title, transform=ax_wt.transAxes, fontsize=10.5,
               fontweight='bold', va='top', ha='left', color=DARK)

    tracks = mpl.gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[i, 2], height_ratios=[1.0, 1.0, 0.78], hspace=0.30,
    )
    ax_ins = fig.add_subplot(tracks[0])
    ax_rna = fig.add_subplot(tracks[1], sharex=ax_ins)
    ax_gene = fig.add_subplot(tracks[2], sharex=ax_ins)
    plot_insulation(ax_ins, loc, ins_ylim, first_row=(i == 0))
    plot_rna_interaction(ax_rna, loc, rna_ylim)
    plot_genes(ax_gene, loc, key_gene_ids)
    # Horizontal track titles avoid the multi-line y-label collisions in the draft.
    for ax in (ax_ins, ax_rna, ax_gene):
        ax.set_ylabel('')
    ax_ins.set_title('Insulation score (log2)', loc='left', fontsize=7.4, pad=2)
    ax_rna.set_title('Genotype × hypoxia interaction log2FC', loc='left', fontsize=7.4, pad=2)
    ax_gene.set_title('Gene annotation', loc='left', fontsize=7.4, pad=2)
    ax_gene.set_xticklabels(['−25', '−12.5', '0', '12.5', '25'])
    assert ax_ins.get_xlim() == ax_rna.get_xlim() == ax_gene.get_xlim() == (-WINDOW_KB, WINDOW_KB)

# One condition legend for the complete figure.
legend_handles = [
    Line2D([0], [0], color=WT_BLUE, ls='-', lw=1.2, label='WT normoxia'),
    Line2D([0], [0], color=WT_BLUE, ls='--', lw=1.2, label='WT hypoxia'),
    Line2D([0], [0], color=KO_ORANGE, ls='-', lw=1.2, label='napM-KO normoxia'),
    Line2D([0], [0], color=KO_ORANGE, ls='--', lw=1.2, label='napM-KO hypoxia'),
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.655, 0.985),
           ncol=4, frameon=False, handlelength=2.0, columnspacing=1.35,
           handletextpad=0.55)
fig.text(0.208, 0.936, 'Hypoxia / normoxia O/E response', ha='center', va='center',
         fontsize=9, fontweight='bold', color=DARK)

# One shared color bar, approximately 40% of the two-heatmap group width.
cax = fig.add_axes([0.105, 0.036, 0.125, 0.0085])
mappable = mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP)
cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
cbar.set_label('log2(O/E hypoxia / O/E normoxia)', fontsize=7.5, labelpad=4)
cbar.ax.tick_params(labelsize=7, length=2.5, pad=2)
cbar.outline.set_visible(False)

png = FIG / 'NapM_hypoxia_locus_panels_FI_redesigned.png'
pdf = FIG / 'NapM_hypoxia_locus_panels_FI_redesigned.pdf'
svg = FIG / 'NapM_hypoxia_locus_panels_FI_redesigned.svg'
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    fig.savefig(png, dpi=600)
    fig.savefig(pdf, format='pdf', metadata={
        'Title': 'NapM hypoxia mechanism figure: redesigned panels F-I',
        'Subject': 'Unsmoothed 5-kb response maps and aligned insulation, RNA interaction, and gene tracks',
        'Author': 'Multi-omics integration analysis',
    })
    fig.savefig(svg, format='svg')
plt.close(fig)

print(f'Display window: {-WINDOW_KB} to +{WINDOW_KB} kb; key region: {-KEY_HALF_KB} to +{KEY_HALF_KB} kb')
print(f'Common insulation y limits: {ins_ylim}')
print(f'Common RNA-interaction y limits: {rna_ylim}')
print(f'Wrote {png} (600 dpi)')
print(f'Wrote editable vector {pdf}')
print(f'Wrote editable vector {svg}')
