#!/usr/bin/env python3
"""Build one complete statistical and in-situ multi-omics mechanism figure."""
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrow, Patch

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC, TAB, FIG = OUT / 'source_data', OUT / 'tables', OUT / 'figures'
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 10,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.linewidth': .65, 'axes.edgecolor': '#444444',
    'xtick.major.width': .55, 'ytick.major.width': .55,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})
WT, KO = '#1F4E79', '#B35845'
HYP, KOHYP = '#8AAFD0', '#E1A294'
PURPLE, GREEN, ORANGE, GREY, LIGHT = '#6B5B7A', '#3F7B70', '#B27A3F', '#8B9298', '#D8DDE1'
CMAP = mpl.colors.LinearSegmentedColormap.from_list('response', ['#31688E', '#FFFFFF', '#A44A4A'])
CMAP.set_bad('#D3D5D6')
CLUSTER_COLORS = {'C1': '#3F7B70', 'C2': '#8C6D9D', 'C3': '#B35845', 'C4': '#D29A45'}

hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
interaction = pd.read_csv(SRC / 'Mechanism_RNA_interaction_clusters.csv')
cluster_summary = pd.read_csv(TAB / 'Mechanism_RNA_cluster_summary.csv')
bins = pd.read_csv(SRC / 'Mechanism_bin_evidence_classes.csv')
candidates = pd.read_csv(TAB / 'Mechanism_structural_candidate_ranking.csv')
maps = pd.read_csv(SRC / 'Mechanism_locus_response_maps.csv')
ins = pd.read_csv(SRC / 'Mechanism_locus_insulation_tracks.csv')
rna = pd.read_csv(SRC / 'Mechanism_locus_RNA_tracks.csv')
genes = pd.read_csv(SRC / 'Mechanism_locus_gene_models.csv')
peaks = pd.read_csv(SRC / 'Mechanism_locus_NapM_peaks.csv')
chip = pd.read_csv(SRC / 'Mechanism_locus_NapM_signal.csv')
expr = pd.read_csv(SRC / 'Mechanism_highlighted_gene_normalized_logCPM.csv')
cors = pd.read_csv(TAB / 'Structure_transcription_correlations.csv')


def clean(ax, grid=False):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#E8E8E8', lw=.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label, title, x=-.14):
    ax.text(x, 1.09, label, transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='top')
    ax.set_title(title, loc='left', fontweight='bold', pad=10)


def focal_shade(ax, loc, xlim=(-32.5, 32.5)):
    z = ins[(ins.locus == loc) & ins.is_focal]
    for row in z.itertuples():
        ax.axvspan(row.position_kb - 2.5, row.position_kb + 2.5,
                   color='#F0ECEF', alpha=.65, zorder=0)
    ax.set_xlim(*xlim)


def plot_response_map(ax, loc, genotype):
    z = maps[(maps.locus == loc) & (maps.genotype == genotype)]
    bins_sorted = sorted(set(z.bin1) | set(z.bin2))
    # Coordinates, rather than numeric bin IDs, define matrix order across oriC.
    positions = sorted(z.position1_kb.unique())
    mat = np.full((len(positions), len(positions)), np.nan)
    lookup = {(r.position1_kb, r.position2_kb): r.log2_hyp_norm_OE for r in z.itertuples()}
    for a, p1 in enumerate(positions):
        for b, p2 in enumerate(positions):
            mat[a, b] = lookup[(p1, p2)]
    im = ax.imshow(mat, origin='lower', extent=[-32.5, 32.5, -32.5, 32.5],
                   cmap=CMAP, vmin=-1.25, vmax=1.25, interpolation='bilinear', aspect='equal')
    ax.axvline(0, color='#333333', lw=.45, ls='--')
    ax.axhline(0, color='#333333', lw=.45, ls='--')
    ax.set_title('WT: hypoxia / normoxia' if genotype == 'WT' else 'napM-KO: hypoxia / normoxia',
                 fontsize=8, pad=4)
    ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30])
    ax.set_xlabel(''); ax.set_ylabel('Position (kb)')
    ax.spines[:].set_linewidth(.45)
    return im


def plot_structure_interaction(ax, loc):
    z = ins[ins.locus == loc].sort_values('position_kb')
    focal_shade(ax, loc)
    ax.axhline(0, color='#999999', lw=.5)
    ax.plot(z.position_kb, z.interaction, color=PURPLE, lw=1.0)
    sig = z.interaction_FDR < .05
    ax.scatter(z.position_kb[sig], z.interaction[sig], s=10, color=PURPLE, zorder=3)
    ax.scatter(z.position_kb[~sig], z.interaction[~sig], s=5, color='#B9C0C5', zorder=3)
    ax.set_ylabel('Insulation\ninteraction')
    ax.set_xlabel('Position (kb)')
    clean(ax, True)


def plot_insulation(ax, loc, show_legend=False):
    z = ins[ins.locus == loc].sort_values('position_kb')
    focal_shade(ax, loc)
    for col, color, ls, label in [
        ('WT', WT, '-', 'WT'), ('KO', KO, '-', 'KO'),
        ('WT_hypoxia', HYP, '--', 'WT hypoxia'), ('KO_hypoxia', KOHYP, '--', 'KO hypoxia')]:
        ax.plot(z.position_kb, z[col], color=color, ls=ls, lw=1.15, label=label)
    ax.set_ylabel('Insulation\n(log2)')
    if show_legend:
        ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.32),
                  handlelength=1.7, columnspacing=1.2)
    clean(ax, True)


def plot_rna(ax, loc, effect, q, color, ylabel, label_genes):
    z = rna[rna.locus == loc].copy()
    focal_shade(ax, loc)
    ax.axhline(0, color='#AAAAAA', lw=.5)
    for row in z.itertuples():
        y = getattr(row, effect)
        if not np.isfinite(y):
            continue
        qq = getattr(row, q)
        significant = np.isfinite(qq) and qq < .05 and abs(y) >= 1
        ax.vlines(row.tss_kb, 0, y, color=color if significant else '#AEB5BA',
                  lw=1.0 if significant else .55, alpha=.95)
        ax.scatter(row.tss_kb, y, s=13 if significant else 4,
                   color=color if significant else '#AEB5BA', edgecolor='white', linewidth=.2, zorder=3)
        if row.gene_id in label_genes:
            name = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
            ax.annotate(name, (row.tss_kb, y), xytext=(0, 3 if y >= 0 else -8),
                        textcoords='offset points', ha='center', va='bottom' if y >= 0 else 'top',
                        fontsize=5.8, rotation=22, color='#333333')
    ax.set_ylabel(ylabel)
    clean(ax, True)


def plot_chip_genes(ax, loc, label_genes):
    focal_shade(ax, loc)
    z = chip[chip.locus == loc].sort_values('start_kb')
    if len(z):
        ax.fill_between(z.start_kb, .45, .45 + z.signal / max(z.signal.max(), 1e-9) * .8,
                        color=PURPLE, alpha=.48, lw=0, step='mid')
    zp = peaks[peaks.locus == loc]
    for row in zp.itertuples():
        ax.vlines(row.center_kb, .4, 1.35, color=PURPLE, lw=.7)
    zg = genes[genes.locus == loc]
    for row in zg.itertuples():
        y = -.05 if row.strand == '+' else -.55
        color = PURPLE if row.binding_class in ['promoter_bound', 'gene_body_bound'] else '#8D9398'
        start, end = sorted([row.start_kb, row.end_kb])
        ax.add_patch(FancyArrow(start, y, max(end-start, .18), 0, width=.10,
                                head_width=.23, head_length=min(.6, max((end-start)*.25, .15)),
                                length_includes_head=True, facecolor=color, edgecolor='none', alpha=.85))
        if row.gene_id in label_genes:
            name = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
            ax.text((start+end)/2, y + (.20 if row.strand == '+' else -.20), name,
                    ha='center', va='bottom' if row.strand == '+' else 'top', fontsize=5.8,
                    fontweight='bold', rotation=15)
    ax.axhline(-.30, color='#D0D0D0', lw=.4)
    ax.set_ylim(-1.0, 1.4); ax.set_yticks([]); ax.set_ylabel('NapM ChIP\nand genes')
    ax.set_xlabel('Position relative to focal bin (kb)')
    clean(ax)


fig = plt.figure(figsize=(18, 22))
outer = fig.add_gridspec(6, 3, height_ratios=[1.12, .68, 1.18, 1.18, 1.18, 1.18],
                         left=.045, right=.985, top=.955, bottom=.035, hspace=.62, wspace=.30)
fig.suptitle('Genotype × hypoxia analysis distinguishes RNA response classes from NapM-associated structural candidates',
             x=.04, y=.988, ha='left', fontsize=17, fontweight='bold')
fig.text(.04, .968,
         'Direct interaction tests, descriptive response clustering and bin-level evidence classification are shown separately from in-situ locus evidence.',
         ha='left', fontsize=9.5, color='#555C61')

# A: interaction volcano shows exactly how the 730 DEGs are obtained.
ax_a = fig.add_subplot(outer[0, 0])
sig = hyp_g.hypoxia_interaction_RNA_class.isin(['detected_increase', 'detected_decrease'])
ax_a.scatter(hyp_g.loc[~sig, 'interaction'], -np.log10(hyp_g.loc[~sig, 'padj_int']),
             s=6, color='#C8CDD1', alpha=.5, edgecolors='none')
ax_a.scatter(hyp_g.loc[sig, 'interaction'], -np.log10(hyp_g.loc[sig, 'padj_int']),
             s=8, color=PURPLE, alpha=.55, edgecolors='none')
ax_a.axvline(-1, color='#888888', lw=.55, ls='--'); ax_a.axvline(1, color='#888888', lw=.55, ls='--')
ax_a.axhline(-np.log10(.05), color='#888888', lw=.55, ls='--')
for gene, dx, dy in [('devR', 10, 4), ('devS', 8, -8), ('sigE', 10, -12)]:
    row = hyp_g[hyp_g.gene_name.eq(gene)].iloc[0]
    ax_a.annotate(gene,
                  (row.interaction, -np.log10(row.padj_int)), xytext=(dx, dy), textcoords='offset points',
                  fontsize=6.6, color='#333333', arrowprops={'arrowstyle': '-', 'lw': .45, 'color': '#777777'})
ax_a.text(.03, .97, '730 interaction DEGs\n212 positive / 518 negative', transform=ax_a.transAxes,
          va='top', fontsize=8, linespacing=1.5)
ax_a.set(xlabel='(KO hyp−KO) − (WT hyp−WT) RNA log2FC', ylabel='−log10 BH FDR')
panel_label(ax_a, 'A', 'Genotype × hypoxia RNA interaction')
clean(ax_a, True)

# B: two-column heatmap and cluster centers for all 730 interaction DEGs.
bs = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], width_ratios=[1.05, 1.25], wspace=.38)
ax_b = fig.add_subplot(bs[0, 0])
ordered = interaction.sort_values(['cluster', 'interaction'], ascending=[True, False])
heat = ordered[['hyp_response_wt', 'hyp_response_ko']].to_numpy()
im = ax_b.imshow(heat, aspect='auto', cmap=CMAP, vmin=-4, vmax=4, interpolation='nearest')
ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(['WT', 'KO'])
ax_b.set_ylabel('730 interaction DEGs')
starts = ordered.groupby('cluster', sort=True).head(1).index
for cluster in ['C1', 'C2', 'C3', 'C4']:
    idx = np.flatnonzero(ordered.cluster.to_numpy() == cluster)
    ax_b.text(1.65, np.mean(idx), f'{cluster} (n={len(idx)})', va='center', fontsize=7,
              color=CLUSTER_COLORS[cluster], fontweight='bold')
    if idx[0] > 0:
        ax_b.axhline(idx[0]-.5, color='white', lw=1.3)
ax_b.set_yticks([])
cb = fig.colorbar(im, ax=ax_b, fraction=.046, pad=.03, ticks=[-4, 0, 4])
cb.set_label('RNA log2FC', fontsize=7)
panel_label(ax_b, 'B', 'Descriptive response-shape clusters', x=-.22)
ax_bc = fig.add_subplot(bs[0, 1])
for y, row in enumerate(cluster_summary.itertuples()):
    yy = len(cluster_summary)-1-y
    ax_bc.plot([row.WT_center, row.KO_center], [yy, yy], color='#C3C8CC', lw=1.5)
    ax_bc.scatter(row.WT_center, yy, s=36, color=WT)
    ax_bc.scatter(row.KO_center, yy, s=36, color=KO)
ax_bc.axvline(0, color='#999999', lw=.5)
ax_bc.set_yticks(range(len(cluster_summary)))
ax_bc.set_yticklabels(cluster_summary.cluster[::-1])
ax_bc.set_xlabel('Cluster-mean RNA log2FC\n(k-means, k = 4)')
clean(ax_bc, True)

# C: operational bin-level evidence classes, not mechanism clustering.
ax_c = fig.add_subplot(outer[0, 2])
class_styles = {
    'NapM-associated_candidate_no_detected_local_RNA_response': (PURPLE, 'NapM structural, no local RNA response'),
    'Transcription-associated_consistent_pattern': (GREEN, 'Transcription-associated'),
    'Mixed_or_unresolved_NapM_structure_RNA': (ORANGE, 'NapM structure/RNA unresolved'),
}
remaining = np.ones(len(bins), bool)
for cls, (color, label) in class_styles.items():
    mask = bins.hypoxia_response_evidence_class.eq(cls).to_numpy()
    remaining &= ~mask
    z = bins[mask]
    ax_c.scatter(z.RNA_interaction_mean_log2FC, z.d_ins_interaction, s=17 if cls.startswith('NapM-associated') else 8,
                 color=color, alpha=.75, edgecolors='white' if cls.startswith('NapM-associated') else 'none',
                 linewidth=.25, label=f'{label} (n={len(z)})')
z = bins[remaining]
ax_c.scatter(z.RNA_interaction_mean_log2FC, z.d_ins_interaction, s=5, color='#C8CDD1', alpha=.45,
             edgecolors='none', label=f'Other bins (n={len(z)})')
ax_c.axhline(0, color='#AAAAAA', lw=.5); ax_c.axvline(0, color='#AAAAAA', lw=.5)
for bin_id, name, dx, dy in [(0, 'oriC', 5, 3), (874, 'ESX-2', 5, -7), (272, 'sigE', 5, 4), (699, 'DosR/S', 5, 3)]:
    row = bins[bins['bin'].eq(bin_id)].iloc[0]
    ax_c.annotate(name, (row.RNA_interaction_mean_log2FC, row.d_ins_interaction),
                  xytext=(dx, dy), textcoords='offset points', fontsize=7, fontweight='bold')
rho_row = cors[cors.contrast.eq('Hypoxia interaction')].iloc[0]
ax_c.text(.03, .97, f"Spearman ρ={rho_row.spearman_rho:.3f}; shift P={rho_row.circular_shift_p:.3f}",
          transform=ax_c.transAxes, va='top', fontsize=7)
ax_c.legend(frameon=False, loc='lower left', fontsize=6.2)
ax_c.set(xlabel='Bin-mean RNA interaction log2FC', ylabel='Insulation interaction (Δ log2)')
panel_label(ax_c, 'C', 'Bin-level joint evidence classification')
clean(ax_c, True)

# D: normalized sample-level expression for the three regulator examples.
ds = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 0:2], wspace=.30)
sample_cols = ['WT-rep1', 'WT-rep2', 'NapM-KO-rep1', 'NapM-KO-rep2',
               'Hypoxia-rep1', 'Hypoxia-rep2', 'NapM_Ko-Hypoxia-rep1', 'NapM_Ko-Hypoxia-rep2']
for j, (gene_id, title) in enumerate([('Rv3133c', 'DosR/DevR'), ('Rv3132c', 'DosS/DevS'), ('Rv1221', 'SigE')]):
    ax = fig.add_subplot(ds[0, j])
    row = expr[expr.gene_id.eq(gene_id)].iloc[0]
    for genotype, cols0, cols1, color in [
        ('WT', sample_cols[0:2], sample_cols[4:6], WT),
        ('KO', sample_cols[2:4], sample_cols[6:8], KO)]:
        v0, v1 = row[cols0].to_numpy(float), row[cols1].to_numpy(float)
        ax.plot([0, 1], [v0.mean(), v1.mean()], color=color, lw=1.4, marker='o', ms=4, label=genotype)
        ax.scatter(np.zeros(2)+.035, v0, s=17, color=color, alpha=.65, edgecolors='white', linewidth=.25)
        ax.scatter(np.ones(2)+.035, v1, s=17, color=color, alpha=.65, edgecolors='white', linewidth=.25)
    grow = hyp_g[hyp_g.gene_id.eq(gene_id)].iloc[0]
    ax.set_title(f'{title}\ninteraction={grow.interaction:+.2f}; FDR={grow.padj_int:.1e}', fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Normoxia', 'Hypoxia'])
    ax.set_xlim(-.20, 1.20)
    if j == 0:
        ax.set_ylabel('Normalized log2CPM')
        ax.text(-.18, 1.35, 'D', transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')
        ax.text(-.04, 1.35, 'Sample-level regulator responses', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
    if j == 2:
        ax.legend(frameon=False, loc='best')
    clean(ax, True)

# E: complete stringent structural-candidate pool and selection context.
ax_e = fig.add_subplot(outer[1, 2])
colors = [KO if b == 874 else WT if b == 0 else PURPLE for b in candidates['bin']]
ax_e.scatter(candidates.d_ins_interaction, -np.log10(candidates.interaction_padj),
             s=20 + 35*candidates.chip_bin_score_max, c=colors, alpha=.75,
             edgecolors='white', linewidth=.35)
ax_e.axvline(0, color='#AAAAAA', lw=.5)
for bin_id, name, dx, dy in [(874, 'ESX-2\nabs-effect rank 1', 6, 2), (0, 'oriC\nLATR01', 6, 2),
                             (119, 'bin 119\nFDR rank 1', 6, 2), (757, 'bin 757', 6, 2)]:
    row = candidates[candidates['bin'].eq(bin_id)].iloc[0]
    ax_e.annotate(name, (row.d_ins_interaction, -np.log10(row.interaction_padj)),
                  xytext=(dx, dy), textcoords='offset points', fontsize=6.5)
ax_e.set(xlabel='Insulation interaction (Δ log2)', ylabel='−log10 BH FDR')
panel_label(ax_e, 'E', 'All 17 stringent structural candidates')
clean(ax_e, True)

# F-I: four aligned in-situ locus blocks with smoothed display of response maps.
loci = [
    ('F', 'DosR_DosS', 'DosR/DosS region', {'Rv3133c'}),
    ('G', 'SigE', 'SigE region', {'Rv1221'}),
    ('H', 'oriC', 'oriC/dnaA region', {'Rv0001'}),
    ('I', 'ESX2', 'ESX-2 region', {'Rv3886c', 'Rv3890c'}),
]
for i, (label, loc, title, label_genes) in enumerate(loci):
    sub = mpl.gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=outer[i+2, :],
            width_ratios=[1.2, 1.2, .06, 3.0], height_ratios=[1.35, .58, .52, .50, .45],
            hspace=.62, wspace=.25)
    ax_m1 = fig.add_subplot(sub[0:4, 0]); ax_m2 = fig.add_subplot(sub[0:4, 1])
    im = plot_response_map(ax_m1, loc, 'WT'); plot_response_map(ax_m2, loc, 'KO')
    ax_m1.text(-.27, 1.25, label, transform=ax_m1.transAxes, fontsize=15,
               fontweight='bold', va='top')
    ax_m1.text(-.12, 1.25, title, transform=ax_m1.transAxes, fontsize=11,
               fontweight='bold', va='top')
    cax = fig.add_subplot(sub[4, 0:2])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
    cb.set_label('log2(O/E hypoxia / O/E normoxia); 5-kb data, bilinear display interpolation', fontsize=6.5)
    cb.outline.set_visible(False)
    ax_ins = fig.add_subplot(sub[0, 3]); plot_insulation(ax_ins, loc, show_legend=(i == 0))
    ax_r1 = fig.add_subplot(sub[1, 3], sharex=ax_ins)
    plot_rna(ax_r1, loc, 'WT_hypoxia_log2FC', 'WT_hypoxia_FDR', WT, 'WT RNA\nlog2FC', label_genes)
    ax_r2 = fig.add_subplot(sub[2, 3], sharex=ax_ins)
    plot_rna(ax_r2, loc, 'KO_hypoxia_log2FC', 'KO_hypoxia_FDR', KO, 'KO RNA\nlog2FC', label_genes)
    ax_r3 = fig.add_subplot(sub[3, 3], sharex=ax_ins)
    plot_rna(ax_r3, loc, 'interaction_log2FC', 'interaction_FDR', PURPLE, 'Interaction\nRNA log2FC', label_genes)
    ax_g = fig.add_subplot(sub[4, 3], sharex=ax_ins); plot_chip_genes(ax_g, loc, label_genes)

fig.text(.045, .012,
         'RNA clusters describe response shapes only. Structural candidates derive from bin-level evidence rules; neither classification establishes direct causation, mediation or pathway independence.',
         fontsize=7.5, color='#555C61', ha='left')

png = FIG / 'NapM_hypoxia_full_mechanism_evidence.png'
pdf = FIG / 'NapM_hypoxia_full_mechanism_evidence.pdf'
svg = FIG / 'NapM_hypoxia_full_mechanism_evidence.svg'
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, format='pdf', metadata={
        'Title': 'NapM hypoxia: statistical and in-situ multi-omics evidence',
        'Author': 'Multi-omics integration analysis',
    })
    fig.savefig(svg, format='svg')
plt.close(fig)
print(f'Wrote {png}')
print(f'Wrote editable vector {pdf}')
print(f'Wrote editable vector {svg}')
