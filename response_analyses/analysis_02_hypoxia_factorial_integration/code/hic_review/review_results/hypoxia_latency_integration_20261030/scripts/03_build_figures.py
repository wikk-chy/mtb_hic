#!/usr/bin/env python3
"""Build publication-style independent hypoxia and latency multi-omics figures."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC, TAB, FIG = OUT / 'source_data', OUT / 'tables', OUT / 'figures'
N, BIN = 883, 5000
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 10,
                     'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.linewidth': .65, 'axes.edgecolor': '#444444',
                     'xtick.major.width': .55, 'ytick.major.width': .55,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
WT, KO = '#1F4E79', '#B35845'
HYP, KOHYP, LAT = '#8AAFD0', '#E1A294', '#8FB6D9'
CHIP, PURPLE, GREEN, GREY, LIGHT = '#6B5B7A', '#725A7A', '#3F7B70', '#8B9298', '#D8DDE1'
CMAP = mpl.colors.LinearSegmentedColormap.from_list('oe', ['#31688E', '#FFFFFF', '#A44A4A'])
CMAP.set_bad('#D3D5D6')

hyp_b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
lat_b = pd.read_csv(SRC / 'Latency_bin_level.csv')
lat_g = pd.read_csv(SRC / 'Latency_gene_level.csv')
orig_b = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
selection = pd.read_csv(TAB / 'Representative_locus_selection.csv')
mean_maps = pd.read_csv(SRC / 'Representative_locus_mean_contact_maps.csv')
ins_tracks = pd.read_csv(SRC / 'Representative_locus_insulation_tracks.csv')
peaks = pd.read_csv(SRC / 'Representative_locus_NapM_peaks.csv')
signal = pd.read_csv(SRC / 'Representative_locus_NapM_signal.csv')
rna = pd.read_csv(SRC / 'Representative_locus_RNA_tracks.csv')
genes = pd.read_csv(SRC / 'Representative_locus_gene_models.csv')
matched = pd.read_csv(SRC / 'FigHypB_matched_bin_pairs.csv')
match_stats = pd.read_csv(TAB / 'Hypoxia_NapM_matched_tests.csv')
cors = pd.read_csv(TAB / 'Structure_transcription_correlations.csv')
regs = pd.read_csv(TAB / 'Structure_transcription_regressions.csv')


def clean(ax, grid=False):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#E7E7E7', lw=.45, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, label, title):
    ax.text(-.16, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title(title, loc='left', fontweight='bold', pad=9)


def focal_shade(ax, loc, xlim):
    z = ins_tracks[(ins_tracks.locus == loc) & ins_tracks.is_focal]
    for row in z.itertuples():
        ax.axvspan(row.position_kb - 2.5, row.position_kb + 2.5, color='#F0ECEF', alpha=.65, zorder=0)
    ax.set_xlim(*xlim)


def map_matrix(locus, condition, bins):
    z = mean_maps[(mean_maps.locus == locus) & (mean_maps.condition == condition)]
    mat = np.full((len(bins), len(bins)), np.nan)
    lookup = {(int(r.bin1), int(r.bin2)): r.mean_OE for r in z.itertuples()}
    for a, b1 in enumerate(bins):
        for c, b2 in enumerate(bins):
            mat[a, c] = lookup[(b1, b2)]
    return mat


def plot_map(ax, locus, condition, bins, pos, title):
    mat = map_matrix(locus, condition, bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        logm = np.log2(mat)
    extent = [pos[0] - 2.5, pos[-1] + 2.5, pos[0] - 2.5, pos[-1] + 2.5]
    im = ax.imshow(logm, origin='lower', extent=extent, cmap=CMAP, vmin=-1.25, vmax=1.25,
                   interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_linewidth(.45)
    return im


def plot_rna_track(ax, loc, effect, q, color, ylabel, focal_only_labels=True):
    z = rna[rna.locus == loc].copy()
    xlim = ax.get_xlim()
    finite = z[effect].abs().replace([np.inf, -np.inf], np.nan).dropna()
    ymax = max(1.25, float(finite.quantile(.98)) * 1.15 if len(finite) else 1.25)
    ax.axhline(0, color='#A9A9A9', lw=.5)
    for row in z.itertuples():
        x = row.tss_kb; y = getattr(row, effect)
        if not np.isfinite(y):
            continue
        qq = getattr(row, q)
        sig = np.isfinite(qq) and qq < .05 and abs(y) >= 1
        ax.vlines(x, 0, y, color=color if sig else '#AEB5BA', lw=1.15 if sig else .65, alpha=.95)
        ax.scatter([x], [y], s=14 if sig else 5, color=color if sig else '#AEB5BA',
                   edgecolor='white', linewidth=.25, zorder=3)
        if row.is_focal_bin and focal_only_labels:
            label = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
            ax.annotate(label, (x, y), xytext=(0, 3 if y >= 0 else -8), textcoords='offset points',
                        ha='center', va='bottom' if y >= 0 else 'top', fontsize=5.5,
                        color='#333333', rotation=25)
    ax.set_ylim(-ymax, ymax); ax.set_ylabel(ylabel)
    clean(ax)


def plot_gene_models(ax, loc, xlim):
    z = genes[genes.locus == loc].copy()
    for row in z.itertuples():
        y = .65 if row.strand == '+' else -.65
        color = CHIP if row.binding_class in ['promoter_bound', 'gene_body_bound'] else '#8D9398'
        start, end = row.start_kb, row.end_kb
        if end < start:
            start, end = end, start
        ax.add_patch(FancyArrow(start, y, max(end - start, .25), 0, width=.14,
                                head_width=.32, head_length=min(.8, max((end-start)*.25, .18)),
                                length_includes_head=True, facecolor=color, edgecolor='none', alpha=.85))
        if row.is_focal_bin:
            label = row.gene_name if isinstance(row.gene_name, str) and row.gene_name else row.gene_id
            ax.text((start + end) / 2, y + (.32 if y > 0 else -.32), label,
                    ha='center', va='bottom' if y > 0 else 'top', fontsize=6, fontweight='bold')
    ax.axhline(0, color='#D0D0D0', lw=.45)
    ax.set_ylim(-1.25, 1.25); ax.set_yticks([-.65, .65]); ax.set_yticklabels(['−', '+'])
    ax.set_ylabel('Genes'); ax.set_xlabel('Position relative to focal bin (kb)')
    ax.set_xlim(*xlim); clean(ax)


def plot_locus_block(subspec, locrow, model, block_label):
    loc = locrow.locus
    zbins = ins_tracks[ins_tracks.locus == loc].sort_values('position_kb')
    bins = zbins['bin'].to_numpy(int); pos = zbins.position_kb.to_numpy(float)
    xlim = (pos[0] - 2.5, pos[-1] + 2.5)
    conditions = ['WT', 'WT_hypoxia', 'KO', 'KO_hypoxia'] if model == 'Hypoxia' else ['WT', 'WT_latent', 'KO']
    titles = {'WT': 'WT normoxia', 'WT_hypoxia': 'WT hypoxia', 'KO': 'napM-KO normoxia',
              'KO_hypoxia': 'napM-KO hypoxia', 'WT_latent': 'WT latent'}
    gs = mpl.gridspec.GridSpecFromSubplotSpec(6, len(conditions), subplot_spec=subspec,
            height_ratios=[1.35, .72, .42, .55, .55, .48], hspace=.20, wspace=.12)
    map_axes = []
    for j, cond in enumerate(conditions):
        ax = plt.gcf().add_subplot(gs[0, j]); map_axes.append(ax)
        map_image = plot_map(ax, loc, cond, bins, pos, titles[cond])
        for f in str(locrow.focal_bins).split(';'):
            fpos = zbins.loc[zbins['bin'].eq(int(f)), 'position_kb']
            if len(fpos):
                ax.axvline(float(fpos.iloc[0]), color='#333333', lw=.45, ls='--')
                ax.axhline(float(fpos.iloc[0]), color='#333333', lw=.45, ls='--')
    if block_label == 'D1':
        cax = map_axes[-1].inset_axes([.15, .05, .70, .055])
        cb = plt.gcf().colorbar(map_image, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
        cb.ax.set_xlabel('log2(O/E)', fontsize=5, labelpad=1)
        cb.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
        cb.outline.set_visible(False)
    # Use explicit add_subplot for broad tracks.
    ax_ins = plt.gcf().add_subplot(gs[1, :])
    x = pos
    cols = {'WT': (WT, '-'), 'KO': (KO, '-')}
    if model == 'Hypoxia':
        cols.update({'ins_hypoxia': (HYP, '--'), 'ins_KO_hypoxia': (KOHYP, '--')})
        labels = {'WT': 'WT', 'KO': 'KO', 'ins_hypoxia': 'WT hypoxia', 'ins_KO_hypoxia': 'KO hypoxia'}
    else:
        cols.update({'ins_latent': (LAT, '--')})
        labels = {'WT': 'WT', 'KO': 'KO', 'ins_latent': 'WT latent'}
    ob = orig_b.set_index('bin')
    for col, (color, ls) in cols.items():
        source_col = 'ins_WT' if col == 'WT' else 'ins_KO' if col == 'KO' else col
        ax_ins.plot(x, ob.loc[bins, source_col], color=color, ls=ls, lw=1.25, label=labels[col])
    focal_shade(ax_ins, loc, xlim); ax_ins.set_ylabel('Insulation\n(log2)')
    ax_ins.legend(ncol=len(cols), frameon=False, loc='upper right', handlelength=1.6)
    clean(ax_ins, True)
    ax_chip = plt.gcf().add_subplot(gs[2, :], sharex=ax_ins)
    zs = signal[signal.locus == loc]
    if len(zs):
        zs = zs.sort_values('start_kb')
        ax_chip.fill_between(zs.start_kb, 0, zs.signal, color=CHIP, alpha=.55, lw=0, step='mid')
    zp = peaks[peaks.locus == loc]
    for row in zp.itertuples():
        ax_chip.vlines(row.center_kb, 0, row.score * (zs.signal.max() if len(zs) else 1), color=CHIP, lw=.7)
    focal_shade(ax_chip, loc, xlim); ax_chip.set_ylabel('NapM\nChIP')
    clean(ax_chip)
    ax_r1 = plt.gcf().add_subplot(gs[3, :], sharex=ax_ins)
    ax_r2 = plt.gcf().add_subplot(gs[4, :], sharex=ax_ins)
    if model == 'Hypoxia':
        focal_shade(ax_r1, loc, xlim); plot_rna_track(ax_r1, loc, 'WT_hypoxia_log2FC', 'WT_hypoxia_FDR', WT, 'WT hypoxia\nRNA log2FC', focal_only_labels=False)
        focal_shade(ax_r2, loc, xlim); plot_rna_track(ax_r2, loc, 'KO_hypoxia_log2FC', 'KO_hypoxia_FDR', KO, 'KO hypoxia\nRNA log2FC', focal_only_labels=False)
    else:
        focal_shade(ax_r1, loc, xlim); plot_rna_track(ax_r1, loc, 'baseline_KO_WT_log2FC', 'baseline_FDR', KO, 'KO−WT\nRNA log2FC', focal_only_labels=False)
        focal_shade(ax_r2, loc, xlim); plot_rna_track(ax_r2, loc, 'WT_latent_log2FC', 'latent_FDR', LAT, 'Latent−WT\nRNA log2FC', focal_only_labels=False)
    ax_g = plt.gcf().add_subplot(gs[5, :], sharex=ax_ins)
    focal_shade(ax_g, loc, xlim); plot_gene_models(ax_g, loc, xlim)
    ax_ins.set_title(f'{block_label}  {locrow.title}  |  {locrow.evidence_class}', loc='left',
                     fontsize=9, fontweight='bold', pad=28)
    return ax_ins

# ------------------------------- Hypoxia figure -------------------------------
fig = plt.figure(figsize=(18, 19))
outer = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.72, 1.72], hspace=.42, wspace=.28,
                         left=.055, right=.98, top=.96, bottom=.045)
# A: RNA response comparison.
ax = fig.add_subplot(outer[0, 0])
x, y = hyp_g.hyp_response_wt, hyp_g.hyp_response_ko
wtdeg = hyp_g.WT_hypoxia_RNA_class.isin(['detected_increase', 'detected_decrease'])
kodeg = hyp_g.KO_hypoxia_RNA_class.isin(['detected_increase', 'detected_decrease'])
colors = np.select([wtdeg & kodeg, wtdeg & ~kodeg, ~wtdeg & kodeg], [PURPLE, WT, KO], default='#C8CDD1')
ax.scatter(x, y, s=5, c=colors, alpha=.42, edgecolors='none', rasterized=True)
lim = 6
ax.plot([-lim, lim], [-lim, lim], color='#777777', lw=.6, ls='--'); ax.axhline(0, color='#BBBBBB', lw=.45); ax.axvline(0, color='#BBBBBB', lw=.45)
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='WT hypoxia RNA log2FC', ylabel='KO hypoxia RNA log2FC')
ax.text(.03, .97, f'WT DEGs: {int(wtdeg.sum()):,}\nKO DEGs: {int(kodeg.sum()):,}\nBoth: {int((wtdeg&kodeg).sum()):,}',
        transform=ax.transAxes, va='top', linespacing=1.5)
panel_label(ax, 'A', 'Hypoxia transcriptional responses'); clean(ax, True)
# B: matched bound/unbound response distributions.
ax = fig.add_subplot(outer[0, 1])
data, positions, facecolors, labels = [], [], [], []
for j, (col, short) in enumerate([('d_ins_hyp_WT', 'WT response'), ('d_ins_hyp_KO', 'KO response'), ('d_ins_interaction', 'Interaction')]):
    for k, group in enumerate(['bound', 'unbound']):
        vals = matched[f'{group}_{col}'].replace([np.inf, -np.inf], np.nan).dropna()
        data.append(vals); positions.append(j * 3 + k); facecolors.append(CHIP if group == 'bound' else '#B9C0C5')
    labels.extend([short, ''])
bp = ax.boxplot(data, positions=positions, widths=.72, patch_artist=True, showfliers=False,
                medianprops={'color': '#222222', 'lw': 1}, whiskerprops={'lw': .6}, capprops={'lw': .6})
for patch, c in zip(bp['boxes'], facecolors):
    patch.set_facecolor(c); patch.set_alpha(.75); patch.set_edgecolor('#555555'); patch.set_linewidth(.5)
for j, name in enumerate(['WT hypoxia response (signed)', 'KO hypoxia response (signed)', 'Hypoxia genotype interaction (signed)']):
    p = match_stats.loc[match_stats.comparison.eq(name), 'circular_shift_p'].iloc[0]
    ax.text(j * 3 + .5, 1.03, f'paired shift P={p:.3f}', transform=ax.get_xaxis_transform(), ha='center', fontsize=6.5)
ax.axhline(0, color='#888888', lw=.55); ax.set_xticks([.5, 3.5, 6.5]); ax.set_xticklabels(['WT hypoxia', 'KO hypoxia', 'Genotype × hypoxia'])
ax.set_ylabel('Δ insulation (log2)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=CHIP, label='NapM-bound'), Patch(facecolor='#B9C0C5', label='Matched unbound')], frameon=False, loc='lower left')
panel_label(ax, 'B', 'Occupancy-matched structural responses'); clean(ax, True)
# C: structure-RNA interaction cross-analysis.
ax = fig.add_subplot(outer[0, 2])
z = hyp_b[['RNA_interaction_mean_log2FC', 'd_ins_interaction', 'hypoxia_response_evidence_class']].dropna()
cc = np.select([z.hypoxia_response_evidence_class.str.startswith('NapM-associated'),
                z.hypoxia_response_evidence_class.str.startswith('Transcription-associated')], [PURPLE, GREEN], default='#B9C0C5')
ax.scatter(z.RNA_interaction_mean_log2FC, z.d_ins_interaction, s=9, c=cc, alpha=.65, edgecolors='none')
ax.axhline(0, color='#AAAAAA', lw=.5); ax.axvline(0, color='#AAAAAA', lw=.5)
rho = cors.loc[cors.contrast.eq('Hypoxia interaction'), 'spearman_rho'].iloc[0]
p = cors.loc[cors.contrast.eq('Hypoxia interaction'), 'circular_shift_p'].iloc[0]
dr2 = regs.loc[regs.model.eq('Hypoxia interaction'), 'incremental_R2_RNA'].iloc[0]
nint = hyp_g.hypoxia_interaction_RNA_class.isin(['detected_increase', 'detected_decrease']).sum()
ax.text(.03, .97, f'Spearman ρ={rho:.3f}; shift P={p:.3f}\nRNA incremental R²={dr2:.5f}\nRNA interaction DEGs: {int(nint):,} (FDR<0.05, |log2FC|≥1)',
        transform=ax.transAxes, va='top', linespacing=1.5)
ax.set(xlabel='Bin-mean RNA interaction log2FC', ylabel='Insulation interaction')
panel_label(ax, 'C', 'Structure–transcription cross-analysis'); clean(ax, True)
for i, locrow in enumerate(selection[selection.model.eq('Hypoxia')].itertuples()):
    plot_locus_block(outer[i+1, :], locrow, 'Hypoxia', f'D{i+1}')
fig.suptitle('Hypoxia: NapM-associated structural responses are not explained by local transcription alone',
             x=.055, y=.992, ha='left', fontsize=15, fontweight='bold')
for ext in ['pdf', 'png']:
    fig.savefig(FIG / f'Hypoxia_multiomics_figure.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

# ------------------------------- Latency figure -------------------------------
fig = plt.figure(figsize=(22, 20))
outer = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.65, 1.65, 1.65], hspace=.42, wspace=.28,
                         left=.05, right=.985, top=.96, bottom=.045)
# A baseline bin relationship.
ax = fig.add_subplot(outer[0, 0])
z = lat_b[['RNA_baseline_mean_log2FC', 'd_ins_KO_WT', 'NapM_bound']].dropna()
ax.scatter(z.RNA_baseline_mean_log2FC, z.d_ins_KO_WT, s=8, c=np.where(z.NapM_bound, CHIP, '#C3C9CD'), alpha=.55, edgecolors='none')
ax.axhline(0, color='#AAAAAA', lw=.5); ax.axvline(0, color='#AAAAAA', lw=.5)
r = cors.loc[cors.contrast.eq('Baseline KO vs WT')].iloc[0]
ax.text(.03, .97, f'ρ={r.spearman_rho:.3f}; shift P={r.circular_shift_p:.3f}', transform=ax.transAxes, va='top')
ax.set(xlabel='Baseline bin-mean RNA log2FC (KO−WT)', ylabel='Baseline Δinsulation (KO−WT)')
panel_label(ax, 'A', 'Baseline genotype differences'); clean(ax, True)
# B latent bin relationship.
ax = fig.add_subplot(outer[0, 1])
z = lat_b[['RNA_latent_mean_log2FC', 'd_ins_latent_WT', 'NapM_bound']].dropna()
ax.scatter(z.RNA_latent_mean_log2FC, z.d_ins_latent_WT, s=8, c=np.where(z.NapM_bound, CHIP, '#C3C9CD'), alpha=.55, edgecolors='none')
ax.axhline(0, color='#AAAAAA', lw=.5); ax.axvline(0, color='#AAAAAA', lw=.5)
r = cors.loc[cors.contrast.eq('WT latent')].iloc[0]
ax.text(.03, .97, f'ρ={r.spearman_rho:.3f}; shift P={r.circular_shift_p:.3f}', transform=ax.transAxes, va='top')
ax.set(xlabel='Latent bin-mean RNA log2FC (latent−WT)', ylabel='Latent Δinsulation (latent−WT)')
panel_label(ax, 'B', 'WT latency-associated responses'); clean(ax, True)
# C1 structure integration.
ax = fig.add_subplot(outer[0, 2])
z = lat_b.copy(); rel = z.latency_structure_direction_relation
cc = np.select([rel.eq('opposite_antagonistic_association'), rel.eq('same_partial_similar_state')], [KO, WT], default='#D1D5D8')
ax.scatter(z.d_ins_KO_WT, z.d_ins_latent_WT, s=8, c=cc, alpha=.52, edgecolors='none')
for locrow in selection[selection.model.eq('Latency')].itertuples():
    for f in str(locrow.focal_bins).split(';'):
        row = lat_b[lat_b['bin'].eq(int(f))].iloc[0]
        ax.scatter(row.d_ins_KO_WT, row.d_ins_latent_WT, s=42, facecolor='none', edgecolor='#222222', lw=.8)
ax.axhline(0, color='#AAAAAA', lw=.5); ax.axvline(0, color='#AAAAAA', lw=.5)
ax.set(xlabel='Baseline Δinsulation (KO−WT)', ylabel='Latent Δinsulation (latent−WT)')
ax.legend(handles=[Patch(facecolor=KO, label='Opposite'), Patch(facecolor=WT, label='Same / partial similarity'), Patch(facecolor='#D1D5D8', label='Not dual detected')], frameon=False, loc='upper left')
panel_label(ax, 'C1', 'Independent evidence: structure'); clean(ax, True)
# C2 gene integration.
ax = fig.add_subplot(outer[0, 3])
z = lat_g.copy(); rel = z.latency_RNA_direction_relation
cc = np.select([rel.eq('opposite_antagonistic_association'), rel.eq('same_partial_similar_state')], [KO, WT], default='#D1D5D8')
ax.scatter(z.log2FC_KOvsWT, z.log2FC_latentvsWT, s=5, c=cc, alpha=.40, edgecolors='none', rasterized=True)
for locrow in selection[selection.model.eq('Latency')].itertuples():
    zz = z[z.bin_tss.isin([int(f) for f in str(locrow.focal_bins).split(';')]) & z.latency_dual_DEG]
    ax.scatter(zz.log2FC_KOvsWT, zz.log2FC_latentvsWT, s=32, facecolor='none', edgecolor='#222222', lw=.65)
ax.axhline(0, color='#AAAAAA', lw=.5); ax.axvline(0, color='#AAAAAA', lw=.5)
ax.set(xlabel='Baseline RNA log2FC (KO−WT)', ylabel='Latent RNA log2FC (latent−WT)', xlim=(-6, 6), ylim=(-7, 7))
panel_label(ax, 'C2', 'Independent evidence: transcription'); clean(ax, True)
for i, locrow in enumerate(selection[selection.model.eq('Latency')].itertuples()):
    plot_locus_block(outer[i+1, :], locrow, 'Latency', f'D{i+1}')
fig.suptitle('Latency-associated candidate loci from independent baseline and WT-latent analyses',
             x=.05, y=.992, ha='left', fontsize=15, fontweight='bold')
fig.text(.05, .975, 'No KO-latent comparison is performed; same-direction patterns denote partial molecular similarity, not a latency defect.',
         ha='left', fontsize=9, color='#555555')
for ext in ['pdf', 'png']:
    fig.savefig(FIG / f'Latency_multiomics_figure.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Figures written to', FIG)
