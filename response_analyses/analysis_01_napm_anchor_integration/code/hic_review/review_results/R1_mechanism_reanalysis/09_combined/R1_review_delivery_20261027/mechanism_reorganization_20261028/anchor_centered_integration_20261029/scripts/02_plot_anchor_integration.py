#!/usr/bin/env python3
"""Plot anchor-centred gene integration and representative multitrack loci."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, PathPatch, Patch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
BASE = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
OUT = BASE / 'anchor_centered_integration_20261029'
TAB = OUT / 'tables'
SRC = OUT / 'source_data'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

TEAL = '#287D8E'
PURPLE = '#6B4C9A'
GOLD = '#D69E2E'
UP = '#C4514B'
DOWN = '#4779A8'
WT = '#287D8E'
KO = '#C05A47'
NAPM = '#73529B'
GREY = '#66717A'
LIGHT = '#D8DEE3'
DET = {'detected_increase', 'detected_decrease'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 1: set overlaps, window sensitivity, UpSet and pathway evidence.
# -------------------------------------------------------------------------
g = pd.read_csv(SRC / 'anchor_and_three_way_gene_membership.csv')
g = g[g.RNA_tested].copy()  # All set overlaps and browser RNA tracks use this universe.
sens = pd.read_csv(TAB / 'anchor_window_sensitivity.csv').set_index('window_bp')
combos = pd.read_csv(TAB / 'three_way_exact_combinations.csv').set_index('combination_code')
enrich = pd.read_csv(TAB / 'anchor_and_three_way_GO_KEGG_enrichment.csv')
assert len(g[g.RNA_tested]) == 4060
n_anchor = int(g.in_changed_anchor.sum())
n_deg = int(g.DEG.sum())
n_both = int((g.in_changed_anchor & g.DEG).sum())
assert (n_anchor, n_deg, n_both) == (406, 820, 79)

fig = plt.figure(figsize=(15.5, 11.7))
gs = fig.add_gridspec(2, 2, left=.075, right=.97, top=.835, bottom=.13,
                      hspace=.52, wspace=.34)

# A: two-set Venn. Areas are schematic; all values are printed explicitly.
ax = fig.add_subplot(gs[0, 0])
ax.add_patch(Circle((-.57, .03), 1.03, facecolor=TEAL, edgecolor=TEAL,
                    alpha=.24, lw=2.0))
ax.add_patch(Circle((.57, .03), 1.03, facecolor=PURPLE, edgecolor=PURPLE,
                    alpha=.20, lw=2.0))
ax.text(-.87, .05, f'{n_anchor-n_both}', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#205E6B')
ax.text(0, .05, f'{n_both}', ha='center', va='center', fontsize=24,
        fontweight='bold', color='#423064')
ax.text(.90, .05, f'{n_deg-n_both}', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#573B79')
ax.text(-.70, 1.22, f'Changed-anchor genes\n(n={n_anchor}; TSS ±10 kb)',
        ha='center', va='bottom', color=TEAL, fontweight='bold')
ax.text(.75, 1.22, f'Strict DEGs\n(n={n_deg})', ha='center', va='bottom',
        color=PURPLE, fontweight='bold')
ax.text(0, -1.25, f'Neither set: {4060-(n_anchor+n_deg-n_both):,}  |  RNA-tested universe: 4,060',
        ha='center', fontsize=8.8, color=GREY)
ax.text(0, -1.43, 'Unique genes; shared anchors are counted once. Circles are not area-scaled.',
        ha='center', fontsize=8.0, color=GREY)
ax.set_xlim(-1.95, 1.95)
ax.set_ylim(-1.55, 1.60)
ax.axis('off')
ax.set_title('A  79 DEGs lie near changed-loop anchors', loc='left', pad=12)

# B: window sensitivity, with total RNA-tested genes and directional DEGs.
ax = fig.add_subplot(gs[0, 1])
x = np.arange(3)
total = sens.loc[[5000, 10000, 25000], 'changed_anchor_RNA_tested'].to_numpy()
up = sens.loc[[5000, 10000, 25000], 'changed_anchor_up'].to_numpy()
down = sens.loc[[5000, 10000, 25000], 'changed_anchor_down'].to_numpy()
ax.axvspan(.55, 1.45, color=GOLD, alpha=.09, lw=0)
ax.bar(x, total, width=.66, color='#E4E9EC', edgecolor='#B7C1C8', lw=.7,
       label='RNA-tested genes near anchors')
ax.bar(x, up, width=.40, color=UP, label='Upregulated DEGs')
ax.bar(x, down, bottom=up, width=.40, color=DOWN, label='Downregulated DEGs')
for i, (n, u, d) in enumerate(zip(total, up, down)):
    ax.text(i, n+20, f'{n}', ha='center', fontweight='bold', color='#45515A')
    ax.text(i, u+d+18, f'{u+d} DEGs', ha='center', fontsize=8.3,
            color='#343D44', fontweight='bold')
ax.set_xticks(x, ['±5 kb', '±10 kb\n(primary)', '±25 kb'])
ax.set_ylim(0, 960)
ax.set_ylabel('Unique genes')
ax.set_xlabel('TSS distance from each changed-loop anchor')
ax.grid(axis='y', color='#E8ECEF', lw=.7, zorder=0)
ax.legend(frameon=False, loc='upper left', fontsize=8.2)
ax.set_title('B  The overlap grows with window size', loc='left', pad=9)
ax.text(.98, .72, '48 up / 31 down\nat ±10 kb', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='#4A4330',
        bbox=dict(boxstyle='round,pad=.35', fc='#FFF9E9', ec='#E6D49C'))

# C: full three-way UpSet with all eight exact combinations.
sub = gs[1, 0].subgridspec(2, 1, height_ratios=[1.55, 1.0], hspace=.06)
ax_bar = fig.add_subplot(sub[0])
ax_mat = fig.add_subplot(sub[1], sharex=ax_bar)
order = ['---', '-I-', 'N--', '--D', '-ID', 'NI-', 'N-D', 'NID']
counts = combos.loc[order, 'n_genes'].to_numpy()
x = np.arange(len(order))
colors = [PURPLE if c == 'NID' else '#AAB5BC' for c in order]
ax_bar.axvspan(6.55, 7.45, color=PURPLE, alpha=.07, lw=0)
ax_mat.axvspan(6.55, 7.45, color=PURPLE, alpha=.07, lw=0)
ax_bar.bar(x, counts, color=colors, width=.67)
for i, n in enumerate(counts):
    ax_bar.text(i, n+45, f'{n}', ha='center', fontsize=8.2,
                color=PURPLE if order[i] == 'NID' else GREY,
                fontweight='bold' if order[i] == 'NID' else 'normal')
ax_bar.set_ylim(0, 2070)
ax_bar.set_ylabel('Exact combination size')
ax_bar.set_xlim(-.6, 7.6)
ax_bar.tick_params(axis='x', labelbottom=False, length=0)
ax_bar.grid(axis='y', color='#E8ECEF', lw=.7, zorder=0)
ax_bar.set_title('C  45 genes share NapM occupancy, insulation change and DEG status',
                 loc='left', pad=9)
set_sizes = [int(g.DEG.sum()), int(g.insulation_changed.sum()), int(g.NapM_occupied.sum())]
ax_mat.set_yticks([2, 1, 0], [f'DEG ({set_sizes[0]})',
                              f'Insulation change ({set_sizes[1]})',
                              f'NapM occupied ({set_sizes[2]})'])
ax_mat.set_ylim(-.65, 2.65)
for i, code in enumerate(order):
    active = [2-j for j, c in enumerate(code) if c != '-']
    # code order is N, I, D; matrix rows are D=2, I=1, N=0.
    active = []
    for j, c in enumerate(code):
        if c == 'N': active.append(0)
        if c == 'I': active.append(1)
        if c == 'D': active.append(2)
    ax_mat.scatter([i]*3, [0, 1, 2], s=31, color='#E0E5E8', zorder=2)
    if active:
        ax_mat.plot([i, i], [min(active), max(active)], color=PURPLE if code == 'NID' else '#596770',
                    lw=1.7, zorder=3)
        ax_mat.scatter([i]*len(active), active, s=48,
                       color=PURPLE if code == 'NID' else '#596770', zorder=4)
ax_mat.set_xticks([])
ax_mat.tick_params(length=0)
ax_mat.spines[['left', 'bottom']].set_visible(False)
ax_mat.grid(axis='y', color='#EEF1F3', lw=.7, zorder=0)
ax_mat.set_xlabel('All 4,060 RNA-tested genes; N = NapM, I = insulation change, D = DEG',
                  fontsize=8.5, color=GREY)

# D: ABC transporters across pre-specified enrichment backgrounds.
ax = fig.add_subplot(gs[1, 1])
context_order = [
    'changed_anchor_up_vs_eligible_anchor_DEGs',
    'changed_anchor_all_vs_eligible_anchor_DEGs',
    'changed_anchor_DEGs_vs_eligible_anchor_RNA_tested',
    'triple_vs_all_DEGs',
    'triple_vs_RNA_tested',
]
labels = [
    'Anchor up-DEGs\nvs other anchor up-DEGs',
    'Anchor all-DEGs\nvs other anchor all-DEGs',
    'Anchor DEGs\nvs anchor RNA-tested genes',
    'Triple intersection\nvs all DEGs',
    'Triple intersection\nvs RNA-tested genes',
]
z = enrich[enrich.term.eq('mtu02010')].set_index('gene_set').loc[context_order].reset_index()
y = np.arange(len(z))[::-1]
for yi, row in zip(y, z.itertuples()):
    primary = row.gene_set == context_order[0]
    color = PURPLE if primary else '#98A4AB'
    ax.hlines(yi, 0, row.fold_enrichment, color=color, lw=2.3)
    ax.scatter(row.fold_enrichment, yi, s=70 if primary else 48,
               color=color, marker='o' if row.overlap else 'x', zorder=3)
    ax.text(5.85, yi, f'{row.overlap} genes; q={row.q_BH_all_tested_terms:.3g}',
            ha='right', va='center', fontsize=8.4,
            color=PURPLE if primary else GREY,
            fontweight='bold' if primary else 'normal')
ax.axvline(1, color='#69747B', lw=.9, ls=(0, (3, 3)))
ax.set_yticks(y, labels, fontsize=8.4)
ax.set_xlim(-.12, 6.0)
ax.set_ylim(-1.25, 4.35)
ax.set_xlabel('ABC transporters: fold enrichment')
ax.grid(axis='x', color='#E8ECEF', lw=.7, zorder=0)
ax.set_title('D  Only the directional conditional test passes FDR', loc='left', pad=9)
ax.text(0, -.62, 'Only FDR-significant GO/KEGG result: 6/11 annotated up-DEGs; q=0.0106.\n'
                 'Zero-fold triple rows have no ABC-transporter genes (q=1), not evidence of depletion.',
        fontsize=8.0, color=GREY, va='top')

fig.suptitle('Anchor-centred integration identifies focused candidate gene sets',
             x=.055, y=.962, ha='left', fontsize=16, fontweight='bold')
fig.text(.055, .916,
         'The analysis uses genes near the 22 anchor instances of 11 changed loops, rather than every gene between two distal anchors.',
         ha='left', fontsize=10.2, color=GREY)
fig.text(.055, .066,
         'Strict DEG: RNA BH FDR < 0.05 and |log₂FC| ≥ 1. NapM denotes baseline promoter/gene-body occupancy, not differential binding. '
         'Insulation change is the existing detected-increase/decrease class.',
         ha='left', fontsize=8.7, color=GREY)
fig.text(.055, .036,
         'Enrichment uses direction-matched eligible-anchor DEG backgrounds where indicated; q values are BH-adjusted across all tested terms. '
         'Overlap and co-variation do not establish direct regulation or contact-to-RNA causality.',
         ha='left', fontsize=8.7, color=GREY)
save(fig, 'R1_anchor_centered_integration')


# -------------------------------------------------------------------------
# Figure 2: representative multitrack loci.
# -------------------------------------------------------------------------
loops = pd.read_csv(BASE / 'loop_change_enrichment/per_loop_change_and_binding.csv').set_index('loop_id')
peaks = pd.read_csv(RA / '09_combined/tables/Fig1_final_A_peaks.csv')
bins = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')

source_rows = []
locus_summaries = []
for loop_id in ['L00', 'L32']:
    row = loops.loc[loop_id]
    source_rows.append(dict(locus=loop_id, loop_id=loop_id, track='loop', id=loop_id,
                            start=row.center1, end=row.center2, center=row.loop_midpoint,
                            value=row.WT_mean, value2=row.KO_mean, delta=row.d_log2OE,
                            condition='', class_='changed_loop', label='', strand='',
                            anchor_id='', RNA_tested=np.nan, DEG=np.nan,
                            NapM_occupied=np.nan, insulation_changed=np.nan,
                            triple_intersection=np.nan, in_changed_anchor=np.nan))


def add_anchor_records(loop_id):
    row = loops.loc[loop_id]
    for side, center, occupied in [
        ('left', row.center1, row.anchor1_bound),
        ('right', row.center2, row.anchor2_bound),
    ]:
        aid = f'{loop_id}_{side}'
        common = dict(locus=loop_id, loop_id=loop_id, id=aid, center=center,
                      condition='', label='', strand='', anchor_id=aid,
                      RNA_tested=np.nan, DEG=np.nan, insulation_changed=np.nan,
                      triple_intersection=np.nan)
        source_rows.append(dict(**common, track='anchor', start=center-2500, end=center+2500,
                                value=np.nan, value2=np.nan, delta=np.nan,
                                class_='NapM_occupied' if occupied else 'NapM_unoccupied',
                                NapM_occupied=occupied, in_changed_anchor=True))
        source_rows.append(dict(**common, track='anchor_window', start=center-10000, end=center+10000,
                                value=np.nan, value2=np.nan, delta=np.nan,
                                class_='TSS_plus_minus_10kb', NapM_occupied=np.nan,
                                in_changed_anchor=True))


for loop_id in ['L00', 'L32']:
    add_anchor_records(loop_id)

configs = [
    dict(loop_id='L00', xlim=(0, 100000), candidate='Rv0001', candidate_label='dnaA',
         title='A  L00 / oriC–dnaA: literature-anchored control locus'),
    dict(loop_id='L32', xlim=(3540000, 3680000), candidate='Rv3196A', candidate_label='Rv3196A',
         title='B  L32 / Rv3196A: independent anchor-centred candidate'),
]

fig = plt.figure(figsize=(15.3, 14.1))
outer = fig.add_gridspec(2, 1, left=.095, right=.975, top=.895, bottom=.075, hspace=.29)

for panel_index, cfg in enumerate(configs):
    loop_id = cfg['loop_id']
    x0, x1 = cfg['xlim']
    lr = loops.loc[loop_id]
    z_peaks = peaks[peaks.center.between(x0, x1)].copy().sort_values('center')
    z_bins = bins[bins.end.gt(x0) & bins.start.lt(x1)].copy().sort_values('start')
    z_genes = g[g.RNA_tested & g.tss.between(x0, x1)].copy().sort_values('tss')
    z_models = g[g.DEG & g.end.gt(x0) & g.start.lt(x1)].copy().sort_values('start')
    candidate = g[g.gene_id.eq(cfg['candidate'])].iloc[0]
    assert candidate.in_changed_anchor and candidate.triple_intersection

    for r in z_peaks.itertuples():
        source_rows.append(dict(locus=loop_id, loop_id=loop_id, track='NapM_peak',
                                id=f'{r.target_gene}:{r.center}', start=r.start, end=r.end,
                                center=r.center, value=r.score, value2=r.vpm, delta=np.nan,
                                condition='baseline', class_='curated_NapM_peak', label=r.target_gene,
                                strand='', anchor_id='', RNA_tested=np.nan, DEG=np.nan,
                                NapM_occupied=True, insulation_changed=np.nan,
                                triple_intersection=np.nan, in_changed_anchor=np.nan))
    for r in z_bins.itertuples():
        source_rows.append(dict(locus=loop_id, loop_id=loop_id, track='insulation_bin',
                                id=str(r.bin), start=r.start, end=r.end,
                                center=(r.start+r.end)/2, value=r.ins_WT, value2=r.ins_KO,
                                delta=r.d_ins_KO_WT, condition='WT_and_KO',
                                class_=r.ins_change_class, label='', strand='', anchor_id='',
                                RNA_tested=np.nan, DEG=np.nan, NapM_occupied=r.chip_bound,
                                insulation_changed=r.ins_change_class in DET,
                                triple_intersection=np.nan, in_changed_anchor=np.nan))
    for r in z_genes.itertuples():
        source_rows.append(dict(locus=loop_id, loop_id=loop_id, track='RNA_gene', id=r.gene_id,
                                start=r.start, end=r.end, center=r.tss, value=r.log2FC_KOvsWT,
                                value2=r.padj_KOvsWT, delta=np.nan, condition='KO_vs_WT',
                                class_='detected_increase' if r.up else ('detected_decrease' if r.down else 'not_DEG'),
                                label=r.gene_name if isinstance(r.gene_name, str) else '',
                                strand=r.strand, anchor_id=r.changed_anchor_ids if isinstance(r.changed_anchor_ids, str) else '',
                                RNA_tested=r.RNA_tested, DEG=r.DEG, NapM_occupied=r.NapM_occupied,
                                insulation_changed=r.insulation_changed,
                                triple_intersection=r.triple_intersection,
                                in_changed_anchor=r.in_changed_anchor))
    locus_summaries.append(dict(
        locus=loop_id, view_start=x0, view_end=x1, candidate=cfg['candidate'],
        candidate_name=cfg['candidate_label'], loop_WT_mean_OE=lr.WT_mean,
        loop_KO_mean_OE=lr.KO_mean, loop_d_log2OE=lr.d_log2OE,
        n_NapM_peaks=len(z_peaks), n_insulation_bins=len(z_bins),
        n_detected_insulation_bins=int(z_bins.ins_change_class.isin(DET).sum()),
        n_RNA_tested_genes_in_view=len(z_genes), n_DEGs_in_view=int(z_genes.DEG.sum()),
        n_anchor_DEGs_in_view=int((z_genes.in_changed_anchor & z_genes.DEG).sum()),
        n_triple_genes_in_view=int(z_genes.triple_intersection.sum()),
        note='Linear 0–100 kb view; wrap-around anchor genes are retained in genome-wide sets but not drawn here.'
             if loop_id == 'L00' else 'Linear local view; all displayed coordinates are H37Rv.'))

    sub = outer[panel_index].subgridspec(5, 1, height_ratios=[1.15, .70, 1.12, 1.28, .72], hspace=.16)
    axes = [fig.add_subplot(sub[i]) for i in range(5)]
    ax_loop, ax_chip, ax_ins, ax_rna, ax_gene = axes
    for ax in axes[1:]:
        ax.sharex(ax_loop)
    for ax in axes:
        ax.set_xlim(x0, x1)
        for center in [lr.center1, lr.center2]:
            ax.axvspan(max(x0, center-10000), min(x1, center+10000),
                       color=GOLD, alpha=.075, lw=0, zorder=0)
        if ax != ax_gene:
            ax.tick_params(axis='x', labelbottom=False)

    # Loop arc and anchor windows.
    mid = (lr.center1+lr.center2)/2
    ax_loop.set_ylim(-.22, 1.10)
    for condition, mean_oe, color, y_text in [
        ('WT', lr.WT_mean, WT, .89), ('napM-KO', lr.KO_mean, KO, .58),
    ]:
        height = .65*mean_oe/lr.WT_mean
        path = MplPath([(lr.center1, .10), (mid, .10+height), (lr.center2, .10)],
                       [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax_loop.add_patch(PathPatch(path, facecolor='none', edgecolor=color,
                                    linewidth=2.8, capstyle='round'))
        ax_loop.text(.99, y_text, f'{condition}: mean O/E={mean_oe:.2f}',
                     transform=ax_loop.transAxes, ha='right', va='top', color=color,
                     fontsize=8.8, fontweight='bold')
    for center, side, occupied in [(lr.center1, 'left', lr.anchor1_bound),
                                   (lr.center2, 'right', lr.anchor2_bound)]:
        ax_loop.add_patch(plt.Rectangle((center-2500, .015), 5000, .10,
                                        facecolor=NAPM if occupied else '#98A4AB',
                                        edgecolor='white', lw=.6, zorder=4))
        ax_loop.text(center, -.065, f'{side} anchor', ha='center', va='top', fontsize=7.8, color=GREY)
    ax_loop.text(mid, .99, f'{loop_id}  |  Δlog₂(KO/WT contact O/E)={lr.d_log2OE:.2f}',
                 ha='center', va='top', fontsize=9.2, color='#303A41', fontweight='bold')
    ax_loop.axis('off')
    ax_loop.set_title(cfg['title'], loc='left', pad=10)

    # Baseline NapM peak track.
    ax_chip.vlines(z_peaks.center, 0, z_peaks.score, color=NAPM, lw=1.2, alpha=.85)
    ax_chip.set_ylim(0, 1.08)
    ax_chip.set_yticks([0, .5, 1.0])
    ax_chip.set_ylabel('NapM\npeak score')
    ax_chip.grid(axis='y', color='#E8ECEF', lw=.6, zorder=0)
    ax_chip.text(.995, .91, f'{len(z_peaks)} baseline peaks', transform=ax_chip.transAxes,
                 ha='right', va='top', fontsize=8, color=NAPM)

    # WT/KO insulation with detected-change bin shading.
    for r in z_bins[z_bins.ins_change_class.isin(DET)].itertuples():
        ax_ins.axvspan(max(x0, r.start), min(x1, r.end),
                       color=UP if r.d_ins_KO_WT > 0 else DOWN, alpha=.15, lw=0)
    centers = (z_bins.start+z_bins.end)/2
    ax_ins.plot(centers, z_bins.ins_WT, color=WT, lw=1.55, marker='o', ms=2.7, label='WT')
    ax_ins.plot(centers, z_bins.ins_KO, color=KO, lw=1.55, marker='o', ms=2.7, label='napM-KO')
    ymin = np.nanmin([z_bins.ins_WT.min(), z_bins.ins_KO.min()])
    ymax = np.nanmax([z_bins.ins_WT.max(), z_bins.ins_KO.max()])
    pad = max(.15, (ymax-ymin)*.13)
    ax_ins.set_ylim(ymin-pad, ymax+pad)
    ax_ins.set_ylabel('Insulation\nscore (5-kb bins)')
    ax_ins.grid(axis='y', color='#E8ECEF', lw=.6, zorder=0)
    ax_ins.legend(frameon=False, loc='upper right', ncol=2, fontsize=8, handlelength=1.5)

    # Gene-level RNA. Values outside +/-4 are clipped visually and labelled.
    values = z_genes.log2FC_KOvsWT.clip(-4, 4)
    nondeg = ~z_genes.DEG
    ax_rna.vlines(z_genes.loc[nondeg, 'tss'], 0, values[nondeg], color='#CBD2D7', lw=.65, alpha=.75)
    for direction, color in [(True, UP), (False, DOWN)]:
        mask = z_genes.DEG & (z_genes.log2FC_KOvsWT.gt(0) == direction)
        ax_rna.vlines(z_genes.loc[mask, 'tss'], 0, values[mask], color=color, lw=1.0, alpha=.90)
        ax_rna.scatter(z_genes.loc[mask, 'tss'], values[mask], s=12, color=color, lw=0, zorder=3)
    anchor_deg = z_genes.in_changed_anchor & z_genes.DEG
    ax_rna.scatter(z_genes.loc[anchor_deg, 'tss'], values[anchor_deg], s=48,
                   facecolor=[UP if v > 0 else DOWN for v in z_genes.loc[anchor_deg, 'log2FC_KOvsWT']],
                   edgecolor='#222A30', linewidth=.65, zorder=5)
    ax_rna.axhline(0, color='#4B555C', lw=.8)
    ax_rna.axhline(1, color='#AAB4BB', lw=.65, ls=(0, (2, 3)))
    ax_rna.axhline(-1, color='#AAB4BB', lw=.65, ls=(0, (2, 3)))
    ax_rna.set_ylim(-4.45, 4.45)
    ax_rna.set_yticks([-4, -2, 0, 2, 4])
    ax_rna.set_ylabel('RNA log₂(KO/WT)\n(clipped at ±4)')
    ax_rna.grid(axis='y', color='#EEF1F3', lw=.6, zorder=0)
    # Candidate annotation.
    cy = float(np.clip(candidate.log2FC_KOvsWT, -4, 4))
    if loop_id == 'L00':
        xytext = (9000, -3.25)
    else:
        xytext = (candidate.tss+16500, -3.35)
    ax_rna.annotate(f"{cfg['candidate_label']}: {candidate.log2FC_KOvsWT:.2f}\nNapM+ / Δinsulation+ / DEG",
                    xy=(candidate.tss, cy), xytext=xytext, fontsize=8.2,
                    color='#2F3338', ha='center', va='center',
                    arrowprops=dict(arrowstyle='-|>', color='#4B555C', lw=.8),
                    bbox=dict(boxstyle='round,pad=.28', fc='white', ec='#C9D1D6', alpha=.94))
    clipped = z_genes[z_genes.log2FC_KOvsWT.abs().gt(4)]
    for r in clipped.itertuples():
        yy = 4 if r.log2FC_KOvsWT > 0 else -4
        ax_rna.scatter(r.tss, yy, marker='^' if yy > 0 else 'v', s=52,
                       color=UP if yy > 0 else DOWN, zorder=6)
        ax_rna.text(r.tss+1200, yy-.35 if yy > 0 else yy+.35,
                    f'{r.gene_id}: {r.log2FC_KOvsWT:.2f} (clipped)',
                    fontsize=7.8, color=UP if yy > 0 else DOWN, va='top' if yy > 0 else 'bottom')

    # Strand-aware models for strict DEGs; only representative candidate is labelled.
    ax_gene.set_ylim(-.20, .94)
    ax_gene.set_yticks([])
    ax_gene.spines[['left', 'top', 'right']].set_visible(False)
    for i, r in enumerate(z_models.itertuples()):
        y0 = .12 if i % 2 == 0 else .51
        color = UP if r.log2FC_KOvsWT > 0 else DOWN
        x_start, dx = (r.start, r.end-r.start) if r.strand == '+' else (r.end, r.start-r.end)
        arrow = FancyArrow(x_start, y0, dx, 0, width=.12, head_width=.25,
                           head_length=min(500, max(130, abs(dx)*.30)),
                           length_includes_head=True, facecolor=color,
                           edgecolor='#20282D' if r.in_changed_anchor else 'white',
                           linewidth=.65 if r.in_changed_anchor else .35, alpha=.90)
        ax_gene.add_patch(arrow)
        if r.gene_id == cfg['candidate']:
            ax_gene.text((r.start+r.end)/2, y0+.19, cfg['candidate_label'],
                         ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#263138')
    ax_gene.set_ylabel('Strict DEG\ngene models')
    if loop_id == 'L00':
        ax_gene.set_xticks(np.arange(0, 100001, 20000))
        ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v/1000:.0f}'))
        ax_gene.set_xlabel('H37Rv position (kb; linear origin-proximal view)')
    else:
        ax_gene.set_xticks(np.arange(3540000, 3680001, 20000))
        ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v/1e6:.2f}'))
        ax_gene.set_xlabel('H37Rv position (Mb)')

fig.suptitle('Representative loci show coincident structural and transcriptional signals',
             x=.055, y=.967, ha='left', fontsize=16, fontweight='bold')
fig.text(.055, .930,
         'Gold shading: anchor-centred ±10 kb gene-assignment windows. Dark rectangles in RNA: anchor-proximal DEGs. '
         'Insulation shading: red increase / blue decrease in KO.',
         ha='left', fontsize=9.4, color=GREY)
fig.text(.095, .032,
         'NapM peaks and gene occupancy are baseline measurements, not KO–WT differential binding. L00/dnaA is a literature-anchored control; '
         'L32/Rv3196A is a high-ranking independent candidate.\n'
         'These tracks integrate loops, anchors, NapM, insulation and RNA at the same coordinates, but do not demonstrate NapM → structure → transcription causality.',
         ha='left', va='bottom', fontsize=8.7, color=GREY)
save(fig, 'R1_representative_anchor_loci')

# Export exactly the plotted source records and locus-level summary.
source = pd.DataFrame(source_rows)
source.to_csv(SRC / 'representative_browser_source.csv', index=False)
pd.DataFrame(locus_summaries).to_csv(TAB / 'representative_locus_summary.csv', index=False)
print('Wrote:', FIG / 'R1_anchor_centered_integration.pdf')
print('Wrote:', FIG / 'R1_representative_anchor_loci.pdf')
print('Browser source rows:', len(source))
print(pd.DataFrame(locus_summaries).to_string(index=False))
