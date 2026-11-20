#!/usr/bin/env python3
"""Plot KO-recovery rates and the gene overlap used for pathway enrichment."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
BASE = ROOT / ('hic_review/review_results/R1_mechanism_reanalysis/09_combined/'
               'R1_review_delivery_20261027/mechanism_reorganization_20261028')
OUT = BASE / 'changed_loop_pathway_enrichment'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


# ---------------------------------------------------------------------
# 1. Independent KO recovery at the prespecified primary +/-1-bin tolerance.
recall = pd.read_csv(BASE / 'loop_recall_sensitivity/wt_loop_ko_recall_per_loop.csv')
groups = [
    ('Both anchors\nNapM-bound', recall.n_bound_anchors.eq(2), '#73529B'),
    ('0 or 1 anchor\nNapM-bound', recall.n_bound_anchors.le(1), '#8798A5'),
]
rows = []
for label, mask, color in groups:
    n = int(mask.sum())
    recovered = int(recall.loc[mask, 'recovered_primary'].sum())
    lo, hi = wilson(recovered, n)
    rows.append(dict(group=label.replace('\n', ' '), n_loops=n,
                     n_recovered=recovered, n_not_recovered=n - recovered,
                     recovery_rate=recovered / n,
                     wilson95_ci_lo=lo, wilson95_ci_hi=hi, color=color))
bar_data = pd.DataFrame(rows)
assert bar_data.n_loops.tolist() == [17, 24]
assert bar_data.n_recovered.tolist() == [6, 14]
bar_data.drop(columns='color').to_csv(OUT / 'recovery_bar_data.csv', index=False)

table = np.array([[14, 10], [6, 11]])  # reference recovered/not; target recovered/not
fisher_p = stats.fisher_exact(table, alternative='two-sided').pvalue
assert np.isclose(fisher_p, 0.20816092687277477)

fig, ax = plt.subplots(figsize=(7.0, 5.8))
fig.subplots_adjust(left=0.15, right=0.96, top=0.77, bottom=0.22)
x = np.arange(2)
rates = 100 * bar_data.recovery_rate.to_numpy()
yerr = np.vstack([
    100 * (bar_data.recovery_rate - bar_data.wilson95_ci_lo),
    100 * (bar_data.wilson95_ci_hi - bar_data.recovery_rate),
])
bars = ax.bar(x, rates, width=0.58, color=[r[2] for r in groups],
              yerr=yerr, capsize=5, error_kw=dict(lw=1.2, capthick=1.2, ecolor='#333333'),
              zorder=2)
for xi, bar, row in zip(x, bars, bar_data.itertuples()):
    ax.text(xi, 7, f'{row.n_recovered}/{row.n_loops}', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold')
    ax.text(xi, rates[xi] + 2.2, f'{rates[xi]:.1f}%', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#222222')
ax.set_xticks(x, [r[0] for r in groups])
ax.set_ylabel('WT-called loops recovered\nin independent KO call (%)')
ax.set_ylim(0, 106)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', color='#E4E7E9', linewidth=0.8, zorder=0)
ax.set_title('Independent KO recovery by NapM occupancy', loc='left', fontweight='bold', pad=12)
ax.plot([0, 0, 1, 1], [91, 93, 93, 91], color='#444444', lw=1.0)
ax.text(0.5, 94.5, f'Two-sided Fisher P = {fisher_p:.3f} (n.s.)',
        ha='center', va='bottom', fontsize=9.5, color='#444444')
fig.text(0.15, 0.105,
         'Recovery: both anchors match an independent KO call within ±1 bin (5 kb).\n'
         'Error bars: Wilson 95% CIs. Non-recovery is not proof of physical loop disappearance.',
         ha='left', va='bottom', fontsize=8.7, color='#51585E')
fig.savefig(OUT / 'R1_KO_recovery_by_NapM_occupancy.pdf', bbox_inches='tight')
fig.savefig(OUT / 'R1_KO_recovery_by_NapM_occupancy.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------
# 2. Two-set Venn: region genes intersect genome-wide strict DEGs.
g = pd.read_csv(OUT / 'changed_loop_gene_membership.csv')
region = g.in_changed_loop.eq(True)
deg = g.is_KO_DEG.eq(True)
intersection = region & deg
n_region = int(region.sum())
n_deg = int(deg.sum())
n_both = int(intersection.sum())
n_region_only = n_region - n_both
n_deg_only = n_deg - n_both
n_up = int((intersection & g.log2FC_KOvsWT.gt(0)).sum())
n_down = int((intersection & g.log2FC_KOvsWT.lt(0)).sum())
assert (n_region, n_both, n_up, n_down) == (1135, 212, 142, 70)
venn_counts = pd.DataFrame([
    dict(category='Genes in 11 changing-loop regions', n_genes=n_region),
    dict(category='All strict KO-vs-WT DEGs', n_genes=n_deg),
    dict(category='Region genes only', n_genes=n_region_only),
    dict(category='DEGs only', n_genes=n_deg_only),
    dict(category='Intersection used for GO/KEGG', n_genes=n_both),
    dict(category='Intersection upregulated', n_genes=n_up),
    dict(category='Intersection downregulated', n_genes=n_down),
])
venn_counts.to_csv(OUT / 'gene_venn_counts.csv', index=False)

fig, ax = plt.subplots(figsize=(8.0, 6.2))
fig.subplots_adjust(left=0.05, right=0.95, top=0.78, bottom=0.16)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')
left = Circle((4.0, 2.85), 2.35, facecolor='#3C7F9E', edgecolor='#2F6C86',
              alpha=0.34, linewidth=2.0)
right = Circle((6.0, 2.85), 2.35, facecolor='#B26A3B', edgecolor='#95562F',
               alpha=0.34, linewidth=2.0)
ax.add_patch(left)
ax.add_patch(right)
ax.text(2.85, 5.25, 'Genes in 11 changing\nloop regions', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color='#2F6C86')
ax.text(7.15, 5.25, 'Strict KO-vs-WT DEGs\ngenome-wide', ha='center', va='center',
        fontsize=11.5, fontweight='bold', color='#95562F')
ax.text(2.85, 3.15, f'{n_region_only:,}', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#234E60')
ax.text(2.85, 2.65, 'region only', ha='center', va='center', fontsize=9.5, color='#234E60')
ax.text(7.15, 3.15, f'{n_deg_only:,}', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#70401F')
ax.text(7.15, 2.65, 'DEGs only', ha='center', va='center', fontsize=9.5, color='#70401F')
ax.text(5.0, 3.55, f'{n_both}', ha='center', va='center',
        fontsize=23, fontweight='bold', color='#222222')
ax.text(5.0, 2.95, 'used for GO/KEGG', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#222222')
ax.text(5.0, 2.45, f'{n_up} up / {n_down} down', ha='center', va='center',
        fontsize=9.5, color='#3E4449')
ax.text(2.85, 0.75, f'{n_region:,} region genes', ha='center', va='center',
        fontsize=9.5, color='#2F6C86')
ax.text(7.15, 0.75, f'{n_deg:,} total DEGs', ha='center', va='center',
        fontsize=9.5, color='#95562F')
ax.set_title('Genes selected from strongly changing loop regions',
             loc='left', fontweight='bold', pad=12, fontsize=14)
fig.text(0.08, 0.065,
         'Region assignment: gene TSS between loop anchors. DEG: RNA BH FDR < 0.05 and |log$_2$FC| ≥ 1.\n'
         'Overlapping loops are deduplicated at the gene level; circle areas are schematic.',
         ha='left', va='bottom', fontsize=8.8, color='#51585E')
fig.savefig(OUT / 'R1_changed_loop_DEG_gene_venn.pdf', bbox_inches='tight')
fig.savefig(OUT / 'R1_changed_loop_DEG_gene_venn.png', dpi=400, bbox_inches='tight')
plt.close(fig)

print('Recovery bars:')
print(bar_data.drop(columns='color').to_string(index=False))
print(f'Two-sided Fisher P = {fisher_p:.12f}')
print('\nVenn counts:')
print(venn_counts.to_string(index=False))
print('\nWrote recovery and Venn figures to', OUT)
