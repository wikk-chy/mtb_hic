#!/usr/bin/env python3
"""Render the independent KO-recall sensitivity without replacing Fig. 1C."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / ('hic_review/review_results/R1_mechanism_reanalysis/09_combined/'
              'R1_review_delivery_20261027/mechanism_reorganization_20261028/'
              'loop_recall_sensitivity')
loops = pd.read_csv(OUT / 'wt_loop_ko_recall_per_loop.csv')
summary = pd.read_csv(OUT / 'ko_recall_group_summary.csv')
tests = pd.read_csv(OUT / 'ko_recall_binding_tests.csv')
continuous = pd.read_csv(OUT / 'ko_recall_continuous_signal_context.csv')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
colors = {0: '#A8AFB8', 1: '#3C8DAD', 2: '#73529B'}
fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.0), gridspec_kw={'width_ratios': [0.9, 1.25]})
fig.subplots_adjust(top=0.80, bottom=0.20, wspace=0.28)

# A: Recovery by occupied-anchor count at the primary +/-1-bin tolerance.
ax = axes[0]
s = summary[(summary.tolerance_bins == 1) & summary.anchor_group.isin(['0', '1', '2'])].copy()
s['anchor_group'] = s.anchor_group.astype(int)
s = s.sort_values('anchor_group')
x = np.arange(3)
rates = 100 * s.recovery_rate.to_numpy()
ax.bar(x, rates, width=0.62, color=[colors[i] for i in s.anchor_group], zorder=2)
for xi, (_, row) in zip(x, s.iterrows()):
    ax.text(xi, row.recovery_rate * 100 + 2.2,
            f"{int(row.n_recovered)}/{int(row.n_loops)}", ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_xticks(x, ['0', '1', '2'])
ax.set_xlabel('NapM-occupied anchors per WT-called loop')
ax.set_ylabel('Loops recovered in independent KO call (%)')
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', color='#E6E8EB', linewidth=0.7, zorder=0)
ax.set_title('A  KO recall by NapM occupancy', loc='left', fontweight='bold')
t = tests[(tests.tolerance_bins == 1) & tests.comparison.eq('both_vs_0_or_1')].iloc[0]
ax.text(0.97, 0.97,
        f"Double-anchor vs 0/1\nNon-recovery: {100*t.target_nonrecovery_rate:.1f}% vs "
        f"{100*t.reference_nonrecovery_rate:.1f}%\n"
        f"Fisher P = {t.fisher_two_sided_p:.3f}\nSpan-stratified P = {t.span_stratified_permutation_two_sided_p:.3f}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.6,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#F5F6F8', edgecolor='#D5D8DC'))

# B: Binary recovery does not map cleanly onto continuous same-coordinate loss.
ax = axes[1]
q = loops[loops.d_log2OE.notna()].copy()
q['status'] = np.where(q.recovered_primary, 0, 1)
rng = np.random.default_rng(42)
for status in [0, 1]:
    subset = q[q.status == status]
    jitter = rng.uniform(-0.13, 0.13, len(subset))
    for group in [0, 1, 2]:
        points = subset[subset.n_bound_anchors == group]
        if points.empty:
            continue
        xj = jitter[subset.n_bound_anchors.to_numpy() == group]
        ax.scatter(np.full(len(points), status) + xj, points.d_log2OE,
                   s=44, color=colors[group], alpha=0.82,
                   edgecolor='white', linewidth=0.55,
                   label=str(group) if status == 0 else '_nolegend_', zorder=3)
    values = subset.d_log2OE.to_numpy()
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    ax.plot([status, status], [q25, q75], color='#222222', linewidth=1.2, zorder=4)
    ax.plot([status - 0.18, status + 0.18], [med, med], color='#222222', linewidth=2.2, zorder=4)
ax.axhline(0, color='#666666', linestyle=(0, (3, 3)), linewidth=0.9, zorder=1)
counts = q.groupby('status').size()
ax.set_xticks([0, 1], [f"Recovered\n(n = {counts.get(0, 0)})",
                       f"Not recovered\n(n = {counts.get(1, 0)})"])
ax.set_ylabel('Same-coordinate loop change\nlog$_2$(KO mean O/E / WT mean O/E)')
ax.set_title('B  Non-recovery is not equivalent to loop loss', loc='left', fontweight='bold')
ax.set_xlim(-0.45, 1.45)
ax.grid(axis='y', color='#E6E8EB', linewidth=0.7, zorder=0)
c = continuous[continuous.tolerance_bins == 1].iloc[0]
ax.text(0.97, 0.97,
        f"Mann–Whitney P = {c.mannwhitney_two_sided_p:.3f}\n"
        f"{int(c.n_not_recovered_with_KO_OE_ge_1)}/{int(c.n_not_recovered)} non-recovered: KO O/E ≥ 1\n"
        f"{int(c.n_not_recovered_with_d_log2OE_ge_0)}/{int(c.n_not_recovered)} non-recovered: no weakening",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.6,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#F5F6F8', edgecolor='#D5D8DC'))
ax.legend(title='Occupied anchors', frameon=False, fontsize=8.2, title_fontsize=8.2,
          loc='lower left', ncol=3, columnspacing=1.0, handletextpad=0.4)

fig.suptitle('Independent KO Chromosight recall: a sensitivity analysis, not a disappearance assay',
             x=0.055, y=0.955, ha='left', fontsize=14, fontweight='bold')
fig.text(0.055, 0.895,
         'WT: 41 calls; napM-KO: 100 calls; primary matching tolerance: ±1 bin (5 kb) per anchor.',
         ha='left', fontsize=9.5, color='#4B5056')
pvals = tests[tests.comparison.eq('both_vs_0_or_1')].sort_values('tolerance_bins')
fig.text(0.055, 0.065,
         'Double-anchor non-recovery, span-stratified P: exact = '
         f"{pvals.iloc[0].span_stratified_permutation_two_sided_p:.3f}; "
         f"±1 bin = {pvals.iloc[1].span_stratified_permutation_two_sided_p:.3f}; "
         f"±2 bins = {pvals.iloc[2].span_stratified_permutation_two_sided_p:.3f}. "
         'Continuous same-coordinate O/E remains the primary evidence.',
         ha='left', fontsize=8.7, color='#4B5056')
fig.savefig(OUT / 'R1_Response_Figure1C_KO_recall_sensitivity.pdf', bbox_inches='tight')
fig.savefig(OUT / 'R1_Response_Figure1C_KO_recall_sensitivity.png', dpi=400, bbox_inches='tight')
print('Wrote independent KO-recall sensitivity figure to', OUT)
