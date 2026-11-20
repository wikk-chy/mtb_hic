#!/usr/bin/env python3
"""
Redraw ED Fig. 5c-e: 20 quantile groups over qrange (0.02, 0.98), strength =
log2(muAA*muBB/muAB^2) over same- vs cross-axis half quadrants (no corner
restriction), jackknife error bars, pairwise jackknife z-test p-values.

Reads outputs of 11_edfig5_20group_compute.py.
Output (300 dpi png + pdf):
  figures/EDfig5cde_saddle_circular20group.{png,pdf}
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUT = BASE + '/review_results/R4_comment1_AB_compartment'
CONDS = ['WT', 'WT-hypoxia', 'WT-latent', 'napM-KO', 'napM-KO-hypoxia']

df = pd.read_csv(OUT + '/tables/control7_circular20group_all_conditions.csv',
                 index_col='condition')
pv = pd.read_csv(OUT + '/tables/control7_circular20group_pairwise_pvalues.csv')

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
NORM = LogNorm(vmin=0.5, vmax=2)
CMAP = 'coolwarm'
TITLES = {
    'WT': 'WT, control', 'WT-hypoxia': 'WT, hypoxia',
    'WT-latent': 'WT, latent', 'napM-KO': '$\\it{napM}$-KO',
    'napM-KO-hypoxia': '$\\it{napM}$-KO, hypoxia',
}


def saddle_panel(ax, Sax, cond):
    ax.imshow(Sax, cmap=CMAP, norm=NORM, origin='upper', extent=[0, 1, 1, 0],
              aspect='auto')
    ax.axvline(0.5, color='black', lw=0.4, ls=':')
    ax.axhline(0.5, color='black', lw=0.4, ls=':')
    ax.set_title(f"{TITLES[cond]}\n"
                 f"strength = {df.loc[cond,'strength_log2']:.2f} log$_2$",
                 fontsize=7, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


# 8-column grid: equal heatmap cells; col 5 is a spacer before the bar chart
fig = plt.figure(figsize=(7.6, 5.2))
gs = fig.add_gridspec(2, 8, width_ratios=[1, 1, 1, 1, 1, 0.35, 1, 1],
                      height_ratios=[1, 1], hspace=0.42, wspace=0.3,
                      left=0.05, right=0.97, top=0.86, bottom=0.1)

axes_hm = {}
for i, c in enumerate(CONDS[:3]):
    axc = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20_{c}.npy'), c)
    axes_hm[c] = axc
for i, c in enumerate(CONDS[3:]):
    axc = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20_{c}.npy'), c)
    axes_hm[c] = axc

# shared horizontal colorbar above the bar-chart region
cax = fig.add_axes([0.72, 0.60, 0.24, 0.018])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                  cax=cax, orientation='horizontal', format='%.1f')
cb.set_label('mean O/E contact frequency', fontsize=6)
cb.set_ticks([0.5, 1, 2])
cb.ax.minorticks_off()
cb.ax.set_xticklabels(['0.5', '1', '2'])
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)

# panel e: bar chart + jackknife CI + pairwise p-value brackets
axb = fig.add_subplot(gs[1, 6:8])
x = np.arange(len(CONDS))
vals = df.loc[CONDS, 'strength_log2'].values
se = df.loc[CONDS, 'jackknife_se'].values
colors = ['#d62728', '#fb6a4a', '#fcae91', '#2c7fb8', '#6baed6']
axb.bar(x, vals, yerr=1.96 * se, color=colors, capsize=2.5,
        error_kw=dict(lw=0.8))
nullmax = df.loc[CONDS, 'shuffle_absmax'].max()
axb.axhspan(-nullmax, nullmax, color='grey', alpha=0.25, lw=0)
axb.axhline(0, color='black', lw=0.6)
axb.set_xticks(x)
axb.set_xticklabels(['WT', 'WT\nhypoxia', 'WT\nlatent', 'napM\nKO',
                     'napM-KO\nhypoxia'], fontsize=5.4)
axb.set_ylabel('segregation strength\nlog$_2$(AA·BB/AB²)', fontsize=6.5)

# p-value brackets (short spans at the bottom)
def fmt_p(p):
    if p < 1e-4:
        return f'{p:.0e}'
    return f'{p:.3f}'.rstrip('0').rstrip('.')

idx = {c: i for i, c in enumerate(CONDS)}
brackets = [('WT', 'WT-hypoxia'), ('napM-KO', 'napM-KO-hypoxia'),
            ('WT', 'WT-latent'), ('WT', 'napM-KO')]
pmap = dict(zip(pv['comparison'], pv['p_value']))
top = (vals + 1.96 * se).max()
y0 = top * 1.08
dy = top * 0.14
for k, (a, b) in enumerate(brackets):
    p = pmap.get(f'{a} vs {b}', np.nan)
    xa, xb = idx[a], idx[b]
    y = y0 + k * dy
    axb.plot([xa, xa, xb, xb], [y, y + dy * 0.25, y + dy * 0.25, y],
             lw=0.7, color='black')
    axb.text((xa + xb) / 2, y + dy * 0.32, f'p={fmt_p(p)}', ha='center',
             va='bottom', fontsize=5.4)
axb.set_ylim(top=y0 + len(brackets) * dy + dy * 0.9)

# panel letters + null-band note
fig.text(0.005, 0.97, 'c', fontsize=11, fontweight='bold', va='top')
fig.text(0.005, 0.47, 'd', fontsize=11, fontweight='bold', va='top')
fig.text(0.70, 0.47, 'e', fontsize=11, fontweight='bold', va='top')
fig.text(0.97, 0.005,
         f'grey band: permutation null (max |100 shuffles| = {nullmax:.2f})',
         fontsize=5, ha='right', va='bottom', color='grey')

fig.savefig(OUT + '/figures/EDfig5cde_saddle_circular20group.png', dpi=300)
fig.savefig(OUT + '/figures/EDfig5cde_saddle_circular20group.pdf')
plt.close(fig)
print('wrote figures/EDfig5cde_saddle_circular20group.{png,pdf}')
