#!/usr/bin/env python3
"""
Redraw main-text Fig. 1c and ED Fig. 5c-e with the unified circular pipeline
(circular-distance O/E + correlation-PCA PC1, 38 quantile groups over
qrange (0.025, 0.975), strength in natural log).

Inputs (from 09_circular38_all_conditions.py):
  data/saddle38_<condition>.npy, data/pc1_<condition>.npy, data/pos_<condition>.npy
  tables/control6_circular38_all_conditions.csv

Outputs (300 dpi png + pdf):
  figures/Fig1c_saddle_circular38.{png,pdf}          WT saddle + E1 margins
  figures/EDfig5cde_saddle_circular38.{png,pdf}      5-condition saddles + strength
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

df = pd.read_csv(OUT + '/tables/control6_circular38_all_conditions.csv',
                 index_col='condition')

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
NORM = LogNorm(vmin=0.5, vmax=2)
CMAP = 'coolwarm'


def saddle_panel(ax, Sax, title, strength_txt):
    ax.imshow(Sax, cmap=CMAP, norm=NORM, origin='upper',
              extent=[0, 1, 1, 0])
    k = max(1, int(round(Sax.shape[0] * 0.2))) / Sax.shape[0]
    for (x0, y0) in [(0, 0), (1 - k, 1 - k), (0, 1 - k), (1 - k, 0)]:
        ax.add_patch(plt.Rectangle((x0, y0), k, k, fill=False,
                                   edgecolor='black', lw=0.7))
    ax.set_title(title + '\n' + strength_txt, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------------------------- Fig. 1c redraw
S = np.load(OUT + '/data/saddle38_WT.npy')
pc1 = np.load(OUT + '/data/pc1_WT.npy')
pos = np.load(OUT + '/data/pos_WT.npy')

fig = plt.figure(figsize=(3.4, 3.9))
gs = fig.add_gridspec(3, 3, width_ratios=[0.16, 1, 0.03],
                      height_ratios=[0.16, 1, 0.05],
                      wspace=0.03, hspace=0.03)
ax_top = fig.add_subplot(gs[0, 1])
ax_left = fig.add_subplot(gs[1, 0])
ax = fig.add_subplot(gs[1, 1])

n = S.shape[0]
lo, hi = np.quantile(pc1, [0.025, 0.975])
sel = np.where((pc1 >= lo) & (pc1 <= hi))[0]
order = sel[np.argsort(pc1[sel])]
groups = np.array_split(order, 38)
gmean = [np.nanmean(pc1[g]) for g in groups]
xb = np.linspace(0, 1, n + 1)
xc = (xb[:-1] + xb[1:]) / 2
ax_top.bar(xc, gmean, width=1 / n, color='#2c7fb8', lw=0)
ax_top.set_xlim(0, 1); ax_top.axis('off')
ax_left.barh(xc, gmean, height=1 / n, color='#2c7fb8', lw=0)
ax_left.set_ylim(1, 0); ax_left.axis('off')

saddle_panel(ax, S, '', '')
_k = max(1, int(round(S.shape[0] * 0.2)))
_aabb = np.nanmean(np.concatenate([S[:_k, :_k].ravel(), S[-_k:, -_k:].ravel()]))
_ab = np.nanmean(np.concatenate([S[:_k, -_k:].ravel(), S[-_k:, :_k].ravel()]))
_str_wt = np.log2(_aabb / _ab)
fig.suptitle('WT $\\it{M. tuberculosis}$ saddle plot\n'
             'sorted by E1 (ori$\\mathrm{-}$ter axis);  '
             f'segregation strength = {_str_wt:.2f} log$_2$',
             fontsize=8)
ax.set_xlabel('E1-sorted bins $\\rightarrow$', fontsize=6.5)
ax.set_ylabel('E1-sorted bins $\\rightarrow$', fontsize=6.5)
fig.savefig(OUT + '/figures/Fig1c_saddle_circular38.png', dpi=300,
            bbox_inches='tight')
fig.savefig(OUT + '/figures/Fig1c_saddle_circular38.pdf', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------- ED Fig. 5c-e redraw
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 7, height_ratios=[1, 1],
                      width_ratios=[1, 1, 1, 1, 0.25, 1, 1],
                      hspace=0.4, wspace=0.3)
titles = {
    'WT': 'WT, control', 'WT-hypoxia': 'WT, hypoxia',
    'WT-latent': 'WT, latent', 'napM-KO': '$\\it{napM}$-KO',
    'napM-KO-hypoxia': '$\\it{napM}$-KO, hypoxia',
}
for i, c in enumerate(CONDS[:3]):
    axc = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    Sc = np.load(OUT + f'/data/saddle38_{c}.npy')
    saddle_panel(axc, Sc, titles[c],
                 f"strength = {df.loc[c,'strength_ln']:.2f} ln")
    if i == 0:
        axc.set_title('c', fontsize=10, fontweight='bold', loc='left',
                      pad=28)
for i, c in enumerate(CONDS[3:]):
    axc = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
    Sc = np.load(OUT + f'/data/saddle38_{c}.npy')
    saddle_panel(axc, Sc, titles[c],
                 f"strength = {df.loc[c,'strength_ln']:.2f} ln")
    if i == 0:
        axc.set_title('d', fontsize=10, fontweight='bold', loc='left',
                      pad=28)

# strength bar chart with jackknife CI + shuffle null band
axb = fig.add_subplot(gs[1, 5:7])
x = np.arange(len(CONDS))
vals = df.loc[CONDS, 'strength_ln'].values
se = df.loc[CONDS, 'jackknife_se'].values
colors = ['#d62728', '#fb6a4a', '#fcae91', '#2c7fb8', '#6baed6']
axb.bar(x, vals, yerr=1.96 * se, color=colors, capsize=3,
        error_kw=dict(lw=0.8))
nullmax = df.loc[CONDS, 'shuffle_absmax'].max()
axb.axhspan(-nullmax, nullmax, color='grey', alpha=0.25, lw=0)
axb.axhline(0, color='black', lw=0.6)
axb.text(0.98, 0.97, f'grey band: permutation null\n(max |100 shuffles| = {nullmax:.2f})',
         fontsize=5.5, ha='right', va='top', color='grey',
         transform=axb.transAxes)
axb.set_xticks(x)
axb.set_xticklabels(['WT', 'WT\nhypoxia', 'WT\nlatent', 'napM\nKO',
                     'napM-KO\nhypoxia'], fontsize=5.2)
axb.set_ylabel('segregation strength,\nln(AA·BB/AB²)', fontsize=6.5)
axb.set_title('e', fontsize=10, fontweight='bold', loc='left', pad=28)
fig.savefig(OUT + '/figures/EDfig5cde_saddle_circular38.png', dpi=300,
            bbox_inches='tight')
fig.savefig(OUT + '/figures/EDfig5cde_saddle_circular38.pdf',
            bbox_inches='tight')
plt.close(fig)
print('wrote figures/Fig1c_saddle_circular38.{png,pdf}')
print('wrote figures/EDfig5cde_saddle_circular38.{png,pdf}')
