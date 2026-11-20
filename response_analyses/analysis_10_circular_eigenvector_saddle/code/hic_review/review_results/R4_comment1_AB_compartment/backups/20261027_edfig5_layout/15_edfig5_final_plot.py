#!/usr/bin/env python3
"""
Final ED Fig. 5c-f: 20 E1-quantile groups, qrange (0.025, 0.975),
outer-20% corner blocks (k=4), strength = log2(mean(AA u BB)/mean(AB)),
jackknife error bars + pairwise z-test p brackets (panel e),
saddle-pattern similarity to WT with replicate ceiling (panel f).

Reads outputs of 14_final_saddle_and_similarity.py.
Output (300 dpi): figures/EDfig5cde_final.{png,pdf}
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

df = pd.read_csv(OUT + '/tables/control9_final_saddle.csv', index_col='condition')
pv = pd.read_csv(OUT + '/tables/control9_pairwise_pvalues.csv')
sim = pd.read_csv(OUT + '/tables/control9_saddle_similarity.csv')

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
NORM = LogNorm(vmin=0.5, vmax=2)
CMAP = 'coolwarm'
TITLES = {
    'WT': 'WT, control', 'WT-hypoxia': 'WT, hypoxia',
    'WT-latent': 'WT, latent', 'napM-KO': '$\\it{napM}$-KO',
    'napM-KO-hypoxia': '$\\it{napM}$-KO, hypoxia',
}
COLORS = ['#d62728', '#fb6a4a', '#fcae91', '#2c7fb8', '#6baed6']


def saddle_panel(ax, Sax, cond):
    ax.imshow(Sax, cmap=CMAP, norm=NORM, origin='upper', extent=[0, 1, 1, 0])
    k = max(1, int(round(Sax.shape[0] * 0.2))) / Sax.shape[0]
    for (x0, y0) in [(0, 0), (1 - k, 1 - k), (0, 1 - k), (1 - k, 0)]:
        ax.add_patch(plt.Rectangle((x0, y0), k, k, fill=False,
                                   edgecolor='black', lw=0.7))
    ax.set_title(f"{TITLES[cond]}\n"
                 f"strength = {df.loc[cond,'strength_log2']:.2f} log$_2$",
                 fontsize=7, pad=3)
    ax.set_xticks([]); ax.set_yticks([])


fig = plt.figure(figsize=(10.4, 5.2))
gs = fig.add_gridspec(2, 8, height_ratios=[1, 1], hspace=0.45, wspace=0.35,
                      left=0.045, right=0.98, top=0.86, bottom=0.12)

for i, c in enumerate(CONDS[:3]):
    axc = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20q_{c}.npy'), c)
for i, c in enumerate(CONDS[3:]):
    axc = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20q_{c}.npy'), c)

cax = fig.add_axes([0.80, 0.62, 0.17, 0.02])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                  cax=cax, orientation='horizontal', format='%.1f')
cb.set_label('mean O/E contact frequency', fontsize=6)
cb.set_ticks([0.5, 1, 2])
cb.ax.minorticks_off()
cb.ax.set_xticklabels(['0.5', '1', '2'])
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)

# ---- panel e: strength bars
axe = fig.add_subplot(gs[1, 4:6])
x = np.arange(len(CONDS))
vals = df.loc[CONDS, 'strength_log2'].values
se = df.loc[CONDS, 'jackknife_se'].values
axe.bar(x, vals, yerr=1.96 * se, color=COLORS, capsize=2.5,
        error_kw=dict(lw=0.8))
nullmax = df.loc[CONDS, 'shuffle_absmax'].max()
axe.axhspan(-nullmax, nullmax, color='grey', alpha=0.25, lw=0)
axe.axhline(0, color='black', lw=0.6)
axe.set_xticks(x)
axe.set_xticklabels(['WT', 'WT\nhypoxia', 'WT\nlatent', 'napM\nKO',
                     'napM-KO\nhypoxia'], fontsize=5.4)
axe.set_ylabel('segregation strength\nlog$_2$((AA+BB)/AB)', fontsize=6.5)


def fmt_p(p):
    if p < 1e-4:
        return f'{p:.0e}'
    return f'{p:.3f}'.rstrip('0').rstrip('.')


idx = {c: i for i, c in enumerate(CONDS)}
brackets = [('WT', 'WT-hypoxia'), ('napM-KO', 'napM-KO-hypoxia'),
            ('WT', 'napM-KO'), ('WT', 'WT-latent')]
pmap = dict(zip(pv['comparison'], pv['p_value']))
top = (vals + 1.96 * se).max()
y0 = top * 1.08
dy = top * 0.14
for k, (a, b) in enumerate(brackets):
    p = pmap.get(f'{a} vs {b}', np.nan)
    xa, xb = idx[a], idx[b]
    y = y0 + k * dy
    axe.plot([xa, xa, xb, xb], [y, y + dy * 0.25, y + dy * 0.25, y],
             lw=0.7, color='black')
    axe.text((xa + xb) / 2, y + dy * 0.32, f'p={fmt_p(p)}', ha='center',
             va='bottom', fontsize=5.4)
axe.set_ylim(top=y0 + len(brackets) * dy + dy * 0.9)

# ---- panel f: saddle similarity to WT
axf = fig.add_subplot(gs[1, 6:8])
sim_conds = ['WT-hypoxia', 'WT-latent', 'napM-KO', 'napM-KO-hypoxia']
sim_v = sim.set_index('condition').loc[sim_conds, 'pearson_r_vs_WT'].values
ceil = sim[sim['condition'].str.contains('ceiling')]['pearson_r_vs_WT'].iloc[0]
xs = np.arange(len(sim_conds))
axf.bar(xs, sim_v, color=[COLORS[1], COLORS[2], COLORS[3], COLORS[4]],
        width=0.65)
axf.axhline(ceil, color='#d62728', lw=1, ls='--')
axf.text(-0.4, ceil + 0.015, f'WT replicate ceiling r = {ceil:.2f}',
         fontsize=5.4, color='#d62728', ha='left', va='bottom')
nullmax_r = sim['null_max'].dropna().iloc[0]
axf.axhspan(0, nullmax_r, color='grey', alpha=0.25, lw=0)
for xi, v in zip(xs, sim_v):
    axf.text(xi, v - 0.03, f'{v:.2f}', ha='center', va='top', fontsize=5.8,
             color='white', fontweight='bold')
axf.set_xticks(xs)
axf.set_xticklabels(['WT\nhypoxia', 'WT\nlatent', 'napM\nKO',
                     'napM-KO\nhypoxia'], fontsize=5.4)
axf.set_ylabel('saddle similarity to WT\n(Pearson r)', fontsize=6.5)
axf.set_ylim(0, 1.12)

# panel letters
fig.text(0.005, 0.97, 'c', fontsize=11, fontweight='bold', va='top')
fig.text(0.005, 0.47, 'd', fontsize=11, fontweight='bold', va='top')
fig.text(0.475, 0.47, 'e', fontsize=11, fontweight='bold', va='top')
fig.text(0.725, 0.47, 'f', fontsize=11, fontweight='bold', va='top')
fig.text(0.98, 0.005,
         f'panel f grey band: permutation null (max r of 100 shuffles = {nullmax_r:.2f}; all conditions p < 0.01)',
         fontsize=5, ha='right', va='bottom', color='grey')

fig.savefig(OUT + '/figures/EDfig5cde_final.png', dpi=300)
fig.savefig(OUT + '/figures/EDfig5cde_final.pdf')
plt.close(fig)
print('wrote figures/EDfig5cde_final.{png,pdf}')
