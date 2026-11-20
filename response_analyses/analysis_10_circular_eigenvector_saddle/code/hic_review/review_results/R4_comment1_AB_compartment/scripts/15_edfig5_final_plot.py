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


fig = plt.figure(figsize=(10.0, 5.0))
# Four-column heatmaps retain their size; three-column e/f panels are ~25% slimmer.
gs = fig.add_gridspec(2, 14, height_ratios=[1, 1], hspace=0.48, wspace=0.40,
                      left=0.05, right=0.965, top=0.84, bottom=0.13)

c_axes = []
for i, c in enumerate(CONDS[:3]):
    axc = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20q_{c}.npy'), c)
    c_axes.append(axc)
d_axes = []
for i, c in enumerate(CONDS[3:]):
    axc = fig.add_subplot(gs[1, 4 * i:4 * i + 4])
    saddle_panel(axc, np.load(OUT + f'/data/saddle20q_{c}.npy'), c)
    d_axes.append(axc)

# Shared vertical colorbar in the space beside the third upper heatmap.
fig.canvas.draw()
pos = c_axes[-1].get_position()
cax = fig.add_axes([pos.x1 + 0.045, pos.y0, 0.012, pos.height])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                  cax=cax, orientation='vertical', format='%.1f')
cb.set_label('mean O/E contact frequency', fontsize=6, labelpad=7)
cb.set_ticks([0.5, 1, 2])
cb.ax.minorticks_off()
cb.ax.set_yticklabels(['0.5', '1', '2'])
cb.ax.tick_params(labelsize=6)
cb.outline.set_linewidth(0.5)

# ---- panel e: strength bars
axe = fig.add_subplot(gs[1, 8:11])
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
axf = fig.add_subplot(gs[1, 11:14])
sim_conds = ['WT-hypoxia', 'WT-latent', 'napM-KO', 'napM-KO-hypoxia']
sim_v = sim.set_index('condition').loc[sim_conds, 'pearson_r_vs_WT'].values
ceil = sim[sim['condition'].str.contains('ceiling')]['pearson_r_vs_WT'].iloc[0]
xs = np.arange(len(sim_conds))
axf.bar(xs, sim_v, color=[COLORS[1], COLORS[2], COLORS[3], COLORS[4]],
        width=0.65)
axf.axhline(ceil, color='#d62728', lw=1, ls='--')
axf.text(-0.4, ceil + 0.008, f'WT replicate ceiling r = {ceil:.2f}',
         fontsize=5.4, color='#d62728', ha='left', va='bottom')
nullmax_r = sim['null_max'].dropna().iloc[0]
for xi, v in zip(xs, sim_v):
    if v - 0.025 >= 0.8:
        axf.text(xi, v - 0.008, f'{v:.2f}', ha='center', va='top', fontsize=5.8,
                 color='white', fontweight='bold')
    else:
        axf.text(xi, v + 0.006, f'{v:.2f}', ha='center', va='bottom', fontsize=5.8,
                 color='#333333', fontweight='bold')
axf.set_xticks(xs)
axf.set_xticklabels(['WT\nhypoxia', 'WT\nlatent', 'napM\nKO',
                     'napM-KO\nhypoxia'], fontsize=5.4)
axf.set_ylabel('saddle similarity to WT\n(Pearson r)', fontsize=6.5)
axf.set_ylim(0.8, 1.03)
axf.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])

# Keep both lower-right panels slim while opening a clear gap for f's y-axis label.
e_pos = axe.get_position()
axe.set_position([e_pos.x0, e_pos.y0, e_pos.width * 0.85, e_pos.height])
f_pos = axf.get_position()
axf.set_position([f_pos.x1 - f_pos.width * 0.85, f_pos.y0,
                  f_pos.width * 0.85, f_pos.height])

# Panel letters and a scale-aware null note (the null band is below the displayed range).
fig.text(0.005, 0.96, 'c', fontsize=11, fontweight='bold', va='top')
fig.text(0.005, 0.47, 'd', fontsize=11, fontweight='bold', va='top')
fig.text(axe.get_position().x0 - 0.025, 0.47, 'e', fontsize=11,
         fontweight='bold', va='top')
fig.text(axf.get_position().x0 - 0.025, 0.47, 'f', fontsize=11,
         fontweight='bold', va='top')
fig.text(0.965, 0.005,
         f'panel f permutation null: max r of 100 shuffles = {nullmax_r:.2f} '
         '(below displayed axis; all conditions p < 0.01)',
         fontsize=5, ha='right', va='bottom', color='grey')

fig.savefig(OUT + '/figures/EDfig5cde_final.png', dpi=300)
fig.savefig(OUT + '/figures/EDfig5cde_final.pdf')
plt.close(fig)
print('wrote figures/EDfig5cde_final.{png,pdf}')
