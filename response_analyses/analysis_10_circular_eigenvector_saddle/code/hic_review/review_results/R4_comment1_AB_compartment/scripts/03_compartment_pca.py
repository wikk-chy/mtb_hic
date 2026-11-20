#!/usr/bin/env python3
"""
A/B compartment plausibility (Reviewer 4, comment 1)
  a. PCA on distance-normalized O/E matrix (5 kb, WT); correlate PC1 with
     GC content and gene density.
  b. Saddle plot (PC1-sorted O/E enrichment) vs shuffled-PC1 control.
  c. Interpretation of PC1 in Mtb.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cooler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUT = BASE + '/review_results/hic_validation'
BIN = 5000
GENOME = 4411532
rng = np.random.default_rng(42)

clr = cooler.Cooler(BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000')
M = clr.matrix(balance=True)[:]
N = M.shape[0]
valid_bins = ~np.isnan(M).all(axis=0)
print(f'valid bins: {valid_bins.sum()}/{N}')
Mv = M[np.ix_(valid_bins, valid_bins)]
n = Mv.shape[0]

# ---------- expected by genomic distance & O/E ----------
exp = np.full(n, np.nan)
for d in range(n):
    diag = np.diagonal(Mv, offset=d)
    diag = diag[np.isfinite(diag) & (diag > 0)]
    exp[d] = diag.mean() if len(diag) else np.nan
OE = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        d = abs(i - j)
        if np.isfinite(exp[d]) and exp[d] > 0 and Mv[i, j] > 0:
            OE[i, j] = Mv[i, j] / exp[d]
oe_valid = np.isfinite(OE).all(axis=0)
OE = OE[np.ix_(oe_valid, oe_valid)]
n2 = OE.shape[0]
print(f'bins after O/E filtering: {n2}')

# ---------- PCA on correlation matrix of O/E ----------
C = np.corrcoef(OE)
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
pc1 = eigvecs[:, 0]
pc2 = eigvecs[:, 1]
var1 = 100 * eigvals[0] / eigvals.sum()
var2 = 100 * eigvals[1] / eigvals.sum()
print(f'PC1 variance explained: {var1:.1f}%, PC2: {var2:.1f}%')

# keep track of which original bins survive
orig_idx = np.where(valid_bins)[0][oe_valid]

# ---------- per-bin GC content and gene density ----------
seq = []
with open(BASE + '/genome.fa') as fh:
    for line in fh:
        if not line.startswith('>'):
            seq.append(line.strip())
seq = ''.join(seq).upper()
assert len(seq) == GENOME

gtf = pd.read_csv(BASE + '/annotation.gtf', sep='\t', comment='#', header=None,
                  usecols=[2, 3, 4], names=['feature', 'start', 'end'])
genes = gtf[gtf.feature == 'gene']

gc = np.zeros(N)
gd = np.zeros(N)
for b in range(N):
    s0, s1 = b * BIN, min((b + 1) * BIN, GENOME)
    sub = seq[s0:s1]
    gc[b] = (sub.count('G') + sub.count('C')) / len(sub) if sub else np.nan
    gd[b] = ((genes['start'] <= s1) & (genes['end'] >= s0)).sum()

gc_v, gd_v = gc[orig_idx], gd[orig_idx]

# orient PC1 so it correlates positively with gene density (arbitrary sign)
if stats.pearsonr(pc1, gd_v)[0] < 0:
    pc1 = -pc1

print('\n== 3a. PC1 correlations ==')
for name, vec in [('GC content', gc_v), ('gene density', gd_v)]:
    pr, pp = stats.pearsonr(pc1, vec)
    sr, sp = stats.spearmanr(pc1, vec)
    print(f'  PC1 vs {name}: Pearson r={pr:.3f} (p={pp:.2e}), '
          f'Spearman rho={sr:.3f} (p={sp:.2e})')
# also GC vs gene density baseline
pr, _ = stats.pearsonr(gc_v, gd_v)
print(f'  (baseline GC vs gene density: r={pr:.3f})')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
pos = (orig_idx * BIN + BIN / 2) / 1e6
axes[0].scatter(pos, pc1, s=8, c=np.where(pc1 > 0, 'red', 'blue'))
axes[0].axhline(0, color='grey', lw=0.5)
axes[0].set_xlabel('genomic position (Mb)')
axes[0].set_ylabel('PC1')
axes[0].set_title(f'PC1 along genome ({var1:.0f}% var.)')
axes[1].scatter(gc_v, pc1, s=8)
axes[1].set_xlabel('GC content per 5 kb bin')
axes[1].set_ylabel('PC1')
axes[1].set_title(f'PC1 vs GC (r={stats.pearsonr(pc1,gc_v)[0]:.2f})')
axes[2].scatter(gd_v, pc1, s=8)
axes[2].set_xlabel('gene count per 5 kb bin')
axes[2].set_ylabel('PC1')
axes[2].set_title(f'PC1 vs gene density (r={stats.pearsonr(pc1,gd_v)[0]:.2f})')
plt.tight_layout()
plt.savefig(OUT + '/PC1_GC_genedensity.png', dpi=200)
plt.close()

# ---------- is PC1 just a Fourier mode of the circular chromosome? ----------
print('\n== 3a2. PC structure vs circular-genome Fourier modes ==')
pos_bp = orig_idx * BIN + BIN / 2
dori = np.minimum(pos_bp, GENOME - pos_bp)  # distance from oriC (position 0)
for k in range(4):
    pc = eigvecs[:, k]
    best, bk = 0, None
    for harm in (1, 2, 3):
        for f, nm in ((np.cos, 'cos'), (np.sin, 'sin')):
            r = abs(stats.pearsonr(pc, f(2 * np.pi * harm * pos_bp / GENOME))[0])
            if r > best:
                best, bk = r, (harm, nm)
    r_ori = stats.pearsonr(pc, dori)[0]
    print(f'  PC{k+1} ({100*eigvals[k]/eigvals.sum():.1f}% var): '
          f'best sinusoid |r|={best:.3f} ({bk[0]}*{bk[1]}), '
          f'r(dist from oriC)={r_ori:.3f}')
print('  top eigenvalue fractions:',
      np.round(eigvals[:6] / eigvals.sum(), 3))

# ---------- b. saddle plot ----------
NGROUP = 50
def saddle(pc, groups=NGROUP):
    order_idx = np.argsort(pc)
    # trim to multiple of groups
    k = (n2 // groups) * groups
    order_idx = order_idx[:k]
    grp = np.repeat(np.arange(groups), k // groups)
    S = np.zeros((groups, groups))
    Cn = np.zeros((groups, groups))
    for a in range(groups):
        ia = order_idx[grp == a]
        for b in range(groups):
            ib = order_idx[grp == b]
            block = OE[np.ix_(ia, ib)]
            vals = block[np.isfinite(block)]
            if len(vals):
                S[a, b] = vals.mean()
                Cn[a, b] = len(vals)
    return S

def saddle_strength(S, corner=5):
    """(AA+BB corner mean)/(AB corner mean), log2."""
    aa = np.concatenate([S[:corner, :corner].ravel(), S[-corner:, -corner:].ravel()])
    ab = np.concatenate([S[:corner, -corner:].ravel(), S[-corner:, :corner].ravel()])
    return np.log2(np.nanmean(aa) / np.nanmean(ab))

S_obs = saddle(pc1)
str_obs = saddle_strength(S_obs)
print(f'\n== 3b. Saddle analysis ==')
print(f'  observed saddle strength (log2((AA+BB)/AB)) = {str_obs:.3f}')

NSHUF = 100
shuf_str = np.empty(NSHUF)
S_shuf_mean = np.zeros_like(S_obs)
for s in range(NSHUF):
    pcs = rng.permutation(pc1)
    Ss = saddle(pcs)
    shuf_str[s] = saddle_strength(Ss)
    S_shuf_mean += Ss
S_shuf_mean /= NSHUF
pct = 100 * np.mean(shuf_str <= str_obs)
print(f'  shuffled-PC1 saddle strength: mean={shuf_str.mean():.3f} '
      f'(95% range {np.percentile(shuf_str,2.5):.3f}..{np.percentile(shuf_str,97.5):.3f})')
print(f'  observed percentile vs shuffle: {pct:.0f}% '
      f'(empirical p ~ {np.mean(shuf_str >= str_obs):.3f})')

# structural controls: is the saddle signal just the ori-ter axis?
cos_axis = np.cos(2 * np.pi * pos_bp / GENOME)
str_cos = saddle_strength(saddle(cos_axis))
str_dori = saddle_strength(saddle(dori))
str_lin = saddle_strength(saddle(pos_bp.astype(float)))
print(f'  control saddles: pure cos(2pi*pos/L) = {str_cos:.3f}, '
      f'dist-from-oriC = {str_dori:.3f}, linear position = {str_lin:.3f}')
print('  => if saddle(PC1) ~ saddle(cos axis), the plaid pattern is fully '
      'explained by the circular distance-decay / ori-ter macro-structure, '
      'not by eukaryotic-style A/B compartments')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
vmax = np.nanpercentile(np.abs(S_obs - 1), 95)
im = axes[0].imshow(S_obs, cmap='RdBu_r', vmin=1 - vmax, vmax=1 + vmax)
axes[0].set_title(f'saddle (PC1-sorted)\nstrength = {str_obs:.2f} log2')
axes[0].set_xlabel('PC1 rank group (low -> high)')
axes[0].set_ylabel('PC1 rank group (low -> high)')
plt.colorbar(im, ax=axes[0], label='mean O/E')
vmax2 = np.nanpercentile(np.abs(S_shuf_mean - 1), 95)
im = axes[1].imshow(S_shuf_mean, cmap='RdBu_r', vmin=1 - vmax2, vmax=1 + vmax2)
axes[1].set_title('saddle (shuffled PC1, mean of 100)')
axes[1].set_xlabel('shuffled rank group')
plt.colorbar(im, ax=axes[1], label='mean O/E')
axes[2].hist(shuf_str, bins=20, color='grey', alpha=0.7, label='shuffled')
axes[2].axvline(str_obs, color='red', lw=2, label=f'observed = {str_obs:.2f}')
axes[2].set_xlabel('saddle strength log2((AA+BB)/AB)')
axes[2].set_ylabel('count')
axes[2].set_title('observed vs shuffled saddle strength')
axes[2].legend()
plt.tight_layout()
plt.savefig(OUT + '/saddle_plot_vs_shuffle.png', dpi=200)
plt.close()

print('\nDone.')
