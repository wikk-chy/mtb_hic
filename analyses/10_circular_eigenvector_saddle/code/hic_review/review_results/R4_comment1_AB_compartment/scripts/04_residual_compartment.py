#!/usr/bin/env python3
"""
Control 3 for Reviewer 4 comment 1: residual compartment analysis.

Pipeline:
  1. Build WT O/E matrix normalized by CIRCULAR genomic distance
     (expected contact per circular distance, balanced 5 kb matrix).
  2. Remove the ori-ter axis component by per-distance regression:
     for every bin pair (i,j) we fit, separately at each circular distance
     d,  O/E[i,j] = a_d + b_d * |cos(theta_i) - cos(theta_j)|,
     theta = 2*pi*pos/L.  This is a first-order, numerically robust model of
     "axis separation" (same-side vs opposite-side of the ori-ter axis).
     The residual matrix R = O/E - model is what cannot be explained by
     distance decay + the ori-ter axis.
  3. PCA on corrcoef(R) -> residual E1; saddle analysis identical to the
     main analysis (20 quantile groups, corners = top/bottom 20%).
  4. Compare residual saddle strength vs original vs shuffled control, and
     characterize residual E1 (Fourier modes? GC? gene density?).

Seed fixed (42). Absolute paths. No external installs.
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
GROUPS = 20          # saddle quantile groups
CORNER_FRAC = 0.20   # A/B corners = top/bottom 20%
NSHUF = 100
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- load data
clr = cooler.Cooler(BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000')
M = clr.matrix(balance=True)[:]
valid = ~np.isnan(M).all(axis=0)
Mv = M[np.ix_(valid, valid)]
n = Mv.shape[0]
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)  # bp coords of kept bins
print(f'valid bins: {n}/{M.shape[0]}')

# --------------------------------------- O/E with circular genomic distance
# pairwise circular distances (in units of bins) between kept bins
dp = np.abs(pos_v[:, None] - pos_v[None, :])
dcirc_bp = np.minimum(dp, GENOME - dp)
dbin = np.round(dcirc_bp / BIN).astype(int)
maxd = dbin.max()

cnt = np.zeros(maxd + 1)
summ = np.zeros(maxd + 1)
nz = Mv > 0
np.add.at(cnt, dbin[nz], 1)
np.add.at(summ, dbin[nz], Mv[nz])
exp = np.where(cnt > 0, summ / np.maximum(cnt, 1), np.nan)

OE = np.full((n, n), np.nan)
ok = nz & np.isfinite(exp[dbin]) & (exp[dbin] > 0)
OE[ok] = Mv[ok] / exp[dbin[ok]]
# keep only bins with finite O/E to everywhere (same filter as main analysis)
fin = np.isfinite(OE).all(axis=0)
OE = OE[np.ix_(fin, fin)]
pos = pos_v[fin]
n2 = OE.shape[0]
print(f'bins after O/E filtering: {n2}')

# ------------------------------------------------------------------- helpers
def pca_eig(X):
    C = np.corrcoef(X)
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return ev[o], vec[:, o], C

def saddle(pc, X, groups=GROUPS):
    order = np.argsort(pc)
    gs = np.array_split(order, groups)
    S = np.zeros((groups, groups))
    for a in range(groups):
        for b in range(groups):
            S[a, b] = X[np.ix_(gs[a], gs[b])].mean()
    return S

def saddle_strength(S, corner_frac=CORNER_FRAC, mode='log2ratio'):
    """Corner contrast of a saddle matrix. mode='log2ratio' for positive O/E
    matrices: log2(mean(AA,BB)/mean(AB)); mode='diff' for matrices with
    non-positive entries (e.g. residuals): mean(AA,BB) - mean(AB)."""
    c = max(1, int(round(S.shape[0] * corner_frac)))
    aa = np.concatenate([S[:c, :c].ravel(), S[-c:, -c:].ravel()])
    ab = np.concatenate([S[:c, -c:].ravel(), S[-c:, :c].ravel()])
    if mode == 'diff':
        return np.nanmean(aa) - np.nanmean(ab)
    return np.log2(np.nanmean(aa) / np.nanmean(ab))

def shuffle_strengths(pc, X, nshuf=NSHUF, mode='log2ratio'):
    out = np.empty(nshuf)
    for s in range(nshuf):
        out[s] = saddle_strength(saddle(rng.permutation(pc), X), mode=mode)
    return out

# ------------------------------------------------- original PC1 and saddle
ev, vec, C = pca_eig(OE)
pc1 = vec[:, 0]
theta = 2 * np.pi * pos / GENOME
cos_axis = np.cos(theta)
if stats.pearsonr(pc1, cos_axis)[0] < 0:
    pc1 = -pc1
S_orig = saddle(pc1, OE)
str_orig = saddle_strength(S_orig)
# legacy metric (50 groups, 10% corners) to bridge to the earlier 1.06 value
str_orig_legacy = saddle_strength(saddle(pc1, OE, groups=50), corner_frac=0.10)
shuf_orig = shuffle_strengths(pc1, OE)
print(f'\n== original O/E matrix ==')
print(f'  PC1 var = {100*ev[0]/ev.sum():.1f}%, |r(PC1, cos axis)| = '
      f'{abs(stats.pearsonr(pc1, cos_axis)[0]):.3f}')
print(f'  saddle strength (20 groups/20% corners) = {str_orig:.3f} log2')
print(f'  saddle strength (legacy 50 groups/10% corners) = {str_orig_legacy:.3f} log2')
print(f'  shuffle: mean = {shuf_orig.mean():.3f}, '
      f'95% max = {np.percentile(shuf_orig, 95):.3f}')

# ------------------------------------- remove ori-ter axis component (residual)
ii, jj = np.triu_indices(n2, k=1)
dc = dbin[np.ix_(fin, fin)][ii, jj]
dax = np.abs(cos_axis[ii] - cos_axis[jj])
y = OE[ii, jj]

resid = np.empty_like(y)
pred = np.empty_like(y)
for d in np.unique(dc):
    m = dc == d
    x = dax[m]
    ym = y[m]
    if m.sum() < 10 or np.var(x) == 0:
        p = np.full(m.sum(), ym.mean())
    else:
        b = np.cov(x, ym, ddof=0)[0, 1] / np.var(x)
        a = ym.mean() - b * x.mean()
        p = a + b * x
    pred[m] = p
    resid[m] = ym - p
r2 = 1 - np.var(resid) / np.var(y)
print(f'\n== axis model O/E ~ a_d + b_d*|cos_i - cos_j| ==')
print(f'  variance of O/E explained by distance+axis model: R2 = {r2:.3f}')

R = np.zeros((n2, n2))
R[ii, jj] = resid
R[jj, ii] = resid
np.fill_diagonal(R, 0.0)

# ------------------------------------------------- PCA and saddle on residual
evr, vecr, Cr = pca_eig(R)
E1 = vecr[:, 0]
# orient E1 towards its best 1st-harmonic for display
if abs(stats.pearsonr(E1, cos_axis)[0]) > abs(stats.pearsonr(E1, np.sin(theta))[0]):
    if stats.pearsonr(E1, cos_axis)[0] < 0:
        E1 = -E1
else:
    if stats.pearsonr(E1, np.sin(theta))[0] < 0:
        E1 = -E1

S_res = saddle(E1, R)
# residual matrix is centered ~0 and contains negatives -> use the
# difference metric (mean(AA,BB) - mean(AB)) instead of a log ratio
str_res = saddle_strength(S_res, mode='diff')
str_orig_diff = saddle_strength(S_orig, mode='diff')
shuf_res = shuffle_strengths(E1, R, mode='diff')
print(f'\n== residual matrix (axis removed) ==')
print(f'  residual E1 var = {100*evr[0]/evr.sum():.1f}%, '
      f'E2 = {100*evr[1]/evr.sum():.1f}%')
print(f'  original saddle contrast (diff metric) = {str_orig_diff:.4f}')
print(f'  residual saddle contrast (diff metric) = {str_res:.4f} '
      f'({100*str_res/str_orig_diff:.0f}% of original) '
      f'(shuffle mean = {shuf_res.mean():.4f}, '
      f'shuffle 95% = {np.percentile(shuf_res,95):.4f})')
p_res = (np.sum(shuf_res >= str_res) + 1) / (NSHUF + 1)
print(f'  residual vs shuffle empirical p = {p_res:.3f}')

# what does residual E1 look like?
seq = []
with open(BASE + '/genome.fa') as fh:
    for line in fh:
        if not line.startswith('>'):
            seq.append(line.strip())
seq = ''.join(seq).upper()
gtf = pd.read_csv(BASE + '/annotation.gtf', sep='\t', comment='#', header=None,
                  usecols=[2, 3, 4], names=['feature', 'start', 'end'])
genes = gtf[gtf.feature == 'gene']
gc = np.array([(seq[int(p - BIN/2):int(p + BIN/2)].count('G') +
                seq[int(p - BIN/2):int(p + BIN/2)].count('C')) / BIN
               for p in pos])
gd = np.array([((genes['start'] <= p + BIN/2) & (genes['end'] >= p - BIN/2)).sum()
               for p in pos])
print(f'\n  residual E1 correlations:')
print(f'    |r| with cos(2pi pos/L)   = {abs(stats.pearsonr(E1, cos_axis)[0]):.3f}')
print(f'    |r| with sin(2pi pos/L)   = {abs(stats.pearsonr(E1, np.sin(theta))[0]):.3f}')
print(f'    |r| with cos(4pi pos/L)   = {abs(stats.pearsonr(E1, np.cos(2*theta))[0]):.3f}')
print(f'    |r| with sin(4pi pos/L)   = {abs(stats.pearsonr(E1, np.sin(2*theta))[0]):.3f}')
print(f'    r with GC content         = {stats.pearsonr(E1, gc)[0]:.3f}')
print(f'    r with gene density       = {stats.pearsonr(E1, gd)[0]:.3f}')

# ------------------------------------------------- figure (a): 3-panel saddle
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
vmax = np.nanpercentile(np.abs(S_orig - 1), 98)
im = axes[0].imshow(S_orig, cmap='RdBu_r', vmin=1 - vmax, vmax=1 + vmax)
axes[0].set_title(f'original O/E, PC1-sorted\nstrength = {str_orig:.2f} log2')
plt.colorbar(im, ax=axes[0], label='mean O/E', shrink=0.8)
vmax_r = max(np.nanpercentile(np.abs(S_res - S_res.mean()), 98), 1e-6)
im = axes[1].imshow(S_res, cmap='RdBu_r',
                    vmin=S_res.mean() - vmax_r, vmax=S_res.mean() + vmax_r)
axes[1].set_title(f'residual (ori-ter axis removed), E1-sorted\n'
                  f'contrast = {str_res:.2f} (diff metric)')
plt.colorbar(im, ax=axes[1], label='mean residual', shrink=0.8)
S_shuf_mean = np.mean([saddle(rng.permutation(pc1), OE) for _ in range(NSHUF)],
                      axis=0)
im = axes[2].imshow(S_shuf_mean, cmap='RdBu_r', vmin=1 - vmax, vmax=1 + vmax)
axes[2].set_title(f'shuffled PC1 (mean of {NSHUF})\n'
                  f'strength ~ {shuf_orig.mean():.2f} log2')
plt.colorbar(im, ax=axes[2], label='mean O/E', shrink=0.8)
for ax in axes:
    ax.set_xlabel('sorted bins (20 quantile groups)')
    ax.set_ylabel('sorted bins (20 quantile groups)')
plt.tight_layout()
plt.savefig(OUT + '/saddle_original_residual_shuffle.png', dpi=300)
plt.savefig(OUT + '/saddle_original_residual_shuffle.pdf')
plt.close()

# --------------------------------------------- figure (c): residual E1 track
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].scatter(pos / 1e6, pc1, s=8, c=np.where(pc1 > 0, 'red', 'blue'))
axes[0].plot(pos / 1e6, cos_axis * np.std(pc1), color='grey', lw=1,
             label='cos(2pi pos/L) (scaled)')
axes[0].axhline(0, color='grey', lw=0.5)
axes[0].set_ylabel('original PC1')
axes[0].set_title('Original PC1 = ori-ter axis (Fourier mode 1)')
axes[0].legend(loc='upper right', fontsize=8)
axes[1].scatter(pos / 1e6, E1, s=8, c=np.where(E1 > 0, 'darkred', 'darkblue'))
axes[1].axhline(0, color='grey', lw=0.5)
axes[1].set_ylabel('residual E1')
axes[1].set_xlabel('genomic position (Mb)')
axes[1].set_title(f'Residual E1 after removing distance + ori-ter axis '
                  f'({100*evr[0]/evr.sum():.1f}% var.; saddle contrast '
                  f'{str_res:.2f} (diff) vs original {str_orig:.2f} log2)')
plt.tight_layout()
plt.savefig(OUT + '/residual_E1_track.png', dpi=300)
plt.savefig(OUT + '/residual_E1_track.pdf')
plt.close()

# ------------------------------------------------------------------- CSV out
summary = pd.DataFrame([
    dict(matrix='original_OE', pc1_var_pct=100*ev[0]/ev.sum(),
         saddle_strength_log2=str_orig,
         saddle_contrast_diff=np.nan,
         saddle_strength_legacy=str_orig_legacy,
         shuffle_mean=shuf_orig.mean(),
         shuffle_p95=np.percentile(shuf_orig, 95),
         axis_model_R2=np.nan,
         pc1_abs_r_cos=abs(stats.pearsonr(pc1, cos_axis)[0]),
         pc1_r_GC=stats.pearsonr(pc1, gc)[0],
         pc1_r_gene_density=stats.pearsonr(pc1, gd)[0]),
    dict(matrix='residual_axis_removed', pc1_var_pct=100*evr[0]/evr.sum(),
         saddle_strength_log2=np.nan,
         saddle_contrast_diff=str_res,
         saddle_strength_legacy=np.nan,
         shuffle_mean=shuf_res.mean(),
         shuffle_p95=np.percentile(shuf_res, 95),
         axis_model_R2=r2,
         pc1_abs_r_cos=abs(stats.pearsonr(E1, cos_axis)[0]),
         pc1_r_GC=stats.pearsonr(E1, gc)[0],
         pc1_r_gene_density=stats.pearsonr(E1, gd)[0]),
])
summary.to_csv(OUT + '/control3_residual_saddle.csv', index=False)
print(f'\nwrote {OUT}/control3_residual_saddle.csv')
print('Done.')
