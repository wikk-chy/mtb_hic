#!/usr/bin/env python3
"""
Control 4 for Reviewer 4 comment 1: synthetic distance-decay-only matrix.

A synthetic matrix is built purely from the WT P(s) curve:
    M_synth[i,j] = P(circular_distance(i,j))
(Poisson-sampled around the mean raw contact count at each circular
distance, to include realistic sampling noise).  It contains NO compartment
or ori-ter axis information by construction.

The identical pipeline (circular-distance O/E -> PCA -> saddle, 20 quantile
groups / 20% corners) is then applied.  If the pipeline itself manufactured
plaid/compartment patterns on circular small genomes, the synthetic matrix
would show a cos-like PC1 and a strong saddle; if it shows neither, the
saddle signal in the real matrix requires structure beyond pure distance
decay.

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
GROUPS = 20
CORNER_FRAC = 0.20
NSHUF = 100
rng = np.random.default_rng(42)

# ------------------------------------------------------------------- helpers
def saddle(pc, X, groups=GROUPS):
    order = np.argsort(pc)
    gs = np.array_split(order, groups)
    S = np.zeros((groups, groups))
    for a in range(groups):
        for b in range(groups):
            S[a, b] = X[np.ix_(gs[a], gs[b])].mean()
    return S

def saddle_strength(S, corner_frac=CORNER_FRAC):
    c = max(1, int(round(S.shape[0] * corner_frac)))
    aa = np.concatenate([S[:c, :c].ravel(), S[-c:, -c:].ravel()])
    ab = np.concatenate([S[:c, -c:].ravel(), S[-c:, :c].ravel()])
    return np.log2(np.nanmean(aa) / np.nanmean(ab))

def oe_pipeline(Mv, pos_v):
    """circular-distance O/E, all-finite-bin filter, PCA. Returns
    (OE2, pos, pc1, eigval_fractions)."""
    n = Mv.shape[0]
    dp = np.abs(pos_v[:, None] - pos_v[None, :])
    dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
    cnt = np.zeros(dbin.max() + 1)
    summ = np.zeros(dbin.max() + 1)
    nz = Mv > 0
    np.add.at(cnt, dbin[nz], 1)
    np.add.at(summ, dbin[nz], Mv[nz])
    exp = np.where(cnt > 0, summ / np.maximum(cnt, 1), np.nan)
    OE = np.full((n, n), np.nan)
    ok = nz & np.isfinite(exp[dbin]) & (exp[dbin] > 0)
    OE[ok] = Mv[ok] / exp[dbin[ok]]
    fin = np.isfinite(OE).all(axis=0)
    OE = OE[np.ix_(fin, fin)]
    pos = pos_v[fin]
    # drop bins whose correlation row is degenerate (zero variance)
    C = np.corrcoef(OE)
    okc = np.isfinite(C).all(axis=0)
    OE, pos, C = OE[np.ix_(okc, okc)], pos[okc], C[np.ix_(okc, okc)]
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    ev, vec = ev[o], vec[:, o]
    return OE, pos, ev / ev.sum(), vec[:, 0]

# ------------------------------------------------- real WT matrix (balanced)
clr = cooler.Cooler(BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000')
Mbal = clr.matrix(balance=True)[:]
valid = ~np.isnan(Mbal).all(axis=0)
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
OE_real, pos_real, frac_real, pc1_real = oe_pipeline(Mbal[np.ix_(valid, valid)],
                                                    pos_v)
theta_real = 2 * np.pi * pos_real / GENOME
if stats.pearsonr(pc1_real, np.cos(theta_real))[0] < 0:
    pc1_real = -pc1_real
str_real = saddle_strength(saddle(pc1_real, OE_real))
shuf_real = np.array([saddle_strength(saddle(rng.permutation(pc1_real), OE_real))
                      for _ in range(NSHUF)])
print('== real WT matrix ==')
print(f'  PC1 var = {100*frac_real[0]:.1f}%, '
      f'|r(PC1, cos axis)| = {abs(stats.pearsonr(pc1_real, np.cos(theta_real))[0]):.3f}')
print(f'  saddle strength = {str_real:.3f} log2, shuffle mean = {shuf_real.mean():.3f}')

# ------------------------------------ synthetic matrix: pure distance decay
Mraw = clr.matrix(balance=False)[:]
Mraw = np.nan_to_num(Mraw, nan=0.0)
Mraw_v = Mraw[np.ix_(valid, valid)]
n = Mraw_v.shape[0]
dp = np.abs(pos_v[:, None] - pos_v[None, :])
dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
cnt = np.zeros(dbin.max() + 1)
summ = np.zeros(dbin.max() + 1)
nz = Mraw_v > 0
np.add.at(cnt, dbin[nz], 1)
np.add.at(summ, dbin[nz], Mraw_v[nz])
Ps = np.where(cnt > 0, summ / np.maximum(cnt, 1), 0.0)  # P(s): mean raw contacts

lam = Ps[dbin]
iu = np.triu_indices(n, k=1)
synth = np.zeros((n, n))
synth[iu] = rng.poisson(lam[iu])
synth[(iu[1], iu[0])] = synth[iu]
np.fill_diagonal(synth, Ps[0])
print(f'\n== synthetic matrix (P(s) only, Poisson noise) ==')
print(f'  total contacts: synth = {synth.sum():.0f}, real raw = {Mraw_v.sum():.0f}')

OE_syn, pos_syn, frac_syn, pc1_syn = oe_pipeline(synth, pos_v)
theta_syn = 2 * np.pi * pos_syn / GENOME
r_cos_syn = abs(stats.pearsonr(pc1_syn, np.cos(theta_syn))[0])
r_sin_syn = abs(stats.pearsonr(pc1_syn, np.sin(theta_syn))[0])
if stats.pearsonr(pc1_syn, np.cos(theta_syn))[0] < 0:
    pc1_syn = -pc1_syn
str_syn = saddle_strength(saddle(pc1_syn, OE_syn))
shuf_syn = np.array([saddle_strength(saddle(rng.permutation(pc1_syn), OE_syn))
                     for _ in range(NSHUF)])
print(f'  PC1 var = {100*frac_syn[0]:.1f}% (real: {100*frac_real[0]:.1f}%)')
print(f'  top-6 eigenvalue fractions synth: {np.round(frac_syn[:6], 3)}')
print(f'  top-6 eigenvalue fractions real : {np.round(frac_real[:6], 3)}')
print(f'  |r(PC1, cos axis)| = {r_cos_syn:.3f}, |r(PC1, sin axis)| = {r_sin_syn:.3f}')
print(f'  saddle strength = {str_syn:.3f} log2, shuffle mean = {shuf_syn.mean():.3f}')

# ---------------- figure (b): PC1 shape comparison + saddle strength bars
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].scatter(pos_real / 1e6, pc1_real / np.std(pc1_real), s=8,
                label=f'real PC1 (|r|cos = {abs(stats.pearsonr(pc1_real, np.cos(theta_real))[0]):.2f})',
                color='red')
axes[0].scatter(pos_syn / 1e6, pc1_syn / np.std(pc1_syn), s=8,
                label=f'synthetic PC1 (|r|cos = {r_cos_syn:.2f})',
                color='grey', alpha=0.7)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_xlabel('genomic position (Mb)')
axes[0].set_ylabel('PC1 (z-scored)')
axes[0].set_title('PC1 shape: real vs synthetic P(s)-only matrix')
axes[0].legend(fontsize=8)
labels = ['real WT', 'synthetic\nP(s) only', 'shuffled PC1\n(real, mean)']
vals = [str_real, str_syn, shuf_real.mean()]
errs = [0, 0, shuf_real.std()]
colors = ['#d62728', '#7f7f7f', '#bbbbbb']
axes[1].bar(labels, vals, yerr=errs, color=colors, capsize=4)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_ylabel('saddle strength log2((AA+BB)/AB)')
axes[1].set_title('saddle strength comparison')
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT + '/synthetic_control_PC1_and_saddle.png', dpi=300)
plt.savefig(OUT + '/synthetic_control_PC1_and_saddle.pdf')
plt.close()

# ------------------------------------------------------------------- CSV out
summary = pd.DataFrame([
    dict(matrix='real_WT', pc1_var_pct=100*frac_real[0],
         pc2_var_pct=100*frac_real[1],
         pc1_abs_r_cos=abs(stats.pearsonr(pc1_real, np.cos(theta_real))[0]),
         saddle_strength_log2=str_real,
         shuffle_mean=shuf_real.mean(),
         shuffle_p95=np.percentile(shuf_real, 95)),
    dict(matrix='synthetic_Ps_only', pc1_var_pct=100*frac_syn[0],
         pc2_var_pct=100*frac_syn[1],
         pc1_abs_r_cos=r_cos_syn,
         saddle_strength_log2=str_syn,
         shuffle_mean=shuf_syn.mean(),
         shuffle_p95=np.percentile(shuf_syn, 95)),
])
summary.to_csv(OUT + '/control4_synthetic_saddle.csv', index=False)
print(f'\nwrote {OUT}/control4_synthetic_saddle.csv')
print('Done.')
