#!/usr/bin/env python3
"""
Reviewer 4 comment 1 - intuitive companion figure (complements
Fig_R4C1_combined, for readers unfamiliar with saddle plots).

  A. WT O/E matrix (circular-distance normalized) with rows/cols ordered by
     E1 (=PC1): the ori-ter axis appears as blocks.
  B. Same matrix ordered by one shuffled E1 (seed=42): structure vanishes.
     (A and B share exactly the same color scale.)
  C. Segregation curve: mean O/E vs |E1_i - E1_j| (20 equal-frequency bins)
     for real data, 100x shuffled E1 (mean + 5-95% band), and the synthetic
     P(s)-only matrix. Real decreases; both controls flat at ~1.
  D. Residual E1 (ori-ter axis removed) four-zone track, simplified port of
     Fig_R4C1_combined panel E.

Self-contained, seed=42, absolute paths.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cooler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from scipy import stats
from scipy.signal import savgol_filter

BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUTDIR = BASE + '/review_results/R4_comment1_AB_compartment'
FIG = OUTDIR + '/figures'
BIN = 5000
GENOME = 4411532
TER_BP = 2030000
NSHUF = 100
NCURVE = 20
rng = np.random.default_rng(42)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
})

# ------------------------------------------------------------- pipeline (as 06)
def oe_pipeline(Mv, pos_v):
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
    OE, pos = OE[np.ix_(fin, fin)], pos_v[fin]
    C = np.corrcoef(OE)
    okc = np.isfinite(C).all(axis=0)
    OE, pos, C = OE[np.ix_(okc, okc)], pos[okc], C[np.ix_(okc, okc)]
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return OE, pos, ev[o] / ev.sum(), vec[:, o[0]]

clr = cooler.Cooler(BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000')
Mbal = clr.matrix(balance=True)[:]
valid = ~np.isnan(Mbal).all(axis=0)
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
OE, pos, frac, E1 = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
n2 = OE.shape[0]
theta = 2 * np.pi * pos / GENOME
if stats.pearsonr(E1, np.cos(theta))[0] < 0:
    E1 = -E1

# synthetic P(s)-only matrix (as 06)
Mraw = np.nan_to_num(clr.matrix(balance=False)[:], nan=0.0)[np.ix_(valid, valid)]
dp = np.abs(pos_v[:, None] - pos_v[None, :])
dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
cnt = np.zeros(dbin.max() + 1); summ = np.zeros(dbin.max() + 1)
nz = Mraw > 0
np.add.at(cnt, dbin[nz], 1); np.add.at(summ, dbin[nz], Mraw[nz])
Ps = np.where(cnt > 0, summ / np.maximum(cnt, 1), 0.0)
lam = Ps[dbin]
iu = np.triu_indices(len(pos_v), k=1)
synth = np.zeros_like(Mraw)
synth[iu] = rng.poisson(lam[iu])
synth[(iu[1], iu[0])] = synth[iu]
np.fill_diagonal(synth, Ps[0])
OE_syn, pos_syn, frac_syn, E1_syn = oe_pipeline(synth, pos_v)

# residual matrix (remove ori-ter axis, as 06) -> residual E1 (mode 2)
dbin2 = np.round(np.minimum(np.abs(pos[:, None] - pos[None, :]),
                            GENOME - np.abs(pos[:, None] - pos[None, :])) / BIN).astype(int)
ii, jj = np.triu_indices(n2, k=1)
dc = dbin2[ii, jj]
dax = np.abs(np.cos(theta[ii]) - np.cos(theta[jj]))
y = OE[ii, jj]
resid = np.empty_like(y)
for d in np.unique(dc):
    m = dc == d
    x, ym = dax[m], y[m]
    if m.sum() < 10 or np.var(x) == 0:
        p = np.full(m.sum(), ym.mean())
    else:
        b = np.cov(x, ym, ddof=0)[0, 1] / np.var(x)
        p = ym.mean() - b * x.mean() + b * x
    resid[m] = ym - p
R = np.zeros((n2, n2)); R[ii, jj] = resid; R[jj, ii] = resid
evr, vecr = np.linalg.eigh(np.corrcoef(R))
o = np.argsort(evr)[::-1]
rE1 = vecr[:, o[0]]
h2 = np.column_stack([np.cos(2 * theta), np.sin(2 * theta), np.ones(n2)])
coef, *_ = np.linalg.lstsq(h2, rE1, rcond=None)
r_rE1_h2 = stats.pearsonr(rE1, h2[:, :2] @ coef[:2])[0]
if r_rE1_h2 < 0:
    rE1, r_rE1_h2 = -rE1, -r_rE1_h2

# residual E1 circular zero-crossings (as 06)
w = 31
rpad = np.concatenate([rE1[-3*w:], rE1, rE1[:3*w]])
rE1s = savgol_filter(rpad, w, 3)[3*w:3*w + n2]
sign_c = np.sign(rE1s)
cross_idx = np.where(np.diff(np.append(sign_c, sign_c[0])) != 0)[0]
cross_bp = []
for k in cross_idx:
    k2 = (k + 1) % n2
    p0 = pos[k]
    p1 = pos[k2] if k2 > k else pos[k2] + GENOME
    v0, v1 = rE1s[k], rE1s[k2]
    cross_bp.append((p0 + (p1 - p0) * (-v0) / (v1 - v0)) % GENOME)
cross_bp = np.sort(cross_bp)
bounds = np.sort(cross_bp)
seg_bounds = [(bounds[s], bounds[(s + 1) % len(bounds)]) for s in range(len(bounds))]

# ------------------------------------------------- segregation curves (Panel C)
def segregation_curve(e1, oe, edges=None, npairs_sub=None):
    """mean O/E in 20 equal-frequency bins of |E1_i - E1_j| (upper triangle)."""
    iu2 = np.triu_indices(len(e1), k=1)
    x = np.abs(e1[iu2[0]] - e1[iu2[1]])
    yy = oe[iu2]
    if edges is None:
        edges = np.quantile(x, np.linspace(0, 1, NCURVE + 1))
        edges[0], edges[-1] = -1e-9, x.max() + 1e-9
    bidx = np.digitize(x, edges) - 1
    means = np.array([yy[bidx == b].mean() for b in range(NCURVE)])
    centers = np.array([x[bidx == b].mean() for b in range(NCURVE)])
    return centers, means, edges

cen, curve_real, edges = segregation_curve(E1, OE)
curves_shuf = np.empty((NSHUF, NCURVE))
for s in range(NSHUF):
    _, curves_shuf[s], _ = segregation_curve(rng.permutation(E1), OE, edges=edges)
shuf_mean = curves_shuf.mean(axis=0)
shuf_lo = np.percentile(curves_shuf, 5, axis=0)
shuf_hi = np.percentile(curves_shuf, 95, axis=0)
cen_syn, curve_syn, _ = segregation_curve(E1_syn, OE_syn, edges=edges)

print('== Panel C segregation curves ==')
print(f'  real: O/E {curve_real[0]:.3f} (|dE1| smallest) -> '
      f'{curve_real[-1]:.3f} (|dE1| largest)')
print(f'  shuffle: mean {shuf_mean.mean():.3f}, '
      f'range of means {shuf_mean.min():.3f}..{shuf_mean.max():.3f}')
print(f'  synthetic: mean {curve_syn.mean():.3f}, '
      f'range {curve_syn.min():.3f}..{curve_syn.max():.3f}')

# ------------------------------------------------------------------- figure
MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 200 * MM))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                       height_ratios=[1.15, 0.85, 0.75])

def panel_label(ax, letter, dx=-0.16):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='left')

# shared color scale for A and B
order = np.argsort(E1)
OE_sorted = OE[np.ix_(order, order)]
vmax = np.nanpercentile(np.abs(OE_sorted - 1), 98)
vmin, vmax = 1 - vmax, 1 + vmax

# one shuffled E1 for panel B (seeded)
E1_shuf1 = rng.permutation(E1)
order_shuf = np.argsort(E1_shuf1)
OE_shuf_sorted = OE[np.ix_(order_shuf, order_shuf)]

def matrix_with_signstrip(sub, mat, e1_sorted_vals, title, letter):
    """heatmap + thin E1-sign strip above and left of it."""
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=sub,
                                             width_ratios=[0.045, 1],
                                             height_ratios=[0.045, 1],
                                             wspace=0.02, hspace=0.02)
    ax_top = fig.add_subplot(inner[0, 1])
    ax_left = fig.add_subplot(inner[1, 0])
    ax = fig.add_subplot(inner[1, 1])
    sign = np.sign(e1_sorted_vals).reshape(1, -1)
    cmap_sign = mcolors.ListedColormap(['#2166ac', '#b2182b'])  # - blue, + red
    ax_top.imshow(sign, aspect='auto', cmap=cmap_sign, vmin=-1, vmax=1)
    ax_left.imshow(sign.T, aspect='auto', cmap=cmap_sign, vmin=-1, vmax=1)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_left.set_xticks([]); ax_left.set_yticks([])
    ax.imshow(mat, cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return ax, ax_left

axA, axA_left = matrix_with_signstrip(
    gs[0, 0], OE_sorted, E1[order],
    'WT O/E matrix, bins ordered by E1', 'A')
axA_left.set_ylabel('bins ordered by E1', fontsize=6.5)
panel_label(axA, 'A')

axB, _ = matrix_with_signstrip(
    gs[0, 1], OE_shuf_sorted, E1_shuf1[order_shuf],
    'E1 labels shuffled (1 of 100)', 'B')
panel_label(axB, 'B')
# shared colorbar
cbax = fig.add_axes([axB.get_position().x1 + 0.012, axB.get_position().y0,
                     0.008, axB.get_position().height])
cb = fig.colorbar(plt.cm.ScalarMappable(
    norm=mcolors.Normalize(vmin, vmax), cmap='coolwarm'), cax=cbax)
cb.set_label('mean O/E', fontsize=6.5)
cb.ax.tick_params(labelsize=6)

# ---- Panel C: segregation curves
axC = fig.add_subplot(gs[1, :])
axC.fill_between(cen, shuf_lo, shuf_hi, color='grey', alpha=0.35, lw=0,
                 label=f'shuffled E1 (n={NSHUF}, 5-95%)')
axC.plot(cen, shuf_mean, color='grey', lw=1)
axC.plot(cen_syn, curve_syn, color='#2ca02c', lw=1.2,
         label=f'synthetic P(s)-only (flat ~{curve_syn.mean():.2f})')
axC.plot(cen, curve_real, color='#c0392b', lw=1.8, marker='o', ms=2.5,
         label=f'real WT ({curve_real[0]:.2f} -> {curve_real[-1]:.2f})')
axC.axhline(1, color='black', lw=0.5, ls='--')
axC.set_xlabel('|E1$_i$ - E1$_j$| (bin-pair E1 difference)')
axC.set_ylabel('mean O/E contact frequency')
axC.set_title('segregation curve: bins with similar E1 contact each other more')
axC.legend(frameon=False, loc='upper right')
panel_label(axC, 'C')

# ---- Panel D: residual four-zone track (simplified port of 06 panel E)
axD = fig.add_subplot(gs[2, :])
seg_colors = {'+': '#f4c7c3', '-': '#c3d4f4'}
for s0, s1 in seg_bounds:
    wrap = s1 <= s0
    end_lin = s1 + GENOME if wrap else s1
    mid = (s0 + end_lin) / 2 % GENOME
    sgn = '+' if np.interp(mid, pos, rE1s, period=GENOME) > 0 else '-'
    if wrap:
        axD.axvspan(s0 / 1e6, GENOME / 1e6, color=seg_colors[sgn], alpha=0.5, lw=0)
        axD.axvspan(0, s1 / 1e6, color=seg_colors[sgn], alpha=0.5, lw=0)
    else:
        axD.axvspan(s0 / 1e6, s1 / 1e6, color=seg_colors[sgn], alpha=0.5, lw=0)
    axD.text(mid, 0.115, f'{s0/1e6:.2f}-{s1/1e6:.2f} Mb\n({sgn})',
             ha='center', va='top', fontsize=6)
xx = np.linspace(0, GENOME, 1000)
fit2 = coef[0] * np.cos(4 * np.pi * xx / GENOME) + coef[1] * np.sin(4 * np.pi * xx / GENOME)
axD.plot(pos / 1e6, rE1, '.', ms=2.5, color='black')
axD.plot(xx / 1e6, fit2, color='#c0392b', lw=1.2,
         label=f'2nd-harmonic fit, |r| = {abs(r_rE1_h2):.2f}')
for c in cross_bp:
    axD.axvline(c / 1e6, color='grey', ls=':', lw=0.8)
axD.axvline(0, color='green', ls='--', lw=1)
axD.text(0.03, -0.085, 'ori', color='green', fontsize=6.5, va='bottom')
axD.axvline(TER_BP / 1e6, color='purple', ls='--', lw=1)
axD.text(TER_BP / 1e6 + 0.02, -0.085, 'ter', color='purple', fontsize=6.5,
         va='bottom')
axD.axhline(0, color='grey', lw=0.5)
axD.set_ylim(-0.09, 0.12)
axD.set_xlabel('genomic position (Mb)')
axD.set_ylabel('residual E1')
axD.set_title('residual E1 after removing the ori-ter axis: four zones '
              '(two full waves)')
axD.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='none')
panel_label(axD, 'D')

plt.savefig(FIG + '/Fig_R4C1_intuitive.pdf')
plt.savefig(FIG + '/Fig_R4C1_intuitive.png', dpi=300)
plt.close()
print('\nwrote Fig_R4C1_intuitive.{pdf,png}')
print('Done.')
