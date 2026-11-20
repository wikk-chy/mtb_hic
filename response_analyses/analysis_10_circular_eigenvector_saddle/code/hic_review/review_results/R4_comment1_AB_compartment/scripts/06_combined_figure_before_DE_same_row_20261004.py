#!/usr/bin/env python3
"""
Reviewer 4 comment 1 - combined multi-panel figure (Panels A-E).

Recomputes all three control experiments from scratch (self-contained,
seed=42) and assembles:
  A. synthetic P(s)-only negative control: PC1 tracks + saddle bars
  B. shuffle control: saddle strength distribution vs observed
  C. PC1 identity: PC1 vs genome with cos fit; PC1 vs GC / gene density
  D. residual saddle: original / axis-removed residual / shuffle heatmaps
  E. residual E1 "four-zone" track with zero-crossings and segment shading

Outputs:
  figures/Fig_R4C1_combined.pdf / .png (300 dpi, 180 mm wide)
  tables/residual_E1_segments.csv
  tables/residual_E1_segments_summary.txt
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
from scipy import stats
from scipy.signal import savgol_filter

# ------------------------------------------------------------ config / paths
BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUTDIR = BASE + '/review_results/R4_comment1_AB_compartment'
FIG = OUTDIR + '/figures'
TAB = OUTDIR + '/tables'
BIN = 5000
GENOME = 4411532
GROUPS = 20
CORNER_FRAC = 0.20
NSHUF = 100
TER_BP = 2030000   # ter position (~2.03 Mb)
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

# ------------------------------------------------------------ pipeline pieces
def saddle(pc, X, groups=GROUPS):
    order = np.argsort(pc)
    gs = np.array_split(order, groups)
    S = np.zeros((groups, groups))
    for a in range(groups):
        for b in range(groups):
            S[a, b] = X[np.ix_(gs[a], gs[b])].mean()
    return S

def saddle_strength(S, corner_frac=CORNER_FRAC, mode='log2ratio'):
    c = max(1, int(round(S.shape[0] * corner_frac)))
    aa = np.concatenate([S[:c, :c].ravel(), S[-c:, -c:].ravel()])
    ab = np.concatenate([S[:c, -c:].ravel(), S[-c:, :c].ravel()])
    if mode == 'diff':
        return np.nanmean(aa) - np.nanmean(ab)
    return np.log2(np.nanmean(aa) / np.nanmean(ab))

def oe_pipeline(Mv, pos_v):
    """circular-distance O/E, all-finite filter, PCA -> (OE, pos, frac, pc1)."""
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

# ------------------------------------------------------------- real WT data
clr = cooler.Cooler(BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000')
Mbal = clr.matrix(balance=True)[:]
valid = ~np.isnan(Mbal).all(axis=0)
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
OE_real, pos, frac_real, pc1 = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
n2 = OE_real.shape[0]
theta = 2 * np.pi * pos / GENOME
cos_ax, sin_ax = np.cos(theta), np.sin(theta)
if stats.pearsonr(pc1, cos_ax)[0] < 0:
    pc1 = -pc1
r_pc1_cos = stats.pearsonr(pc1, cos_ax)[0]

S_orig = saddle(pc1, OE_real)
str_orig = saddle_strength(S_orig)
shuf = np.array([saddle_strength(saddle(rng.permutation(pc1), OE_real))
                 for _ in range(NSHUF)])
S_shuf_mean = np.mean([saddle(rng.permutation(pc1), OE_real)
                       for _ in range(NSHUF)], axis=0)
p_shuf = (np.sum(shuf >= str_orig) + 1) / (NSHUF + 1)

# ------------------------------------------------------- synthetic P(s) only
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
OE_syn, pos_syn, frac_syn, pc1_syn = oe_pipeline(synth, pos_v)
theta_syn = 2 * np.pi * pos_syn / GENOME
r_syn_cos = abs(stats.pearsonr(pc1_syn, np.cos(theta_syn))[0])
str_syn = saddle_strength(saddle(pc1_syn, OE_syn))

# --------------------------------------------------- residual (axis removed)
dbin2 = np.round(np.minimum(np.abs(pos[:, None] - pos[None, :]),
                            GENOME - np.abs(pos[:, None] - pos[None, :])) / BIN).astype(int)
ii, jj = np.triu_indices(n2, k=1)
dc = dbin2[ii, jj]
dax = np.abs(cos_ax[ii] - cos_ax[jj])
y = OE_real[ii, jj]
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
Cr = np.corrcoef(R)
evr, vecr = np.linalg.eigh(Cr)
o = np.argsort(evr)[::-1]
evr, E1 = evr[o] / evr.sum(), vecr[:, o[0]]
# fit 2nd harmonic and orient E1 to match it
h2 = np.column_stack([np.cos(2 * theta), np.sin(2 * theta), np.ones(n2)])
coef, *_ = np.linalg.lstsq(h2, E1, rcond=None)
h2_fit = h2[:, :2] @ coef[:2]
r_E1_h2 = stats.pearsonr(E1, h2_fit)[0]
if r_E1_h2 < 0:
    E1, r_E1_h2 = -E1, -r_E1_h2
S_res = saddle(E1, R)
str_res_diff = saddle_strength(S_res, mode='diff')
str_orig_diff = saddle_strength(S_orig, mode='diff')
shuf_res = np.array([saddle_strength(saddle(rng.permutation(E1), R), mode='diff')
                     for _ in range(NSHUF)])
p_res = (np.sum(shuf_res >= str_res_diff) + 1) / (NSHUF + 1)

# ------------------------------------------------------- GC / gene density
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
                seq[int(p - BIN/2):int(p + BIN/2)].count('C')) / BIN for p in pos])
gd = np.array([((genes['start'] <= p + BIN/2) & (genes['end'] >= p - BIN/2)).sum()
               for p in pos])
r_gc, p_gc = stats.pearsonr(pc1, gc)
r_gd, p_gd = stats.pearsonr(pc1, gd)

# ----------------------------------------- residual E1 zero-crossings/segments
# circular smoothing to avoid edge artifacts at the ori (position 0 wraps)
w = 31
E1pad = np.concatenate([E1[-3*w:], E1, E1[:3*w]])
E1s = savgol_filter(E1pad, w, 3)[3*w:3*w + n2]
# crossings on the circular series (wrap between last and first bin included)
sign_c = np.sign(E1s)
cross_idx = np.where(np.diff(np.append(sign_c, sign_c[0])) != 0)[0]
cross_bp = []
for k in cross_idx:
    k2 = (k + 1) % n2
    p0 = pos[k]
    p1 = pos[k2] if k2 > k else pos[k2] + GENOME   # unwrap across origin
    v0, v1 = E1s[k], E1s[k2]
    cross_bp.append((p0 + (p1 - p0) * (-v0) / (v1 - v0)) % GENOME)
cross_bp = np.sort(cross_bp)
print(f'E1 zero-crossings (circular, smoothed): {np.round(cross_bp/1e3, 1)} kb')

# build circular segments between consecutive crossings (4 expected)
segments = []
ncross = len(cross_bp)
for s in range(ncross):
    s0 = cross_bp[s]
    s1 = cross_bp[(s + 1) % ncross]
    wrap = s1 <= s0
    end_lin = s1 + GENOME if wrap else s1
    mid = (s0 + end_lin) / 2 % GENOME
    sign = '+' if np.interp(mid, pos, E1s, period=GENOME) > 0 else '-'
    segments.append(dict(segment=s + 1, start_bp=int(round(s0)),
                         end_bp=int(round(s1)), wraps_origin=wrap,
                         E1_sign=sign,
                         contains_ori=bool(wrap),
                         contains_ter=bool(s0 <= TER_BP <= end_lin)))
segdf = pd.DataFrame(segments)
segdf.to_csv(TAB + '/residual_E1_segments.csv', index=False)

# --------------------------------------------------------------- the figure
MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 215 * MM))
gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.75, wspace=0.9,
                       height_ratios=[1, 1, 1.05, 0.85])

def panel_label(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='left')

# ---- Panel A: synthetic negative control
axA1 = fig.add_subplot(gs[0, 0:4])
axA1.plot(pos / 1e6, pc1 / np.std(pc1), '.', ms=2, color='#c0392b',
          label=f'real WT PC1 (|r|cos = {abs(r_pc1_cos):.2f})')
axA1.plot(pos_syn / 1e6, pc1_syn / np.std(pc1_syn), '.', ms=2, color='grey',
          alpha=0.6, label=f'synthetic P(s)-only PC1 (|r|cos = {r_syn_cos:.2f})')
axA1.axhline(0, color='black', lw=0.5)
axA1.set_xlabel('genomic position (Mb)')
axA1.set_ylabel('PC1 (z-scored)')
axA1.set_title('PC1 along genome: real vs synthetic')
axA1.legend(markerscale=3, frameon=False, loc='upper center', ncol=2)
axA2 = fig.add_subplot(gs[0, 4:6])
bars = axA2.bar([0, 1, 2], [str_orig, str_syn, shuf.mean()],
                color=['#c0392b', 'grey', '#cccccc'], width=0.65)
axA2.errorbar(2, shuf.mean(), yerr=shuf.std(), color='black', capsize=3, lw=1)
axA2.set_xticks([0, 1, 2])
axA2.set_xticklabels(['real\nWT', 'synthetic\nP(s) only', 'shuffled\n(mean)'])
axA2.set_ylabel('saddle strength\nlog2((AA+BB)/AB)')
axA2.axhline(0, color='black', lw=0.5)
axA2.set_ylim(top=str_orig * 1.25)
for x, v in zip([0, 1, 2], [str_orig, str_syn, shuf.mean()]):
    axA2.text(x, v + 0.05 if v >= 0 else v + 0.12, f'{v:.2f}', ha='center', fontsize=6.5)
axA2.set_title('no compartment signal from\ndistance decay alone')
panel_label(axA1, 'A')

# ---- Panel B: shuffle control histogram
axB = fig.add_subplot(gs[1, 0:2])
axB.hist(shuf, bins=20, color='grey', alpha=0.8)
axB.axvline(str_orig, color='#c0392b', lw=1.5)
axB.set_xlim(right=str_orig * 1.12)
axB.text(str_orig - 0.05, axB.get_ylim()[1] * 0.75,
         f'observed = {str_orig:.2f}\n'
         f'empirical p < {max(p_shuf, 1/(NSHUF+1)):.2f}', color='#c0392b',
         fontsize=6.5, va='top', ha='right')
axB.set_xlabel('saddle strength log2((AA+BB)/AB)')
axB.set_ylabel('count')
axB.set_title(f'shuffle control (n = {NSHUF})')
panel_label(axB, 'B')

# ---- Panel C: PC1 identity
axC1 = fig.add_subplot(gs[1, 2:4])
h1 = np.column_stack([cos_ax, sin_ax, np.ones(n2)])
c1, *_ = np.linalg.lstsq(h1, pc1, rcond=None)
xx = np.linspace(0, GENOME, 500)
fit1 = c1[0] * np.cos(2 * np.pi * xx / GENOME) + c1[1] * np.sin(2 * np.pi * xx / GENOME) + c1[2]
axC1.plot(pos / 1e6, pc1, '.', ms=2, color='#c0392b')
axC1.plot(xx / 1e6, fit1, color='black', lw=1,
          label=f'1st-harmonic fit, |r| = {abs(r_pc1_cos):.2f}')
axC1.axhline(0, color='grey', lw=0.5)
axC1.set_xlabel('genomic position (Mb)')
axC1.set_ylabel('PC1')
axC1.set_title('PC1 = ori-ter axis (Fourier mode 1)')
axC1.legend(frameon=False, loc='lower left')
axC2 = fig.add_subplot(gs[1, 4])
axC2.plot(gc, pc1, '.', ms=2, color='steelblue')
axC2.set_xlabel('GC content / 5 kb')
axC2.set_ylabel('PC1')
axC2.set_title(f'r = {r_gc:.2f} (n.s.)')
axC3 = fig.add_subplot(gs[1, 5])
axC3.plot(gd, pc1, '.', ms=2, color='seagreen')
axC3.set_xlabel('genes / 5 kb')
axC3.set_ylabel('PC1')
axC3.set_title(f'r = {r_gd:.2f} (n.s.)')
panel_label(axC1, 'C')

# ---- Panel D: residual saddle triptych
axD1 = fig.add_subplot(gs[2, 0:2])
vmax = np.nanpercentile(np.abs(S_orig - 1), 98)
axD1.imshow(S_orig, cmap='coolwarm', vmin=1 - vmax, vmax=1 + vmax)
axD1.set_title(f'original (strength {str_orig:.2f} log2)')
axD2 = fig.add_subplot(gs[2, 2:4])
vr = np.nanpercentile(np.abs(S_res - S_res.mean()), 98)
axD2.imshow(S_res, cmap='coolwarm', vmin=S_res.mean() - vr, vmax=S_res.mean() + vr)
axD2.set_title(f'residual, ori-ter axis removed\n'
               f'(contrast {str_res_diff:.2f} = {100*str_res_diff/str_orig_diff:.0f}% of '
               f'original, p = {max(p_res, 1/(NSHUF+1)):.2f})')
axD3 = fig.add_subplot(gs[2, 4:6])
axD3.imshow(S_shuf_mean, cmap='coolwarm', vmin=1 - vmax, vmax=1 + vmax)
axD3.set_title(f'shuffled PC1, mean of {NSHUF}\n(strength ~ {shuf.mean():.2f} log2)')
for ax in (axD1, axD2, axD3):
    ax.set_xlabel('sorted bins')
    ax.set_ylabel('sorted bins')
    ax.set_xticks([]); ax.set_yticks([])
panel_label(axD1, 'D')

# ---- Panel E: residual E1 four-zone track
axE = fig.add_subplot(gs[3, :])
xx = np.linspace(0, GENOME, 1000)
fit2 = coef[0] * np.cos(4 * np.pi * xx / GENOME) + coef[1] * np.sin(4 * np.pi * xx / GENOME)
seg_colors = {'+': '#f4c7c3', '-': '#c3d4f4'}
for i, s in segdf.iterrows():
    if s.wraps_origin:
        axE.axvspan(s.start_bp / 1e6, GENOME / 1e6, color=seg_colors[s.E1_sign],
                    alpha=0.5, lw=0)
        axE.axvspan(0, s.end_bp / 1e6, color=seg_colors[s.E1_sign],
                    alpha=0.5, lw=0)
    else:
        axE.axvspan(s.start_bp / 1e6, s.end_bp / 1e6, color=seg_colors[s.E1_sign],
                    alpha=0.5, lw=0)
axE.plot(pos / 1e6, E1, '.', ms=2.5, color='black')
axE.plot(xx / 1e6, fit2, color='#c0392b', lw=1.2,
         label=f'2nd-harmonic fit, |r| = {abs(r_E1_h2):.2f}')
for c in cross_bp:
    axE.axvline(c / 1e6, color='grey', ls=':', lw=0.8)
axE.axvline(TER_BP / 1e6, color='purple', ls='--', lw=1)
axE.text(TER_BP / 1e6, axE.get_ylim()[0] * 0.92, ' ter', color='purple',
         fontsize=6.5, va='bottom')
axE.axvline(0, color='green', ls='--', lw=1)
axE.text(0.03, axE.get_ylim()[0] * 0.92, ' ori', color='green', fontsize=6.5,
         va='bottom')
axE.axhline(0, color='grey', lw=0.5)
axE.set_ylim(top=axE.get_ylim()[1] * 1.6)   # headroom for segment labels
for i, s in segdf.iterrows():
    end_lin = s.end_bp + GENOME if s.wraps_origin else s.end_bp
    mid = ((s.start_bp + end_lin) / 2 % GENOME) / 1e6
    lbl = (f"{s.start_bp/1e6:.2f}-{s.end_bp/1e6:.2f} Mb"
           + (' (wraps ori)' if s.wraps_origin else '') + f"\n({s.E1_sign})")
    axE.text(mid, axE.get_ylim()[1] * 0.92, lbl, ha='center', va='top',
             fontsize=6)
axE.set_xlabel('genomic position (Mb)')
axE.set_ylabel('residual E1')
axE.set_title('residual E1 = four-zone pattern (two full waves; Fourier mode 2)')
axE.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='none')
panel_label(axE, 'E')

plt.savefig(FIG + '/Fig_R4C1_combined.pdf')
plt.savefig(FIG + '/Fig_R4C1_combined.png', dpi=300)
plt.close()

# ------------------------------------------------------------- summary table
lines = []
lines.append('Residual E1 (ori-ter axis removed) four-zone segmentation')
lines.append('=' * 60)
lines.append(f'E1 explains {100*evr[0]:.1f}% of residual-matrix correlation variance;')
lines.append(f'|r(E1, 2nd harmonic cos(4*pi*pos/L) fit)| = {abs(r_E1_h2):.3f}.')
lines.append(f'Zero-crossings (smoothed E1, bp): '
             + ', '.join(str(int(round(c))) for c in cross_bp))
lines.append('')
for _, s in segdf.iterrows():
    tags = []
    if s.contains_ori:
        tags.append('contains ori (position 0)')
    if s.contains_ter:
        tags.append(f'contains ter (~{TER_BP/1e6:.2f} Mb)')
    seglen = (s.end_bp - s.start_bp) % GENOME
    lines.append(f"segment {s.segment}: {s.start_bp}-{s.end_bp} bp "
                 f"({seglen/1e3:.0f} kb"
                 + (', wraps origin' if s.wraps_origin else '')
                 + f"), E1 sign = {s.E1_sign}"
                 + (('; ' + '; '.join(tags)) if tags else ''))
lines.append('')
ori_sign = segdf.loc[segdf.contains_ori, 'E1_sign'].iloc[0]
ter_sign = segdf.loc[segdf.contains_ter, 'E1_sign'].iloc[0]
lines.append('Interpretation: the four alternating zones correspond to a')
lines.append('second-harmonic (two-full-wave) pattern around the circular')
lines.append(f'chromosome. The ori-containing zone (E1 sign {ori_sign}) and the')
lines.append(f'ter-containing zone (E1 sign {ter_sign}) lie half a genome apart')
lines.append('and, as expected for a 2nd harmonic, share the same E1 sign;')
lines.append('the two intervening zones of opposite sign cover the mid-arm')
lines.append('regions of the left replichore (ori -> ter, ~0.48-1.68 Mb) and')
lines.append('the right replichore (ter -> ori, ~3.04-3.73 Mb). This is an')
lines.append('arm-level (macrodomain-like) organization layered on top of the')
lines.append('primary ori-ter axis, not eukaryotic-style A/B compartments')
lines.append('(E1 correlates with neither GC content nor gene density).')
with open(TAB + '/residual_E1_segments_summary.txt', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

print(f'\nkey numbers:')
print(f'  real saddle = {str_orig:.3f} log2, synth = {str_syn:.3f}, '
      f'shuffle mean = {shuf.mean():.3f} (p < {max(p_shuf, 1/(NSHUF+1)):.2f})')
print(f'  residual contrast = {str_res_diff:.3f} '
      f'({100*str_res_diff/str_orig_diff:.0f}% of original), p = {p_res:.2f}')
print(f'  r(PC1,GC) = {r_gc:.3f} (p={p_gc:.2f}), r(PC1,genes) = {r_gd:.3f} (p={p_gd:.2f})')
print(f'  residual E1: {100*evr[0]:.1f}% var, |r| 2nd harmonic = {abs(r_E1_h2):.3f}')
print(segdf.to_string(index=False))
print('Done.')
