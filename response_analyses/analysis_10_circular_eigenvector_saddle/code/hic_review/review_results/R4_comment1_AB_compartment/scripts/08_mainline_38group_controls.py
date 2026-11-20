#!/usr/bin/env python3
"""
Re-run the Reviewer-4-comment-1 controls with the MAIN-text saddle pipeline
(notebook ABcompartment.ipynb cell 4 / cell 6 parameter set):

    cooltools.eigs_cis (GC-phased, balanced)  ->
    cooltools.expected_cis                    ->
    cooltools.saddle(n_bins=38, qrange=(0.025, 0.975))   [min_diag default=2, no max_diag]
    strength = ln( (mu_AA * mu_BB + eps) / (mu_AB + eps)^2 ),  corner frac = 0.2, eps = 1e-6

Three runs on the identical pipeline:
  1. real WT matrix (wt_5k.cool::/resolutions/5000)
  2. synthetic matrix built purely from the WT circular P(s) curve
     (Poisson-sampled, total contacts matched; seed 42)
  3. 100 label permutations of the real WT E1 track (and of the synthetic E1)

Outputs:
  tables/control5_mainline_38group_saddle.csv    summary numbers
  tables/control5_mainline_shuffle_strengths.csv  per-permutation strengths
Seed fixed (42). Absolute paths.
"""
import os
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict

import numpy as np
import pandas as pd
import cooler
import cooltools
import bioframe
from cooltools.lib.common import make_cooler_view
from scipy import stats

BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUT = BASE + '/review_results/R4_comment1_AB_compartment'
WT_URI = BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000'
SYNTH_URI = OUT + '/data/synth_Ps_5k.cool'
os.makedirs(OUT + '/data', exist_ok=True)
FASTA = BASE + '/genome.fa'

BIN = 5000
GENOME = 4411532
N_GROUPS = 38
QRANGE = (0.025, 0.975)
CORNER_FRAC = 0.2
EPS = 1e-6
NSHUF = 100
SEED = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------- helpers
def gc_track_for(clr):
    fasta = bioframe.load_fasta(FASTA)
    genome = OrderedDict({'Chromosome': fasta['Chromosome']})
    bins = clr.bins()[:][['chrom', 'start', 'end']]
    gc_cov = bioframe.frac_gc(bins, genome)
    return gc_cov[['chrom', 'start', 'end', 'GC']].copy()


def eigs_and_saddle(clr, gc_track, e1_override=None):
    """Run eigs_cis (+ optional E1 override) and cooltools.saddle with the
    38-group main-line parameters. Returns (M38, e1_values, positions)."""
    view_df = make_cooler_view(clr)
    eigvals, eig_df = cooltools.eigs_cis(clr, gc_track, view_df=view_df, n_eigs=3)
    track = eig_df[['chrom', 'start', 'end', 'E1']].copy()
    if e1_override is not None:
        track['E1'] = e1_override
    cvd = cooltools.expected_cis(clr=clr, view_df=view_df)
    interaction_sum, interaction_count = cooltools.saddle(
        clr, cvd, track, 'cis',
        n_bins=N_GROUPS, qrange=QRANGE, view_df=view_df,
    )
    M = interaction_sum / interaction_count
    M = np.asarray(M, dtype=float)
    # cooltools.saddle returns (n_bins + 2) x (n_bins + 2) with flanking
    # outlier rows/cols -> strip to the 38 x 38 core
    if M.shape[0] == N_GROUPS + 2:
        M = M[1:-1, 1:-1]
    assert M.shape == (N_GROUPS, N_GROUPS), M.shape
    e1 = track['E1'].values.astype(float)
    pos = (track['start'].values + BIN / 2).astype(float)
    return M, e1, pos


def strength_ln(M, frac=CORNER_FRAC, eps=EPS):
    """Code-canonical compartment strength: ln((muAA*muBB+eps)/(muAB+eps)^2)."""
    k = max(1, int(round(M.shape[0] * frac)))
    BB = M[:k, :k].ravel()
    AA = M[-k:, -k:].ravel()
    AB = np.concatenate([M[:k, -k:].ravel(), M[-k:, :k].ravel()])
    AA = AA[~np.isnan(AA)]; BB = BB[~np.isnan(BB)]; AB = AB[~np.isnan(AB)]
    muAA, muBB, muAB = AA.mean(), BB.mean(), AB.mean()
    cs = np.log((muAA * muBB + eps) / ((muAB + eps) ** 2))
    legacy = np.log2((muAA + muBB) / 2 / muAB)  # for cross-ref with 20-group run
    return cs, legacy, (muAA, muBB, muAB)


def shuffle_strengths(clr, gc_track, e1, nshuf=NSHUF):
    """Permute E1 labels among valid (non-NaN) bins; recompute full pipeline."""
    valid = np.isfinite(e1)
    vals = e1[valid]
    out = np.empty(nshuf)
    for i in range(nshuf):
        e1p = e1.copy()
        e1p[valid] = rng.permutation(vals)
        M, _, _ = eigs_and_saddle(clr, gc_track, e1_override=e1p)
        out[i] = strength_ln(M)[0]
    return out


# ==================================================================== real WT
print('== real WT, 38-group main pipeline ==')
clr = cooler.Cooler(WT_URI)
gc_track = gc_track_for(clr)
M_real, e1_real, pos_real = eigs_and_saddle(clr, gc_track)
theta = 2 * np.pi * pos_real[np.isfinite(e1_real)] / GENOME
fin = np.isfinite(e1_real)
def _harm(e, pos):
    m = np.isfinite(e)
    th = 2 * np.pi * pos[m] / GENOME
    ev = e[m]
    return (stats.pearsonr(ev, np.cos(th))[0],
            stats.pearsonr(ev, np.sin(th))[0],
            stats.pearsonr(ev, np.cos(2 * th))[0],
            stats.pearsonr(ev, np.sin(2 * th))[0])
rh_real = _harm(e1_real, pos_real)
r_gc = stats.pearsonr(e1_real[fin], gc_track['GC'].values[fin])[0]
cs_real, leg_real, mus_real = strength_ln(M_real)
print(f'  E1 harmonics: cos1={rh_real[0]:.3f} sin1={rh_real[1]:.3f} '
      f'cos2={rh_real[2]:.3f} sin2={rh_real[3]:.3f}; r(E1, GC) = {r_gc:.3f}')
print(f'  strength = {cs_real:.3f} ln  (legacy log2 (AA+BB)/AB = {leg_real:.3f})')
print(f'  muAA={mus_real[0]:.4f} muBB={mus_real[1]:.4f} muAB={mus_real[2]:.4f}')

print(f'  {NSHUF} label permutations ...')
shuf_real = shuffle_strengths(clr, gc_track, e1_real)
emp_p = (np.sum(np.abs(shuf_real) >= abs(cs_real)) + 1) / (NSHUF + 1)
print(f'  shuffle mean = {shuf_real.mean():.4f}, p95 = '
      f'{np.percentile(shuf_real, 95):.4f}, max|.| = '
      f'{np.abs(shuf_real).max():.4f}, two-sided empirical p = {emp_p:.4f}')

# ====================================================== synthetic P(s) matrix
print('\n== synthetic P(s)-only matrix ==')
Mraw = clr.matrix(balance=False)[:]
Mraw = np.nan_to_num(Mraw, nan=0.0)
w = clr.bins()[:]['weight'].values
valid = ~np.isnan(w)
Mraw_v = Mraw[np.ix_(valid, valid)]
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
n_v = Mraw_v.shape[0]

dp = np.abs(pos_v[:, None] - pos_v[None, :])
dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
cnt = np.zeros(dbin.max() + 1)
summ = np.zeros(dbin.max() + 1)
nz = Mraw_v > 0
np.add.at(cnt, dbin[nz], 1)
np.add.at(summ, dbin[nz], Mraw_v[nz])
Ps = np.where(cnt > 0, summ / np.maximum(cnt, 1), 0.0)

lam = Ps[dbin]
iu = np.triu_indices(n_v, k=1)
synth_v = np.zeros((n_v, n_v))
synth_v[iu] = rng.poisson(lam[iu])
synth_v[(iu[1], iu[0])] = synth_v[iu]
np.fill_diagonal(synth_v, Ps[0])
print(f'  total contacts: synth = {synth_v.sum():.0f}, real raw = {Mraw_v.sum():.0f}')

# embed back into full 883 x 883 grid (invalid bins stay zero) and write .cool
n = Mraw.shape[0]
synth = np.zeros((n, n))
synth[np.ix_(valid, valid)] = synth_v
ii, jj = np.triu_indices(n)
counts = synth[ii, jj]
keep = counts > 0
bins_df = clr.bins()[:][['chrom', 'start', 'end']]
pixels = pd.DataFrame({'bin1_id': ii[keep], 'bin2_id': jj[keep],
                       'count': counts[keep].astype(np.int64)})
cooler.create_cooler(SYNTH_URI, bins_df, pixels, assembly='ASM19595v2')
clr_syn = cooler.Cooler(SYNTH_URI)
cooler.balance_cooler(clr_syn, store=True)
print(f'  wrote + balanced {SYNTH_URI}')

gc_syn = gc_track_for(clr_syn)
M_syn, e1_syn, pos_syn = eigs_and_saddle(clr_syn, gc_syn)
rh_syn = _harm(e1_syn, pos_syn)
cs_syn, leg_syn, mus_syn = strength_ln(M_syn)
print(f'  E1 harmonics: cos1={rh_syn[0]:.3f} sin1={rh_syn[1]:.3f} '
      f'cos2={rh_syn[2]:.3f} sin2={rh_syn[3]:.3f}')
print(f'  strength = {cs_syn:.3f} ln  (legacy log2 = {leg_syn:.3f})')
print(f'  muAA={mus_syn[0]:.4f} muBB={mus_syn[1]:.4f} muAB={mus_syn[2]:.4f}')

print(f'  {NSHUF} label permutations (synthetic E1) ...')
shuf_syn = shuffle_strengths(clr_syn, gc_syn, e1_syn)
print(f'  shuffle mean = {shuf_syn.mean():.4f}, p95 = '
      f'{np.percentile(shuf_syn, 95):.4f}, max = {shuf_syn.max():.4f}')

# -------------------------------------------------------------------- outputs
summary = pd.DataFrame([
    dict(matrix='real_WT', pipeline='cooltools 38grp qrange(0.025,0.975) ln',
         e1_r_cos1=rh_real[0], e1_r_sin1=rh_real[1],
         e1_r_cos2=rh_real[2], e1_r_sin2=rh_real[3], e1_r_GC=r_gc,
         strength_ln=cs_real, strength_log2_legacy=leg_real,
         muAA=mus_real[0], muBB=mus_real[1], muAB=mus_real[2],
         shuffle_mean=shuf_real.mean(),
         shuffle_p95=np.percentile(shuf_real, 95),
         shuffle_absmax=np.abs(shuf_real).max(), empirical_p=emp_p),
    dict(matrix='synthetic_Ps_only', pipeline='cooltools 38grp qrange(0.025,0.975) ln',
         e1_r_cos1=rh_syn[0], e1_r_sin1=rh_syn[1],
         e1_r_cos2=rh_syn[2], e1_r_sin2=rh_syn[3], e1_r_GC=np.nan,
         strength_ln=cs_syn, strength_log2_legacy=leg_syn,
         muAA=mus_syn[0], muBB=mus_syn[1], muAB=mus_syn[2],
         shuffle_mean=shuf_syn.mean(),
         shuffle_p95=np.percentile(shuf_syn, 95),
         shuffle_absmax=np.abs(shuf_syn).max(), empirical_p=np.nan),
])
summary.to_csv(OUT + '/tables/control5_mainline_38group_saddle.csv', index=False)

pd.DataFrame({'perm': np.arange(1, NSHUF + 1),
              'real_WT_strength_ln': shuf_real,
              'synthetic_strength_ln': shuf_syn}).to_csv(
    OUT + '/tables/control5_mainline_shuffle_strengths.csv', index=False)

print(f'\nwrote {OUT}/tables/control5_mainline_38group_saddle.csv')
print(f'wrote {OUT}/tables/control5_mainline_shuffle_strengths.csv')
print('Done.')
