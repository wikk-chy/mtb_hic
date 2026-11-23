#!/usr/bin/env python3
"""
Final unified saddle analysis (per user decision):
  20 E1-quantile groups, qrange (0.025, 0.975), outer-20% corner blocks,
  strength = log2( mean(AA u BB) / mean(AB) )   [the metric that gave WT 1.32]
on the circular pipeline (circular O/E + correlation-PCA PC1 = ori-ter axis).

Also: saddle-pattern similarity of each condition to WT (Pearson r of the
flattened 20x20 saddle), with
  - null: r of 100 label-shuffled WT saddles to the true WT saddle
  - ceiling: mean pairwise r among the four WT biological replicates
    (review_data/rep_icedmatrix/wt_[1-4]_5000_iced.matrix)

Outputs:
  data/saddle20q_<condition>.npy  (+ WT replicates)
  tables/control9_final_saddle.csv
  tables/control9_saddle_similarity.csv
Seed 42. Absolute paths.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import cooler
from scipy import stats

BASE = '/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review'
OUT = BASE + '/review_results/R4_comment1_AB_compartment'
BIN = 5000
GENOME = 4411532
N_GROUPS = 20
Q_LO, Q_HI = 0.025, 0.975
NSHUF = 100
SEED = 42
rng = np.random.default_rng(SEED)

CONDITIONS = {
    'WT': BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000',
    'WT-hypoxia': BASE + '/review_data/cool_file/hypoxia_5k.mcool::/resolutions/5000',
    'WT-latent': BASE + '/review_data/cool_file/latent_5k.mcool::/resolutions/5000',
    'napM-KO': BASE + '/review_data/cool_file/napm_ko_5k.cool::/resolutions/5000',
    'napM-KO-hypoxia': BASE + '/review_data/cool_file/napmhypoxia_5k.mcool::/resolutions/5000',
}


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
    OE = OE[np.ix_(fin, fin)]
    pos = pos_v[fin]
    C = np.corrcoef(OE)
    okc = np.isfinite(C).all(axis=0)
    OE, pos, C = OE[np.ix_(okc, okc)], pos[okc], C[np.ix_(okc, okc)]
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return OE, pos, ev[o] / ev[o].sum(), vec[:, o[0]]


def saddleN(pc, X, groups=N_GROUPS):
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    g = np.full(len(pc), -1)
    for gi, grp in enumerate(np.array_split(order, groups)):
        g[grp] = gi
    gi, gj = np.meshgrid(g, g, indexing='ij')
    okm = (gi >= 0) & (gj >= 0) & np.isfinite(X)
    lin = (gi[okm] * groups + gj[okm]).ravel()
    vals = X[okm].ravel()
    sums = np.bincount(lin, weights=vals, minlength=groups ** 2)
    cnts = np.bincount(lin, minlength=groups ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (sums / np.maximum(cnts, 1)).reshape(groups, groups)
        S[cnts.reshape(groups, groups) == 0] = np.nan
    return S


def strength_corner(S):
    """log2( mean(AA u BB) / mean(AB) ), outer-20% corner blocks."""
    k = max(1, int(round(S.shape[0] * 0.2)))
    aabb = np.concatenate([S[:k, :k].ravel(), S[-k:, -k:].ravel()])
    ab = np.concatenate([S[:k, -k:].ravel(), S[-k:, :k].ravel()])
    return np.log2(np.nanmean(aabb) / np.nanmean(ab))


def jackknife(pc, X):
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    grps = np.array_split(order, N_GROUPS)
    thetas = []
    for drop in range(N_GROUPS):
        g = np.full(len(pc), -1)
        sub = grps[:drop] + grps[drop + 1:]
        for gi, grp in enumerate(sub):
            g[grp] = gi
        gi, gj = np.meshgrid(g, g, indexing='ij')
        okm = (gi >= 0) & (gj >= 0) & np.isfinite(X)
        lin = (gi[okm] * (N_GROUPS - 1) + gj[okm]).ravel()
        vals = X[okm].ravel()
        sums = np.bincount(lin, weights=vals, minlength=(N_GROUPS - 1) ** 2)
        cnts = np.bincount(lin, minlength=(N_GROUPS - 1) ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            S = (sums / np.maximum(cnts, 1)).reshape(N_GROUPS - 1,
                                                     N_GROUPS - 1)
        thetas.append(strength_corner(S))
    thetas = np.array(thetas)
    se = np.sqrt((N_GROUPS - 1) / N_GROUPS * np.sum((thetas - thetas.mean()) ** 2))
    return thetas.mean(), se


def flat(S):
    v = S.ravel()
    return v


def load_iced_triplets(path, nbins=883):
    M = np.zeros((nbins, nbins))
    dat = np.loadtxt(path)
    i = dat[:, 0].astype(int) - 1
    j = dat[:, 1].astype(int) - 1
    M[i, j] = dat[:, 2]
    M[j, i] = dat[:, 2]
    return M


# ---------------------------------------------------------------- conditions
rows, shuf_tab = [], {}
saddles = {}
for name, uri in CONDITIONS.items():
    clr = cooler.Cooler(uri)
    Mbal = clr.matrix(balance=True)[:]
    valid = ~np.isnan(Mbal).all(axis=0)
    pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
    OE, pos, frac, pc1 = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
    dp = np.abs(pos[:, None] - pos[None, :])
    dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
    OE[dbin < 2] = np.nan
    theta = 2 * np.pi * pos / GENOME
    if stats.pearsonr(pc1, np.cos(theta))[0] < 0:
        pc1 = -pc1
    S = saddleN(pc1, OE)
    saddles[name] = S
    np.save(OUT + f'/data/saddle20q_{name}.npy', S)
    cs = strength_corner(S)
    shuf = np.array([strength_corner(saddleN(rng.permutation(pc1), OE))
                     for _ in range(NSHUF)])
    emp_p = (np.sum(np.abs(shuf) >= abs(cs)) + 1) / (NSHUF + 1)
    shuf_tab[name] = shuf
    jk, se = jackknife(pc1, OE)
    rows.append(dict(condition=name, pc1_r_cos1=stats.pearsonr(pc1, np.cos(theta))[0],
                     strength_log2=cs, jackknife_mean=jk, jackknife_se=se,
                     shuffle_absmax=np.abs(shuf).max(), empirical_p=emp_p))
    print(f'{name:16s} strength = {cs:.3f} log2  jk {jk:.3f}+-{se:.3f}  '
          f'shufmax {np.abs(shuf).max():.3f}  p={emp_p:.4f}')

df = pd.DataFrame(rows).set_index('condition')
df.to_csv(OUT + '/tables/control9_final_saddle.csv')
pd.DataFrame(shuf_tab).to_csv(OUT + '/tables/control9_shuffle_strengths.csv')

print('\npairwise (jackknife z):')
pairs = [('WT', 'WT-hypoxia'), ('WT', 'WT-latent'), ('WT', 'napM-KO'),
         ('napM-KO', 'napM-KO-hypoxia'), ('WT-hypoxia', 'napM-KO-hypoxia')]
prows = []
for a, b in pairs:
    d = df.loc[a, 'strength_log2'] - df.loc[b, 'strength_log2']
    se = np.sqrt(df.loc[a, 'jackknife_se'] ** 2 + df.loc[b, 'jackknife_se'] ** 2)
    p = 2 * stats.norm.sf(abs(d / se))
    prows.append(dict(comparison=f'{a} vs {b}', diff_log2=d, p_value=p))
    print(f'  {a:12s} vs {b:16s} diff={d:+.3f} p={p:.3g}')
pd.DataFrame(prows).to_csv(OUT + '/tables/control9_pairwise_pvalues.csv',
                           index=False)

# ------------------------------------------------- saddle similarity to WT
WTv = flat(saddles['WT'])
ok = np.isfinite(WTv)

# null: shuffled WT E1 saddles vs true WT saddle
clr = cooler.Cooler(CONDITIONS['WT'])
Mbal = clr.matrix(balance=True)[:]
valid = ~np.isnan(Mbal).all(axis=0)
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
OE_wt, pos_wt, _, pc1_wt = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
dp = np.abs(pos_wt[:, None] - pos_wt[None, :])
dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
OE_wt[dbin < 2] = np.nan
null_r = np.array([
    stats.pearsonr(flat(saddleN(rng.permutation(pc1_wt), OE_wt))[ok],
                   WTv[ok])[0]
    for _ in range(NSHUF)])

sim_rows = []
for name in CONDITIONS:
    if name == 'WT':
        continue
    v = flat(saddles[name])
    m = ok & np.isfinite(v)
    r = stats.pearsonr(v[m], WTv[m])[0]
    p = (np.sum(null_r >= r) + 1) / (NSHUF + 1)
    sim_rows.append(dict(condition=name, pearson_r_vs_WT=r,
                         null_max=null_r.max(), p_value=p))
    print(f'sim {name:16s} r = {r:.3f}  (null max {null_r.max():.3f}, p={p:.4f})')

# WT replicate ceiling
def oe_pipeline_sparse(Mv, pos_v, max_nan_frac=0.2):
    """Variant of oe_pipeline tolerating missing pairs (iced replicate
    matrices): drop bins with >20% NaN O/E columns, fill residual NaN with
    1.0 (neutral) for PCA only; return NaN-preserving OE for the saddle."""
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
    fin = np.isnan(OE).mean(axis=0) < max_nan_frac
    OE = OE[np.ix_(fin, fin)]
    pos = pos_v[fin]
    OE_f = np.where(np.isfinite(OE), OE, 1.0)
    C = np.corrcoef(OE_f)
    okc = np.isfinite(C).all(axis=0)
    OE, pos, C = OE[np.ix_(okc, okc)], pos[okc], C[np.ix_(okc, okc)]
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return OE, pos, ev[o] / ev[o].sum(), vec[:, o[0]]


rep_rs = []
rep_S = []
for i in (1, 2, 3, 4):
    M = load_iced_triplets(BASE + f'/review_data/rep_icedmatrix/wt_{i}_5000_iced.matrix')
    validr = M.sum(axis=0) > 0
    posr = (np.where(validr)[0] * BIN + BIN / 2).astype(float)
    OEr, posr2, _, pc1r = oe_pipeline_sparse(M[np.ix_(validr, validr)], posr)
    dpr = np.abs(posr2[:, None] - posr2[None, :])
    dbr = np.round(np.minimum(dpr, GENOME - dpr) / BIN).astype(int)
    OEr[dbr < 2] = np.nan
    thr = 2 * np.pi * posr2 / GENOME
    if stats.pearsonr(pc1r, np.cos(thr))[0] < 0:
        pc1r = -pc1r
    Sr = saddleN(pc1r, OEr)
    rep_S.append(Sr)
    np.save(OUT + f'/data/saddle20q_WT_rep{i}.npy', Sr)
for a in range(4):
    for b in range(a + 1, 4):
        va, vb = flat(rep_S[a]), flat(rep_S[b])
        m = np.isfinite(va) & np.isfinite(vb)
        rep_rs.append(stats.pearsonr(va[m], vb[m])[0])
rep_rs = np.array(rep_rs)
print(f'WT replicate pairwise saddle r: {np.round(rep_rs, 3)} '
      f'(mean {rep_rs.mean():.3f} +- {rep_rs.std():.3f})')

sim_rows.append(dict(condition='WT replicate ceiling (mean pairwise r)',
                     pearson_r_vs_WT=rep_rs.mean(),
                     null_max=np.nan, p_value=np.nan))
pd.DataFrame(sim_rows).to_csv(OUT + '/tables/control9_saddle_similarity.csv',
                              index=False)
print('\nwrote tables/control9_final_saddle.csv, control9_pairwise_pvalues.csv,'
      ' control9_saddle_similarity.csv, control9_shuffle_strengths.csv')
print('Done.')
