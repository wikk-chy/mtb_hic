#!/usr/bin/env python3
"""
Sensitivity check: qrange (0.025, 0.975), corner blocks = outer 20% of groups,
strength = log2((muAA*muBB+eps)/(muAB+eps)^2), jackknife SE, pairwise z-tests.
Computed for both 20-group (k=4) and 38-group (k=8) saddles.
Outputs: tables/control8_corner20pct_q0025.csv
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
CONDITIONS = {
    'WT': BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000',
    'WT-hypoxia': BASE + '/review_data/cool_file/hypoxia_5k.mcool::/resolutions/5000',
    'WT-latent': BASE + '/review_data/cool_file/latent_5k.mcool::/resolutions/5000',
    'napM-KO': BASE + '/review_data/cool_file/napm_ko_5k.cool::/resolutions/5000',
    'napM-KO-hypoxia': BASE + '/review_data/cool_file/napmhypoxia_5k.mcool::/resolutions/5000',
}
Q_LO, Q_HI = 0.025, 0.975
EPS = 1e-6


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


def saddleN(pc, X, groups):
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


def strength_corner(S, eps=EPS):
    k = max(1, int(round(S.shape[0] * 0.2)))
    BB = np.nanmean(S[:k, :k])
    AA = np.nanmean(S[-k:, -k:])
    AB = np.nanmean(np.concatenate([S[:k, -k:].ravel(), S[-k:, :k].ravel()]))
    return np.log2((AA * BB + eps) / ((AB + eps) ** 2))


def jackknife(pc, X, groups):
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    grps = np.array_split(order, groups)
    thetas = []
    for drop in range(groups):
        g = np.full(len(pc), -1)
        sub = grps[:drop] + grps[drop + 1:]
        for gi, grp in enumerate(sub):
            g[grp] = gi
        gi, gj = np.meshgrid(g, g, indexing='ij')
        okm = (gi >= 0) & (gj >= 0) & np.isfinite(X)
        lin = (gi[okm] * (groups - 1) + gj[okm]).ravel()
        vals = X[okm].ravel()
        sums = np.bincount(lin, weights=vals, minlength=(groups - 1) ** 2)
        cnts = np.bincount(lin, minlength=(groups - 1) ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            S = (sums / np.maximum(cnts, 1)).reshape(groups - 1, groups - 1)
        thetas.append(strength_corner(S))
    thetas = np.array(thetas)
    se = np.sqrt((groups - 1) / groups * np.sum((thetas - thetas.mean()) ** 2))
    return thetas.mean(), se


rows = []
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
    for groups in (20, 38):
        S = saddleN(pc1, OE, groups)
        cs = strength_corner(S)
        jk, se = jackknife(pc1, OE, groups)
        rows.append(dict(condition=name, groups=groups,
                         corner_k=max(1, int(round(groups * 0.2))),
                         strength_log2=cs, jackknife_mean=jk, jackknife_se=se))
        print(f'{name:16s} {groups}grp k={max(1,int(round(groups*0.2)))}  '
              f'strength = {cs:.3f} log2  (jk {jk:.3f} +- {se:.3f})')

df = pd.DataFrame(rows)
df.to_csv(OUT + '/tables/control8_corner20pct_q0025.csv', index=False)

print('\npairwise (jackknife z):')
for groups in (20, 38):
    sub = df[df.groups == groups].set_index('condition')
    for a, b in [('WT-hypoxia', 'napM-KO-hypoxia'), ('WT', 'WT-latent'),
                 ('WT', 'WT-hypoxia'), ('WT', 'napM-KO'),
                 ('napM-KO', 'napM-KO-hypoxia')]:
        d = sub.loc[a, 'strength_log2'] - sub.loc[b, 'strength_log2']
        se = np.sqrt(sub.loc[a, 'jackknife_se'] ** 2 + sub.loc[b, 'jackknife_se'] ** 2)
        p = 2 * stats.norm.sf(abs(d / se))
        print(f'  {groups}grp  {a:12s} vs {b:16s} diff={d:+.3f} p={p:.3g}')
print('Done.')
