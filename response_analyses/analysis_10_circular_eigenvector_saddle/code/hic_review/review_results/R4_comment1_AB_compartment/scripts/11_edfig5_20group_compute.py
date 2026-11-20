#!/usr/bin/env python3
"""
ED Fig. 5c-e with the Methods-text parameter set: 20 quantile groups over
qrange (0.02, 0.98), outer-20% corner blocks (k = 4 groups), on the unified
circular pipeline (circular-distance O/E + correlation-PCA PC1 = ori-ter axis).

Fixes relative to the original notebook cell 6:
  - uses the proper MEAN saddle matrix (interaction_sum / interaction_count),
    not saddle[0] (the sum matrix);
  - no 500-kb max-distance cap (with the cap, cross-corner group pairs have
    zero pixels on this genome -> muAB = 0 -> artifactual strength);
  - consistent axis across conditions (PC1 = ori-ter in all five).

Outputs:
  tables/control7_circular20group_all_conditions.csv
  tables/control7_circular20group_shuffle_strengths.csv
  data/saddle20_<condition>.npy
Seed fixed (42). Absolute paths.
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
Q_LO, Q_HI = 0.02, 0.98
CORNER_FRAC = 0.2
EPS = 1e-6
NSHUF = 100
NBOOT = 10000
SEED = 42
rng = np.random.default_rng(SEED)

CONDITIONS = {
    'WT': BASE + '/review_data/cool_file/wt_5k.cool::/resolutions/5000',
    'WT-hypoxia': BASE + '/review_data/cool_file/hypoxia_5k.mcool::/resolutions/5000',
    'WT-latent': BASE + '/review_data/cool_file/latent_5k.mcool::/resolutions/5000',
    'napM-KO': BASE + '/review_data/cool_file/napm_ko_5k.cool::/resolutions/5000',
    'napM-KO-hypoxia': BASE + '/review_data/cool_file/napmhypoxia_5k.mcool::/resolutions/5000',
}

# ------------------------------------------------------------------- helpers
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
    ev, vec = ev[o], vec[:, o]
    return OE, pos, ev / ev.sum(), vec[:, 0]


def saddleN(pc, X, groups=N_GROUPS, qlo=Q_LO, qhi=Q_HI):
    lo, hi = np.quantile(pc, [qlo, qhi])
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


def strength_quadrant(S, eps=EPS):
    """Half-quadrant segregation strength (no corner restriction):
    AA/BB = same-axis half quadrants, AB = cross-axis quadrants;
    log2((muAA*muBB+eps)/(muAB+eps)^2)."""
    h = S.shape[0] // 2
    AA = S[h:, h:].ravel()
    BB = S[:h, :h].ravel()
    AB = np.concatenate([S[:h, h:].ravel(), S[h:, :h].ravel()])
    muAA, muBB, muAB = np.nanmean(AA), np.nanmean(BB), np.nanmean(AB)
    return np.log2((muAA * muBB + eps) / ((muAB + eps) ** 2)), (muAA, muBB, muAB)


def pixel_bootstrap(S, n_boot=NBOOT, seed=SEED, eps=EPS):
    h = S.shape[0] // 2
    AA = S[h:, h:].ravel(); AA = AA[~np.isnan(AA)]
    BB = S[:h, :h].ravel(); BB = BB[~np.isnan(BB)]
    AB = np.concatenate([S[:h, h:].ravel(), S[h:, :h].ravel()])
    AB = AB[~np.isnan(AB)]
    r = np.random.default_rng(seed)
    ABz = np.where(AB == 0, eps, AB)
    iA = r.integers(0, len(AA), (n_boot, len(AA)))
    iB = r.integers(0, len(BB), (n_boot, len(BB)))
    iX = r.integers(0, len(ABz), (n_boot, len(ABz)))
    cs = np.log2((AA[iA].mean(1) * BB[iB].mean(1) + eps)
                 / ((ABz[iX].mean(1) + eps) ** 2))
    return cs.mean(), np.percentile(cs, [2.5, 97.5])


def group_jackknife(pc, X):
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    groups = np.array_split(order, N_GROUPS)
    thetas = []
    for drop in range(N_GROUPS):
        g = np.full(len(pc), -1)
        sub = groups[:drop] + groups[drop + 1:]
        for gi, grp in enumerate(sub):
            g[grp] = gi
        gi, gj = np.meshgrid(g, g, indexing='ij')
        okm = (gi >= 0) & (gj >= 0) & np.isfinite(X)
        lin = (gi[okm] * (N_GROUPS - 1) + gj[okm]).ravel()
        vals = X[okm].ravel()
        sums = np.bincount(lin, weights=vals, minlength=(N_GROUPS - 1) ** 2)
        cnts = np.bincount(lin, minlength=(N_GROUPS - 1) ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            S = (sums / np.maximum(cnts, 1)).reshape(N_GROUPS - 1, N_GROUPS - 1)
        thetas.append(strength_quadrant(S)[0])
    thetas = np.array(thetas)
    se = np.sqrt((N_GROUPS - 1) / N_GROUPS * np.sum((thetas - thetas.mean()) ** 2))
    return thetas.mean(), se

# ====================================================================== main
rows = []
shuf_table = {}
for name, uri in CONDITIONS.items():
    print(f'== {name} ==')
    clr = cooler.Cooler(uri)
    Mbal = clr.matrix(balance=True)[:]
    valid = ~np.isnan(Mbal).all(axis=0)
    pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
    OE, pos, frac, pc1 = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
    dp = np.abs(pos[:, None] - pos[None, :])
    dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
    OE_s = OE.copy()
    OE_s[dbin < 2] = np.nan
    theta = 2 * np.pi * pos / GENOME
    if stats.pearsonr(pc1, np.cos(theta))[0] < 0:
        pc1 = -pc1
    r_cos = stats.pearsonr(pc1, np.cos(theta))[0]

    S = saddleN(pc1, OE_s)
    np.save(OUT + f'/data/saddle20_{name}.npy', S)
    cs, mus = strength_quadrant(S)

    shuf = np.array([
        strength_quadrant(saddleN(rng.permutation(pc1), OE_s))[0]
        for _ in range(NSHUF)])
    emp_p = (np.sum(np.abs(shuf) >= abs(cs)) + 1) / (NSHUF + 1)
    shuf_table[name] = shuf

    boot_mean, boot_ci = pixel_bootstrap(S)
    jk_mean, jk_se = group_jackknife(pc1, OE_s)

    print(f'  PC1 var = {100*frac[0]:.1f}%, r cos1 = {r_cos:.3f}')
    print(f'  strength = {cs:.3f} log2; '
          f'shuffle max|.| = {np.abs(shuf).max():.3f}, p = {emp_p:.4f}')
    print(f'  bootstrap {boot_mean:.3f} [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]; '
          f'jackknife {jk_mean:.3f} +- {jk_se:.3f} (SE)')

    rows.append(dict(condition=name, pc1_var_pct=100 * frac[0],
                     pc1_r_cos1=r_cos,
                     strength_log2=cs,
                     muAA=mus[0], muBB=mus[1], muAB=mus[2],
                     shuffle_mean=shuf.mean(),
                     shuffle_absmax=np.abs(shuf).max(), empirical_p=emp_p,
                     boot_mean=boot_mean, boot_lo=boot_ci[0], boot_hi=boot_ci[1],
                     jackknife_mean=jk_mean, jackknife_se=jk_se))

df = pd.DataFrame(rows).set_index('condition')
df.to_csv(OUT + '/tables/control7_circular20group_all_conditions.csv')

# pairwise comparisons via jackknife z-test
pairs = [('WT', 'WT-hypoxia'), ('WT', 'WT-latent'), ('WT', 'napM-KO'),
         ('napM-KO', 'napM-KO-hypoxia')]
prow = []
for a, b in pairs:
    d = df.loc[a, 'strength_log2'] - df.loc[b, 'strength_log2']
    se = np.sqrt(df.loc[a, 'jackknife_se'] ** 2 + df.loc[b, 'jackknife_se'] ** 2)
    z = d / se
    p = 2 * stats.norm.sf(abs(z))
    prow.append(dict(comparison=f'{a} vs {b}', diff_log2=d, z=z, p_value=p))
    print(f'  {a} vs {b}: diff = {d:.3f}, z = {z:.2f}, p = {p:.2e}')
pd.DataFrame(prow).to_csv(
    OUT + '/tables/control7_circular20group_pairwise_pvalues.csv', index=False)

pd.DataFrame(shuf_table).to_csv(
    OUT + '/tables/control7_circular20group_shuffle_strengths.csv', index=False)
print('\nwrote tables/control7_circular20group_all_conditions.csv')
print('wrote tables/control7_circular20group_pairwise_pvalues.csv')
print('wrote tables/control7_circular20group_shuffle_strengths.csv')
print('Done.')
