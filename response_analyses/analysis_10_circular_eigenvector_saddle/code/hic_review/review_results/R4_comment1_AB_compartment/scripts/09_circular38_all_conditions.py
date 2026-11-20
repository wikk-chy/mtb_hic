#!/usr/bin/env python3
"""
Unified circular-distance pipeline for the redrawn main figures (decision:
harmonic framework in the response letter; main Fig. 1c and ED Fig. 5c-e are
redrawn with the reproducible circular pipeline instead of cooltools E1).

Pipeline per condition (5-kb balanced matrix):
  circular-distance O/E (first two diagonals excluded)
  -> PCA of the correlation matrix -> PC1 (oriented to the ori-ter cosine axis)
  -> saddle: 38 equisized E1-quantile groups spanning qrange (0.025, 0.975)
  -> strength = ln( (mu_AA*mu_BB + eps) / (mu_AB + eps)^2 ), corner frac 0.2

Controls / uncertainty:
  - 100 label permutations per condition -> two-sided empirical p on |strength|
  - pixel bootstrap (10,000; kept for continuity with the old figure)
  - leave-one-group-out jackknife over the 38 groups (conservative SE that
    accounts for spatial autocorrelation; reported in the figure)

Outputs:
  tables/control6_circular38_all_conditions.csv   per-condition summary
  tables/control6_circular38_shuffle_strengths.csv  per-permutation strengths
  data/saddle38_<condition>.npy                   38x38 saddle matrices (for plotting)
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
N_GROUPS = 38
Q_LO, Q_HI = 0.025, 0.975
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
    'napM-KO-hypoxia': BASE + '/review_data/cool_file/naphypoxia_5k.mcool::/resolutions/5000',
}
CONDITIONS['napM-KO-hypoxia'] = BASE + '/review_data/cool_file/napmhypoxia_5k.mcool::/resolutions/5000'

# ------------------------------------------------------------------- helpers
def oe_pipeline(Mv, pos_v):
    """circular-distance O/E (diag >= 2 kept), all-finite filter, corr PCA."""
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


def saddle38(pc, X):
    """38 equisized quantile groups over qrange; vectorized aggregation."""
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    g = np.full(len(pc), -1)
    for gi, grp in enumerate(np.array_split(order, N_GROUPS)):
        g[grp] = gi
    gi, gj = np.meshgrid(g, g, indexing='ij')
    okm = (gi >= 0) & (gj >= 0) & np.isfinite(X)
    lin = (gi[okm] * N_GROUPS + gj[okm]).ravel()
    vals = X[okm].ravel()
    sums = np.bincount(lin, weights=vals, minlength=N_GROUPS ** 2)
    cnts = np.bincount(lin, minlength=N_GROUPS ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (sums / np.maximum(cnts, 1)).reshape(N_GROUPS, N_GROUPS)
        S[cnts.reshape(N_GROUPS, N_GROUPS) == 0] = np.nan
    return S


def strength_blocks(S, frac=CORNER_FRAC):
    """Return AA, BB, AB corner pixel vectors (top-left = low E1 = B)."""
    k = max(1, int(round(S.shape[0] * frac)))
    BB = S[:k, :k].ravel()
    AA = S[-k:, -k:].ravel()
    AB = np.concatenate([S[:k, -k:].ravel(), S[-k:, :k].ravel()])
    return (AA[~np.isnan(AA)], BB[~np.isnan(BB)], AB[~np.isnan(AB)])


def strength_ln(AA, BB, AB, eps=EPS):
    muAA, muBB, muAB = AA.mean(), BB.mean(), AB.mean()
    return (np.log((muAA * muBB + eps) / ((muAB + eps) ** 2)),
            (muAA, muBB, muAB))


def pixel_bootstrap(AA, BB, AB, n_boot=NBOOT, seed=SEED, eps=EPS):
    r = np.random.default_rng(seed)
    ABz = np.where(AB == 0, eps, AB)
    iA = r.integers(0, len(AA), (n_boot, len(AA)))
    iB = r.integers(0, len(BB), (n_boot, len(BB)))
    iX = r.integers(0, len(ABz), (n_boot, len(ABz)))
    cs = np.log((AA[iA].mean(1) * BB[iB].mean(1) + eps)
                / ((ABz[iX].mean(1) + eps) ** 2))
    return cs.mean(), np.percentile(cs, [2.5, 97.5])


def group_jackknife(pc, X):
    """Leave-one-group-out over the 38 saddle groups -> jackknife SE of strength."""
    lo, hi = np.quantile(pc, [Q_LO, Q_HI])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    groups = np.array_split(order, N_GROUPS)
    thetas = []
    for drop in range(N_GROUPS):
        keep = np.concatenate([groups[i] for i in range(N_GROUPS) if i != drop])
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
        AA, BB, AB = strength_blocks(S)
        thetas.append(strength_ln(AA, BB, AB)[0])
    thetas = np.array(thetas)
    full = thetas.mean()
    se = np.sqrt((N_GROUPS - 1) / N_GROUPS * np.sum((thetas - full) ** 2))
    return full, se

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
    # exclude the first two diagonals from the saddle aggregation only
    # (Methods: "the first two diagonals were excluded"); PCA uses full O/E
    dp = np.abs(pos[:, None] - pos[None, :])
    dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
    OE_s = OE.copy()
    OE_s[dbin < 2] = np.nan
    theta = 2 * np.pi * pos / GENOME
    if stats.pearsonr(pc1, np.cos(theta))[0] < 0:
        pc1 = -pc1
    r_cos = stats.pearsonr(pc1, np.cos(theta))[0]
    r_sin = stats.pearsonr(pc1, np.sin(theta))[0]
    r_cos2 = stats.pearsonr(pc1, np.cos(2 * theta))[0]

    S = saddle38(pc1, OE_s)
    np.save(OUT + f'/data/saddle38_{name}.npy', S)
    np.save(OUT + f'/data/pc1_{name}.npy', pc1)
    np.save(OUT + f'/data/pos_{name}.npy', pos)
    AA, BB, AB = strength_blocks(S)
    cs, mus = strength_ln(AA, BB, AB)
    legacy = np.log2((mus[0] + mus[1]) / 2 / mus[2])

    shuf = np.array([
        strength_ln(*strength_blocks(saddle38(rng.permutation(pc1), OE_s)))[0]
        for _ in range(NSHUF)])
    emp_p = (np.sum(np.abs(shuf) >= abs(cs)) + 1) / (NSHUF + 1)
    shuf_table[name] = shuf

    boot_mean, boot_ci = pixel_bootstrap(AA, BB, AB)
    jk_mean, jk_se = group_jackknife(pc1, OE_s)

    print(f'  PC1 var = {100*frac[0]:.1f}%, r cos1 = {r_cos:.3f}, '
          f'sin1 = {r_sin:.3f}, cos2 = {r_cos2:.3f}')
    print(f'  strength = {cs:.3f} ln (legacy log2 {legacy:.3f}); '
          f'shuffle max|.| = {np.abs(shuf).max():.3f}, p = {emp_p:.4f}')
    print(f'  bootstrap {boot_mean:.3f} [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]; '
          f'jackknife {jk_mean:.3f} +- {jk_se:.3f} (SE)')

    rows.append(dict(condition=name, pc1_var_pct=100 * frac[0],
                     pc1_r_cos1=r_cos, pc1_r_sin1=r_sin, pc1_r_cos2=r_cos2,
                     strength_ln=cs, strength_log2_legacy=legacy,
                     muAA=mus[0], muBB=mus[1], muAB=mus[2],
                     shuffle_mean=shuf.mean(),
                     shuffle_absmax=np.abs(shuf).max(), empirical_p=emp_p,
                     boot_mean=boot_mean, boot_lo=boot_ci[0], boot_hi=boot_ci[1],
                     jackknife_mean=jk_mean, jackknife_se=jk_se))

pd.DataFrame(rows).to_csv(OUT + '/tables/control6_circular38_all_conditions.csv',
                          index=False)
pd.DataFrame(dict([(k, v) for k, v in shuf_table.items()])).to_csv(
    OUT + '/tables/control6_circular38_shuffle_strengths.csv', index=False)
print('\nwrote tables/control6_circular38_all_conditions.csv')
print('wrote tables/control6_circular38_shuffle_strengths.csv')
print('Done.')
