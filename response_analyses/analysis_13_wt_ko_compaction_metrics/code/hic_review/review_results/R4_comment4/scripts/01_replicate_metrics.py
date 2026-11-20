#!/usr/bin/env python3
"""Replicate-level metrics for Reviewer 4 Comment 4 (replaces block bootstrap).

Input: HiC-Pro iced sparse matrices (3 cols: bin_i, bin_j, value; 1-based;
5 kb; 883 bins; circular H37Rv 4,411,532 bp; ori=0, ter=2.03 Mb).
4 conditions x 4 replicates (no replicates for napM-KO-hypoxia).

Metrics per replicate (definitions identical to the merged-matrix version):
  - frac_short / frac_mid / frac_long: fraction of total contact probability
    (unique i<j pairs, value-weighted) with circular distance <100 kb,
    100 kb-1 Mb, >1 Mb.
  - sec_diag_OE: mean observed/expected over mirror-symmetric pairs
    (x vs L-x, +/-10 kb band) with circular distance >500 kb; expected =
    same-replicate P(s) at the pair's distance.
  - ori_ter_OE: mean contact over ori (0+/-50 kb, circular) x ter
    (2.03 Mb+/-50 kb) pairs / same-replicate P(s) at 2.03 Mb.
  - compaction_slope: log10-log10 slope of P(s) over 100 kb-1 Mb.
Also: within-condition pairwise Pearson correlation of log1p upper-triangle
values between replicates (reproducibility).

Outputs: tables/replicate_metrics.csv, tables/ps_curves_per_replicate.csv,
tables/replicate_correlations.csv, tables/stats_tests.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

DATA = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_data/rep_icedmatrix"
OUT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment4"
GENOME_LEN = 4_411_532
TER_POS = 2_030_000
BIN = 5_000
N_BINS = 883

CONDITIONS = {"WT": "wt", "napM-KO": "napm_ko", "hypoxia": "hypoxia",
              "latent": "latent"}
COND_ORDER = ["WT", "napM-KO", "hypoxia", "latent"]
REPS = [1, 2, 3, 4]
METRICS = ["frac_short_lt100kb", "frac_mid_100kb_1Mb", "frac_long_gt1Mb",
           "sec_diag_OE", "ori_ter_OE", "compaction_slope"]
COMPARISONS = [("napM-KO", "WT"), ("hypoxia", "WT"), ("latent", "WT")]

# bin midpoints (last bin shorter)
starts = np.arange(N_BINS) * BIN
ends = np.minimum(starts + BIN, GENOME_LEN)
mids = (starts + ends) / 2.0

# circular distance matrix
D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)
IU = np.triu_indices(N_BINS, k=1)
D_UT = D[IU]
DBIN_UT = np.clip((D_UT // BIN).astype(int), 0, 440)

# masks on unique pairs (computed once)
mirror_i = (GENOME_LEN - mids[IU[0]]) % GENOME_LEN
off = np.abs(mids[IU[1]] - mirror_i)
off = np.minimum(off, GENOME_LEN - off)
SEC_MASK = (off <= 10_000) & (D_UT > 500_000)

d_ori_i = np.minimum(mids[IU[0]], GENOME_LEN - mids[IU[0]])
d_ori_j = np.minimum(mids[IU[1]], GENOME_LEN - mids[IU[1]])
in_ori_i, in_ori_j = d_ori_i <= 50_000, d_ori_j <= 50_000
in_ter_i = np.abs(mids[IU[0]] - TER_POS) <= 50_000
in_ter_j = np.abs(mids[IU[1]] - TER_POS) <= 50_000
TER_MASK = (in_ori_i & in_ter_j) | (in_ter_i & in_ori_j)
TER_DBIN = int(TER_POS // BIN)

RANGE_IDX = np.digitize(D_UT, [100_000, 1_000_000])  # 0/1/2
SLOPE_BINS = np.arange(20, 200)                       # 100 kb..1 Mb
SLOPE_X = np.log10((SLOPE_BINS + 0.5) * BIN)

# log-spaced bins for P(s) curves (panel A)
PS_EDGES = np.unique(np.round(
    np.logspace(np.log10(5000), np.log10(GENOME_LEN / 2), 60)).astype(int))
PS_IDX_UT = np.clip(np.digitize(D_UT, PS_EDGES) - 1, 0, len(PS_EDGES) - 2)


def load_matrix(cond_key, rep):
    path = f"{DATA}/{cond_key}_{rep}_5000_iced.matrix"
    a = np.loadtxt(path)
    i = a[:, 0].astype(int) - 1
    j = a[:, 1].astype(int) - 1
    v = a[:, 2]
    M = np.zeros((N_BINS, N_BINS))
    M[i, j] = v
    M[j, i] = v
    # bins with zero total contact were filtered by ICE -> invalid (NaN),
    # mirroring the NaN-weight handling of the merged-matrix version
    bad = M.sum(axis=0) == 0
    M[bad, :] = np.nan
    M[:, bad] = np.nan
    return M, bad.sum()


def compute_metrics(M):
    v = M[IU]
    ok = np.isfinite(v)
    v, dbin, ridx = v[ok], DBIN_UT[ok], RANGE_IDX[ok]
    sec, ter = SEC_MASK[ok], TER_MASK[ok]

    tot = v.sum()
    fr = np.bincount(ridx, weights=v, minlength=3) / tot

    ps_sum = np.bincount(dbin, weights=v, minlength=441)
    ps_cnt = np.bincount(dbin, minlength=441)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = ps_sum / ps_cnt

    oe = v[sec] / ps[dbin[sec]]
    ok2 = np.isfinite(oe) & (ps[dbin[sec]] > 0)
    sec_oe = oe[ok2].mean()

    ori_ter_obs = v[ter].mean()
    ori_ter_oe = ori_ter_obs / ps[TER_DBIN]

    y = ps[SLOPE_BINS]
    ok3 = np.isfinite(y) & (y > 0)
    slope = np.polyfit(SLOPE_X[ok3], np.log10(y[ok3]), 1)[0]

    # raw (unnormalised) secondary-diagonal mean, >500 kb subset (Panel C)
    sec_raw = v[sec].mean()    # P(s) on log-spaced bins for plotting
    idx = PS_IDX_UT[ok]
    pl_sum = np.bincount(idx, weights=v, minlength=len(PS_EDGES) - 1)
    pl_cnt = np.bincount(idx, minlength=len(PS_EDGES) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps_log = pl_sum / pl_cnt

    return dict(frac_short_lt100kb=fr[0], frac_mid_100kb_1Mb=fr[1],
                frac_long_gt1Mb=fr[2], sec_diag_OE=sec_oe,
                ori_ter_OE=ori_ter_oe, compaction_slope=slope,
                sec_diag_raw=sec_raw, sec_diag_raw_per1M=sec_raw / tot * 1e6,
                total_contacts=tot,
                ps_log=ps_log)


# ---------- run ----------
rows, ps_rows, mats = [], [], {}
for cond in COND_ORDER:
    for rep in REPS:
        M, nbad = load_matrix(CONDITIONS[cond], rep)
        mats[(cond, rep)] = M
        r = compute_metrics(M)
        ps_log = r.pop("ps_log")
        rows.append(dict(condition=cond, replicate=rep, n_bad_bins=nbad, **r))
        for k, val in enumerate(ps_log):
            ps_rows.append(dict(condition=cond, replicate=rep,
                                dist_lo=PS_EDGES[k], dist_hi=PS_EDGES[k + 1],
                                ps=val))
        print(f"{cond} rep{rep}: done (bad bins: {nbad}, "
              f"total contacts: {r['total_contacts']:.0f})")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/replicate_metrics.csv", index=False)
pd.DataFrame(ps_rows).to_csv(f"{OUT}/tables/ps_curves_per_replicate.csv",
                             index=False)

# ---------- within-condition replicate correlations ----------
corr_rows = []
for cond in COND_ORDER:
    vecs = {}
    for rep in REPS:
        v = mats[(cond, rep)][IU]
        vecs[rep] = np.log1p(np.nan_to_num(v, nan=0.0))
    for a in range(len(REPS)):
        for b in range(a + 1, len(REPS)):
            ra, rb = REPS[a], REPS[b]
            ok = np.isfinite(mats[(cond, ra)][IU]) & np.isfinite(mats[(cond, rb)][IU])
            r, _ = stats.pearsonr(vecs[ra][ok], vecs[rb][ok])
            corr_rows.append(dict(condition=cond, rep_a=ra, rep_b=rb,
                                  pearson_r=r))
df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(f"{OUT}/tables/replicate_correlations.csv", index=False)
print("\nWithin-condition replicate correlations (log1p, upper triangle):")
print(df_corr.groupby("condition")["pearson_r"]
      .agg(["min", "mean", "max"]).round(4))

# ---------- full 16x16 correlation matrix (Panel D) ----------
samples = [(c, r) for c in COND_ORDER for r in REPS]
C = np.full((16, 16), np.nan)
logv = {s: np.log1p(np.nan_to_num(mats[s][IU], nan=0.0)) for s in samples}
for ai, sa in enumerate(samples):
    for bi, sb in enumerate(samples):
        ok = np.isfinite(mats[sa][IU]) & np.isfinite(mats[sb][IU])
        C[ai, bi] = stats.pearsonr(logv[sa][ok], logv[sb][ok])[0]
np.save(f"{OUT}/tables/correlation_matrix_16x16.npy", C)
pd.DataFrame(C, index=[f"{c}_{r}" for c, r in samples],
             columns=[f"{c}_{r}" for c, r in samples]
             ).to_csv(f"{OUT}/tables/correlation_matrix_16x16.csv")

# ---------- replicate-level statistics ----------
stat_rows = []
for cond, ref in COMPARISONS:
    for m in METRICS:
        x = df.loc[df.condition == cond, m].values
        y = df.loc[df.condition == ref, m].values
        t, p_t = stats.ttest_ind(x, y, equal_var=False)
        u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
        stat_rows.append(dict(
            comparison=f"{cond} vs {ref}", metric=m,
            mean_cond=x.mean(), sd_cond=x.std(ddof=1),
            mean_ref=y.mean(), sd_ref=y.std(ddof=1),
            diff=x.mean() - y.mean(),
            welch_t=t, welch_p=p_t, mw_U=u, mw_p=p_u))
df_stat = pd.DataFrame(stat_rows)
df_stat.to_csv(f"{OUT}/tables/stats_tests.csv", index=False)

# ---------- summary table (condition x metric: mean ± SD) ----------
summ = {}
for cond in COND_ORDER:
    sub = df[df.condition == cond]
    summ[cond] = {m: f"{sub[m].mean():.4f} ± {sub[m].std(ddof=1):.4f}"
                  for m in METRICS + ["sec_diag_raw", "sec_diag_raw_per1M",
                                      "total_contacts"]}
pd.DataFrame(summ).T.to_csv(f"{OUT}/tables/summary_mean_sd.csv")

print("\n== replicate-level stats ==")
print(df_stat[["comparison", "metric", "mean_ref", "mean_cond", "diff",
               "welch_p", "mw_p"]].to_string(index=False,
             float_format=lambda x: f"{x:.4g}"))
print("\nSaved tables to", f"{OUT}/tables/")
