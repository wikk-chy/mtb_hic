#!/usr/bin/env python3
"""R4C5 polymer_openmm - experimental reference metrics.

Computes the replicate-level Hi-C metrics that the polymer model is fitted
to / validated against. Definitions are copied verbatim from
R4_comment4/scripts/01_replicate_metrics.py (circular short-arc distance,
ICE-balanced 5 kb matrices, ori = 0 +/- 50 kb circular, ter = 2.03 Mb
+/- 50 kb, arm-arm = mirror pairs (x, L-x) +/- 10 kb with circular
distance > 500 kb, slope = log-log P(s) over 100 kb-1 Mb).

Inputs : hic_review/review_data/rep_icedmatrix/{wt,hypoxia,latent}_N_5000_iced.matrix
Outputs:
  tables/experiment_metrics.csv      one row per condition x replicate
  tables/experiment_summary.csv      condition x metric mean +/- SD
  tables/experiment_ps_curves.csv    per-replicate P(s), linear 5 kb bins
                                     (used as the fitting target; fitting
                                     uses only WT / latent P(s))
"""
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
DATA = f"{BASE}/review_data/rep_icedmatrix"
OUT = f"{BASE}/review_results/R4_comment5/polymer_openmm"
GENOME_LEN = 4_411_532
TER_POS = 2_030_000
BIN = 5_000
N_BINS = 883

CONDITIONS = {"WT": "wt", "hypoxia": "hypoxia", "latent": "latent"}
REPS = [1, 2, 3, 4]

starts = np.arange(N_BINS) * BIN
ends = np.minimum(starts + BIN, GENOME_LEN)
mids = (starts + ends) / 2.0

D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)              # circular short-arc distance
IU = np.triu_indices(N_BINS, k=1)
D_UT = D[IU]
DBIN_UT = np.clip((D_UT // BIN).astype(int), 0, 440)

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

RANGE_IDX = np.digitize(D_UT, [100_000, 1_000_000])
SLOPE_BINS = np.arange(20, 200)                # 100 kb .. 1 Mb
SLOPE_X = np.log10((SLOPE_BINS + 0.5) * BIN)


def load_matrix(cond_key, rep):
    a = np.loadtxt(f"{DATA}/{cond_key}_{rep}_5000_iced.matrix")
    i, j, v = a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1, a[:, 2]
    M = np.zeros((N_BINS, N_BINS))
    M[i, j] = v
    M[j, i] = v
    bad = M.sum(axis=0) == 0
    M[bad, :] = np.nan
    M[:, bad] = np.nan
    return M


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
    ps_prob = ps_sum / tot                     # probability-normalised P(s)

    oe = v[sec] / ps[dbin[sec]]
    ok2 = np.isfinite(oe) & (ps[dbin[sec]] > 0)
    sec_oe = oe[ok2].mean()

    ori_ter_oe = v[ter].mean() / ps[TER_DBIN]

    y = ps[SLOPE_BINS]
    ok3 = np.isfinite(y) & (y > 0)
    slope = np.polyfit(SLOPE_X[ok3], np.log10(y[ok3]), 1)[0]

    return dict(frac_short_lt100kb=fr[0], frac_mid_100kb_1Mb=fr[1],
                frac_long_gt1Mb=fr[2], sec_diag_OE=sec_oe,
                ori_ter_OE=ori_ter_oe, compaction_slope=slope,
                ps_prob=ps_prob)


rows, ps_rows = [], []
for cond, key in CONDITIONS.items():
    for rep in REPS:
        M = load_matrix(key, rep)
        r = compute_metrics(M)
        ps_prob = r.pop("ps_prob")
        rows.append(dict(condition=cond, replicate=rep, **r))
        for b in range(1, 441):
            ps_rows.append(dict(condition=cond, replicate=rep,
                                dist_bin=b, dist_bp=(b + 0.5) * BIN,
                                ps=ps_prob[b]))
        print(f"{cond} rep{rep} done")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/experiment_metrics.csv", index=False)
pd.DataFrame(ps_rows).to_csv(f"{OUT}/tables/experiment_ps_curves.csv",
                             index=False)

METRICS = ["frac_short_lt100kb", "frac_mid_100kb_1Mb", "frac_long_gt1Mb",
           "sec_diag_OE", "ori_ter_OE", "compaction_slope"]
summ = {}
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    summ[cond] = {m: f"{sub[m].mean():.6f} ± {sub[m].std(ddof=1):.6f}"
                  for m in METRICS}
    for m in METRICS:
        summ[cond + "__" + m + "_mean"] = None  # placeholder removed below
summ_df = df.groupby("condition")[METRICS].agg(["mean", "std"])
summ_df.to_csv(f"{OUT}/tables/experiment_summary.csv")
print("\n== condition x metric (mean ± SD, n=4) ==")
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    print(cond, {m: f"{sub[m].mean():.4f}±{sub[m].std(ddof=1):.4f}"
                 for m in METRICS})

# latent/WT and hypoxia/WT folds for quick reference
for cond in ["latent", "hypoxia"]:
    print(f"\n{cond}/WT folds:",
          {m: f"{df.loc[df.condition == cond, m].mean() / df.loc[df.condition == 'WT', m].mean():.3f}"
           for m in METRICS})
print("\nSaved tables/experiment_metrics.csv, experiment_summary.csv, "
      "experiment_ps_curves.csv")
