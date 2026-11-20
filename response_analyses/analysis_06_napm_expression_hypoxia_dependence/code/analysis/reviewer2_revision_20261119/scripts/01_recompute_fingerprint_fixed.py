#!/usr/bin/env python3
"""
Recompute the 9-metric multiscale chromosome response fingerprint with:
  1. Consistent genome_mean_insulation matching Panel C (E_independent_sample_genome_mean_insulation.csv),
     fixing the Jensen-inequality sign-flip bug caused by mean(log2(win / mean(win))).
  2. Consistent circular distance cutoffs matching main.tex Methods and R4 compaction metrics:
     - Short-range fraction: <100 kb
     - Intermediate fraction: 100 kb-1 Mb
     - Long-range fraction: >1 Mb
     - P(s) decay slope: 100 kb-1 Mb (bins 20..199)
  3. Replicate-level z-score response vectors, bootstrap 95% CI, and exact/10,000-round
     within-genotype oxygen-label permutation tests for Euclidean distance and cosine similarity.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 10000
N_PERM = 10000

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
HIC = ROOT / "hic_review"
RA = HIC / "review_results/R1_mechanism_reanalysis"
MAT = HIC / "review_data/rep_icedmatrix"
FINAL_SD = RA / "09_combined/R1_review_delivery_20261027/final_response_figure_20261027/source_data"
OUT = ROOT / "analysis/reviewer2_revision_20261119/tables"
OUT.mkdir(parents=True, exist_ok=True)

GENOME_LEN = 4_411_532
BIN = 5_000
N_BINS = 883
INS_W = 10

loops = pd.read_csv(RA / "02_loop_curated/loops_41_curated.tsv", sep="\t")
cid = pd.read_csv(RA / "02_loop_curated/wt_cid_boundaries_curated.tsv", sep="\t")
bounds = np.sort(cid["bin"].values)
assert len(bounds) == 18

# True independent-sample genome-mean insulation matching Panel C
ins_panel_c = pd.read_csv(FINAL_SD / "E_independent_sample_genome_mean_insulation.csv")
ins_map = {
    (row["condition"], int(row["replicate"])): float(row["genome_mean_insulation"])
    for _, row in ins_panel_c.iterrows()
}

# Shared circular distance geometry (exact match to R4 / main.tex Methods)
starts = np.arange(N_BINS) * BIN
mids = (starts + np.minimum(starts + BIN, GENOME_LEN)) / 2.0
D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)
IU = np.triu_indices(N_BINS, k=1)
D_UT = D[IU]
RANGE_IDX = np.digitize(D_UT, [100_000, 1_000_000])  # 0: <100kb, 1: 100kb-1Mb, 2: >1Mb
SLOPE_BINS = np.arange(20, 200)                      # 100 kb..1 Mb
SLOPE_X = np.log10((SLOPE_BINS + 0.5) * BIN)

# Domain assignment per bin
DOM = np.empty(N_BINS, dtype=int)
for k in range(len(bounds)):
    a = bounds[k]
    b = bounds[(k + 1) % len(bounds)]
    if b > a:
        DOM[a:b] = k
    else:
        DOM[a:] = k
        DOM[:b] = k
SAME_DOM_UT = DOM[IU[0]] == DOM[IU[1]]


def load_matrix(path: Path) -> np.ndarray:
    d = np.loadtxt(path)
    i = d[:, 0].astype(int) - 1
    j = d[:, 1].astype(int) - 1
    v = d[:, 2]
    m = np.zeros((N_BINS, N_BINS))
    m[i, j] = v
    m = m + m.T - np.diag(np.diag(m))
    return m


def circ_diag_sums(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    out = np.empty(n // 2 + 1)
    out[0] = np.trace(m)
    for d in range(1, n // 2 + 1):
        out[d] = np.sum(m[np.arange(n), (np.arange(n) + d) % n])
    return out


def local_insulation_profile(m: np.ndarray, w: int = INS_W) -> np.ndarray:
    n = m.shape[0]
    idx = np.arange(-w + 1, w + 1)
    win = np.empty(n)
    for i in range(n):
        ii = (i + idx) % n
        win[i] = m[np.ix_(ii, ii)].sum()
    mu = win.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(win / mu)


SAMPLES = (
    [("wt", "WT", i) for i in range(1, 5)]
    + [("napm_ko", "KO", i) for i in range(1, 5)]
    + [("hypoxia", "HYP", i) for i in range(1, 5)]
    + [("ko-hypoxia", "KO-HYP", i) for i in range(1, 4)]
)

rows = []
for fname, cond, rep in SAMPLES:
    sample = f"{fname}_{rep}"
    m = load_matrix(MAT / f"{sample}_5000_iced.matrix")

    # P(s) circular diagonal sums
    ds = circ_diag_sums(m)
    dmax = N_BINS // 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = ds[1:] / N_BINS
    total_diag = ds[1:].sum()

    # 1. P(s) slope over 100 kb - 1 Mb (bins 20..199, matching main.tex Methods)
    ps_slope_vals = ps[SLOPE_BINS - 1]
    sel = np.isfinite(ps_slope_vals) & (ps_slope_vals > 0)
    slope = float(np.polyfit(SLOPE_X[sel], np.log10(ps_slope_vals[sel]), 1)[0])

    # 2-4. Distance fractions (<100 kb, 100 kb-1 Mb, >1 Mb; matching main.tex Methods)
    vals_ut = m[IU]
    tot_ut = vals_ut.sum()
    short_frac = float(vals_ut[RANGE_IDX == 0].sum() / tot_ut)
    inter_frac = float(vals_ut[RANGE_IDX == 1].sum() / tot_ut)
    long_frac = float(vals_ut[RANGE_IDX == 2].sum() / tot_ut)

    # 5. True genome-mean insulation (exact match to Panel C!)
    ins_mean = ins_map[(cond, rep)]

    # 6. CID boundary depth from local normalized insulation profile
    ins_prof = local_insulation_profile(m)
    flank = []
    for b in bounds:
        fi = np.concatenate([[(b - k) % N_BINS for k in range(1, 4)],
                             [(b + k) % N_BINS for k in range(1, 4)]])
        flank.append(np.nanmean(ins_prof[fi]) - ins_prof[b])
    boundary_depth = float(np.nanmean(flank))

    # 7. Loop O/E (41 curated loops)
    oes = []
    for _, r in loops.iterrows():
        a, b = int(r["bin1"]), int(r["bin2"])
        dd = min(abs(a - b), N_BINS - abs(a - b))
        e = ps[dd - 1] if dd >= 1 else np.nan
        o = m[a, b]
        if e and np.isfinite(e) and e > 0 and o > 0:
            oes.append(o / e)
        else:
            oes.append(np.nan)
    loop_oe = float(np.nanmean(oes))

    # 8. CID intra-domain contact fraction
    cid_intra_frac = float(vals_ut[SAME_DOM_UT].sum() / tot_ut)

    # 9. Distance entropy (bits)
    p = ds[1:] / total_diag
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    rows.append(dict(
        sample=sample, condition=cond, replicate=rep,
        ps_slope=slope, short_frac=short_frac, inter_frac=inter_frac,
        long_frac=long_frac, ins_mean=ins_mean, boundary_depth=boundary_depth,
        loop_oe=loop_oe, cid_intra_frac=cid_intra_frac, entropy=entropy,
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT / "fingerprint_15samples_fixed.csv", index=False)

METRICS = [
    "ps_slope", "short_frac", "inter_frac", "long_frac",
    "ins_mean", "boundary_depth", "loop_oe", "cid_intra_frac", "entropy",
]
LABELS = {
    "ps_slope": "P(s) slope (100 kb–1 Mb)",
    "short_frac": "Short-range (<100 kb)",
    "inter_frac": "Intermediate (100 kb–1 Mb)",
    "long_frac": "Long-range (>1 Mb)",
    "ins_mean": "Mean insulation",
    "boundary_depth": "CID boundary depth",
    "loop_oe": "Loop O/E (41 loops)",
    "cid_intra_frac": "CID intra-domain frac.",
    "entropy": "Distance entropy",
}

X = df[METRICS].to_numpy(dtype=float)
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
cond = df["condition"].to_numpy()


def resp_vectors(Z_mat, c_arr):
    rw = Z_mat[c_arr == "HYP"].mean(axis=0) - Z_mat[c_arr == "WT"].mean(axis=0)
    rk = Z_mat[c_arr == "KO-HYP"].mean(axis=0) - Z_mat[c_arr == "KO"].mean(axis=0)
    return rw, rk


def calc_stats(rw, rk):
    cos = float(rw @ rk / (np.linalg.norm(rw) * np.linalg.norm(rk)))
    ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    euc = float(np.linalg.norm(rw - rk))
    pea = float(np.corrcoef(rw, rk)[0, 1])
    return cos, ang, euc, pea


R_WT, R_KO = resp_vectors(Z, cond)
cos_o, ang_o, euc_o, pea_o = calc_stats(R_WT, R_KO)

# Replicate bootstrap (10,000)
idx_by_cond = {c: np.where(cond == c)[0] for c in ["WT", "HYP", "KO", "KO-HYP"]}
boot_cos = np.empty(N_BOOT)
boot_euc = np.empty(N_BOOT)
for b in range(N_BOOT):
    sel = np.concatenate([RNG.choice(v, size=len(v), replace=True) for v in idx_by_cond.values()])
    rw, rk = resp_vectors(Z[sel], cond[sel])
    boot_cos[b], _, boot_euc[b], _ = calc_stats(rw, rk)

# Exact within-genotype oxygen-label enumeration!
# WT block has 8 samples (choose 4 as HYP -> C(8,4) = 70)
# KO block has 7 samples (choose 3 as KO-HYP -> C(7,3) = 35)
# Total exact permutations = 70 * 35 = 2,450! Exact test is even better than Monte Carlo!
from itertools import combinations
wt_idx = np.where((cond == "WT") | (cond == "HYP"))[0]
ko_idx = np.where((cond == "KO") | (cond == "KO-HYP"))[0]
wt_set = set(wt_idx)
ko_set = set(ko_idx)

exact_cos = []
exact_euc = []
for hyp_sub in combinations(wt_idx, 4):
    hyp_arr = np.array(hyp_sub)
    wt_arr = np.array(sorted(wt_set - set(hyp_sub)))
    rw_p = Z[hyp_arr].mean(axis=0) - Z[wt_arr].mean(axis=0)
    for kohyp_sub in combinations(ko_idx, 3):
        kohyp_arr = np.array(kohyp_sub)
        ko_arr = np.array(sorted(ko_set - set(kohyp_sub)))
        rk_p = Z[kohyp_arr].mean(axis=0) - Z[ko_arr].mean(axis=0)
        c_p, _, e_p, _ = calc_stats(rw_p, rk_p)
        exact_cos.append(c_p)
        exact_euc.append(e_p)

exact_cos = np.array(exact_cos)
exact_euc = np.array(exact_euc)
p_euc_exact = float(np.mean(exact_euc >= euc_o - 1e-12))
p_cos_exact = float(np.mean(exact_cos <= cos_o + 1e-12))

contrib = pd.DataFrame({
    "metric": METRICS,
    "metric_label": [LABELS[m] for m in METRICS],
    "R_WT": R_WT,
    "R_KO": R_KO,
    "abs_diff": np.abs(R_WT - R_KO),
})
contrib.to_csv(OUT / "fingerprint_response_vectors_fixed.csv", index=False)

stats_df = pd.DataFrame([
    {"statistic": "euclidean_distance", "observed": euc_o, "ci_lo": np.percentile(boot_euc, 2.5),
     "ci_hi": np.percentile(boot_euc, 97.5), "exact_perm_P": p_euc_exact, "n_exact_perms": len(exact_euc)},
    {"statistic": "cosine_similarity", "observed": cos_o, "ci_lo": np.percentile(boot_cos, 2.5),
     "ci_hi": np.percentile(boot_cos, 97.5), "exact_perm_P": p_cos_exact, "n_exact_perms": len(exact_cos)},
    {"statistic": "angle_deg", "observed": ang_o, "ci_lo": np.nan, "ci_hi": np.nan,
     "exact_perm_P": p_cos_exact, "n_exact_perms": len(exact_cos)},
    {"statistic": "pearson_r", "observed": pea_o, "ci_lo": np.nan, "ci_hi": np.nan,
     "exact_perm_P": np.nan, "n_exact_perms": len(exact_cos)},
])
stats_df.to_csv(OUT / "fingerprint_response_stats_fixed.csv", index=False)

print("=== Per-condition mean raw metrics ===")
print(df.groupby("condition")[METRICS].mean().to_string())
print("\n=== Response Vectors (z-scored over 15 samples) ===")
print(contrib.to_string(index=False))
print("\n=== Global Tests (Exact 2,450 within-genotype permutations) ===")
print(stats_df.to_string(index=False))
