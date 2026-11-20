#!/usr/bin/env python3
"""
s02b_interaction_update.py — re-do the Hi-C genotype x condition interaction
analysis now that KO-hypoxia biological replicates exist (n=3; WT/KO/hypoxia
n=4 each; slightly unbalanced design, noted in all outputs).

Per-replicate endpoints (identical definitions to s01_build_tables.py):
  * circular diamond log2 insulation score, 25 kb window (w=5 bins, primary)
  * long-range contact fraction (>100 kb circular distance)

Formal per-bin interaction test (Welch, unbalanced):
  interaction_i = mean(KO-hyp) - mean(KO) - (mean(hyp) - mean(WT))
  SE from per-group replicate variances; Welch-Satterthwaite df;
  two-sided t-test; BH-FDR across 883 bins.
Three-way classification with pre-registered equivalence bound 0.25 log2:
  detected increase / detected decrease / supports small change /
  cannot determine (90% CI rule, as in s01).

Updates bin_level_integrated.csv columns:
  ins_KO_hypoxia_rep1..3, lrfrac_KO_hypoxia_rep1..3, ins_KO_hypoxia_repmean,
  d_ins_hyp_KO{,_lo,_hi}        (now replicate-based Welch estimates)
  d_ins_interaction{,_lo,_hi}   (now full Welch CI on BOTH contrasts)
  interaction_p, interaction_padj, interaction_class
  d_lrfrac_interaction{,_lo,_hi} (long-range fraction interaction)
and rewrites the stats_fig3_* tables with replicate counts noted.

Random seed 42. Environment: .venv_cooltools (numpy 1.26.4, pandas 2.2.2,
scipy 1.13.1).
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{ROOT}/review_results/R1_mechanism_zhenghe"
TAB = f"{OUT}/tables"
N_BINS = 883
EQ_INS = 0.25
N_PERM = 1000

REP_FILES = {
    "WT": [f"wt_rep_icedmatrix/wt_{i}_5000_iced.matrix" for i in range(1, 5)],
    "KO": [f"wt_rep_icedmatrix/napm_ko_{i}_5000_iced.matrix" for i in range(1, 5)],
    "hypoxia": [f"wt_rep_icedmatrix/hypoxia_{i}_5000_iced.matrix" for i in range(1, 5)],
    "KO_hypoxia": [f"wt_rep_icedmatrix/ko-hypoxia_{i}_5000_iced.matrix"
                   for i in range(1, 4)],          # NEW: 3 replicates
}
N_REPS = {k: len(v) for k, v in REP_FILES.items()}


def load_dense_iced(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["b1", "b2", "v"])
    mat = np.zeros((N_BINS, N_BINS))
    i = df.b1.values - 1
    j = df.b2.values - 1
    mat[i, j] = df.v.values
    mat[j, i] = df.v.values
    return mat


def circ_diamond_insulation(mat, w=5):
    n = mat.shape[0]
    score = np.full(n, np.nan)
    for i in range(n):
        up = [(i - w + 1 + k) % n for k in range(w)]
        dn = [(i + 1 + k) % n for k in range(w)]
        sub = mat[np.ix_(up, dn)]
        if np.isfinite(sub).sum() >= 0.5 * sub.size:
            score[i] = np.nanmean(sub)
    return np.log2(score / np.nanmedian(score))


def long_range_fraction(mat, dist_bins=20):
    n = mat.shape[0]
    idx = np.arange(n)
    D = np.abs(idx[:, None] - idx[None, :])
    D = np.minimum(D, n - D)
    lr = np.nansum(np.where(D > dist_bins, mat, 0.0), axis=1)
    tot = np.nansum(np.where(np.isfinite(mat), mat, 0.0), axis=1)
    return np.where(tot > 0, lr / tot, np.nan)


def bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    n = len(idx)
    order = idx[np.argsort(p[idx])]
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def classify_3way(est, lo, hi, bound=EQ_INS):
    if not np.isfinite(est):
        return "cannot_determine"
    if lo > bound:
        return "detected_increase"
    if hi < -bound:
        return "detected_decrease"
    if lo >= -bound and hi <= bound:
        return "supports_small_change"
    return "cannot_determine"


def welch_contrast(groups, signs):
    """Contrast sum(signs[g] * mean(g)) with Welch SE / per-bin df / p."""
    est, var, df_num, df_den = 0.0, 0.0, 0.0, 0.0
    for s, arr in zip(signs, groups):
        n = np.isfinite(arr).sum(axis=0)              # per-bin finite n
        m = np.nanmean(arr, axis=0)
        v = np.nanvar(arr, axis=0, ddof=1)
        est = est + s * m
        var = var + v / n
        df_num += v / n
        df_den += np.where(n > 1, (v / n) ** 2 / (n - 1), 0.0)
    se = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = est / se
        df = np.where(df_den > 0, df_num ** 2 / df_den, np.nan)
        df = np.where(df >= 1, df, np.nan)
        p = 2 * stats.t.sf(np.abs(t), df)
    bad = ~np.isfinite(est) | ~np.isfinite(se) | (se == 0)
    p[bad] = np.nan
    return est, se, p


def sanitize(a):
    """Non-finite insulation (zero-coverage diamond -> -inf) becomes NaN."""
    return np.where(np.isfinite(a), a, np.nan)


def pair_ci(a_reps, b_reps):
    """all pairwise differences; 5-95 percentile as 90% CI (NaN-safe)"""
    ds = np.array([[b - a for b in b_reps] for a in a_reps])
    return (np.nanmedian(ds, axis=(0, 1)),
            np.nanpercentile(ds, 5, axis=(0, 1)),
            np.nanpercentile(ds, 95, axis=(0, 1)))


def main():
    # ---- per-replicate endpoints (non-finite -> NaN: zero-coverage bins)
    ins_rep, lr_rep = {}, {}
    for cond, files in REP_FILES.items():
        ims = [load_dense_iced(f"{ROOT}/review_data/{f}") for f in files]
        ins_rep[cond] = sanitize(np.array([circ_diamond_insulation(m)
                                           for m in ims]))
        lr_rep[cond] = np.array([long_range_fraction(m) for m in ims])
        # QC: replicate reproducibility (finite bins only)
        cc = []
        for a in range(N_REPS[cond]):
            for bb in range(a + 1, N_REPS[cond]):
                x, y = ins_rep[cond][a], ins_rep[cond][bb]
                ok = np.isfinite(x) & np.isfinite(y)
                cc.append(np.corrcoef(x[ok], y[ok])[0, 1])
        print(f"{cond}: n={N_REPS[cond]}, rep-rep insulation r = "
              f"{np.round(cc, 3).tolist()}, "
              f"bins NaN in any rep: {int((~np.isfinite(ins_rep[cond]).all(axis=0)).sum())}")

    b = pd.read_csv(f"{TAB}/bin_level_integrated.csv")

    # ---- recompute the replicate-based baseline contrasts with sanitized
    # values (s01 stored -inf-contaminated pair stats at 2 zero-coverage bins)
    pt, lo, hi = pair_ci(ins_rep["WT"], ins_rep["KO"])
    b["d_ins_KO_WT"], b["d_ins_KO_WT_lo"], b["d_ins_KO_WT_hi"] = pt, lo, hi
    b["ins_change_class"] = [classify_3way(e, l, h)
                             for e, l, h in zip(pt, lo, hi)]
    pt, lo, hi = pair_ci(ins_rep["WT"], ins_rep["hypoxia"])
    b["d_ins_hyp_WT"], b["d_ins_hyp_WT_lo"], b["d_ins_hyp_WT_hi"] = pt, lo, hi

    for r in range(3):
        b[f"ins_KO_hypoxia_rep{r+1}"] = ins_rep["KO_hypoxia"][r]
        b[f"lrfrac_KO_hypoxia_rep{r+1}"] = lr_rep["KO_hypoxia"][r]
    b["ins_KO_hypoxia_repmean"] = np.nanmean(ins_rep["KO_hypoxia"], axis=0)

    # ---- KO hypoxia response (KO-hyp - KO), Welch
    est_kr, se_kr, p_kr = welch_contrast([ins_rep["KO_hypoxia"], ins_rep["KO"]],
                                         [1, -1])
    b["d_ins_hyp_KO"] = est_kr
    b["d_ins_hyp_KO_lo"] = est_kr - 1.645 * se_kr   # 90% CI (for 3-way calls)
    b["d_ins_hyp_KO_hi"] = est_kr + 1.645 * se_kr
    b["d_ins_hyp_KO_p"] = p_kr

    # ---- formal interaction: (KO-hyp - KO) - (hyp - WT)
    est_i, se_i, p_i = welch_contrast(
        [ins_rep["KO_hypoxia"], ins_rep["KO"], ins_rep["hypoxia"],
         ins_rep["WT"]], [1, -1, -1, 1])
    padj_i = bh_fdr(p_i)
    b["d_ins_interaction"] = est_i
    b["d_ins_interaction_lo"] = est_i - 1.645 * se_i
    b["d_ins_interaction_hi"] = est_i + 1.645 * se_i
    b["interaction_p"] = p_i
    b["interaction_padj"] = padj_i
    b["interaction_class"] = [classify_3way(e, l, h)
                              for e, l, h in zip(est_i, est_i - 1.645 * se_i,
                                                 est_i + 1.645 * se_i)]

    # ---- long-range fraction interaction (secondary endpoint)
    est_lr, se_lr, p_lr = welch_contrast(
        [lr_rep["KO_hypoxia"], lr_rep["KO"], lr_rep["hypoxia"], lr_rep["WT"]],
        [1, -1, -1, 1])
    b["d_lrfrac_interaction"] = est_lr
    b["d_lrfrac_interaction_lo"] = est_lr - 1.645 * se_lr
    b["d_lrfrac_interaction_hi"] = est_lr + 1.645 * se_lr

    b.to_csv(f"{TAB}/bin_level_integrated.csv", index=False)

    # ---- per-bin interaction test table
    itab = pd.DataFrame(dict(bin=np.arange(N_BINS), interaction=est_i,
                             se=se_i, p=p_i, padj=padj_i,
                             cls=b["interaction_class"]))
    itab.to_csv(f"{TAB}/stats_fig3_interaction_test_perbin.csv", index=False)
    print("\ninteraction classes:", b["interaction_class"].value_counts().to_dict())
    print("bins with interaction padj<0.05:", int((padj_i < 0.05).sum()))
    print("median |interaction|:", round(float(np.nanmedian(np.abs(est_i))), 4),
          "| 95% of |interaction| <", round(float(np.nanpercentile(np.abs(est_i), 95)), 4))

    # ---- bound vs matched summaries (replicate-based values)
    bound = b.chip_bound.values.astype(bool)
    matched = b.matched_background.values.astype(bool)
    rows = []
    for lab, mask in [("NapM_bound", bound), ("matched_background", matched)]:
        for col, tag in [("d_ins_hyp_WT", "WT hypoxia response"),
                         ("d_ins_hyp_KO", "KO hypoxia response"),
                         ("d_ins_interaction", "interaction")]:
            v = b[col].values[mask]
            v = v[np.isfinite(v)]
            bs = RNG.choice(v, size=(2000, len(v)), replace=True)
            lo, hi = np.percentile(np.median(bs, axis=1), [2.5, 97.5])
            rows.append(dict(set=lab, metric=tag, n_bins=len(v),
                             median=float(np.median(v)),
                             ci_lo=float(lo), ci_hi=float(hi)))
    cetab = pd.DataFrame(rows)
    cetab["replicates"] = "WT=4, KO=4, hypoxia=4, KO-hypoxia=3"
    cetab.to_csv(f"{TAB}/stats_fig3_condition_effects_by_occupancy.csv",
                 index=False)

    # ---- circular-shift permutation on the replicate-based interaction
    iv = est_i
    obs = np.nanmedian(iv[bound]) - np.nanmedian(iv[matched])
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        s = RNG.integers(1, N_BINS)
        null[k] = np.nanmedian(iv[np.roll(bound, s)]) - \
            np.nanmedian(iv[np.roll(matched, s)])
    p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    np.save(f"{TAB}/fig3_null_interaction.npy", null)
    pd.DataFrame([dict(obs=obs, p_circular_shift=p_perm,
                       null_mean=float(null.mean()), null_sd=float(null.std()),
                       replicates="WT=4, KO=4, hypoxia=4, KO-hypoxia=3")]).to_csv(
        f"{TAB}/stats_fig3_interaction_permutation.csv", index=False)
    print(f"\nbound vs matched interaction: obs={obs:+.4f}, "
          f"circular-shift p={p_perm:.3f}")

    # ---- latent descriptive unchanged; rewrite summary json entry
    summ = json.load(open(f"{TAB}/stats_summary.json"))
    summ["fig3"] = dict(
        replicates=N_REPS,
        interaction_per_bin=dict(
            n_fdr05=int((padj_i < 0.05).sum()),
            median_abs=float(np.nanmedian(np.abs(est_i))),
            p95_abs=float(np.nanpercentile(np.abs(est_i), 95))),
        interaction_perm_p=float(p_perm),
        rna_interaction_fdr05=0)
    json.dump(summ, open(f"{TAB}/stats_summary.json", "w"), indent=2)

    # ---- the audit premise changed: update unanalyzable table
    un = pd.read_csv(f"{TAB}/unanalyzable_items.csv")
    un = un[un["item"] != "KO-hypoxia Hi-C replicates"]
    un.loc[len(un)] = dict(
        item="KO-hypoxia Hi-C replicates", n=1,
        reason="only 3 replicates (vs 4 in other conditions); design slightly "
               "unbalanced, Welch tests used")
    un.to_csv(f"{TAB}/unanalyzable_items.csv", index=False)
    print("\ntables updated")


if __name__ == "__main__":
    main()
