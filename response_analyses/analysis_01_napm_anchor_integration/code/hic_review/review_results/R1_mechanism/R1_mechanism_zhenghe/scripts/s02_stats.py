#!/usr/bin/env python3
"""
s02_stats.py — all pre-registered statistical comparisons for the three main
figures. Reads tables produced by s01_build_tables.py, writes stats tables and
a machine-readable summary to tables/.

Random seed: 42. Permutations: 1000 restricted circular shifts (preserve the
genome's circular structure and the spatial clustering of occupancy labels).
Genome resampling is used only for null calibration, never as additional
biological replicates.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{ROOT}/review_results/R1_mechanism_zhenghe"
TAB = f"{OUT}/tables"
N_PERM = 1000
N_BINS = 883

b = pd.read_csv(f"{TAB}/bin_level_integrated.csv")
g = pd.read_csv(f"{TAB}/gene_level_integrated.csv")
inter = pd.read_csv(f"{TAB}/rna_interaction_model.csv")
summary = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def circ_shift_null(labels, stat_func, n_perm=N_PERM):
    """Null distribution of stat_func(shifted_labels) under circular shifts."""
    obs = stat_func(labels)
    null = np.empty(n_perm)
    for k in range(n_perm):
        s = RNG.integers(1, N_BINS)
        null[k] = stat_func(np.roll(labels, s))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, null, p


# ============================================================ S1: Fig 1 stats
bound = b.chip_bound.values
matched = b.matched_background.values
dins = b.d_ins_KO_WT.values

# paired comparison (matched pairs)
pairs = b.loc[b.chip_bound, ["bin", "matched_to_bin"]].dropna()
pb = b.set_index("bin")
diff = (pb.loc[pairs.bin, "d_ins_KO_WT"].values -
        pb.loc[pairs.matched_to_bin.astype(int), "d_ins_KO_WT"].values)
wil = stats.wilcoxon(diff)
boot = RNG.choice(diff, size=(10000, len(diff))).mean(axis=1)
s1_paired = dict(n_pairs=len(diff), median_paired_diff=float(np.median(diff)),
                 mean_paired_diff=float(diff.mean()),
                 ci_lo=float(np.percentile(boot, 2.5)),
                 ci_hi=float(np.percentile(boot, 97.5)),
                 wilcoxon_p=float(wil.pvalue))
pd.DataFrame([s1_paired]).to_csv(f"{TAB}/stats_fig1_paired_bound_vs_matched.csv",
                                 index=False)

# circular-shift permutation on median(bound) - median(matched)
def stat_medshift(lab_b, lab_m):
    return np.nanmedian(dins[lab_b]) - np.nanmedian(dins[lab_m])

obs = stat_medshift(bound, matched)
null = np.empty(N_PERM)
for k in range(N_PERM):
    s = RNG.integers(1, N_BINS)
    null[k] = stat_medshift(np.roll(bound, s), np.roll(matched, s))
p_shift = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
s1_perm = dict(obs=obs, p_circular_shift=p_shift,
               null_mean=float(null.mean()), null_sd=float(null.std()))
np.save(f"{TAB}/fig1_null_shift.npy", null)
pd.DataFrame([s1_perm]).to_csv(f"{TAB}/stats_fig1_permutation.csv", index=False)

# class proportions with Wilson CIs
rows = []
for lab, name in [(bound, "NapM_bound"), (matched, "matched_background")]:
    sub = b[lab]
    n = len(sub)
    for cls in ["detected_increase", "detected_decrease",
                "supports_small_change", "cannot_determine"]:
        k = int((sub.ins_change_class == cls).sum())
        lo, hi = wilson(k, n)
        rows.append(dict(set=name, change_class=cls, k=k, n=n,
                         frac=k / n, ci_lo=lo, ci_hi=hi))
pd.DataFrame(rows).to_csv(f"{TAB}/stats_fig1_class_proportions.csv", index=False)

# aggregate profile around bound-region centres (+-10 bins) vs matched centres
reg = pd.read_csv(f"{TAB}/napm_bound_regions.csv")
centers = (reg.center // 5000).clip(0, N_BINS - 1).values
mcenters = pairs.matched_to_bin.astype(int).values
half = 10
def profile(cents):
    prof = np.full((len(cents), 2 * half + 1), np.nan)
    for i, c in enumerate(cents):
        for j, o in enumerate(range(-half, half + 1)):
            prof[i, j] = dins[(c + o) % N_BINS]
    return np.nanmean(prof, axis=0), np.nanstd(prof, axis=0) / np.sqrt(len(cents))

prof_b, prof_b_se = profile(centers)
prof_m, prof_m_se = profile(mcenters)
pd.DataFrame(dict(offset_bins=np.arange(-half, half + 1),
                  offset_kb=np.arange(-half, half + 1) * 5,
                  bound_mean=prof_b, bound_se=prof_b_se,
                  matched_mean=prof_m, matched_se=prof_m_se)).to_csv(
    f"{TAB}/fig1_aggregate_profile.csv", index=False)
# centre enrichment: mean |dins| within +-2 bins of centres vs shift null
def stat_center(lab_centers):
    vals = []
    for c in lab_centers:
        for o in range(-2, 3):
            vals.append(abs(dins[(c + o) % N_BINS]))
    return np.nanmean(vals)

obs_c = stat_center(centers)
null_c = np.array([stat_center((centers + s) % N_BINS)
                   for s in RNG.integers(1, N_BINS, N_PERM)])
p_c = (np.sum(null_c >= obs_c) + 1) / (N_PERM + 1)
np.save(f"{TAB}/fig1_null_center.npy", null_c)
pd.DataFrame([dict(obs_mean_abs_dins_at_centers=obs_c,
                   p_circular_shift=p_c,
                   null_mean=float(null_c.mean()))]).to_csv(
    f"{TAB}/stats_fig1_center_enrichment.csv", index=False)

# distance from bound bins to confirmed CID boundaries vs shift null
cid = pd.read_csv(f"{ROOT}/review_results/R4_comment2_CID/tables/cid_boundaries_recalled.csv")
cid_bins = cid.loc[cid.confirmed == True, "bin"].values
def dist_to_cid(cents_bins):
    out = []
    for cb in cents_bins:
        d = np.abs(cid_bins - cb)
        d = np.minimum(d, N_BINS - d)
        out.append(d.min())
    return np.mean(out)

bound_bin_idx = np.where(bound)[0]
obs_d = dist_to_cid(bound_bin_idx)
null_d = np.array([dist_to_cid((bound_bin_idx + s) % N_BINS)
                   for s in RNG.integers(1, N_BINS, N_PERM)])
p_d = (np.sum(null_d <= obs_d) + 1) / (N_PERM + 1)
pd.DataFrame([dict(obs_mean_dist_bins=obs_d, null_mean=float(null_d.mean()),
                   p_circular_shift=p_d, n_cid=len(cid_bins))]).to_csv(
    f"{TAB}/stats_fig1_cid_boundary_distance.csv", index=False)
np.save(f"{TAB}/fig1_null_cid.npy", null_d)

summary["fig1"] = dict(paired=s1_paired, perm=s1_perm,
                       center_enrichment_p=p_c, cid_distance_p=p_d)

# ============================================================ S2: Fig 2 stats
# bin-level RNA class
def bin_rna_class(r):
    if r.n_genes == 0:
        return "no_data"
    if r.n_rna_up > 0 or r.n_rna_down > 0:
        return "detected_change"
    if r.n_rna_small == r.n_genes:
        return "supports_small_change"
    return "cannot_determine"

b["rna_bin_class"] = b.apply(bin_rna_class, axis=1)
b["structure_detected"] = b.ins_change_class.isin(
    ["detected_increase", "detected_decrease"])

# cross-classification (occupancy x structure x RNA)
rows = []
for lab, name in [(bound, "NapM_bound"), (matched, "matched_background"),
                  (~bound & ~matched, "other_unbound")]:
    sub = b[lab]
    for sc in [True, False]:
        for rc in ["detected_change", "supports_small_change",
                   "cannot_determine", "no_data"]:
            k = int(((sub.structure_detected == sc) & (sub.rna_bin_class == rc)).sum())
            rows.append(dict(set=name, structure_detected=sc, rna_class=rc,
                             k=k, n=len(sub)))
ctab = pd.DataFrame(rows)
ctab.to_csv(f"{TAB}/stats_fig2_cross_classification.csv", index=False)

# RNA vs structure correlation (bins with both), stratified
rows = []
for lab, name in [(bound, "NapM_bound"), (matched, "matched_background"),
                  (~bound, "all_unbound")]:
    sub = b[lab & np.isfinite(b.rna_mean_log2FC_KOvsWT)]
    r, p = stats.spearmanr(sub.d_ins_KO_WT, sub.rna_mean_log2FC_KOvsWT)
    rows.append(dict(set=name, n=len(sub), spearman_r=r, p=p))
pd.DataFrame(rows).to_csv(f"{TAB}/stats_fig2_rna_structure_corr.csv", index=False)

# "limited RNA change but structure changed" regions
limited = b[(b.rna_bin_class == "supports_small_change") & b.structure_detected]
pd.DataFrame([dict(
    n_limitedRNA_structureChanged=len(limited),
    n_bound_in_set=int(limited.chip_bound.sum()),
    median_d_ins=float(limited.d_ins_KO_WT.median()) if len(limited) else np.nan,
    frac_of_structure_changed=float(len(limited) / max(b.structure_detected.sum(), 1)),
)]).to_csv(f"{TAB}/stats_fig2_limited_rna_structure_change.csv", index=False)

# ---- M0 / M1 / M2 nested models with contiguous-block cross-validation
model_df = b[["d_ins_KO_WT", "weight_WT", "ins_WT", "expr_wt_mean",
              "rna_mean_log2FC_KOvsWT", "chip_bound"]].copy()
model_df["expr_wt_mean"] = model_df.expr_wt_mean.fillna(model_df.expr_wt_mean.median())
model_df["rna_mean_log2FC_KOvsWT"] = model_df.rna_mean_log2FC_KOvsWT.fillna(0.0)
# listwise deletion of bins with non-finite technical covariates (12-14 bins
# with NaN ICE weights in the merged coolers)
finite = np.isfinite(model_df.drop(columns=["chip_bound"]).values).all(axis=1)
model_df = model_df[finite].reset_index(drop=True)
n_model = len(model_df)
y = model_df.d_ins_KO_WT.values
X0 = model_df[["weight_WT", "ins_WT", "expr_wt_mean"]].values
X1 = np.column_stack([X0, model_df.rna_mean_log2FC_KOvsWT.values])
X2 = np.column_stack([X1, model_df.chip_bound.values.astype(float)])

N_BLOCKS = 10
block_id = np.arange(n_model) * N_BLOCKS // n_model  # contiguous blocks

def cv_r2(X, y, blocks):
    preds = np.full(len(y), np.nan)
    for blk in np.unique(blocks):
        tr = blocks != blk
        te = blocks == blk
        Xtr = np.column_stack([np.ones(tr.sum()), X[tr]])
        Xte = np.column_stack([np.ones(te.sum()), X[te]])
        beta = np.linalg.lstsq(Xtr, y[tr], rcond=None)[0]
        preds[te] = Xte @ beta
    ss_res = np.nansum((y - preds) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return 1 - ss_res / ss_tot

r2 = [cv_r2(X, y, block_id) for X in (X0, X1, X2)]
# block bootstrap CI for dR2 (M2 vs M1)
boots = []
for _ in range(1000):
    blk_sample = RNG.choice(np.unique(block_id), size=N_BLOCKS, replace=True)
    bs = np.concatenate([np.where(block_id == bl)[0] for bl in blk_sample])
    # rebuild block labels within the bootstrap sample from original blocks
    orig_blk = np.concatenate([np.full((block_id == bl).sum(), k)
                               for k, bl in enumerate(blk_sample)])
    r1 = cv_r2(X1[bs], y[bs], orig_blk)
    r2b = cv_r2(X2[bs], y[bs], orig_blk)
    boots.append(r2b - r1)
boots = np.array(boots)
model_tbl = pd.DataFrame(dict(
    model=["M0_baseline", "M1_plus_RNA", "M2_plus_occupancy"],
    cv_R2=r2))
model_tbl["n_bins"] = n_model
model_tbl.to_csv(f"{TAB}/stats_fig2_nested_models.csv", index=False)
pd.DataFrame([dict(dR2_M2_vs_M1=r2[2] - r2[1],
                   ci_lo=float(np.percentile(boots, 2.5)),
                   ci_hi=float(np.percentile(boots, 97.5)),
                   dR2_M1_vs_M0=r2[1] - r2[0])]).to_csv(
    f"{TAB}/stats_fig2_model_comparison.csv", index=False)
summary["fig2"] = dict(cv_R2=r2, dR2_M2_M1=float(r2[2] - r2[1]),
                       dR2_ci=[float(np.percentile(boots, 2.5)),
                               float(np.percentile(boots, 97.5))])

# in-sample coefficient of occupancy in M2 (context, not causal)
Xf = np.column_stack([np.ones(n_model), X2])
beta = np.linalg.lstsq(Xf, y, rcond=None)[0]
resid = y - Xf @ beta
s2 = resid @ resid / (n_model - Xf.shape[1])
se = np.sqrt(np.diag(np.linalg.inv(Xf.T @ Xf)) * s2)
pd.DataFrame(dict(term=["intercept", "weight_WT", "ins_WT", "expr_wt_mean",
                        "rna_log2FC", "napm_occupancy"],
                  beta=beta, se=se)).to_csv(f"{TAB}/stats_fig2_m2_coefficients.csv",
                                            index=False)

# ============================================================ S3: Fig 3 stats
# hypoxia response (WT side, replicate-supported) at bound vs matched
rows = []
for lab, name in [(bound, "NapM_bound"), (matched, "matched_background")]:
    sub = b[lab]
    rows.append(dict(set=name, n=len(sub),
                     median_d_ins_hyp_WT=float(sub.d_ins_hyp_WT.median()),
                     median_d_ins_hyp_KO=float(sub.d_ins_hyp_KO.median()),
                     median_interaction=float(sub.d_ins_interaction.median())))
pd.DataFrame(rows).to_csv(f"{TAB}/stats_fig3_condition_effects_by_occupancy.csv",
                          index=False)
# permutation for interaction difference bound vs matched
iv = b.d_ins_interaction.values
obs_i = np.nanmedian(iv[bound]) - np.nanmedian(iv[matched])
null_i = np.empty(N_PERM)
for k in range(N_PERM):
    s = RNG.integers(1, N_BINS)
    null_i[k] = np.nanmedian(iv[np.roll(bound, s)]) - np.nanmedian(iv[np.roll(matched, s)])
p_i = (np.sum(np.abs(null_i) >= abs(obs_i)) + 1) / (N_PERM + 1)
pd.DataFrame([dict(obs=obs_i, p_circular_shift=p_i)]).to_csv(
    f"{TAB}/stats_fig3_interaction_permutation.csv", index=False)

# RNA interaction summary
inter_sig = inter[inter.padj_int < 0.05]
pd.DataFrame([dict(
    n_genes_tested=len(inter),
    n_interaction_fdr05=len(inter_sig),
    n_interaction_nominal_p01=int((inter.p_int < 0.01).sum()),
    median_abs_interaction=float(inter.interaction.abs().median()),
    note="no variance moderation (df=4); low power by construction")] ).to_csv(
    f"{TAB}/stats_fig3_rna_interaction_summary.csv", index=False)

# latent descriptive
lat_desc = dict(median_ins_latent=float(b.ins_latent.median()),
                median_d_ins_latent_WT=float((b.ins_latent - b.ins_WT).median()),
                note="latent has no matched KO condition; descriptive only")
pd.DataFrame([lat_desc]).to_csv(f"{TAB}/stats_fig3_latent_descriptive.csv",
                                index=False)
summary["fig3"] = dict(interaction_perm_p=float(p_i),
                       rna_interaction_fdr05=int(len(inter_sig)))

# ---- sensitivity of the primary Fig-1 comparison
sens_rows = []
for col, lab in [("d_ins_KO_WT", "w25kb_primary"),
                 ("d_ins_KO_WT_w2", "w10kb"), ("d_ins_KO_WT_w10", "w50kb")]:
    bb = b.loc[b.chip_bound, col].values
    mm = b.loc[b.matched_background, col].values
    stat = np.nanmedian(bb) - np.nanmedian(mm)
    vv = b[col].values
    nulls = np.empty(500)
    for k in range(500):
        s = RNG.integers(1, N_BINS)
        nulls[k] = np.nanmedian(vv[np.roll(bound, s)]) - np.nanmedian(vv[np.roll(matched, s)])
    sens_rows.append(dict(analysis=lab, stat=stat,
                          p=(np.sum(np.abs(nulls) >= abs(stat)) + 1) / 501))
# alternative occupancy threshold
ba = b.chip_bound_alt.values
stat = np.nanmedian(dins[ba]) - np.nanmedian(dins[~ba])
nulls = np.empty(500)
for k in range(500):
    s = RNG.integers(1, N_BINS)
    nulls[k] = np.nanmedian(dins[np.roll(ba, s)]) - np.nanmedian(dins[~np.roll(ba, s)])
sens_rows.append(dict(analysis="alt_occupancy_q97.5", stat=stat,
                      p=(np.sum(np.abs(nulls) >= abs(stat)) + 1) / 501))
pd.DataFrame(sens_rows).to_csv(f"{TAB}/stats_sensitivity.csv", index=False)

b.to_csv(f"{TAB}/bin_level_integrated.csv", index=False)  # with rna_bin_class
json.dump(summary, open(f"{TAB}/stats_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("sensitivity:", pd.DataFrame(sens_rows).to_string())
