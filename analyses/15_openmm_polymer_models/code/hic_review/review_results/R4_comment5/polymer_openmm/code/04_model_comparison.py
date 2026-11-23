#!/usr/bin/env python3
"""R4C5 polymer_openmm - model comparison across scenarios.

Protocol (held-out discipline):
  * baseline (WT-like): confinement scale chosen by P(s) shape RMSE vs WT
    (tables/baseline_grid.csv, written by 03_analyze.py --mode grid).
  * scenario strength: chosen by P(s) shape RMSE vs LATENT only. P(s) is a
    fit metric; ori-ter O/E and arm-arm O/E are NEVER used for selection.
  * comparison: Gaussian pseudo-likelihood of the simulated metrics given
    the latent experimental replicate mean +/- SD, separately for the four
    fit metrics (frac_short, frac_mid, frac_long, compaction slope) and the
    two held-out metrics (ori_ter_OE, sec_diag_OE). Because experimental
    replicate SDs are very small, we also report a regularised version with
    a 5%-of-mean variance floor; the regularised version is primary.
  * AIC = 2k - 2 logL, BIC = k*ln(n_obs) - 2 logL, n_obs = 6 metrics,
    k = number of scenario-specific strength parameters (0 for the
    "no-change" baseline null; the WT-fitted confinement scale is shared by
    all models and adds a constant +1 that cancels in Delta-AIC).

Outputs:
  tables/model_comparison.csv   per-scenario z-scores, logL, AIC/BIC
  tables/fish_predictions.csv   3D distance predictions for the proposed
                                FISH probe pairs (best run per scenario)
"""
import json
import numpy as np
import pandas as pd

POLY = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm"

FIT_METRICS = ["frac_short", "frac_mid", "frac_long", "slope"]
HELDOUT_METRICS = ["ori_ter_OE", "arm_arm_OE"]
# sim name -> experiment name
EXP_NAME = dict(frac_short="frac_short_lt100kb", frac_mid="frac_mid_100kb_1Mb",
                frac_long="frac_long_gt1Mb", slope="compaction_slope",
                ori_ter_OE="ori_ter_OE", arm_arm_OE="sec_diag_OE")
K_PARAMS = {"baseline": 0, "S1": 1, "S2": 1, "S3": 1, "S3app": 2}

sim = pd.read_csv(f"{POLY}/tables/sim_metrics.csv")
if "init" in sim.columns:
    sim_main = sim[sim.init != "helix"]   # helix-init = convergence check
else:
    sim_main = sim
# S3 lambda=1, seed 42 underwent the folding transition mid-production
# (block analysis); superseded by the extended-burn-in rerun (seed 142,
# equil 1e6 steps)
SUPERSEDED = {"S3_str1_str20_scale0.7_seed42"}
sim_main = sim_main[~sim_main.tag.isin(SUPERSEDED)]
exp = pd.read_csv(f"{POLY}/tables/experiment_metrics.csv")
lat = exp[exp.condition == "latent"]
EXP_MU = {m: lat[EXP_NAME[m]].mean() for m in FIT_METRICS + HELDOUT_METRICS}
EXP_SD = {m: lat[EXP_NAME[m]].std(ddof=1) for m in FIT_METRICS + HELDOUT_METRICS}


def pick_best(scenario):
    """Best run of a scenario by P(s) shape RMSE vs latent (fit metric)."""
    sub = sim_main[sim_main.scenario == scenario]
    if scenario == "S3app":
        # lambda fixed at the S3 optimum; select tether strength only
        lam = pick_best("S3")["strength"]
        sub = sub[np.isclose(sub.strength, lam)]
    if scenario == "baseline":
        grid = pd.read_csv(f"{POLY}/tables/baseline_grid.csv")
        best_scale = grid.iloc[grid.ps_rmse_WT.argmin()]["scale"]
        sub = sub[np.isclose(sub.scale, best_scale)]
    sub42 = sub[(sub.seed == 42)
                | (sub.tag == "S3_str1_str20_scale0.7_seed142")]
    best = sub42.iloc[sub42.ps_rmse_latent.argmin()]
    return best


rows = []
for sc in ["baseline", "S1", "S2", "S3", "S3app"]:
    best = pick_best(sc)
    row = dict(scenario=sc, tag=best["tag"], strength=best["strength"],
               strength2=best["strength2"], k=K_PARAMS[sc],
               n_eff=best["n_eff"])
    for grp, ms in [("fit", FIT_METRICS), ("heldout", HELDOUT_METRICS)]:
        z2, z2r = 0.0, 0.0
        for m in ms:
            z = (best[m] - EXP_MU[m]) / EXP_SD[m]
            sd_r = np.sqrt(EXP_SD[m] ** 2 + (0.05 * EXP_MU[m]) ** 2)
            zr = (best[m] - EXP_MU[m]) / sd_r
            row[f"z_{m}"] = z
            row[f"sim_{m}"] = best[m]
            row[f"exp_{m}"] = EXP_MU[m]
            z2 += z ** 2
            z2r += zr ** 2
        row[f"sumz2_{grp}"] = z2
        row[f"sumz2_{grp}_reg"] = z2r
    rows.append(row)

df = pd.DataFrame(rows)
for suffix in ["", "_reg"]:
    zf = df["sumz2_fit" + suffix]
    zh = df["sumz2_heldout" + suffix]
    df["logL" + suffix] = -0.5 * (zf + zh)      # constants cancel in deltas
    df["logL_heldout" + suffix] = -0.5 * zh
    df["AIC" + suffix] = 2 * df.k - 2 * df["logL" + suffix]
    df["BIC" + suffix] = df.k * np.log(6) - 2 * df["logL" + suffix]
df["dAIC_reg"] = df.AIC_reg - df.AIC_reg.min()
df["dBIC_reg"] = df.BIC_reg - df.BIC_reg.min()
df.to_csv(f"{POLY}/tables/model_comparison.csv", index=False)

cols = ["scenario", "tag", "k", "n_eff", "sumz2_fit_reg", "sumz2_heldout_reg",
        "AIC_reg", "dAIC_reg", "dBIC_reg"]
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print("\nheld-out metrics (sim vs latent experiment):")
for sc in df.scenario:
    r = df[df.scenario == sc].iloc[0]
    print(f"  {sc:9s} ori-ter O/E {r.sim_ori_ter_OE:.3f} "
          f"(exp {r.exp_ori_ter_OE:.3f}), arm-arm O/E {r.sim_arm_arm_OE:.3f} "
          f"(exp {r.exp_arm_arm_OE:.3f})")

# ---- FISH predictions from saved per-conformation distances ----
fish_rows = []
for sc in ["baseline", "S1", "S2", "S3", "S3app"]:
    tag = df.loc[df.scenario == sc, "tag"].iloc[0]
    z = np.load(f"{POLY}/runs/{tag}.npz")
    names = [str(s) for s in z["fish_names"]]
    for k, nm in enumerate(names):
        d = z["prod_fish"][:, k]
        fish_rows.append(dict(scenario=sc, pair=nm,
                              mean_3D_sigma=d.mean(),
                              sd_3D_sigma=d.std(ddof=1),
                              median_3D_sigma=np.median(d),
                              frac_within_5sigma=float((d < 5).mean())))
pd.DataFrame(fish_rows).to_csv(f"{POLY}/tables/fish_predictions.csv",
                               index=False)
# ---- seed independence: seed 42 vs 777 at matched parameter points ----
pairs = []
for (sc, s1, s2_, sc_), grp in sim_main.groupby(
        ["scenario", "strength", "strength2", "scale"]):
    seeds = set(grp.seed)
    # S3 lambda=1: seed-42 row superseded by the extended-burn-in seed-142
    # run; pair seed 777 with seed 142 (folded basin, disclosed)
    ref_seed = 42
    if 42 not in seeds and 142 in seeds:
        ref_seed = 142
    if ref_seed in seeds and 777 in seeds:
        a = grp[grp.seed == ref_seed].iloc[0]
        b = grp[grp.seed == 777].iloc[0]
        pairs.append(dict(scenario=sc, strength=s1, strength2=s2_,
                          tag_a=a.tag, tag_b=b.tag,
                          ori_ter_OE_a=a.ori_ter_OE, ori_ter_OE_b=b.ori_ter_OE,
                          arm_arm_OE_a=a.arm_arm_OE, arm_arm_OE_b=b.arm_arm_OE,
                          slope_a=a.slope, slope_b=b.slope,
                          frac_long_a=a.frac_long, frac_long_b=b.frac_long))
pd.DataFrame(pairs).to_csv(f"{POLY}/tables/seed_check.csv", index=False)
print("\nseed 42 vs 777 agreement:")
print(pd.DataFrame(pairs).to_string(index=False,
                                    float_format=lambda x: f"{x:.3g}"))
print("\nsaved tables/model_comparison.csv, tables/fish_predictions.csv, "
      "tables/seed_check.csv")
