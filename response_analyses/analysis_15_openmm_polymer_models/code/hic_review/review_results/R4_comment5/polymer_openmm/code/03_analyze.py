#!/usr/bin/env python3
"""R4C5 polymer_openmm - analysis of OpenMM run outputs.

Bead-unit metric definitions mirror the experimental ones exactly
(01_experiment_metrics.py; 2 x 5 kb bins = 1 bead, so bead distance db
corresponds to genomic distance db x 10 kb; circular short-arc throughout):
  P(s)          probability-normalised contact sum per circular bead
                distance db = 1..220; also per-pair mean ps_mean
  fractions     short db < 10 (<100 kb), mid 10 <= db < 100, long db >= 100
  ori-ter O/E   mean contact over ori beads (0 +/- 5 circular) x ter beads
                (203 +/- 5) / ps_mean[203]
  arm-arm O/E   mean over mirror pairs (i, (N-i) mod N) +/- 1 bead band,
                circular db > 50 (>500 kb), of C / ps_mean[db]
  slope         log10-log10 slope of ps_mean over db = 10..99
                (100 kb - 1 Mb)
  geometry      Rg, principal-axis elongation, axial z extent
  n_eff         effective independent conformations from the Rg
                autocorrelation time: nconf * sample_every / (2*tau_int)

Modes:
  --mode cutoff --run <tag>     calibrate contact cutoff on a baseline run
                                against WT frac_short = 0.3769
  --mode grid                   P(s) shape RMSE (log10, mean-centred,
                                10 kb-2.2 Mb) of every baseline run vs WT
  --mode all                    full metrics for every runs/*.npz; needs
                                tables/contact_cutoff.json
Outputs: tables/baseline_grid.csv, tables/contact_cutoff.json,
         tables/sim_metrics.csv, tables/maps/<tag>_cmap.npz
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

POLY = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm"
N = 441
TER_BEAD = 203
KB_PER_BEAD = 10.0
WT_FRAC_SHORT = 0.3769
CUTOFF_CANDIDATES = [round(1.2 + 0.1 * i, 1) for i in range(34)]  # 1.2..4.5

IDX = np.arange(N)
DB = np.minimum(np.abs(IDX[:, None] - IDX[None, :]),
                N - np.abs(IDX[:, None] - IDX[None, :]))
IU = np.triu_indices(N, k=1)
DB_UT = DB[IU]
ORI_BEADS = np.arange(-5, 6) % N
TER_BEADS = (TER_BEAD + np.arange(-5, 6)) % N

# mirror-pair mask (unique pairs, +/-1 bead band, db > 50)
SEC_IJ = []
for i in range(N):
    for dj in (-1, 0, 1):
        j = (N - i + dj) % N
        if i < j and DB[i, j] > 50:
            SEC_IJ.append((i, j))
SEC_IJ = np.array(SEC_IJ)
SEC_DB = DB[SEC_IJ[:, 0], SEC_IJ[:, 1]]
SLOPE_DB = np.arange(10, 100)

# experimental P(s): replicate-mean probability-normalised, per 5 kb bin
_ps = pd.read_csv(f"{POLY}/tables/experiment_ps_curves.csv")
PS_EXP = {}
for cond in ["WT", "hypoxia", "latent"]:
    sub = _ps[_ps.condition == cond].groupby("dist_bin")["ps"].mean()
    arr = np.full(441, np.nan)
    arr[sub.index.values] = sub.values
    PS_EXP[cond] = arr          # index = dbin (5 kb), 1..440


def contact_map(positions, cutoff, max_conf=None):
    """Mean contact frequency map over (subsampled) conformations."""
    pos = positions if max_conf is None else positions[::max(1, len(positions) // max_conf)]
    C = np.zeros((N, N))
    for x in pos:
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        C += d2 < cutoff ** 2
    C /= len(pos)
    np.fill_diagonal(C, 0.0)
    return C


def ps_from_map(C):
    v = C[IU]
    ps_sum = np.bincount(DB_UT, weights=v, minlength=N // 2 + 1)
    ps_cnt = np.bincount(DB_UT, minlength=N // 2 + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps_mean = ps_sum / ps_cnt
    ps_prob = ps_sum / v.sum()
    return ps_mean, ps_prob


def ps_shape_rmse(ps_prob, cond):
    """Mean-centred log10 RMSE vs experiment, dbin 2..440 (10 kb-2.2 Mb).

    Only the *shape* of P(s) is compared: the absolute scale of simulated
    contacts depends on the arbitrary contact cutoff, so both curves are
    centred to zero mean in log10 over the fit range.
    """
    exp = PS_EXP[cond][2::2]        # dbin 2,4,...,440 -> db 1..220
    sim = ps_prob[1:221]
    ok = np.isfinite(exp) & (sim > 0)
    le, ls = np.log10(exp[ok]), np.log10(sim[ok])
    le -= le.mean()
    ls -= ls.mean()
    return float(np.sqrt(np.mean((ls - le) ** 2)))


def metrics_from_positions(positions, cutoff, prod_rg=None,
                           sample_every=3000):
    C = contact_map(positions, cutoff)
    ps_mean, ps_prob = ps_from_map(C)
    v = C[IU]
    tot = v.sum()
    fr_short = v[DB_UT < 10].sum() / tot
    fr_mid = v[(DB_UT >= 10) & (DB_UT < 100)].sum() / tot
    fr_long = v[DB_UT >= 100].sum() / tot
    ot_obs = C[np.ix_(ORI_BEADS, TER_BEADS)].mean()
    ot_oe = ot_obs / ps_mean[TER_BEAD]
    sec_oe = float(np.mean(C[SEC_IJ[:, 0], SEC_IJ[:, 1]] / ps_mean[SEC_DB]))
    y = ps_mean[SLOPE_DB]
    slope = np.polyfit(np.log10(SLOPE_DB * KB_PER_BEAD),
                       np.log10(y), 1)[0]

    # geometry over all saved conformations
    x = positions
    rg = np.sqrt(((x - x.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
    cov = np.einsum("cni,cnj->cij", x - x.mean(1, keepdims=True),
                    x - x.mean(1, keepdims=True)) / N
    eig = np.linalg.eigvalsh(cov)          # ascending
    elong = (eig[:, 2] / eig[:, :2].mean(1))
    zext = np.percentile(x[:, :, 2], 95, axis=1) - \
        np.percentile(x[:, :, 2], 5, axis=1)

    out = dict(frac_short=fr_short, frac_mid=fr_mid, frac_long=fr_long,
               ori_ter_OE=ot_oe, arm_arm_OE=sec_oe, slope=slope,
               Rg=float(rg.mean()), Rg_sd=float(rg.std(ddof=1)),
               elongation=float(elong.mean()),
               axial_len=float(zext.mean()),
               ps_rmse_WT=ps_shape_rmse(ps_prob, "WT"),
               ps_rmse_latent=ps_shape_rmse(ps_prob, "latent"))
    if prod_rg is not None and len(prod_rg) > 20:
        vv = prod_rg - prod_rg.mean()
        n = len(vv)
        f = np.fft.rfft(vv, n=2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac /= ac[0]
        neg = np.argmax(ac < 0)
        neg = n // 2 if neg == 0 else neg
        tau_int = (2 * np.sum(ac[:neg]) - 1) * sample_every
        out["tau_int_steps"] = float(tau_int)
        out["n_eff"] = float(np.clip(n * sample_every / (2 * tau_int),
                                     1, n))
    return out, C, ps_prob


def run_tag(path):
    return os.path.basename(path).replace(".npz", "")


def mode_cutoff(tag):
    z = np.load(f"{POLY}/runs/{tag}.npz")
    pos = z["positions"][::5]              # 200 conformations
    rows = []
    for cut in CUTOFF_CANDIDATES:
        C = contact_map(pos, cut)
        ps_mean, _ = ps_from_map(C)
        v = C[IU]
        fr_short = v[DB_UT < 10].sum() / v.sum()
        rows.append(dict(cutoff=cut, frac_short=fr_short))
        print(f"cutoff {cut}: frac_short = {fr_short:.4f} "
              f"(target {WT_FRAC_SHORT})")
    df = pd.DataFrame(rows)
    df.to_csv(f"{POLY}/tables/cutoff_calibration.csv", index=False)
    best = df.iloc[(df.frac_short - WT_FRAC_SHORT).abs().argmin()]
    with open(f"{POLY}/tables/contact_cutoff.json", "w") as fh:
        json.dump(dict(cutoff=float(best.cutoff),
                       frac_short_sim=float(best.frac_short),
                       frac_short_WT=WT_FRAC_SHORT, calibrated_on=tag), fh,
                  indent=2)
    print(f"-> cutoff = {best.cutoff} (sim frac_short {best.frac_short:.4f})")


def mode_grid():
    rows = []
    for path in sorted(glob.glob(f"{POLY}/runs/baseline_*.npz")):
        if "helixinit" in path:      # initial-condition cross-check runs
            continue
        z = np.load(path)
        C = contact_map(z["positions"][::5], 1.5)   # cutoff cancels in
        _, ps_prob = ps_from_map(C)                 # mean-centred RMSE
        p = json.loads(str(z["params"]))
        rows.append(dict(tag=run_tag(path), scale=p["scale"],
                         ps_rmse_WT=ps_shape_rmse(ps_prob, "WT"),
                         ps_rmse_latent=ps_shape_rmse(ps_prob, "latent"),
                         Rg=float(z["prod_rg"].mean())))
        print(rows[-1])
    df = pd.DataFrame(rows).sort_values("scale")
    df.to_csv(f"{POLY}/tables/baseline_grid.csv", index=False)
    best = df.iloc[df.ps_rmse_WT.argmin()]
    print(f"\nBest baseline scale = {best.scale} "
          f"(P(s) shape RMSE vs WT = {best.ps_rmse_WT:.4f})")


def mode_all():
    with open(f"{POLY}/tables/contact_cutoff.json") as fh:
        cutoff = json.load(fh)["cutoff"]
    print(f"contact cutoff = {cutoff}")
    os.makedirs(f"{POLY}/tables/maps", exist_ok=True)
    rows = []
    for path in sorted(glob.glob(f"{POLY}/runs/*.npz")):
        tag = run_tag(path)
        z = np.load(path)
        p = json.loads(str(z["params"]))
        m, C, ps_prob = metrics_from_positions(
            z["positions"], cutoff, z["prod_rg"], p["sample_every"])
        # block stationarity: first-200 vs last-200 conformation drift,
        # in units of the overall SD (>1 = suspicious)
        for nm, series in [("Rg", z["prod_rg"]),
                           ("ot3d", z["prod_fish"][:, 0])]:
            sd = series.std(ddof=1)
            m[f"drift_{nm}"] = float(
                (series[-200:].mean() - series[:200].mean()) / max(sd, 1e-9))
        fish = z["prod_fish"]
        names = [str(s) for s in z["fish_names"]]
        for k, nm in enumerate(names):
            m[f"fish_{nm}_mean"] = float(fish[:, k].mean())
            m[f"fish_{nm}_sd"] = float(fish[:, k].std(ddof=1))
        np.savez_compressed(f"{POLY}/tables/maps/{tag}_cmap.npz",
                            contact_map=C, ps_prob=ps_prob)
        rows.append(dict(tag=tag, scenario=p["scenario"],
                         init=p.get("init", "longitudinal"),
                         strength=p["strength"], strength2=p["strength2"],
                         scale=p["scale"], seed=p["seed"],
                         nconf=p["nconf"], **m))
        print(f"{tag}: otOE={m['ori_ter_OE']:.3f} aaOE={m['arm_arm_OE']:.3f} "
              f"slope={m['slope']:.3f} frL={m['frac_long']:.3f} "
              f"rmseWT={m['ps_rmse_WT']:.3f} rmseLat={m['ps_rmse_latent']:.3f} "
              f"n_eff={m.get('n_eff', np.nan):.0f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{POLY}/tables/sim_metrics.csv", index=False)
    print("saved tables/sim_metrics.csv + tables/maps/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["cutoff", "grid", "all"])
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.mode == "cutoff":
        assert a.run, "--run <tag> required"
        mode_cutoff(a.run)
    elif a.mode == "grid":
        mode_grid()
    else:
        mode_all()
