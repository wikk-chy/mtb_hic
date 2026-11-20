#!/usr/bin/env python3
"""R4C5 polymer_openmm - figures (300 dpi png + pdf).

Requires tables/sim_metrics.csv + tables/model_comparison.csv
(03_analyze.py --mode all, 04_model_comparison.py).

  fig_contact_maps    WT/latent experimental maps + best sim maps
  fig_ps              P(s): experiment vs baseline / scenarios
  fig_heldout         held-out metrics, sim folds vs experimental folds
  fig_comparison      AIC bar chart + per-metric z-score heatmap
  fig_convergence     burn-in plateaus, production trace, autocorrelation
  fig_postpred        posterior-predictive check (exp mean+/-SD vs sim)
  fig_fish            predicted 3D distance distributions for FISH probes
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLY = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm"
DATA = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_data/rep_icedmatrix"
N = 441
SCEN = ["baseline", "S1", "S2", "S3", "S3app"]
SCEN_LABEL = {"baseline": "baseline (WT-like)", "S1": "S1 global collapse",
              "S2": "S2 ori-ter tether", "S3": "S3 toroid",
              "S3app": "S3 + apposition"}

mc = pd.read_csv(f"{POLY}/tables/model_comparison.csv")
sim = pd.read_csv(f"{POLY}/tables/sim_metrics.csv")
exp = pd.read_csv(f"{POLY}/tables/experiment_metrics.csv")
BEST = {r.scenario: r.tag for r in mc.itertuples()}


def save(fig, name):
    fig.savefig(f"{POLY}/figures/{name}.png", dpi=300)
    fig.savefig(f"{POLY}/figures/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def load_cmap(tag):
    return np.load(f"{POLY}/tables/maps/{tag}_cmap.npz")


def exp_map(cond_key):
    M = np.zeros((883, 883))
    for rep in [1, 2, 3, 4]:
        a = np.loadtxt(f"{DATA}/{cond_key}_{rep}_5000_iced.matrix")
        i, j, v = a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1, a[:, 2]
        M[i, j] += v
        M[j, i] += v
    M[M == 0] = np.nan
    return M / 4


# ---------- fig_contact_maps ----------
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for ax, key, ttl in [(axes[0, 0], "wt", "experiment WT"),
                     (axes[0, 1], "latent", "experiment latent")]:
    M = exp_map(key)
    im = ax.imshow(np.log10(M), cmap="Reds", vmin=-1, vmax=2.5)
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ax, sc in zip(list(axes[0, 2:]) + list(axes[1, :2]), SCEN):
    C = load_cmap(BEST[sc])["contact_map"]
    Cp = C.copy(); Cp[Cp == 0] = np.nan
    ax.imshow(np.log10(Cp), cmap="Reds")
    ax.set_title(SCEN_LABEL[sc], fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes[1, 2:]:
    ax.axis("off")
fig.suptitle("Contact maps: experiment (5 kb, log10 balanced contacts) vs "
             "best-fit simulations (10 kb beads, log10 contact frequency)")
save(fig, "fig_contact_maps")

# ---------- fig_ps ----------
ps_exp = pd.read_csv(f"{POLY}/tables/experiment_ps_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for cond, col in [("WT", "black"), ("latent", "firebrick")]:
    sub = ps_exp[ps_exp.condition == cond].groupby("dist_bp")["ps"].mean()
    axes[0].loglog(sub.index, sub.values, ".", ms=2, color=col,
                   label=f"exp {cond}", alpha=0.6)
C = load_cmap(BEST["baseline"])
ps = C["ps_prob"]
axes[0].loglog((np.arange(1, 221)) * 1e4, ps[1:221], "-", color="steelblue",
               lw=1.5, label="sim baseline")
axes[0].set_xlabel("genomic distance s (bp)")
axes[0].set_ylabel("P(s)")
axes[0].legend(fontsize=8)
axes[0].set_title("baseline vs WT")
axes[1].loglog(ps_exp[ps_exp.condition == "latent"].groupby("dist_bp")
               ["ps"].mean().index,
               ps_exp[ps_exp.condition == "latent"].groupby("dist_bp")
               ["ps"].mean().values, ".", ms=2, color="firebrick",
               label="exp latent", alpha=0.6)
for sc, col in zip(SCEN[1:], ["C1", "C2", "C3", "C4"]):
    ps = load_cmap(BEST[sc])["ps_prob"]
    axes[1].loglog((np.arange(1, 221)) * 1e4, ps[1:221], "-", lw=1.2,
                   color=col, label=f"sim {SCEN_LABEL[sc]}")
axes[1].set_xlabel("genomic distance s (bp)")
axes[1].legend(fontsize=7)
axes[1].set_title("scenarios vs latent (best strength each)")
save(fig, "fig_ps")

# ---------- fig_heldout ----------
lat = exp[exp.condition == "latent"]
wt = exp[exp.condition == "WT"]
exp_fold = dict(ori_ter_OE=lat.ori_ter_OE.mean() / wt.ori_ter_OE.mean(),
                arm_arm_OE=lat.sec_diag_OE.mean() / wt.sec_diag_OE.mean())
base = mc[mc.scenario == "baseline"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, m, ttl in [(axes[0], "ori_ter_OE", "ori-ter O/E (held-out)"),
                   (axes[1], "arm_arm_OE", "arm-arm O/E (held-out)")]:
    folds = [mc[mc.scenario == sc].iloc[0][f"sim_{m}"] / base[f"sim_{m}"]
             for sc in SCEN]
    ax.bar(range(len(SCEN)), folds, color="steelblue", alpha=0.8,
           label="simulation (fold vs baseline)")
    ax.axhline(exp_fold[m], color="firebrick", lw=2,
               label=f"experiment latent/WT = {exp_fold[m]:.2f}")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels(SCEN, rotation=30, ha="right", fontsize=8)
    ax.set_title(ttl)
    ax.set_ylabel("fold change")
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_heldout")

# ---------- fig_comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(mc.scenario, mc.AIC_reg, color="teal")
axes[0].set_ylabel("AIC (regularised pseudo-likelihood)")
axes[0].set_title("model comparison (lower = better)")
metrics = ["frac_short", "frac_mid", "frac_long", "slope",
           "ori_ter_OE", "arm_arm_OE"]
Z = mc[[f"z_{m}" for m in metrics]].values
Z = np.clip(Z, -10, 10)
im = axes[1].imshow(Z, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
axes[1].set_xticks(range(len(metrics)))
axes[1].set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
axes[1].set_yticks(range(len(mc)))
axes[1].set_yticklabels(mc.scenario, fontsize=8)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        axes[1].text(j, i, f"{mc.iloc[i][f'z_{metrics[j]}']:.1f}",
                     ha="center", va="center", fontsize=7)
axes[1].set_title("z-scores vs latent experiment\n(held-out: last two cols)")
fig.colorbar(im, ax=axes[1], label="z (clipped at ±10)")
fig.tight_layout()
save(fig, "fig_comparison")

# ---------- fig_convergence (baseline best) ----------
z = np.load(f"{POLY}/runs/{BEST['baseline']}.npz")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
t_burn = (np.arange(len(z["burn_e"])) + 1) * 2000
axes[0].plot(t_burn, z["burn_e"], lw=0.6, label="energy")
ax2 = axes[0].twinx()
ax2.plot(t_burn, z["burn_rg"], color="C1", lw=0.6, label="Rg")
ax2.plot(t_burn, z["burn_ot"], color="C2", lw=0.6, label="ori-ter dist")
axes[0].set_xlabel("burn-in step")
axes[0].set_title("burn-in plateaus (energy, Rg, ori-ter)")
axes[0].legend(fontsize=7, loc="upper left")
ax2.legend(fontsize=7, loc="lower right")
rg = z["prod_rg"]
axes[1].plot(np.arange(len(rg)) * 3000, rg, lw=0.3)
axes[1].set_xlabel("production step")
axes[1].set_ylabel("Rg")
axes[1].set_title("production Rg trace")
v = rg - rg.mean()
n = len(v)
f = np.fft.rfft(v, n=2 * n)
ac = np.fft.irfft(f * np.conj(f))[:n]
ac /= ac[0]
neg = np.argmax(ac < 0)
neg = n // 2 if neg == 0 else neg
tau_int = (2 * np.sum(ac[:neg]) - 1) * 3000
axes[2].plot(np.arange(min(n, 200)) * 3000, ac[:min(n, 200)], lw=1)
axes[2].axvline(tau_int, color="firebrick", ls="--",
                label=f"$\\tau_{{int}}$ = {tau_int:.0f} steps")
n_eff = np.clip(n * 3000 / (2 * tau_int), 1, n)
axes[2].axvline(3000, color="grey", ls=":", label="sampling interval")
axes[2].set_xlabel("lag (steps)")
axes[2].set_ylabel("Rg autocorrelation")
axes[2].set_title(f"autocorrelation; n_eff = {n_eff:.0f} / {n} confs")
axes[2].legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_convergence")

# ---------- fig_postpred ----------
fig, ax = plt.subplots(figsize=(8, 5))
labels = {"frac_short": "frac short", "frac_mid": "frac mid",
          "frac_long": "frac long", "slope": "P(s) slope",
          "ori_ter_OE": "ori-ter O/E *", "arm_arm_OE": "arm-arm O/E *"}
EXP_NAME = dict(frac_short="frac_short_lt100kb", frac_mid="frac_mid_100kb_1Mb",
                frac_long="frac_long_gt1Mb", slope="compaction_slope",
                ori_ter_OE="ori_ter_OE", arm_arm_OE="sec_diag_OE")
ys = np.arange(len(metrics))[::-1]
for y, m in zip(ys, metrics):
    mu = lat[EXP_NAME[m]].mean()
    sd = lat[EXP_NAME[m]].std(ddof=1)
    lo, hi = mu - 3 * sd, mu + 3 * sd
    span = max(abs(hi - lo), 1e-12)
    for sc, col in zip(SCEN, ["C0", "C1", "C2", "C3", "C4"]):
        v_ = mc[mc.scenario == sc].iloc[0][f"sim_{m}"]
        xc = np.clip((v_ - lo) / span / 6 + 0.5, -0.4, 1.4)
        ax.plot(xc, y, "o", color=col, ms=6,
                label=SCEN_LABEL[sc] if m == metrics[0] else None)
    ax.plot([0.5, 0.5], [y - 0.3, y + 0.3], color="black", lw=2)
    ax.plot([1 / 6, 5 / 6], [y, y], color="grey", lw=4, alpha=0.4)
ax.set_yticks(ys)
ax.set_yticklabels([labels[m] for m in metrics], fontsize=9)
ax.set_xlabel("position relative to latent experiment\n"
              "(black tick = mean, grey band = mean ± 3 SD; "
              "points clipped if outside)")
ax.set_title("posterior-predictive check (* = held-out metrics)")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_postpred")

# ---------- fig_fish ----------
fish = pd.read_csv(f"{POLY}/tables/fish_predictions.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pairs = fish.pair.unique()
for sc, col in zip(SCEN, ["C0", "C1", "C2", "C3", "C4"]):
    sub = fish[fish.scenario == sc].set_index("pair").loc[pairs]
    axes[0].plot(range(len(pairs)), sub.mean_3D_sigma, "o-", color=col,
                 label=SCEN_LABEL[sc])
axes[0].set_xticks(range(len(pairs)))
axes[0].set_xticklabels(pairs, rotation=30, ha="right", fontsize=8)
axes[0].set_ylabel("mean 3D distance (sigma ~ 50 nm)")
axes[0].set_title("predicted FISH distance means")
axes[0].legend(fontsize=8)
for sc, col in zip(SCEN, ["C0", "C1", "C2", "C3", "C4"]):
    sub = fish[fish.scenario == sc].set_index("pair").loc[pairs]
    axes[1].plot(range(len(pairs)), sub.frac_within_5sigma, "o-", color=col,
                 label=SCEN_LABEL[sc])
axes[1].set_xticks(range(len(pairs)))
axes[1].set_xticklabels(pairs, rotation=30, ha="right", fontsize=8)
axes[1].set_ylabel("fraction within 5 sigma (~250 nm)")
axes[1].set_title("predicted co-localisation fractions")
fig.tight_layout()
save(fig, "fig_fish")
