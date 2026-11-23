#!/usr/bin/env python3
"""R4C5 polymer_openmm - combined multi-panel figure (all results except FISH).

Panels:
  A  contact maps: exp WT / exp latent / baseline / S1 / S2 / S3 / S3app
  B  P(s): experiment WT+latent vs baseline sim
  C  P(s): latent experiment vs scenario sims (best strength each)
  D  held-out ori-ter O/E (sim fold vs baseline; exp latent/WT line)
  E  held-out arm-arm O/E (same)
  F  model comparison AIC (lower = better)
  G  per-metric z-scores vs latent experiment
  H  posterior-predictive check
  I  convergence diagnostics (burn-in plateaus, production Rg, autocorr)

Reuses data products of 01-04 (tables/ + runs/ + tables/maps/).
Output: figures/fig_combined_polymer.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

POLY = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm"
DATA = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_data/rep_icedmatrix"
SCEN = ["baseline", "S1", "S2", "S3", "S3app"]
SCEN_LABEL = {"baseline": "baseline (WT-like)", "S1": "S1 global collapse",
              "S2": "S2 ori-ter tether", "S3": "S3 toroid",
              "S3app": "S3 + apposition"}

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

mc = pd.read_csv(f"{POLY}/tables/model_comparison.csv")
sim = pd.read_csv(f"{POLY}/tables/sim_metrics.csv")
exp = pd.read_csv(f"{POLY}/tables/experiment_metrics.csv")
BEST = {r.scenario: r.tag for r in mc.itertuples()}


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


fig = plt.figure(figsize=(16.5, 15))
gs = gridspec.GridSpec(4, 14, figure=fig, hspace=0.55, wspace=0.55,
                       left=0.05, right=0.985, top=0.94, bottom=0.05)


def plabel(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


# ---------------- A: contact maps (7 in a row) ----------------
map_specs = [("exp", "wt", "experiment WT"), ("exp", "latent", "experiment latent")]
map_specs += [("sim", sc, SCEN_LABEL[sc]) for sc in SCEN]
first_map_ax = None
for k, (kind, key, ttl) in enumerate(map_specs):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    if kind == "exp":
        M = exp_map(key)
        im = ax.imshow(np.log10(M), cmap="Reds", vmin=-1, vmax=2.5)
    else:
        C = load_cmap(BEST[key])["contact_map"]
        Cp = C.copy()
        Cp[Cp == 0] = np.nan
        im = ax.imshow(np.log10(Cp), cmap="Reds")
    ax.set_title(ttl, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        plabel(ax, "A")
caxm = fig.add_axes([0.985, 0.755, 0.008, 0.16])
cbm = fig.colorbar(im, cax=caxm)
cbm.set_label("log10 contact", fontsize=6.5)
cbm.ax.tick_params(labelsize=6)

# ---------------- B: P(s) baseline vs WT ----------------
ps_exp = pd.read_csv(f"{POLY}/tables/experiment_ps_curves.csv")
axB = fig.add_subplot(gs[1, 0:4])
for cond, col in [("WT", "black"), ("latent", "firebrick")]:
    sub = ps_exp[ps_exp.condition == cond].groupby("dist_bp")["ps"].mean()
    axB.loglog(sub.index, sub.values, ".", ms=2, color=col,
               label=f"exp {cond}", alpha=0.6)
C = load_cmap(BEST["baseline"])
axB.loglog((np.arange(1, 221)) * 1e4, C["ps_prob"][1:221], "-",
           color="steelblue", lw=1.5, label="sim baseline")
axB.set_xlabel("genomic distance s (bp)")
axB.set_ylabel("P(s)")
axB.legend(fontsize=7)
axB.set_title("P(s): baseline vs experiment", fontsize=8)
plabel(axB, "B")

# ---------------- C: P(s) scenarios vs latent ----------------
axC = fig.add_subplot(gs[1, 4:8])
sub = ps_exp[ps_exp.condition == "latent"].groupby("dist_bp")["ps"].mean()
axC.loglog(sub.index, sub.values, ".", ms=2, color="firebrick",
           label="exp latent", alpha=0.6)
for sc, col in zip(SCEN[1:], ["C1", "C2", "C3", "C4"]):
    ps = load_cmap(BEST[sc])["ps_prob"]
    axC.loglog((np.arange(1, 221)) * 1e4, ps[1:221], "-", lw=1.2, color=col,
               label=f"sim {SCEN_LABEL[sc]}")
axC.set_xlabel("genomic distance s (bp)")
axC.legend(fontsize=6.5)
axC.set_title("P(s): scenarios vs latent (best strength)", fontsize=8)
plabel(axC, "C")

# ---------------- D/E: held-out metrics ----------------
lat = exp[exp.condition == "latent"]
wt = exp[exp.condition == "WT"]
exp_fold = dict(ori_ter_OE=lat.ori_ter_OE.mean() / wt.ori_ter_OE.mean(),
                arm_arm_OE=lat.sec_diag_OE.mean() / wt.sec_diag_OE.mean())
base = mc[mc.scenario == "baseline"].iloc[0]
for k, (m, ttl, letter) in enumerate([
        ("ori_ter_OE", "ori-ter O/E (held-out)", "D"),
        ("arm_arm_OE", "arm-arm O/E (held-out)", "E")]):
    ax = fig.add_subplot(gs[1, 8 + 3 * k:11 + 3 * k])
    folds = [mc[mc.scenario == sc].iloc[0][f"sim_{m}"] / base[f"sim_{m}"]
             for sc in SCEN]
    ax.bar(range(len(SCEN)), folds, color="steelblue", alpha=0.8)
    ax.axhline(exp_fold[m], color="firebrick", lw=1.5,
               label=f"exp latent/WT = {exp_fold[m]:.2f}")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels(["base", "S1", "S2", "S3", "S3app"], fontsize=6.5)
    ax.set_title(ttl, fontsize=8)
    ax.set_ylabel("fold change (sim vs baseline)")
    ax.legend(fontsize=6.5, loc="upper right")
    plabel(ax, letter)

# ---------------- F: AIC bar ----------------
axF = fig.add_subplot(gs[2, 0:4])
axF.bar(mc.scenario, mc.AIC_reg, color="teal")
axF.set_ylabel("AIC (regularised pseudo-likelihood)")
axF.set_title("model comparison (lower = better)", fontsize=8)
axF.tick_params(axis="x", rotation=30)
plabel(axF, "F")

# ---------------- G: z-score heatmap ----------------
axG = fig.add_subplot(gs[2, 4:9])
metrics = ["frac_short", "frac_mid", "frac_long", "slope",
           "ori_ter_OE", "arm_arm_OE"]
Z = np.clip(mc[[f"z_{m}" for m in metrics]].values, -10, 10)
im = axG.imshow(Z, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
axG.set_xticks(range(len(metrics)))
axG.set_xticklabels(metrics, rotation=30, ha="right", fontsize=6.5)
axG.set_yticks(range(len(mc)))
axG.set_yticklabels(mc.scenario, fontsize=7)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        axG.text(j, i, f"{mc.iloc[i][f'z_{metrics[j]}']:.1f}", ha="center",
                 va="center", fontsize=6)
axG.set_title("z-scores vs latent experiment (held-out: last two cols)",
              fontsize=8)
fig.colorbar(im, ax=axG, label="z (clipped at ±10)", shrink=0.8)
plabel(axG, "G")

# ---------------- H: posterior-predictive check ----------------
axH = fig.add_subplot(gs[2, 9:14])
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
        axH.plot(xc, y, "o", color=col, ms=5,
                 label=SCEN_LABEL[sc] if m == metrics[0] else None)
    axH.plot([0.5, 0.5], [y - 0.3, y + 0.3], color="black", lw=1.5)
    axH.plot([1 / 6, 5 / 6], [y, y], color="grey", lw=3, alpha=0.4)
axH.set_yticks(ys)
axH.set_yticklabels([labels[m] for m in metrics], fontsize=7)
axH.set_xlabel("position relative to latent experiment\n(tick = mean, band = ±3 SD)", fontsize=7)
axH.set_title("posterior-predictive check (* = held-out)", fontsize=8)
axH.legend(fontsize=6, loc="lower right")
plabel(axH, "H")

# ---------------- I: convergence (baseline best) ----------------
z = np.load(f"{POLY}/runs/{BEST['baseline']}.npz")
axI1 = fig.add_subplot(gs[3, 0:5])
t_burn = (np.arange(len(z["burn_e"])) + 1) * 2000
axI1.plot(t_burn, z["burn_e"], lw=0.6, label="energy")
axI1b = axI1.twinx()
axI1b.plot(t_burn, z["burn_rg"], color="C1", lw=0.6, label="Rg")
axI1b.plot(t_burn, z["burn_ot"], color="C2", lw=0.6, label="ori-ter dist")
axI1.set_xlabel("burn-in step")
axI1.set_title("burn-in plateaus", fontsize=8)
axI1.legend(fontsize=6, loc="upper left")
axI1b.legend(fontsize=6, loc="lower right")
plabel(axI1, "I")

axI2 = fig.add_subplot(gs[3, 5:9])
rg = z["prod_rg"]
axI2.plot(np.arange(len(rg)) * 3000, rg, lw=0.3)
axI2.set_xlabel("production step")
axI2.set_ylabel("Rg")
axI2.set_title("production Rg trace", fontsize=8)

axI3 = fig.add_subplot(gs[3, 9:14])
v = rg - rg.mean()
n = len(v)
f = np.fft.rfft(v, n=2 * n)
ac = np.fft.irfft(f * np.conj(f))[:n]
ac /= ac[0]
neg = np.argmax(ac < 0)
neg = n // 2 if neg == 0 else neg
tau_int = (2 * np.sum(ac[:neg]) - 1) * 3000
axI3.plot(np.arange(min(n, 200)) * 3000, ac[:min(n, 200)], lw=1)
axI3.axvline(tau_int, color="firebrick", ls="--",
             label=f"$\\tau_{{int}}$ = {tau_int:.0f} steps")
n_eff = np.clip(n * 3000 / (2 * tau_int), 1, n)
axI3.axvline(3000, color="grey", ls=":", label="sampling interval")
axI3.set_xlabel("lag (steps)")
axI3.set_ylabel("Rg autocorrelation")
axI3.set_title(f"autocorrelation; n_eff = {n_eff:.0f} / {n} confs", fontsize=8)
axI3.legend(fontsize=6.5)

fig.suptitle("Polymer-model discrimination of ori–ter contact mechanisms "
             "(R4 comment 5; FISH predictions shown separately)",
             fontsize=12, fontweight="bold")
fig.savefig(f"{POLY}/figures/fig_combined_polymer.png", dpi=300)
fig.savefig(f"{POLY}/figures/fig_combined_polymer.pdf")
print("saved figures/fig_combined_polymer.{png,pdf}")
