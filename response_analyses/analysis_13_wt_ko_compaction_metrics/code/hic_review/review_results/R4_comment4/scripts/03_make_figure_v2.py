#!/usr/bin/env python3
"""Fig_R4C4 v2: WT vs napM-KO only, "what the reviewer saw vs what
quantification shows".

Panels:
  A  merged-matrix contact maps (log colour scale, shared): WT, napM-KO,
     and log2(KO/WT) ratio map (diverging); secondary diagonal (arm-arm
     mirror line, x vs L-x) marked with a dashed line.
  B  P(s) decay, 4-replicate mean +/- SD; inset log2(KO/WT) ratio curve.
  C  three-level decomposition of the secondary diagonal: raw mean contact,
     depth-normalised (per 1M contacts), distance-normalised O/E.
  D  arm-arm O/E profile along the arm (mirror pairs x vs L-x, distance
     >500 kb), replicate mean +/- SD.
  E  four metrics, per-replicate dots, Welch t and Mann-Whitney p.
Also writes tables/summary_WT_vs_KO.csv and tables/arm_profile_sec_diag.csv.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import cooler

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.linewidth": 0.6,
})

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{BASE}/review_results/R4_comment4"
DATA = f"{BASE}/review_data/rep_icedmatrix"
GENOME_LEN = 4_411_532
TER_POS = 2_030_000
BIN = 5_000
N_BINS = 883
COLORS = {"WT": "#1f77b4", "napM-KO": "#d62728"}

df = pd.read_csv(f"{OUT}/tables/replicate_metrics.csv")
df = df[df.condition.isin(["WT", "napM-KO"])].reset_index(drop=True)
ps_tab = pd.read_csv(f"{OUT}/tables/ps_curves_per_replicate.csv")
ps_tab = ps_tab[ps_tab.condition.isin(["WT", "napM-KO"])]

# ---------- Panel A data: merged balanced matrices ----------
def load_cool(name):
    c = cooler.Cooler(f"{BASE}/review_data/cool_file/{name}::/resolutions/5000")
    return c.matrix(balance=True, sparse=False)[:].astype(float)

M_wt, M_ko = load_cool("wt_5k.cool"), load_cool("napm_ko_5k.cool")


def inpaint(M):
    """Fill ICE-filtered (NaN) pixels by local 3x3 mean; display only."""
    from scipy.ndimage import uniform_filter
    w = np.isfinite(M).astype(float)
    M0 = np.nan_to_num(M, nan=0.0)
    sm = uniform_filter(M0, size=3) / np.maximum(uniform_filter(w, size=3),
                                                 1e-12)
    return np.where(np.isfinite(M), M, sm)


M_wt_d, M_ko_d = inpaint(M_wt), inpaint(M_ko)   # display copies
with np.errstate(divide="ignore"):
    L_wt, L_ko = np.log10(M_wt_d), np.log10(M_ko_d)
finite = np.isfinite(L_wt) & np.isfinite(L_ko)
vmin, vmax = np.nanpercentile(np.concatenate([L_wt[finite], L_ko[finite]]),
                              [5, 99.5])
with np.errstate(divide="ignore", invalid="ignore"):
    RATIO = np.log2(M_ko_d / M_wt_d)
rlim = 1.5
# smooth the ratio map (5x5 bins = 25 kb) for display only
from scipy.ndimage import uniform_filter
_r = np.where(np.isfinite(RATIO), RATIO, np.nan)
RATIO_SM = uniform_filter(np.nan_to_num(_r, nan=np.nanmean(_r)), size=5)
RATIO_SM[~np.isfinite(RATIO)] = np.nan
EXT = [0, GENOME_LEN / 1e6, GENOME_LEN / 1e6, 0]  # Mb, origin upper

# ---------- Panel D data: arm profile per replicate ----------
starts = np.arange(N_BINS) * BIN
mids = (starts + np.minimum(starts + BIN, GENOME_LEN)) / 2.0
D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)


def load_rep(cond_key, rep):
    a = np.loadtxt(f"{DATA}/{cond_key}_{rep}_5000_iced.matrix")
    i, j, v = a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1, a[:, 2]
    M = np.zeros((N_BINS, N_BINS))
    M[i, j] = v
    M[j, i] = v
    bad = M.sum(axis=0) == 0
    M[bad, :] = np.nan
    M[:, bad] = np.nan
    return M


ARM_WIN = 20
arm_edges = np.linspace(250_000, GENOME_LEN / 2 - 250_000, ARM_WIN + 1)
arm_rows = []
for cond, key in [("WT", "wt"), ("napM-KO", "napm_ko")]:
    for rep in (1, 2, 3, 4):
        M = load_rep(key, rep)
        # same-replicate P(s) at 5 kb
        iu = np.triu_indices(N_BINS, k=1)
        v = M[iu]
        dbin = np.clip((D[iu] // BIN).astype(int), 0, 440)
        ok = np.isfinite(v)
        ps_sum = np.bincount(dbin[ok], weights=v[ok], minlength=441)
        ps_cnt = np.bincount(dbin[ok], minlength=441)
        with np.errstate(invalid="ignore", divide="ignore"):
            ps = ps_sum / ps_cnt
        # mirror partner per bin
        order = np.argsort(mids)
        jmir = order[np.searchsorted(mids[order], (GENOME_LEN - mids) % GENOME_LEN) % N_BINS]
        for w in range(ARM_WIN):
            lo, hi = arm_edges[w], arm_edges[w + 1]
            sel = np.where((mids >= lo) & (mids < hi))[0]
            oes = []
            for i in sel:
                j = jmir[i]
                if not (np.isfinite(M[i, j])):
                    continue
                if abs(mids[j] - ((GENOME_LEN - mids[i]) % GENOME_LEN)) > 10_000:
                    continue
                d = D[i, j]
                if d <= 500_000:
                    continue
                e = ps[min(int(d // BIN), 440)]
                if np.isfinite(e) and e > 0:
                    oes.append(M[i, j] / e)
            if oes:
                arm_rows.append(dict(condition=cond, replicate=rep,
                                     arm_pos=(lo + hi) / 2, oe=np.mean(oes)))
df_arm = pd.DataFrame(arm_rows)
df_arm.to_csv(f"{OUT}/tables/arm_profile_sec_diag.csv", index=False)

# ---------- WT vs KO summary table (incl. raw sec-diag levels) ----------
M_ALL = ["frac_short_lt100kb", "frac_mid_100kb_1Mb", "frac_long_gt1Mb",
         "sec_diag_OE", "ori_ter_OE", "compaction_slope",
         "sec_diag_raw", "sec_diag_raw_per1M"]
rows = []
for m in M_ALL:
    x = df.loc[df.condition == "napM-KO", m].values
    y = df.loc[df.condition == "WT", m].values
    t, p_t = stats.ttest_ind(x, y, equal_var=False)
    _, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
    rows.append(dict(metric=m,
                     WT_mean=y.mean(), WT_sd=y.std(ddof=1),
                     KO_mean=x.mean(), KO_sd=x.std(ddof=1),
                     diff_KO_minus_WT=x.mean() - y.mean(),
                     fold_KO_over_WT=x.mean() / y.mean(),
                     welch_p=p_t, mw_p=p_u))
df_sum = pd.DataFrame(rows)
df_sum.to_csv(f"{OUT}/tables/summary_WT_vs_KO.csv", index=False)
print(df_sum.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ================= figure =================
MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 225 * MM))
gs = GridSpec(4, 12, figure=fig, hspace=0.9, wspace=0.75,
              left=0.065, right=0.985, top=0.955, bottom=0.055)


def plabel(ax, s, x=-0.04):
    ax.text(x, 1.05, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


# ---- Panel A ----
cmap = plt.cm.viridis.copy(); cmap.set_bad("#d9d9d9")
rcmap = plt.cm.RdBu_r.copy(); rcmap.set_bad("#d9d9d9")
axesA = []
for k, (L, ttl) in enumerate([(L_wt, "WT"), (L_ko, "napM-KO")]):
    ax = fig.add_subplot(gs[0, k * 4:(k + 1) * 4])
    im = ax.imshow(np.ma.masked_invalid(L), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=EXT, interpolation="none")
    ax.grid(False)
    ax.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "w--", lw=0.9)
    ax.set_title(ttl, fontsize=8)
    ax.set_xlabel("genomic position (Mb)")
    if k == 0:
        ax.set_ylabel("genomic position (Mb)")
    ax.text(0.42, 3.62, "secondary diagonal (arm-arm)", color="white",
            fontsize=5.5, ha="center", rotation=-45, rotation_mode="anchor")
    ax.text(0.12, 0.32, "ori", color="white", fontsize=6, fontweight="bold")
    ax.text(TER_POS / 1e6 - 0.08, TER_POS / 1e6 + 0.28, "ter", color="white",
            fontsize=6, fontweight="bold")
    axesA.append(ax)
cb = fig.colorbar(im, ax=axesA, fraction=0.02, pad=0.02)
cb.set_label("log10 balanced contact", fontsize=6)
cb.ax.tick_params(labelsize=5.5)

axR = fig.add_subplot(gs[0, 8:12])
imR = axR.imshow(np.ma.masked_invalid(np.clip(RATIO_SM, -rlim, rlim)),
                 cmap=rcmap, vmin=-rlim, vmax=rlim, extent=EXT,
                 interpolation="none")
axR.grid(False)
axR.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "k--", lw=0.9)
axR.set_title("log2(KO/WT), 25 kb smooth", fontsize=8)
axR.set_xlabel("genomic position (Mb)")
cbR = fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.02)
cbR.ax.tick_params(labelsize=5.5)
cbR.set_label("log2(KO/WT)", fontsize=6)
plabel(axesA[0], "A", x=-0.12)

# ---- Panel B ----
axB = fig.add_subplot(gs[1, 0:6])
ps_tab = ps_tab.copy()
ps_tab["center"] = np.sqrt(ps_tab.dist_lo * ps_tab.dist_hi)
_tot = df.set_index(["condition", "replicate"])["total_contacts"]
ps_tab["ps"] = ps_tab.apply(
    lambda r: r.ps / _tot.loc[(r.condition, r.replicate)], axis=1)
curves = {}
for cond in ["WT", "napM-KO"]:
    piv = (ps_tab[ps_tab.condition == cond]
           .groupby("center")["ps"].agg(["mean", "std"]).dropna())
    curves[cond] = piv
    axB.loglog(piv.index, piv["mean"], "-", lw=1.3, color=COLORS[cond],
               label=cond)
    axB.fill_between(piv.index, piv["mean"] - piv["std"],
                     piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2,
                     lw=0)
axB.set_xlabel("genomic distance s (bp)")
axB.set_ylabel("contact probability P(s)")
axB.legend(frameon=False, fontsize=6.5, loc="lower left")
axB.grid(alpha=0.25, which="both", lw=0.3)
axB.set_title("contact-decay curve P(s), mean ± SD (n = 4)", fontsize=7.5)
plabel(axB, "B")

axBi = axB.inset_axes([0.40, 0.45, 0.57, 0.42])
ratio = curves["napM-KO"]["mean"] / curves["WT"]["mean"]
axBi.semilogx(ratio.index, np.log2(ratio), "-", lw=1, color="black")
axBi.axhline(0, color="gray", lw=0.6)
axBi.axvspan(5e3, 1e5, color="#1f77b4", alpha=0.12)
axBi.axvspan(1e6, 2.2e6, color="#d62728", alpha=0.12)
axBi.set_ylabel("log2(KO/WT)", fontsize=6)
axBi.set_xlabel("s (bp)", fontsize=6, labelpad=1)
axBi.tick_params(labelsize=5.5)
axBi.text(0.03, 0.05, "<100 kb: short-range loss", fontsize=5,
          color="#1f77b4", transform=axBi.transAxes)
axBi.text(0.97, 0.05, ">1 Mb: long-range gain", fontsize=5,
          color="#d62728", transform=axBi.transAxes, ha="right")

# ---- Panel C ----
C_ITEMS = [("sec_diag_raw", "(i) raw mean contact", None),
           ("sec_diag_raw_per1M", "(ii) per 10$^6$ contacts\n(depth-normalised)", None),
           ("sec_diag_OE", "(iii) distance-normalised O/E\n(expected = P(s))", None)]
for k, (m, label, _) in enumerate(C_ITEMS):
    ax = fig.add_subplot(gs[1, 6 + k * 2: 8 + k * 2])
    for x, cond in enumerate(["WT", "napM-KO"]):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=7,
                   color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=7,
                    color="black", capsize=2, lw=0.8, zorder=4)
    wt_m = df.loc[df.condition == "WT", m].mean()
    ko_m = df.loc[df.condition == "napM-KO", m].mean()
    fc = ko_m / wt_m
    ax.set_title(label, fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"], fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.margins(x=0.25, y=0.3)
    ax.text(0.5, 0.97, f"KO/WT = {fc:.2f}×", transform=ax.transAxes,
            ha="center", va="top", fontsize=6, fontweight="bold",
            color="#d62728" if fc > 1 else "#1f77b4")
    if k == 0:
        plabel(ax, "C")

# ---- Panel D ----
axD = fig.add_subplot(gs[2, 0:6])
for cond in ["WT", "napM-KO"]:
    piv = (df_arm[df_arm.condition == cond]
           .groupby("arm_pos")["oe"].agg(["mean", "std"]))
    axD.plot(piv.index / 1e6, piv["mean"], "o-", ms=2.5, lw=1.2,
             color=COLORS[cond], label=cond)
    axD.fill_between(piv.index / 1e6, piv["mean"] - piv["std"],
                     piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2,
                     lw=0)
axD.axvline(TER_POS / 1e6, color="gray", lw=0.8, ls=":")
axD.text(TER_POS / 1e6 + 0.02, axD.get_ylim()[1], "ter", fontsize=6,
         color="gray", va="top")
axD.set_xlabel("arm coordinate x (Mb); pair = (x, L−x)")
axD.set_ylabel("arm-arm O/E (>500 kb)")
axD.set_title("secondary-diagonal O/E along the arm, mean ± SD (n = 4)",
              fontsize=7.5)
axD.legend(frameon=False, fontsize=6.5)
axD.grid(alpha=0.25, lw=0.3)
plabel(axD, "D")

# per-position sign test for D
piv_wt = df_arm[df_arm.condition == "WT"].groupby("arm_pos")["oe"].mean()
piv_ko = df_arm[df_arm.condition == "napM-KO"].groupby("arm_pos")["oe"].mean()
n_lower = int((piv_ko < piv_wt).sum())
axD.text(0.98, 0.03, f"KO < WT at {n_lower}/{len(piv_wt)} arm positions",
         transform=axD.transAxes, ha="right", fontsize=6)

# ---- Panel E ----
E_METRICS = [("frac_short_lt100kb", "short-range fraction\n(<100 kb)"),
             ("sec_diag_OE", "secondary-diagonal O/E"),
             ("ori_ter_OE", "ori-ter O/E"),
             ("compaction_slope", "compaction slope\n(P(s), 100 kb-1 Mb)")]
for k, (m, label) in enumerate(E_METRICS):
    ax = fig.add_subplot(gs[3, k * 3:(k + 1) * 3])
    for x, cond in enumerate(["WT", "napM-KO"]):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=7,
                   color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=7,
                    color="black", capsize=2, lw=0.8, zorder=4)
    r = df_sum[df_sum.metric == m].iloc[0]
    ax.set_title(label, fontsize=6.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"], fontsize=6)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.tick_params(labelsize=6)
    ax.margins(x=0.2, y=0.3)
    ax.text(0.5, 0.97, f"Welch p = {r.welch_p:.1e}\nMW p = {r.mw_p:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
    if k == 0:
        plabel(ax, "E")

fig.savefig(f"{OUT}/figures/Fig_R4C4_replicate_metrics.pdf")
fig.savefig(f"{OUT}/figures/Fig_R4C4_replicate_metrics.png", dpi=300)
print("Saved figures/Fig_R4C4_replicate_metrics.{pdf,png} (v2, WT vs KO)")
