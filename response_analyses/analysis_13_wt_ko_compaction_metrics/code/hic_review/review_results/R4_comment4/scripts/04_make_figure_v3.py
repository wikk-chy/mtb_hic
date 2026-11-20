#!/usr/bin/env python3
"""Fig_R4C4 v3: WT vs napM-KO, 7 panels (A-G), reworked layout.

New analyses relative to v2:
  F  3D-model size quantification (LorDG reconstructions, one per condition;
     radius of gyration Rg, pairwise bead-distance distribution, arm
     extension). Single reconstruction per condition -> NO p-values.
  G  cumulative contact-distance distribution per replicate (5 kb-2.2 Mb)
     + median contact distance with Welch/MW tests.

Layout (180 mm wide):
  row 1: A  three equal square heatmaps (WT | KO | log2(KO/WT)) + shared
            colour handling, dashed secondary diagonal, ori/ter marks
  row 2: B  P(s) mean+/-SD with log2(KO/WT) inset | G cumulative distance
            distribution with median-distance annotation
  row 3: C  three-level secondary-diagonal decomposition | D arm profile
  row 4: E  four metric dot plots (2/3 width) | F Rg + 3D distance
            distribution (1/3 width)
Outputs: figures/Fig_R4C4_v3.{pdf,png}, tables/median_contact_distance.csv,
tables/g_cumulative_curves.csv, tables/model_3d_metrics.csv
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
    "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{BASE}/review_results/R4_comment4"
DATA = f"{BASE}/review_data/rep_icedmatrix"
PDB = f"{BASE}/review_data/3d_model_pdbfile"
GENOME_LEN = 4_411_532
TER_POS = 2_030_000
BIN = 5_000
N_BINS = 883
COLORS = {"WT": "#1f77b4", "napM-KO": "#d62728"}
CONDS = ["WT", "napM-KO"]
KEYS = {"WT": "wt", "napM-KO": "napm_ko"}

df = pd.read_csv(f"{OUT}/tables/replicate_metrics.csv")
df = df[df.condition.isin(CONDS)].reset_index(drop=True)
ps_tab = pd.read_csv(f"{OUT}/tables/ps_curves_per_replicate.csv")
ps_tab = ps_tab[ps_tab.condition.isin(CONDS)]
df_arm = pd.read_csv(f"{OUT}/tables/arm_profile_sec_diag.csv")

# ---------- shared geometry ----------
starts = np.arange(N_BINS) * BIN
mids = (starts + np.minimum(starts + BIN, GENOME_LEN)) / 2.0
D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)
IU = np.triu_indices(N_BINS, k=1)


def load_rep(cond, rep):
    a = np.loadtxt(f"{DATA}/{KEYS[cond]}_{rep}_5000_iced.matrix")
    i, j, v = a[:, 0].astype(int) - 1, a[:, 1].astype(int) - 1, a[:, 2]
    M = np.zeros((N_BINS, N_BINS))
    M[i, j] = v
    M[j, i] = v
    bad = M.sum(axis=0) == 0
    M[bad, :] = np.nan
    M[:, bad] = np.nan
    return M


# ---------- Panel A data (merged matrices) ----------
def load_cool(name):
    c = cooler.Cooler(f"{BASE}/review_data/cool_file/{name}::/resolutions/5000")
    return c.matrix(balance=True, sparse=False)[:].astype(float)


def fill_with_expected(M):
    """Fill ICE-filtered (NaN) pixels with the same-matrix P(s) expected
    value at their circular distance (display only; gaps of consecutive
    filtered bins cannot be filled by local interpolation)."""
    iu = IU
    v = M[iu]
    d = D[iu]
    ok = np.isfinite(v)
    dbin = np.clip((d // BIN).astype(int), 0, 440)
    ps_sum = np.bincount(dbin[ok], weights=v[ok], minlength=441)
    ps_cnt = np.bincount(dbin[ok], minlength=441)
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = ps_sum / ps_cnt
    out = M.copy()
    bad = ~np.isfinite(out)
    db_full = np.clip((D // BIN).astype(int), 0, 440)
    out[bad] = ps[db_full[bad]]
    return out


M_wt, M_ko = load_cool("wt_5k.cool"), load_cool("napm_ko_5k.cool")
M_wt_d, M_ko_d = fill_with_expected(M_wt), fill_with_expected(M_ko)
with np.errstate(divide="ignore"):
    L_wt, L_ko = np.log10(M_wt_d), np.log10(M_ko_d)
fin = np.isfinite(L_wt) & np.isfinite(L_ko)
vmin, vmax = np.nanpercentile(np.concatenate([L_wt[fin], L_ko[fin]]), [5, 99.5])
with np.errstate(divide="ignore", invalid="ignore"):
    RATIO = np.log2(M_ko_d / M_wt_d)
from scipy.ndimage import uniform_filter
_r = np.where(np.isfinite(RATIO), RATIO, np.nan)
RATIO_SM = uniform_filter(np.nan_to_num(_r, nan=np.nanmean(_r)), size=5)
RLIM = 1.5
EXT = [0, GENOME_LEN / 1e6, GENOME_LEN / 1e6, 0]

# ---------- Panel G data: cumulative contact-distance distribution ----------
GRID = np.logspace(np.log10(5000), np.log10(GENOME_LEN / 2), 120)
g_rows, med_rows = [], []
for cond in CONDS:
    for rep in (1, 2, 3, 4):
        M = load_rep(cond, rep)
        v = M[IU]
        d = D[IU]
        ok = np.isfinite(v)
        v, d = v[ok], d[ok]
        order = np.argsort(d)
        d_s, v_s = d[order], v[order]
        cum = np.cumsum(v_s) / v_s.sum()
        cum_g = np.interp(GRID, d_s, cum)
        med = np.interp(0.5, cum_g, GRID)
        med_rows.append(dict(condition=cond, replicate=rep,
                             median_contact_distance_bp=med))
        for g, cv in zip(GRID, cum_g):
            g_rows.append(dict(condition=cond, replicate=rep,
                               distance_bp=g, cum_fraction=cv))
df_med = pd.DataFrame(med_rows)
df_med.to_csv(f"{OUT}/tables/median_contact_distance.csv", index=False)
df_g = pd.DataFrame(g_rows)
df_g.to_csv(f"{OUT}/tables/g_cumulative_curves.csv", index=False)
x_ko = df_med.loc[df_med.condition == "napM-KO", "median_contact_distance_bp"]
x_wt = df_med.loc[df_med.condition == "WT", "median_contact_distance_bp"]
t_g, p_g_t = stats.ttest_ind(x_ko, x_wt, equal_var=False)
_, p_g_u = stats.mannwhitneyu(x_ko, x_wt, alternative="two-sided")
print("median contact distance: WT %.0f ± %.0f kb, KO %.0f ± %.0f kb, "
      "Welch p=%.2e, MW p=%.4f"
      % (x_wt.mean() / 1e3, x_wt.std(ddof=1) / 1e3, x_ko.mean() / 1e3,
         x_ko.std(ddof=1) / 1e3, p_g_t, p_g_u))

# ---------- Panel F data: 3D models ----------
def load_pdb(path):
    c = []
    for line in open(path):
        if line.startswith("ATOM"):
            c.append([float(line[30:38]), float(line[38:46]),
                      float(line[46:54])])
    return np.array(c)


f_rows, dist_distr = [], {}
for cond, fn in [("WT", "wt_5k_1701486311672.pdb"),
                 ("napM-KO", "napm_ko_5k_1704965399048.pdb")]:
    xyz = load_pdb(f"{PDB}/{fn}")
    cen = xyz.mean(axis=0)
    rg = float(np.sqrt(((xyz - cen) ** 2).sum(axis=1).mean()))
    dd = np.linalg.norm(xyz[IU[0]] - xyz[IU[1]], axis=1)
    ori_c = xyz[0:10].mean(axis=0)
    ter_c = xyz[400:413].mean(axis=0)
    a1, a2 = xyz[203], xyz[644]          # arm midpoints ~1.015 / ~3.22 Mb
    f_rows.append(dict(condition=cond, Rg=rg,
                       pairdist_mean=dd.mean(), pairdist_median=np.median(dd),
                       arm1mid_to_ori=np.linalg.norm(a1 - ori_c),
                       arm1mid_to_ter=np.linalg.norm(a1 - ter_c),
                       arm2mid_to_ori=np.linalg.norm(a2 - ori_c),
                       arm2mid_to_ter=np.linalg.norm(a2 - ter_c)))
    dist_distr[cond] = dd
df_f = pd.DataFrame(f_rows)
df_f.to_csv(f"{OUT}/tables/model_3d_metrics.csv", index=False)
rg_fc = df_f.loc[1, "Rg"] / df_f.loc[0, "Rg"]
print("3D model Rg: WT %.3f, KO %.3f (%.3fx); pairdist mean %.3f -> %.3f"
      % (df_f.Rg[0], df_f.Rg[1], rg_fc, df_f.pairdist_mean[0],
         df_f.pairdist_mean[1]))

# ================= figure =================
MM = 1 / 25.4
fig = plt.figure(figsize=(180 * MM, 240 * MM))


def plabel(ax, s):
    ax.text(-0.03, 1.04, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


# ---- row 1: Panel A, manual equal squares ----
side = 0.195                     # figure-height fraction; square in mm
wsq = side * 240 / 180           # matching width fraction
y0, ytop = 0.74, 0.74 + side
x0s = [0.05, 0.335, 0.665]
cmap = plt.cm.viridis.copy()
rcmap = plt.cm.RdBu_r.copy()
axA = []
for k, (L, ttl) in enumerate([(L_wt, "WT"), (L_ko, "napM-KO")]):
    ax = fig.add_axes([x0s[k], y0, wsq, side])
    im = ax.imshow(L, cmap=cmap, vmin=vmin, vmax=vmax, extent=EXT,
                   interpolation="none")
    ax.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "w--", lw=0.9)
    ax.set_title(ttl, fontsize=8)
    ax.set_xlabel("genomic position (Mb)")
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 1, 2, 3, 4])
    if k == 0:
        ax.set_ylabel("genomic position (Mb)")
    ax.text(0.45, 3.60, "secondary diagonal (arm-arm)", color="white",
            fontsize=7, ha="center", rotation=-45, rotation_mode="anchor")
    ax.text(0.14, 0.34, "ori", color="white", fontsize=7, fontweight="bold")
    ax.text(TER_POS / 1e6 - 0.10, TER_POS / 1e6 + 0.30, "ter", color="white",
            fontsize=7, fontweight="bold")
    axA.append(ax)
cbA = fig.add_axes([x0s[1] + wsq + 0.008, y0, 0.011, side])
cb1 = fig.colorbar(im, cax=cbA)
cb1.ax.tick_params(labelsize=7)
cb1.ax.set_title("log10\nbalanced\ncontact", fontsize=7, pad=3)

axR = fig.add_axes([x0s[2], y0, wsq, side])
imR = axR.imshow(np.clip(RATIO_SM, -RLIM, RLIM), cmap=rcmap, vmin=-RLIM,
                 vmax=RLIM, extent=EXT, interpolation="none")
axR.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "k--", lw=0.9)
axR.set_title("log2(KO/WT), 25 kb smooth", fontsize=8)
axR.set_xlabel("genomic position (Mb)")
axR.set_xticks([0, 1, 2, 3, 4]); axR.set_yticks([0, 1, 2, 3, 4])
cbR = fig.add_axes([x0s[2] + wsq + 0.008, y0, 0.011, side])
cb2 = fig.colorbar(imR, cax=cbR)
cb2.ax.tick_params(labelsize=7)
cb2.set_label("log2(KO/WT)", fontsize=7)
axA[0].text(-0.18, 1.04, "A", transform=axA[0].transAxes, fontsize=10,
            fontweight="bold", va="bottom")

# ---- rows 2-4 via GridSpec ----
gs = GridSpec(3, 12, figure=fig, top=0.685, bottom=0.055, left=0.075,
              right=0.975, hspace=0.62, wspace=0.65)

# ---- Panel B ----
axB = fig.add_subplot(gs[0, 0:6])
ps_tab = ps_tab.copy()
ps_tab["center"] = np.sqrt(ps_tab.dist_lo * ps_tab.dist_hi)
_tot = df.set_index(["condition", "replicate"])["total_contacts"]
ps_tab["ps"] = ps_tab.apply(
    lambda r: r.ps / _tot.loc[(r.condition, r.replicate)], axis=1)
curves = {}
for cond in CONDS:
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
axB.legend(frameon=False, loc="lower left")
axB.grid(alpha=0.25, which="both", lw=0.3)
axB.set_title("contact-decay curve P(s), mean ± SD (n = 4)")
plabel(axB, "B")
axBi = axB.inset_axes([0.40, 0.44, 0.57, 0.42])
ratio = curves["napM-KO"]["mean"] / curves["WT"]["mean"]
axBi.semilogx(ratio.index, np.log2(ratio), "-", lw=1, color="black")
axBi.axhline(0, color="gray", lw=0.6)
axBi.axvspan(5e3, 1e5, color="#1f77b4", alpha=0.12)
axBi.axvspan(1e6, 2.2e6, color="#d62728", alpha=0.12)
axBi.set_ylabel("log2(KO/WT)", fontsize=7)
axBi.set_xlabel("s (bp)", fontsize=7, labelpad=1)
axBi.tick_params(labelsize=7)
axBi.set_ylim(np.nanmin(np.log2(ratio)) - 0.6, None)
axBi.text(0.02, 0.03, "<100 kb\nshort-range loss", fontsize=7,
          color="#1f77b4", transform=axBi.transAxes, va="bottom")
axBi.text(0.98, 0.03, ">1 Mb\nlong-range gain", fontsize=7,
          color="#d62728", transform=axBi.transAxes, ha="right", va="bottom")

# ---- Panel G ----
axG = fig.add_subplot(gs[0, 6:12])
for cond in CONDS:
    piv = (df_g[df_g.condition == cond]
           .groupby("distance_bp")["cum_fraction"].agg(["mean", "std"]))
    axG.semilogx(piv.index, piv["mean"], "-", lw=1.3, color=COLORS[cond],
                 label=cond)
    axG.fill_between(piv.index, piv["mean"] - piv["std"],
                     piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2,
                     lw=0)
axG.axhline(0.5, color="gray", lw=0.6, ls=":")
for cond in CONDS:
    m = df_med.loc[df_med.condition == cond, "median_contact_distance_bp"]
    axG.axvline(m.mean(), color=COLORS[cond], lw=0.8, ls="--")
axG.set_xlabel("genomic distance s (bp)")
axG.set_ylabel("cumulative contact fraction ≤ s")
axG.set_title("cumulative contact-distance distribution, mean ± SD (n = 4)")
axG.legend(frameon=False, loc="upper left")
axG.grid(alpha=0.25, which="both", lw=0.3)
axG.text(0.97, 0.30,
         "median contact distance:\nWT %.0f ± %.0f kb\nKO %.0f ± %.0f kb\n"
         "Welch p = %.1e; MW p = %.3f"
         % (x_wt.mean() / 1e3, x_wt.std(ddof=1) / 1e3, x_ko.mean() / 1e3,
            x_ko.std(ddof=1) / 1e3, p_g_t, p_g_u),
         transform=axG.transAxes, ha="right", fontsize=7,
         va="top", bbox=dict(fc="white", ec="0.8", lw=0.5))
plabel(axG, "G")

# ---- Panel C ----
C_ITEMS = [("sec_diag_raw", "(i) raw mean contact"),
           ("sec_diag_raw_per1M", "(ii) per 10$^6$ contacts\n(depth-normalised)"),
           ("sec_diag_OE", "(iii) distance-normalised O/E\n(expected = P(s))")]
for k, (m, label) in enumerate(C_ITEMS):
    ax = fig.add_subplot(gs[1, k * 2:(k + 1) * 2])
    for x, cond in enumerate(CONDS):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=8,
                   color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=7,
                    color="black", capsize=2, lw=0.8, zorder=4)
    fc = df.loc[df.condition == "napM-KO", m].mean() / \
        df.loc[df.condition == "WT", m].mean()
    ax.set_title(label, fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"])
    ax.margins(x=0.28, y=0.3)
    ax.text(0.5, 0.97, f"KO/WT = {fc:.2f}×", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, fontweight="bold",
            color="#d62728" if fc > 1 else "#1f77b4")
    if k == 0:
        ax.text(-0.42, 1.04, "C", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="left")

# ---- Panel D ----
axD = fig.add_subplot(gs[1, 6:12])
for cond in CONDS:
    piv = (df_arm[df_arm.condition == cond]
           .groupby("arm_pos")["oe"].agg(["mean", "std"]))
    axD.plot(piv.index / 1e6, piv["mean"], "o-", ms=2.5, lw=1.2,
             color=COLORS[cond], label=cond)
    axD.fill_between(piv.index / 1e6, piv["mean"] - piv["std"],
                     piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2,
                     lw=0)
axD.axvline(TER_POS / 1e6, color="gray", lw=0.8, ls=":")
axD.text(TER_POS / 1e6 - 0.02, axD.get_ylim()[1], "ter ", fontsize=7,
         color="gray", va="top", ha="right")
axD.set_xlabel("arm coordinate x (Mb); pair = (x, L−x)")
axD.set_ylabel("arm-arm O/E (>500 kb)")
axD.set_title("secondary-diagonal O/E along the arm, mean ± SD (n = 4)")
axD.legend(frameon=False, loc="lower left")
axD.grid(alpha=0.25, lw=0.3)
piv_wt = df_arm[df_arm.condition == "WT"].groupby("arm_pos")["oe"].mean()
piv_ko = df_arm[df_arm.condition == "napM-KO"].groupby("arm_pos")["oe"].mean()
axD.text(0.98, 0.03, f"KO < WT at {(piv_ko < piv_wt).sum()}/{len(piv_wt)} "
         "arm positions", transform=axD.transAxes, ha="right", fontsize=7)
plabel(axD, "D")

# ---- Panel E ----
E_METRICS = [("frac_short_lt100kb", "short-range fraction\n(<100 kb)"),
             ("sec_diag_OE", "secondary-diagonal O/E"),
             ("ori_ter_OE", "ori-ter O/E"),
             ("compaction_slope", "compaction slope\n(P(s), 100 kb-1 Mb)")]
stat = pd.read_csv(f"{OUT}/tables/stats_tests.csv")
for k, (m, label) in enumerate(E_METRICS):
    ax = fig.add_subplot(gs[2, k * 2:(k + 1) * 2])
    for x, cond in enumerate(CONDS):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=8,
                   color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=7,
                    color="black", capsize=2, lw=0.8, zorder=4)
    r = stat.query("comparison == 'napM-KO vs WT' and metric == @m").iloc[0]
    ax.set_title(label, fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.margins(x=0.28, y=0.32)
    ax.text(0.5, 0.97, f"Welch p={r.welch_p:.0e}\nMW p={r.mw_p:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7)
    if k == 0:
        plabel(ax, "E")

# ---- Panel F ----
axF1 = fig.add_subplot(gs[2, 8:10])
axF1.bar([0, 1], df_f.Rg, color=[COLORS[c] for c in CONDS], width=0.6)
for x, v in enumerate(df_f.Rg):
    axF1.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
axF1.set_xticks([0, 1]); axF1.set_xticklabels(["WT", "KO"])
axF1.set_ylabel("radius of gyration")
axF1.set_title("3D model size (Rg)\nKO/WT = %.3f×" % rg_fc, fontsize=7)
axF1.margins(y=0.25)
plabel(axF1, "F")

axF2 = fig.add_subplot(gs[2, 10:12])
all_d = np.concatenate([dist_distr["WT"], dist_distr["napM-KO"]])
bins = np.linspace(0, np.percentile(all_d, 99.9), 80)
for cond in CONDS:
    h, e = np.histogram(dist_distr[cond], bins=bins, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    axF2.plot(c, h, "-", lw=1.2, color=COLORS[cond], label=cond)
axF2.set_xlabel("bead-bead 3D distance")
axF2.set_ylabel("density")
axF2.set_title("pairwise 3D distance\n(single model per cond.)", fontsize=7)
axF2.legend(frameon=False)
axF2.tick_params(labelsize=7)

fig.savefig(f"{OUT}/figures/Fig_R4C4_v3.pdf")
fig.savefig(f"{OUT}/figures/Fig_R4C4_v3.png", dpi=300)
print("Saved figures/Fig_R4C4_v3.{pdf,png}")
