#!/usr/bin/env python3
"""
Rebuild Response Fig. 14 (Reviewer 4, Comment 4 — WT vs napM-KO expansion & compaction):
100% vector-editable TrueType Arial (pdf.fonttype=42), generous print scale (9.2 x 11.4 in),
clean spacing across all 4 rows, and ZERO overlaps (fixes Panel A colorbar crowding,
Panel B inset overlap, Panel C/E y-tick & title crowding, Panel F legend overlap,
Panel G box overlap, and Panel H dagger collision with dashed line).
Exact same underlying WT/napM-KO Hi-C & 3D PDB datasets as 06_make_figure_v4.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from scipy.ndimage import uniform_filter
import cooler
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
OUT_SRC = BASE / "review_results/R4_comment4"
DATA = BASE / "review_data/rep_icedmatrix"
PDB = BASE / "review_data/3d_model_pdbfile"
OUT_FIG = ROOT / "analysis/reviewer4_fig_polish_20261124/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8.2,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.1,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.72,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

GENOME_LEN = 4_411_532
TER_POS = 2_030_000
BIN = 5_000
N_BINS = 883
COLORS = {"WT": "#1F77B4", "napM-KO": "#D62728"}
CONDS = ["WT", "napM-KO"]
C_TEXT = "#1F2933"

df = pd.read_csv(OUT_SRC / "tables/replicate_metrics.csv")
df = df[df.condition.isin(CONDS)].reset_index(drop=True)
ps_tab = pd.read_csv(OUT_SRC / "tables/ps_curves_per_replicate.csv")
ps_tab = ps_tab[ps_tab.condition.isin(CONDS)].copy()
df_arm = pd.read_csv(OUT_SRC / "tables/arm_profile_sec_diag.csv")
df_g = pd.read_csv(OUT_SRC / "tables/g_cumulative_curves.csv")
df_med = pd.read_csv(OUT_SRC / "tables/median_contact_distance.csv")
df_f = pd.read_csv(OUT_SRC / "tables/model_3d_metrics.csv")
stat = pd.read_csv(OUT_SRC / "tables/stats_tests.csv")
rec = pd.read_csv(OUT_SRC / "tables/model_map_reconciliation.csv")

starts = np.arange(N_BINS) * BIN
mids = (starts + np.minimum(starts + BIN, GENOME_LEN)) / 2.0
D = np.abs(mids[:, None] - mids[None, :])
D = np.minimum(D, GENOME_LEN - D)
IU = np.triu_indices(N_BINS, k=1)


def load_cool(name):
    c = cooler.Cooler(str(BASE / f"review_data/cool_file/{name}") + "::/resolutions/5000")
    return c.matrix(balance=True, sparse=False)[:].astype(float)


def fill_with_expected(M):
    v = M[IU]
    d = D[IU]
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
_r = np.where(np.isfinite(RATIO), RATIO, np.nan)
RATIO_SM = uniform_filter(np.nan_to_num(_r, nan=np.nanmean(_r)), size=5)
RLIM = 1.5
EXT = [0, GENOME_LEN / 1e6, GENOME_LEN / 1e6, 0]

x_ko = df_med.loc[df_med.condition == "napM-KO", "median_contact_distance_bp"]
x_wt = df_med.loc[df_med.condition == "WT", "median_contact_distance_bp"]
_, p_g_t = stats.ttest_ind(x_ko, x_wt, equal_var=False)
_, p_g_u = stats.mannwhitneyu(x_ko, x_wt, alternative="two-sided")


def load_pdb(path):
    c = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                c.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(c)


dist_distr = {}
for cond, fn in [("WT", "wt_5k_1701486311672.pdb"), ("napM-KO", "napm_ko_5k_1704965399048.pdb")]:
    xyz = load_pdb(PDB / fn)
    dist_distr[cond] = np.linalg.norm(xyz[IU[0]] - xyz[IU[1]], axis=1)
rg_fc = float(df_f.loc[1, "Rg"] / df_f.loc[0, "Rg"])

# ==================== Build wide, large-font 4-row figure (10.6 x 9.4 in) ====================
fig = plt.figure(figsize=(10.6, 9.4))

# Row 1: Panel A — 3 large equal squares across full width with clean colorbar clearance
side = 0.212
wsq = side * 9.4 / 10.6
y0 = 0.742
x0s = [0.068, 0.305, 0.642]
cmap = plt.cm.viridis.copy()
rcmap = plt.cm.RdBu_r.copy()
axA = []
for k, (L, ttl) in enumerate([(L_wt, "WT"), (L_ko, "napM-KO")]):
    ax = fig.add_axes([x0s[k], y0, wsq, side])
    im = ax.imshow(L, cmap=cmap, vmin=vmin, vmax=vmax, extent=EXT, interpolation="nearest")
    ax.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "w--", lw=1.0)
    ax.set_title(ttl, fontsize=9.2, fontweight="bold", pad=5, color=C_TEXT)
    ax.set_xlabel("Genomic position (Mb)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    if k == 0:
        ax.set_ylabel("Genomic position (Mb)")
    else:
        ax.set_yticklabels([])
    ax.text(1.18, 3.23, "secondary diagonal (arm–arm)", color="white",
            fontsize=7.0, ha="center", va="center", rotation=-45, rotation_mode="anchor")
    ax.text(0.15, 0.38, "ori", color="white", fontsize=8.0, fontweight="bold")
    ax.text(TER_POS / 1e6 - 0.10, TER_POS / 1e6 + 0.34, "ter", color="white", fontsize=8.0, fontweight="bold")
    axA.append(ax)

cbA = fig.add_axes([x0s[1] + wsq + 0.010, y0, 0.012, side])
cb1 = fig.colorbar(im, cax=cbA)
cb1.ax.tick_params(labelsize=7.2)
cb1.set_label(r"$\log_{10}$ balanced contact", fontsize=7.6, labelpad=4)

axR = fig.add_axes([x0s[2], y0, wsq, side])
imR = axR.imshow(np.clip(RATIO_SM, -RLIM, RLIM), cmap=rcmap, vmin=-RLIM, vmax=RLIM, extent=EXT, interpolation="nearest")
axR.plot([0, GENOME_LEN / 1e6], [GENOME_LEN / 1e6, 0], "k--", lw=1.0)
axR.set_title(r"$\log_2$(KO/WT), 25 kb smooth", fontsize=9.2, fontweight="bold", pad=5, color=C_TEXT)
axR.set_xlabel("Genomic position (Mb)")
axR.set_xticks([0, 1, 2, 3, 4])
axR.set_yticks([0, 1, 2, 3, 4])
axR.set_yticklabels([])
cbR = fig.add_axes([x0s[2] + wsq + 0.010, y0, 0.012, side])
cb2 = fig.colorbar(imR, cax=cbR)
cb2.ax.tick_params(labelsize=7.2)
cb2.set_label(r"$\log_2$(KO/WT)", fontsize=7.6, labelpad=4)

# Rows 2–4 using clean nested GridSpec
outer = GridSpec(
    3, 1, figure=fig,
    top=0.662, bottom=0.052, left=0.066, right=0.985,
    hspace=0.52, height_ratios=[1.02, 1.02, 1.06],
)
gs_r2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.22, width_ratios=[1.05, 1.25])
gs_r2_c = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_r2[0, 1], wspace=0.45)

gs_r3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.22, width_ratios=[0.96, 1.50])
gs_r3_e = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_r3[0, 1], wspace=0.48)

gs_r4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.26, width_ratios=[0.96, 1.06, 0.96])
gs_r4_f = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_r4[0, 0], wspace=0.38, width_ratios=[0.9, 1.15])


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.6)


# ---- Panel B: P(s) with clean non-overlapping inset ----
axB = fig.add_subplot(gs_r2[0, 0])
ps_tab["center"] = np.sqrt(ps_tab.dist_lo * ps_tab.dist_hi)
_tot = df.set_index(["condition", "replicate"])["total_contacts"]
ps_tab["ps"] = ps_tab.apply(lambda r: r.ps / _tot.loc[(r.condition, r.replicate)], axis=1)
curves = {}
for cond in CONDS:
    piv = ps_tab[ps_tab.condition == cond].groupby("center")["ps"].agg(["mean", "std"]).dropna()
    curves[cond] = piv
    axB.loglog(piv.index, piv["mean"], "-", lw=1.4, color=COLORS[cond], label=cond)
    axB.fill_between(piv.index, piv["mean"] - piv["std"], piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2, lw=0)
axB.set_ylim(5e-7, 1.8e-3)
axB.set_xlabel("Genomic distance s (bp)")
axB.set_ylabel("Contact probability P(s)")
axB.legend(frameon=False, loc="lower left", fontsize=7.6)
axB.grid(alpha=0.25, which="both", lw=0.35)
axB.set_title("Contact-decay curve P(s), mean ± SD (n = 4)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axB)

axBi = axB.inset_axes([0.44, 0.48, 0.54, 0.47])
axBi.set_facecolor("white")
ratio = curves["napM-KO"]["mean"] / curves["WT"]["mean"]
axBi.semilogx(ratio.index, np.log2(ratio), "-", lw=1.2, color="black")
axBi.axhline(0, color="gray", lw=0.6)
axBi.axvspan(5e3, 1e5, color="#1f77b4", alpha=0.12)
axBi.axvspan(1e6, 2.2e6, color="#d62728", alpha=0.12)
axBi.set_ylabel(r"$\log_2$(KO/WT)", fontsize=6.9, labelpad=1)
axBi.set_xlabel("s (bp)", fontsize=6.9, labelpad=1)
axBi.tick_params(labelsize=6.6, length=2.2)
axBi.set_ylim(np.nanmin(np.log2(ratio)) - 0.72, 0.75)
axBi.text(0.03, 0.04, "<100 kb\nloss", fontsize=6.6, color="#1f77b4", transform=axBi.transAxes, va="bottom")
axBi.text(0.97, 0.04, ">1 Mb\ngain", fontsize=6.6, color="#d62728", transform=axBi.transAxes, ha="right", va="bottom")

# ---- Panel C: 3-level secondary-diagonal decomposition ----
C_ITEMS = [
    ("sec_diag_raw", "(i) Raw mean\ncontact"),
    ("sec_diag_raw_per1M", "(ii) Per 10$^6$ contacts\n(depth-norm.)"),
    ("sec_diag_OE", "(iii) Distance-norm.\nO/E (exp. = P(s))"),
]
axC_list = []
for k, (m, label) in enumerate(C_ITEMS):
    ax = fig.add_subplot(gs_r2_c[0, k])
    axC_list.append(ax)
    for x, cond in enumerate(CONDS):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=18, color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=9, color="black", capsize=2.8, lw=1.0, zorder=4)
    fc = df.loc[df.condition == "napM-KO", m].mean() / df.loc[df.condition == "WT", m].mean()
    ax.set_title(label, fontsize=7.8, pad=5, fontweight="bold", color=C_TEXT)
    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.margins(x=0.30, y=0.38)
    ax.text(0.5, 0.96, f"KO/WT = {fc:.2f}×", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.6, fontweight="bold",
            color="#d62728" if fc > 1 else "#1f77b4")
    clean_spines(ax)

# ---- Panel D: Arm profile ----
axD = fig.add_subplot(gs_r3[0, 0])
for cond in CONDS:
    piv = df_arm[df_arm.condition == cond].groupby("arm_pos")["oe"].agg(["mean", "std"])
    axD.plot(piv.index / 1e6, piv["mean"], "o-", ms=3.0, lw=1.4, color=COLORS[cond], label=cond)
    axD.fill_between(piv.index / 1e6, piv["mean"] - piv["std"], piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2, lw=0)
axD.axvline(TER_POS / 1e6, color="gray", lw=0.9, ls=":")
axD.set_ylim(1.18, 3.35)
axD.text(TER_POS / 1e6 - 0.03, 3.28, "ter", fontsize=7.6, color="gray", va="top", ha="right")
axD.set_xlabel("Arm coordinate x (Mb); pair = (x, L−x)")
axD.set_ylabel("Arm–arm O/E (>500 kb)")
axD.set_title("Secondary-diagonal O/E along arm, mean ± SD (n = 4)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axD.legend(frameon=False, loc="lower left", fontsize=7.5)
axD.grid(alpha=0.25, lw=0.35)
piv_wt = df_arm[df_arm.condition == "WT"].groupby("arm_pos")["oe"].mean()
piv_ko = df_arm[df_arm.condition == "napM-KO"].groupby("arm_pos")["oe"].mean()
axD.text(0.97, 0.04, f"KO < WT at {(piv_ko < piv_wt).sum()}/{len(piv_wt)} positions",
         transform=axD.transAxes, ha="right", va="bottom", fontsize=7.4, color=C_TEXT,
         bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.88))
clean_spines(axD)

# ---- Panel E: 4 metric dot plots with generous horizontal clearance ----
E_METRICS = [
    ("frac_short_lt100kb", "Short-range\n(<100 kb)"),
    ("sec_diag_OE", "Arm–arm\nO/E"),
    ("ori_ter_OE", "ori–ter\nO/E"),
    ("compaction_slope", "P(s) slope\n(100 kb–1 Mb)"),
]
axE_list = []
for k, (m, label) in enumerate(E_METRICS):
    ax = fig.add_subplot(gs_r3_e[0, k])
    axE_list.append(ax)
    for x, cond in enumerate(CONDS):
        y = df.loc[df.condition == cond, m].values
        ax.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=18, color=COLORS[cond], lw=0, zorder=3)
        ax.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=9, color="black", capsize=2.8, lw=1.0, zorder=4)
    r = stat.query("comparison == 'napM-KO vs WT' and metric == @m").iloc[0]
    ax.set_title(label, fontsize=7.8, pad=5, fontweight="bold", color=C_TEXT)
    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.margins(x=0.30, y=0.42)
    ax.text(0.5, 0.97, f"Welch P={r.welch_p:.0e}\nMW P={r.mw_p:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.9, linespacing=1.22)
    clean_spines(ax)

# ---- Panel F: 3D model Rg + pairwise distance ----
axF1 = fig.add_subplot(gs_r4_f[0, 0])
axF1.bar([0, 1], df_f.Rg, color=[COLORS[c] for c in CONDS], edgecolor="black", lw=0.55, width=0.56)
for x, v in enumerate(df_f.Rg):
    axF1.text(x, v + 0.14, f"{v:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
axF1.set_xticks([0, 1], ["WT", "KO"])
axF1.set_ylabel("Radius of gyration")
axF1.set_ylim(0, 7.6)
axF1.set_title(f"3D model Rg\nKO/WT = {rg_fc:.3f}×", fontsize=7.9, pad=5, fontweight="bold", color=C_TEXT)
clean_spines(axF1)

axF2 = fig.add_subplot(gs_r4_f[0, 1])
all_d = np.concatenate([dist_distr["WT"], dist_distr["napM-KO"]])
bins = np.linspace(0, np.percentile(all_d, 99.9), 80)
ymax_f2 = 0.0
for cond in CONDS:
    h, e = np.histogram(dist_distr[cond], bins=bins, density=True)
    c = 0.5 * (e[:-1] + e[1:])
    axF2.plot(c, h, "-", lw=1.45, color=COLORS[cond], label=cond)
    ymax_f2 = max(ymax_f2, float(h.max()))
axF2.set_ylim(0, ymax_f2 * 1.36)
axF2.set_xlabel("Bead–bead 3D dist.")
axF2.set_ylabel("Density", labelpad=2)
axF2.set_title("Pairwise 3D dist.\n(single model)", fontsize=7.9, pad=5, fontweight="bold", color=C_TEXT)
axF2.legend(frameon=False, fontsize=7.0, loc="upper right", handlelength=1.2)
axF2.set_yticks([])
clean_spines(axF2)

# ---- Panel G: Cumulative contact-distance distribution ----
axG = fig.add_subplot(gs_r4[0, 1])
for cond in CONDS:
    piv = df_g[df_g.condition == cond].groupby("distance_bp")["cum_fraction"].agg(["mean", "std"])
    axG.semilogx(piv.index, piv["mean"], "-", lw=1.5, color=COLORS[cond], label=cond)
    axG.fill_between(piv.index, piv["mean"] - piv["std"], piv["mean"] + piv["std"], color=COLORS[cond], alpha=0.2, lw=0)
axG.axhline(0.5, color="gray", lw=0.7, ls=":")
for cond in CONDS:
    m = df_med.loc[df_med.condition == cond, "median_contact_distance_bp"]
    axG.axvline(m.mean(), color=COLORS[cond], lw=0.9, ls="--")
axG.set_xlabel("Genomic distance s (bp)")
axG.set_ylabel("Cumulative contact fraction")
axG.set_ylim(0, 1.06)
axG.set_title("Cumulative contact-distance distribution\nmean ± SD (n = 4)", loc="left", pad=5, fontweight="bold", color=C_TEXT)
axG.legend(frameon=False, loc="upper left", fontsize=7.4)
axG.grid(alpha=0.25, which="both", lw=0.35)
axG.text(
    0.97, 0.04,
    f"Median distance:\nWT {x_wt.mean()/1e3:.0f} ± {x_wt.std(ddof=1)/1e3:.0f} kb\n"
    f"KO {x_ko.mean()/1e3:.0f} ± {x_ko.std(ddof=1)/1e3:.0f} kb\n"
    f"Welch P = {p_g_t:.1e}; MW P = {p_g_u:.3f}",
    transform=axG.transAxes, ha="right", va="bottom", fontsize=6.8, linespacing=1.22,
    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.55),
)
clean_spines(axG)

# ---- Panel H: Hi-C vs 3D-model fold changes ----
axH = fig.add_subplot(gs_r4[0, 2])
rw, rk = rec.iloc[0], rec.iloc[1]
sec_fold = df.loc[df.condition == "napM-KO", "sec_diag_OE"].mean() / df.loc[df.condition == "WT", "sec_diag_OE"].mean()
ot_fold = df.loc[df.condition == "napM-KO", "ori_ter_OE"].mean() / df.loc[df.condition == "WT", "ori_ter_OE"].mean()
med_fold = x_ko.mean() / x_wt.mean()
groups = ["Arm–arm\njuxtap.", "ori–ter\njuxtap.", "Overall\nexpansion"]
map_folds = [sec_fold, ot_fold, med_fold]
model_folds = [rw.mirror_arm_3D_mean / rk.mirror_arm_3D_mean, rw.ori_ter_3D / rk.ori_ter_3D, rk.Rg / rw.Rg]
xg = np.arange(3)
wbar = 0.34
b1 = axH.bar(xg - wbar / 2, map_folds, wbar, color="#4A5568", edgecolor="black", lw=0.45, label="Hi-C metric (KO/WT)")
b2 = axH.bar(xg + wbar / 2, model_folds, wbar, color="#CBD5E1", edgecolor="black", lw=0.45, label="3D model (KO/WT)")
for b_idx, bars in enumerate((b1, b2)):
    for j_idx, b in enumerate(bars):
        suffix = "†" if (b_idx == 1 and j_idx == 1) else ""
        axH.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.045,
            f"{b.get_height():.2f}{suffix}", ha="center", va="bottom",
            fontsize=7.2, fontweight="bold" if suffix else "normal", zorder=6,
            bbox=dict(boxstyle="square,pad=0.07", facecolor="white", edgecolor="none", alpha=0.92),
        )
axH.axhline(1.0, color="#333333", lw=0.85, ls="--")
axH.set_xticks(xg, groups)
axH.set_ylabel("Fold change (KO/WT)")
axH.set_ylim(0, 2.42)
axH.legend(frameon=False, fontsize=7.0, loc="upper left")
axH.set_title("Hi-C vs 3D-model fold changes\n(† ori–ter direction differs)", loc="left", pad=5, fontweight="bold", color=C_TEXT)
clean_spines(axH)

# Perfectly aligned figure-level panel letters A-H
pB, pC = axB.get_position(), axC_list[0].get_position()
pD, pE = axD.get_position(), axE_list[0].get_position()
pF, pG, pH = axF1.get_position(), axG.get_position(), axH.get_position()
fig.text(0.012, y0 + side + 0.028, "A", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pB.y1 + 0.032, "B", fontsize=13.0, fontweight="bold", va="top")
fig.text(pC.x0 - 0.044, pC.y1 + 0.036, "C", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pD.y1 + 0.032, "D", fontsize=13.0, fontweight="bold", va="top")
fig.text(pE.x0 - 0.046, pE.y1 + 0.036, "E", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pF.y1 + 0.036, "F", fontsize=13.0, fontweight="bold", va="top")
fig.text(pG.x0 - 0.048, pG.y1 + 0.036, "G", fontsize=13.0, fontweight="bold", va="top")
fig.text(pH.x0 - 0.046, pH.y1 + 0.036, "H", fontsize=13.0, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_14.pdf"
png_p = OUT_FIG / "Response_Fig_14.png"
jpg_p = OUT_FIG / "Response_Fig_14.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_14.{pdf,png,jpg}")
