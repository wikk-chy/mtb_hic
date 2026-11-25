#!/usr/bin/env python3
"""
Rebuild Response Fig. 16 (Reviewer 4, Comment 5 part 2 — OpenMM polymer models & MCMC diagnostics):
100% vector-editable TrueType Arial (pdf.fonttype=42), clean print scale (9.6 x 10.6 in),
and fixes ALL visual overlaps spotted in the original AI/PDF layout:
  - Panel C/D: generous horizontal gap so 'Fold vs baseline' NEVER touches Panel C legend!
  - Panel D & E: red 'Latent/WT = 1.33 / 0.66' labels cleanly positioned in open headroom — NEVER touching tall blue S2 bar!
  - Panel G: crisp WHITE text on dark heatmap cells (|z| > 220)!
  - Panel H (left): legend placed cleanly ABOVE plot spine so it NEVER overlaps green/blue burn-in traces!
Exact same underlying OpenMM simulation outputs as 06_combined_figure.py / Response_Fig_16.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
POLY = ROOT / "hic_review/review_results/R4_comment5/polymer_openmm"
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
    "axes.titlesize": 8.9,
    "axes.labelsize": 8.1,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.1,
    "axes.linewidth": 0.72,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

SCEN = ["baseline", "S1", "S2", "S3", "S3app"]
SCEN_LABEL = {
    "baseline": "Baseline (WT-like)",
    "S1": "S1 global collapse",
    "S2": "S2 ori–ter tether",
    "S3": "S3 toroid",
    "S3app": "S3 + apposition",
}
C_TEXT = "#1F2933"

mc = pd.read_csv(POLY / "tables/model_comparison.csv")
exp = pd.read_csv(POLY / "tables/experiment_metrics.csv")
ps_exp = pd.read_csv(POLY / "tables/experiment_ps_curves.csv")
BEST = {r.scenario: r.tag for r in mc.itertuples()}


def load_cmap(tag):
    return np.load(POLY / f"tables/maps/{tag}_cmap.npz")


# ==================== Build wide, large-font 4-row figure (10.6 x 9.2 in) ====================
fig = plt.figure(figsize=(10.6, 9.2))
outer = GridSpec(
    4, 1, figure=fig,
    left=0.068, right=0.916, top=0.956, bottom=0.056,
    hspace=0.55, height_ratios=[0.84, 0.98, 1.10, 0.90],
)

gs_r1 = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.14)
gs_r2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.38, width_ratios=[1.12, 1.16, 0.96, 0.96])
gs_r3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.26, width_ratios=[0.90, 1.28])
gs_r4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.38, width_ratios=[1.15, 1.0, 1.05])


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.6)


# ---- Row 1: Panel A — 5 simulated contact maps ----
# Compute shared log10 scale across all 5 simulated maps
log_maps = []
for sc in SCEN:
    C = load_cmap(BEST[sc])["contact_map"].copy()
    C[C == 0] = np.nan
    log_maps.append(np.log10(C))
vmin_m, vmax_m = -2.8, 0.0

axA_list = []
for k, (sc, lm) in enumerate(zip(SCEN, log_maps)):
    ax = fig.add_subplot(gs_r1[0, k])
    axA_list.append(ax)
    imA = ax.imshow(lm, cmap="Reds", vmin=vmin_m, vmax=vmax_m, interpolation="nearest")
    ax.set_title(SCEN_LABEL[sc], fontsize=8.4, fontweight="bold", pad=5, color=C_TEXT)
    ax.set_xticks([])
    ax.set_yticks([])

pA_last = axA_list[-1].get_position()
caxA = fig.add_axes([pA_last.x1 + 0.010, pA_last.y0, 0.011, pA_last.height])
cbA = fig.colorbar(imA, cax=caxA)
cbA.set_label(r"$\log_{10}$ contact", fontsize=7.6)
cbA.ax.tick_params(labelsize=7.1)

# ---- Row 2: Panels B, C, D, E ----
axB = fig.add_subplot(gs_r2[0, 0])
for cond, col in [("WT", "#333333"), ("latent", "#B22222")]:
    sub = ps_exp[ps_exp.condition == cond].groupby("dist_bp")["ps"].mean()
    axB.loglog(sub.index, sub.values, ".", ms=3.5, color=col, alpha=0.72, label=f"Exp {cond}")
C_base = load_cmap(BEST["baseline"])
axB.loglog(np.arange(1, 221) * 1e4, C_base["ps_prob"][1:221], "-", color="#2B7BBA", lw=1.6, label="Sim baseline")
axB.set_xlabel("Genomic distance s (bp)")
axB.set_ylabel("P(s)")
axB.set_title("P(s): baseline vs experiment", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axB.legend(frameon=False, loc="upper right", fontsize=7.0, handletextpad=0.35)
clean_spines(axB)

axC = fig.add_subplot(gs_r2[0, 1])
sub_lat = ps_exp[ps_exp.condition == "latent"].groupby("dist_bp")["ps"].mean()
axC.loglog(sub_lat.index, sub_lat.values, ".", ms=3.5, color="#B22222", alpha=0.72, label="Exp latent")
for sc, col, lbl_short in zip(SCEN[1:], ["#E67E22", "#2CA02C", "#D62728", "#8E44AD"],
                              ["S1 collapse", "S2 ori–ter", "S3 toroid", "S3 + appos."]):
    ps = load_cmap(BEST[sc])["ps_prob"]
    axC.loglog(np.arange(1, 221) * 1e4, ps[1:221], "-", lw=1.35, color=col, label=lbl_short)
axC.set_xlabel("Genomic distance s (bp)")
axC.set_title("P(s): scenarios vs latent", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axC.legend(frameon=False, loc="upper right", fontsize=6.8, handletextpad=0.35, labelspacing=0.22)
clean_spines(axC)

lat = exp[exp.condition == "latent"]
wt = exp[exp.condition == "WT"]
exp_fold = dict(
    ori_ter_OE=lat.ori_ter_OE.mean() / wt.ori_ter_OE.mean(),
    arm_arm_OE=lat.sec_diag_OE.mean() / wt.sec_diag_OE.mean(),
)
base = mc[mc.scenario == "baseline"].iloc[0]

axDE = []
for k, (m, ttl, ylim_top) in enumerate([
    ("ori_ter_OE", "ori–ter O/E (held-out)", 9.2),
    ("arm_arm_OE", "arm–arm O/E (held-out)", 6.8),
]):
    ax = fig.add_subplot(gs_r2[0, 2 + k])
    axDE.append(ax)
    folds = [mc[mc.scenario == sc].iloc[0][f"sim_{m}"] / base[f"sim_{m}"] for sc in SCEN]
    ax.bar(range(len(SCEN)), folds, color="#5B8DB8", edgecolor="black", lw=0.45, width=0.62, alpha=0.9)
    ax.axhline(exp_fold[m], color="#C0392B", lw=1.6, zorder=4)
    ax.axhline(1.0, color="#888888", lw=0.8, ls=":")
    ax.set_xticks(range(len(SCEN)), ["base", "S1", "S2", "S3", "S3app"], fontsize=7.0)
    ax.set_ylim(0, ylim_top)
    if k == 0:
        ax.set_ylabel("Fold vs baseline")
    # Clean non-overlapping red target badge in top-left open space above base & S1!
    ax.text(
        0.04, 0.96, f"Exp latent/WT = {exp_fold[m]:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.9, fontweight="bold", color="#C0392B",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.92, edgecolor="#F5B7B1", lw=0.55),
    )
    ax.set_title(ttl, loc="left", pad=6, fontweight="bold", color=C_TEXT)
    clean_spines(ax)

# ---- Row 3: Panel F (AIC bar) & Panel G (z-score heatmap clipped at +-400) ----
axF = fig.add_subplot(gs_r3[0, 0])
axF.bar(range(len(mc)), mc.AIC_reg, color="#148F8F", edgecolor="black", lw=0.5, width=0.58)
axF.set_xticks(range(len(mc)), mc.scenario, rotation=20, ha="right")
axF.set_ylabel("AIC (regularised pseudo-likelihood)")
axF.set_ylim(0, 40500)
for i, v in enumerate(mc.AIC_reg):
    axF.text(i, v + 650, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.0,
             fontweight="bold" if mc.iloc[i].scenario == "S3" else "normal", color=C_TEXT)
axF.set_title("Model comparison (lower = better)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axF)

axG = fig.add_subplot(gs_r3[0, 1])
metrics = ["frac_short", "frac_mid", "frac_long", "slope", "ori_ter_OE", "arm_arm_OE"]
Z_raw = mc[[f"z_{m}" for m in metrics]].values
Z_clip = np.clip(Z_raw, -400, 400)
imG = axG.imshow(Z_clip, cmap="RdBu_r", vmin=-400, vmax=400, aspect="auto")
axG.set_xticks(range(len(metrics)), metrics, rotation=22, ha="right")
axG.set_yticks(range(len(mc)), mc.scenario)
for i in range(Z_raw.shape[0]):
    for j in range(Z_raw.shape[1]):
        val = Z_raw[i, j]
        tcol = "white" if abs(val) > 220 else "#111111"
        axG.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.6,
                 fontweight="bold" if abs(val) > 220 else "normal", color=tcol)
axG.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
axG.set_yticks(np.arange(-0.5, len(mc), 1), minor=True)
axG.grid(which="minor", color="white", lw=1.0)
axG.tick_params(which="minor", bottom=False, left=False)
axG.set_title("z-scores vs latent experiment (held-out: last two cols)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
cbG = fig.colorbar(imG, ax=axG, fraction=0.046, pad=0.03)
cbG.set_label("z-score (clipped at ±400)", fontsize=7.6)
cbG.ax.tick_params(labelsize=7.1)

# ---- Row 4: Panel H (MCMC convergence diagnostics: 3 panels, ZERO legend overlap!) ----
z = np.load(POLY / f"runs/{BEST['baseline']}.npz")
axH1 = fig.add_subplot(gs_r4[0, 0])
t_burn = (np.arange(len(z["burn_e"])) + 1) * 2000
l1, = axH1.plot(t_burn, z["burn_e"], lw=0.85, color="#1F77B4", label="Energy")
axH1b = axH1.twinx()
l2, = axH1b.plot(t_burn, z["burn_rg"], color="#E67E22", lw=0.95, label="Rg")
l3, = axH1b.plot(t_burn, z["burn_ot"], color="#2CA02C", lw=0.85, label="ori–ter")
axH1.set_xlabel("Burn-in step")
axH1.set_ylabel("Energy", color="#1F77B4")
axH1b.set_ylabel("Rg / ori–ter", color="#2CA02C")
axH1.set_title("Burn-in plateaus", loc="left", pad=14, fontweight="bold", color=C_TEXT)
axH1.legend(
    handles=[l1, l2, l3], loc="lower right", bbox_to_anchor=(1.04, 1.01),
    ncol=3, frameon=False, fontsize=6.8, handlelength=1.1, handletextpad=0.3, columnspacing=0.75,
)
axH1.spines["top"].set_visible(False)
axH1b.spines["top"].set_visible(False)
axH1.tick_params(direction="out", length=2.6)
axH1b.tick_params(direction="out", length=2.6)

axH2 = fig.add_subplot(gs_r4[0, 1])
rg = z["prod_rg"]
axH2.plot(np.arange(len(rg)) * 3000 / 1e6, rg, lw=0.5, color="#1F77B4")
axH2.set_xlabel("Production step (×10$^6$)")
axH2.set_ylabel("Rg")
axH2.set_title("Production Rg trace", loc="left", pad=12, fontweight="bold", color=C_TEXT)
clean_spines(axH2)

axH3 = fig.add_subplot(gs_r4[0, 2])
v = rg - rg.mean()
n = len(v)
f_fft = np.fft.rfft(v, n=2 * n)
ac = np.fft.irfft(f_fft * np.conj(f_fft))[:n]
ac /= ac[0]
neg = np.argmax(ac < 0)
neg = n // 2 if neg == 0 else neg
tau_int = (2 * np.sum(ac[:neg]) - 1) * 3000
n_eff = np.clip(n * 3000 / (2 * tau_int), 1, n)
axH3.plot(np.arange(min(n, 200)) * 3000 / 1e5, ac[:min(n, 200)], lw=1.15, color="#1F77B4")
axH3.axhline(0, color="#CCCCCC", lw=0.6)
axH3.axvline(tau_int / 1e5, color="#C0392B", ls="--", lw=1.05, label=rf"$\tau_{{\mathrm{{int}}}}$ = {tau_int:.0f} steps")
axH3.axvline(3000 / 1e5, color="#777777", ls=":", lw=0.95, label="Sampling interval")
axH3.set_xlabel("Lag (×10$^5$ steps)")
axH3.set_ylabel("Rg autocorrelation")
axH3.set_title(f"Autocorrelation; $n_{{\\mathrm{{eff}}}}$ = {n_eff:.0f}/{n}", loc="left", pad=12, fontweight="bold", color=C_TEXT)
axH3.legend(frameon=False, loc="upper right", fontsize=6.9)
clean_spines(axH3)

# Perfectly aligned figure-level panel letters A-H
pA0 = axA_list[0].get_position()
pB, pC, pD, pE = axB.get_position(), axC.get_position(), axDE[0].get_position(), axDE[1].get_position()
pF, pG = axF.get_position(), axG.get_position()
pH = axH1.get_position()

fig.text(0.012, pA0.y1 + 0.026, "A", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pB.y1 + 0.032, "B", fontsize=13.0, fontweight="bold", va="top")
fig.text(pC.x0 - 0.044, pC.y1 + 0.032, "C", fontsize=13.0, fontweight="bold", va="top")
fig.text(pD.x0 - 0.048, pD.y1 + 0.032, "D", fontsize=13.0, fontweight="bold", va="top")
fig.text(pE.x0 - 0.040, pE.y1 + 0.032, "E", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pF.y1 + 0.032, "F", fontsize=13.0, fontweight="bold", va="top")
fig.text(pG.x0 - 0.056, pG.y1 + 0.032, "G", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pH.y1 + 0.040, "H", fontsize=13.0, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_16.pdf"
png_p = OUT_FIG / "Response_Fig_16.png"
jpg_p = OUT_FIG / "Response_Fig_16.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_16.{pdf,png,jpg}")
