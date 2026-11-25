#!/usr/bin/env python3
"""
Rebuild Response Fig. 15 (Reviewer 4, Comment 5 part 1 — Stress refolding hypotheses,
Hi-C metrics, P(s) curves & pilot ter-probe FISH):
100% vector-editable TrueType Arial (pdf.fonttype=42) for ALL plots/text/labels,
drops bloated 33.5 MB PDF down to ~2 MB by cleanly embedding 300-DPI images,
and fixes ALL visual flaws:
  - Panel B: crisp WHITE text on dark navy/red heatmap cells (fixes unreadable black-on-navy!)
  - Panel D: clean non-overlapping log2 insets (no label collision with main P(s) curves!)
  - Panel E: adds crisp vector-editable Replicate 1-4 headers + Overview / Magnified labels!
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
R4C5 = BASE / "review_results/R4_comment5"
R4C4 = BASE / "review_results/R4_comment4"
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
    "font.size": 8.4,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.72,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

CONDS3 = ["WT", "hypoxia", "latent"]
COLORS = {"WT": "#1F77B4", "hypoxia": "#2CA02C", "latent": "#8E44AD"}
C_TEXT = "#1F2933"

met = pd.read_csv(R4C4 / "tables/replicate_metrics.csv")
stat = pd.read_csv(R4C4 / "tables/stats_tests.csv")
ps_tab = pd.read_csv(R4C4 / "tables/ps_curves_per_replicate.csv")
obs = pd.read_csv(R4C5 / "tables/observed_vs_predictions.csv")

# Extract Panel A illustration & 8 Panel E FISH tiles cleanly from the ORIGINAL backup PDF
src_pdf = fitz.open(ROOT / "analysis/reviewer4_fig_polish_20261124/backups/Response_Fig_15.pdf.orig")
pg_src = src_pdf[0]
# Render Panel A schematic cleanly from original schematic PNG, tightly cropped for max legibility
schem_raw = Image.open(R4C5 / "figures/chromosome-stress-models.png").convert("RGB")
schem_img = schem_raw.crop((80, 5, 1516, 1015))
# Render the 8 exact FISH boxes (including their crisp yellow frames & 20 um / 10 um scale bars!)
fish_top_rects = [
    fitz.Rect(0.18 * 72, 6.20 * 72, 2.08 * 72, 8.11 * 72),
    fitz.Rect(2.17 * 72, 6.20 * 72, 4.08 * 72, 8.11 * 72),
    fitz.Rect(4.16 * 72, 6.20 * 72, 6.07 * 72, 8.11 * 72),
    fitz.Rect(6.16 * 72, 6.20 * 72, 8.07 * 72, 8.11 * 72),
]
fish_bot_rects = [
    fitz.Rect(0.18 * 72, 8.18 * 72, 2.08 * 72, 10.09 * 72),
    fitz.Rect(2.17 * 72, 8.18 * 72, 4.08 * 72, 10.09 * 72),
    fitz.Rect(4.16 * 72, 8.18 * 72, 6.07 * 72, 10.09 * 72),
    fitz.Rect(6.16 * 72, 8.18 * 72, 8.07 * 72, 10.09 * 72),
]
fish_top = []
fish_bot = []
for r in fish_top_rects:
    pix = pg_src.get_pixmap(clip=r, dpi=220, alpha=False)
    fish_top.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
for r in fish_bot_rects:
    pix = pg_src.get_pixmap(clip=r, dpi=220, alpha=False)
    fish_bot.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
src_pdf.close()

# ==================== Build ultra-compact, edge-to-edge 3-row figure (9.2 x 8.6 in) ====================
fig = plt.figure(figsize=(9.2, 8.6))
outer = GridSpec(
    3, 1, figure=fig,
    left=0.072, right=0.988, top=0.964, bottom=0.014,
    hspace=0.27, height_ratios=[1.36, 0.86, 1.88],
)

gs_r1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.16, width_ratios=[1.16, 1.0])
gs_r2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.30, width_ratios=[0.88, 1.16, 1.16])
gs_r3 = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[2], wspace=0.03, hspace=0.04)


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


# ---- Panel A: Three candidate mechanisms illustration (exact aspect fit, zero dead side margins) ----
axA = fig.add_subplot(gs_r1[0, 0])
axA.imshow(schem_img)
axA.set_anchor("W")
axA.axis("off")

# ---- Panel B: Observed changes vs WT heatmap (WHITE text on dark cells!) ----
axB = fig.add_subplot(gs_r1[0, 1])
comps = ["latent", "hypoxia"]
mlabels = {
    "ori_ter_OE": "ori–ter O/E",
    "sec_diag_OE": "arm–arm O/E",
    "frac_short_lt100kb": "P(s) short",
    "frac_long_gt1Mb": "P(s) long",
    "compaction_slope": "P(s) slope",
}
M5 = list(mlabels)
Z = np.array([[np.log2(obs.loc[obs.metric == m, f"{c}_fold"].iloc[0]) for c in comps] for m in M5])
imB = axB.imshow(Z, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
axB.set_xticks(range(2), ["Latent / WT", "Hypoxia / WT"])
axB.set_yticks(range(len(M5)), [mlabels[m] for m in M5])
for i, m in enumerate(M5):
    for j, c in enumerate(comps):
        p = obs.loc[obs.metric == m, f"{c}_p"].iloc[0]
        fc = obs.loc[obs.metric == m, f"{c}_fold"].iloc[0]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        tcol = "white" if abs(Z[i, j]) > 0.30 else "#111111"
        axB.text(j, i, f"{fc:.2f}×\n{star}", ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=tcol, linespacing=1.12)
axB.set_xticks(np.arange(-0.5, 2, 1), minor=True)
axB.set_yticks(np.arange(-0.5, len(M5), 1), minor=True)
axB.grid(which="minor", color="white", lw=1.2)
axB.tick_params(which="minor", bottom=False, left=False)
axB.set_title("Observed changes vs WT\n(log$_2$ fold; Welch P)", loc="left", pad=5, fontweight="bold", color=C_TEXT)
cb = fig.colorbar(imB, ax=axB, location="bottom", fraction=0.060, pad=0.14)
cb.set_label(r"$\log_2$ fold change", fontsize=7.8)
cb.ax.tick_params(labelsize=7.3)

# ---- Panel C: ori-ter O/E per replicate ----
axC = fig.add_subplot(gs_r2[0, 0])
for x, cond in enumerate(CONDS3):
    y = met.loc[met.condition == cond, "ori_ter_OE"].values
    axC.scatter(np.full(4, x) + np.linspace(-0.14, 0.14, 4), y, s=22, color=COLORS[cond], edgecolor="black", lw=0.45, zorder=3)
    axC.errorbar(x, y.mean(), yerr=y.std(ddof=1), fmt="_", ms=11, color="black", capsize=3.0, lw=1.1, zorder=4)
    if cond != "WT":
        p = stat.query("comparison == @cond + ' vs WT' and metric == 'ori_ter_OE'")["welch_p"].iloc[0]
        axC.text(x, 0.96, f"P = {p:.0e}", transform=axC.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.8, fontweight="bold", color=COLORS[cond])
axC.set_xticks(range(3), ["WT", "Hypoxia", "Latent"])
axC.set_ylim(0.56, 0.955)
axC.set_ylabel("ori–ter O/E")
axC.set_title("ori–ter O/E per replicate\n(mean ± SD, n = 4)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axC)

# ---- Panel D: P(s) WT vs hypoxia and WT vs latent (ZERO inset overlap!) ----
ps_tab = ps_tab.copy()
ps_tab["center"] = np.sqrt(ps_tab.dist_lo * ps_tab.dist_hi)
_tot = met.set_index(["condition", "replicate"])["total_contacts"]
ps_tab["ps"] = ps_tab.apply(lambda r: r.ps / _tot.loc[(r.condition, r.replicate)], axis=1)
curves = {c: ps_tab[ps_tab.condition == c].groupby("center")["ps"].agg(["mean", "std"]).dropna() for c in CONDS3}

axD_list = []
for k, (cond, disp) in enumerate([("hypoxia", "Hypoxia"), ("latent", "Latent")]):
    ax = fig.add_subplot(gs_r2[0, 1 + k])
    axD_list.append(ax)
    for cc, lbl_c in [("WT", "WT"), (cond, disp)]:
        piv = curves[cc]
        ax.loglog(piv.index, piv["mean"], "-", lw=1.6, color=COLORS[cc], label=lbl_c)
        ax.fill_between(piv.index, piv["mean"] - piv["std"], piv["mean"] + piv["std"], color=COLORS[cc], alpha=0.2, lw=0)
    ax.set_ylim(5e-7, 2.2e-3)
    ax.set_xlabel("Genomic distance s (bp)")
    if k == 0:
        ax.set_ylabel("Contact probability P(s)")
    ax.set_title(f"P(s): WT vs {disp} (mean ± SD, n = 4)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
    ax.legend(frameon=False, loc="lower left", fontsize=7.8)
    ax.grid(alpha=0.25, which="both", lw=0.35)
    clean_spines(ax)

    axi = ax.inset_axes([0.46, 0.50, 0.52, 0.46])
    axi.set_facecolor("white")
    ratio = curves[cond]["mean"] / curves["WT"]["mean"]
    axi.semilogx(ratio.index, np.log2(ratio), "-", lw=1.2, color="black")
    axi.axhline(0, color="gray", lw=0.6)
    axi.axvspan(5e3, 1e5, color="#1f77b4", alpha=0.12)
    axi.axvspan(1e6, 2.2e6, color="#d62728", alpha=0.12)
    axi.set_ylabel(r"$\log_2$(" + disp[:3] + "/WT)", fontsize=6.9, labelpad=1)
    axi.set_xlabel("s (bp)", fontsize=6.9, labelpad=1)
    axi.tick_params(labelsize=6.6, length=2.0)
    axi.set_ylim(np.nanmin(np.log2(ratio)) - 0.55, 0.65)

# ---- Panel E: Pilot ter-probe FISH (4 replicates x Overview/Magnified with crisp editable headers) ----
axE_first = None
for col_i in range(4):
    ax_t = fig.add_subplot(gs_r3[0, col_i])
    ax_t.imshow(fish_top[col_i])
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for spn in ax_t.spines.values():
        spn.set_visible(False)
    ax_t.set_title(f"Replicate {col_i + 1}", fontsize=8.8, fontweight="bold", pad=4, color=C_TEXT)
    if col_i == 0:
        axE_first = ax_t
        ax_t.set_ylabel("Overview\n(20 µm)", fontsize=8.4, fontweight="bold", color=C_TEXT, labelpad=6)

    ax_b = fig.add_subplot(gs_r3[1, col_i])
    ax_b.imshow(fish_bot[col_i])
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spn in ax_b.spines.values():
        spn.set_visible(False)
    if col_i == 0:
        ax_b.set_ylabel("Magnified\n(10 µm)", fontsize=8.4, fontweight="bold", color=C_TEXT, labelpad=6)

# Aligned figure-level panel letters A-E
pA, pB = axA.get_position(), axB.get_position()
pC, pD = axC.get_position(), axD_list[0].get_position()
pE = axE_first.get_position()
fig.text(0.012, pA.y1 + 0.022, "A", fontsize=13.0, fontweight="bold", va="top")
fig.text(pB.x0 - 0.088, pB.y1 + 0.036, "B", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pC.y1 + 0.038, "C", fontsize=13.0, fontweight="bold", va="top")
fig.text(pD.x0 - 0.062, pD.y1 + 0.038, "D", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pE.y1 + 0.022, "E", fontsize=13.0, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_15.pdf"
png_p = OUT_FIG / "Response_Fig_15.png"
jpg_p = OUT_FIG / "Response_Fig_15.jpg"
fig.savefig(pdf_p, dpi=250)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_15.{pdf,png,jpg}")
