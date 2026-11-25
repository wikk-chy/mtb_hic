#!/usr/bin/env python3
"""
Rebuild Response Fig. 13 (Reviewer 4, Comment 3 — Chromosight loop validation & APA):
FIXES critical Type 3 uneditable font issue -> 100% vector-editable TrueType Arial (pdf.fonttype=42),
standard print scale (10.4 x 8.4 in vs previous oversized 16.2 x 11.7 in), NO redundant suptitle,
and ZERO overlaps (fixes Panel B '100 kb', Panel D 'loop pixel', Panel I legend over rep4 bar).
Exact same underlying Chromosight loops, APA pileups, and seed=42 computations.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cooler
import cooltools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from scipy import stats
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
OUT_SRC = BASE / "review_results/R4_comment3_loops"
OUT_FIG = ROOT / "analysis/reviewer4_fig_polish_20261124/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

WT_COOL = str(BASE / "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = str(BASE / "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
CS_WT = OUT_SRC / "chromosight_out/wt.tsv"
REP_DIR = BASE / "review_data/rep_icedmatrix"

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

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
    "font.size": 7.2,
    "axes.titlesize": 7.7,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.68,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

C_WT, C_RAND = "#0072B2", "#94A3B8"
C_TEXT = "#1F2933"
cmap_hic = LinearSegmentedColormap.from_list("hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list("oe", ["#2166ac", "#f7f7f7", "#b2182b"])

clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)
exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)
exp_wt_diag = exp_wt.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()
exp_ko_diag = exp_ko.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()

mat_wt = clr_wt.matrix(sparse=False).fetch("Chromosome")
mat_ko = clr_ko.matrix(sparse=False).fetch("Chromosome")
n = mat_wt.shape[0]

oe_wt = np.full_like(mat_wt, np.nan, dtype=float)
for d in range(n):
    e = exp_wt_diag.get(d, np.nan)
    if e and e > 0:
        rows = np.arange(n - d)
        cols = rows + d
        vals = np.diag(mat_wt, k=d) / e
        oe_wt[rows, cols] = vals
        oe_wt[cols, rows] = vals

cs = pd.read_csv(CS_WT, sep="\t")
cs["mid1"] = ((cs.start1 + cs.end1) // 2).astype(int)
cs["mid2"] = ((cs.start2 + cs.end2) // 2).astype(int)
cs["dist_kb"] = (cs.mid2 - cs.mid1) // 1000
cs["oe"] = [oe_wt[int(r.bin1), int(r.bin2)] if int(r.bin2) < n else np.nan for _, r in cs.iterrows()]

ok = (cs.start1 - FLANK >= 0) & (cs.end2 + FLANK < GENOME)
cs_apa = cs[ok].reset_index(drop=True)


def bedpe_of(df):
    return pd.DataFrame({
        "chrom1": "Chromosome", "start1": df.start1, "end1": df.end1,
        "chrom2": "Chromosome", "start2": df.start2, "end2": df.end2,
    })


bp = bedpe_of(cs_apa)
rand_rows = []
for d in (cs_apa.bin2 - cs_apa.bin1).values:
    lo, hi = FLANK // BINSIZE, (GENOME - FLANK) // BINSIZE - int(d) - 1
    if hi <= lo:
        continue
    b1 = int(rng.integers(lo, hi + 1))
    rand_rows.append(("Chromosome", b1 * BINSIZE, b1 * BINSIZE + BINSIZE,
                      "Chromosome", (b1 + int(d)) * BINSIZE, (b1 + int(d)) * BINSIZE + BINSIZE))
rand_bp = pd.DataFrame(rand_rows, columns=bp.columns)
shift_bp = bp.copy()
shift_bp[["start1", "end1", "start2", "end2"]] += 100000
shift_bp = shift_bp[shift_bp.end2 + FLANK < GENOME].reset_index(drop=True)


def aggregate(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    return np.nanmean(s, axis=0)


pile_wt = aggregate(cooltools.pileup(clr_wt, bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_rand = aggregate(cooltools.pileup(clr_wt, rand_bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_shift = aggregate(cooltools.pileup(clr_wt, shift_bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_ko = aggregate(cooltools.pileup(clr_ko, bp, expected_df=exp_ko, flank=FLANK, nproc=1))


def center_pixel(m):
    c = m.shape[0] // 2
    return float(m[c, c])


real_oe = cs_apa.oe.values[np.isfinite(cs_apa.oe.values)]
# 1,000 span-matched shifted controls (>=100 kb = 20 bins from each original anchor)
rng_shift = np.random.default_rng(SEED)
shift_oe_list = []
for _, r in cs_apa.iterrows():
    i0, j0 = int(r.bin1), int(r.bin2)
    sp = j0 - i0
    valid_starts = [s for s in range(0, n - sp) if min(abs(s - i0), n - abs(s - i0)) >= 20]
    picks = rng_shift.choice(valid_starts, size=1000, replace=True)
    vs = oe_wt[picks, picks + sp]
    shift_oe_list.append(vs[np.isfinite(vs)])
shift_oe = np.concatenate(shift_oe_list)

ko_oe = []
for _, r in cs_apa.iterrows():
    b1, b2 = int(r.bin1), int(r.bin2)
    e = exp_ko_diag.get(abs(b2 - b1), np.nan)
    ko_oe.append(mat_ko[b1, b2] / e if e and e > 0 else np.nan)
cs_apa["oe_ko"] = ko_oe
okp = np.isfinite(cs_apa.oe) & np.isfinite(cs_apa.oe_ko)
_, w_p = stats.wilcoxon(cs_apa.oe[okp], cs_apa.oe_ko[okp])
_, u_p = stats.mannwhitneyu(real_oe, shift_oe, alternative="greater")

fdr_rows = []
for thr in [1.25, 1.5, 2.0]:
    fp = float((shift_oe > thr).mean())
    f_real = float((real_oe > thr).mean())
    fdr_rows.append(dict(threshold=thr, frac_null=fp, frac_real=f_real,
                         fdr=fp / max(f_real, 1e-12)))
fdr_df = pd.DataFrame(fdr_rows)


def load_rep_matrix(path, n_bins=883):
    m = np.zeros((n_bins, n_bins))
    with open(path) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            try:
                i, j, v = int(p[0]) - 1, int(p[1]) - 1, float(p[2])
                if 0 <= i < n_bins and 0 <= j < n_bins:
                    m[i, j] = v
                    m[j, i] = v
            except ValueError:
                continue
    return m


rep_mats = [load_rep_matrix(REP_DIR / f)
            for f in sorted(os.listdir(REP_DIR))
            if f.startswith("wt_") and f.endswith(".matrix")]
rep_names = [f"rep{i+1}" for i in range(len(rep_mats))]


def rep_expected(m):
    exp = {}
    for d in range(m.shape[0]):
        vals = np.diag(m, k=d)
        vals = vals[vals > 0]
        exp[d] = np.mean(vals) if len(vals) else np.nan
    return exp


def apa_on_dense(m, exp_map, loops_df, flank_bins=10):
    stack = []
    for _, r in loops_df.iterrows():
        b1, b2 = int(r.bin1), int(r.bin2)
        if b1 - flank_bins < 0 or b2 + flank_bins >= m.shape[0]:
            continue
        win = m[b1 - flank_bins:b1 + flank_bins + 1, b2 - flank_bins:b2 + flank_bins + 1]
        win_oe = np.full_like(win, np.nan, dtype=float)
        for ii in range(win.shape[0]):
            for jj in range(win.shape[1]):
                d = abs((b1 - flank_bins + ii) - (b2 - flank_bins + jj))
                e = exp_map.get(d, np.nan)
                if e and e > 0:
                    win_oe[ii, jj] = win[ii, jj] / e
        stack.append(win_oe)
    return np.nanmean(np.array(stack), axis=0) if stack else np.full((21, 21), np.nan)


rep_center, rep_center_rand = [], []
rand_bp_bins = rand_bp.assign(bin1=rand_bp.start1 // BINSIZE, bin2=rand_bp.start2 // BINSIZE)
for m in rep_mats:
    e_map = rep_expected(m)
    rep_center.append(center_pixel(apa_on_dense(m, e_map, cs_apa)))
    rep_center_rand.append(center_pixel(apa_on_dense(m, e_map, rand_bp_bins)))

# ==================== Build clean, TrueType-editable 3x4 figure ====================
fig = plt.figure(figsize=(10.6, 8.6))
gs = gridspec.GridSpec(
    3, 4, figure=fig,
    hspace=0.48, wspace=0.42,
    left=0.065, right=0.965, top=0.935, bottom=0.068,
)

best = cs_apa.dropna(subset=["oe"]).sort_values("oe", ascending=False).iloc[0]
bmid1, bmid2 = int(best.mid1), int(best.mid2)
half_w = 25000
sub = oe_wt[(bmid1 - half_w) // BINSIZE:(bmid1 + half_w) // BINSIZE + 1,
            (bmid2 - half_w) // BINSIZE:(bmid2 + half_w) // BINSIZE + 1]
vmax5 = np.nanpercentile(sub, 97)


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.6)


# Row 1: A-D
axA = fig.add_subplot(gs[0, 0])
axA.imshow(mat_wt, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat_wt, 99), interpolation="nearest")
axA.scatter(cs.bin2, cs.bin1, s=16, facecolor="none", edgecolor=C_WT, lw=0.9,
            label="Chromosight loops", alpha=0.95)
xt = np.arange(0, n, 200)
axA.set_xticks(xt, [f"{v*5/1000:.0f}" for v in xt])
axA.set_yticks(xt, [f"{v*5/1000:.0f}" for v in xt])
axA.set_xlabel("Genomic position (Mb)")
axA.set_ylabel("Genomic position (Mb)")
axA.legend(loc="upper right", framealpha=0.92, edgecolor="#DDDDDD", fontsize=6.1, handletextpad=0.3)
axA.set_title("Loop positions on WT map", loc="left", pad=6, fontweight="bold", color=C_TEXT)

axB = fig.add_subplot(gs[0, 1])
bins_hist = np.arange(0, 1050, 50)
axB.hist(cs.dist_kb, bins=bins_hist, density=True, alpha=0.80, color=C_WT,
         edgecolor="white", lw=0.4, label="Chromosight loops")
axB.axvline(100, color="#333333", ls=":", lw=0.9)
ymaxB = axB.get_ylim()[1]
axB.set_ylim(0, ymaxB * 1.18)
axB.text(122, ymaxB * 1.06, "100 kb", fontsize=6.6, color="#333333", va="top", ha="left")
axB.set_xlabel("Anchor distance (kb)")
axB.set_ylabel("Density")
axB.legend(frameon=False, loc="upper right", fontsize=6.3)
axB.set_title("Anchor distance distribution\n(span 30–995 kb)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axB)

reg_lo, reg_hi = 1400000, 2900000
rb1, rb2 = reg_lo // BINSIZE, reg_hi // BINSIZE
sub_reg = oe_wt[rb1:rb2, rb1:rb2]
axC = fig.add_subplot(gs[0, 2])
vmax_reg = max(1.5, np.nanpercentile(sub_reg, 98))
imC = axC.imshow(sub_reg, cmap=cmap_oe, vmin=0.5, vmax=vmax_reg, interpolation="nearest")
ga1 = (bmid1 - reg_lo) / (reg_hi - reg_lo) * sub_reg.shape[0]
ga2 = (bmid2 - reg_lo) / (reg_hi - reg_lo) * sub_reg.shape[0]
axC.axhline(ga1, color=C_WT, ls="--", lw=0.75)
axC.axvline(ga1, color=C_WT, ls="--", lw=0.75)
axC.axhline(ga2, color=C_WT, ls="--", lw=0.75)
axC.axvline(ga2, color=C_WT, ls="--", lw=0.75)
box_half = 80000
bwp = 2 * box_half / (reg_hi - reg_lo) * sub_reg.shape[0]
axC.add_patch(Rectangle((ga1 - bwp / 2, ga2 - bwp / 2), bwp, bwp, fill=False, edgecolor="black", lw=1.3))
axC.text(ga1 + bwp * 0.85, ga2 - bwp * 0.55, "zoomed in D", fontsize=6.3, fontweight="bold",
         ha="left", va="center", bbox=dict(boxstyle="round,pad=0.14", facecolor="white", alpha=0.82, edgecolor="none"))
nb = sub_reg.shape[0]
tick_pos = np.linspace(0, nb, 4)
tick_lab = [f"{(reg_lo + p / nb * (reg_hi - reg_lo)) / 1e6:.1f}" for p in tick_pos]
axC.set_xticks(tick_pos, tick_lab)
axC.set_yticks(tick_pos, tick_lab)
axC.set_xlabel("Genomic position (Mb)")
axC.set_title(f"Loop: {bmid1//1000} × {bmid2//1000} kb\n(dashes = anchors)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
cbC = plt.colorbar(imC, ax=axC, fraction=0.048, pad=0.04)
cbC.set_label("O/E", fontsize=6.6)
cbC.ax.tick_params(labelsize=6.2)

axD = fig.add_subplot(gs[0, 3])
imD = axD.imshow(sub, cmap=cmap_oe, vmin=1, vmax=vmax5, interpolation="nearest")
cx_d, cy_d = sub.shape[1] // 2, sub.shape[0] // 2
axD.add_patch(Circle((cx_d, cy_d), radius=2.1, fill=False, edgecolor="black", lw=1.3))
axD.annotate(
    "loop pixel", xy=(cx_d + 1.5, cy_d - 1.5), xytext=(0.95, 0.92),
    textcoords="axes fraction", ha="right", va="top",
    fontsize=6.5, fontweight="bold", color="black",
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.88, edgecolor="#CCCCCC", lw=0.5),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.85),
)
axD.set_xticks([])
axD.set_yticks([])
axD.set_title("Zoom-in of boxed region\n(±25 kb around crossing)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
cbD = plt.colorbar(imD, ax=axD, fraction=0.048, pad=0.04)
cbD.set_label("O/E", fontsize=6.6)
cbD.ax.tick_params(labelsize=6.2)

# Row 2: APA (E-H)
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50", "-25", "0", "+25", "+50"]
vmax_apa = max(center_pixel(pile_wt), 1.8)
apa_panels = [
    (pile_wt, "E", f"APA: loops on WT\n(centre O/E = {center_pixel(pile_wt):.2f})"),
    (pile_rand, "F", f"APA: random pairs (WT)\n(centre O/E = {center_pixel(pile_rand):.2f})"),
    (pile_shift, "G", f"APA: +100 kb shifted\n(centre O/E = {center_pixel(pile_shift):.2f})"),
    (pile_ko, "H", f"APA: loops on napM-KO\n(centre O/E = {center_pixel(pile_ko):.2f})"),
]
ax_apa = []
for i, (m, letter, title) in enumerate(apa_panels):
    ax = fig.add_subplot(gs[1, i])
    ax_apa.append(ax)
    im = ax.imshow(m, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
    c = m.shape[0] // 2
    ax.add_patch(Rectangle((c - 2.5, c - 2.5), 5, 5, fill=False, edgecolor="black", lw=0.9))
    ax.set_xticks(ticks, ticklabels)
    ax.set_yticks(ticks, ticklabels)
    ax.set_xlabel("kb from anchor 1")
    if i == 0:
        ax.set_ylabel("kb from anchor 2")
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color=C_TEXT)
    cb = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cb.set_label("agg. O/E", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.1)

# Row 3: Statistics (I-L)
axI = fig.add_subplot(gs[2, 0])
xp = np.arange(len(rep_names))
w = 0.36
axI.bar(xp - w / 2, rep_center, w, color=C_WT, edgecolor="black", lw=0.5, label="Loops")
axI.bar(xp + w / 2, rep_center_rand, w, color=C_RAND, edgecolor="black", lw=0.5, label="Random")
axI.axhline(1.0, color="#444444", ls=":", lw=0.75)
axI.set_xticks(xp, rep_names)
axI.set_ylim(0, 2.22)
axI.set_ylabel("APA centre O/E")
axI.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.9, handletextpad=0.35, fontsize=6.4)
axI.set_title("Per-replicate APA (4 WT reps)\n1.62–1.69 vs random 1.01–1.13", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axI)

axJ = fig.add_subplot(gs[2, 1])
xp = np.arange(len(fdr_df))
axJ.bar(xp - 0.19, fdr_df.frac_real, 0.38, color=C_WT, edgecolor="black", lw=0.5, label="Loops")
axJ.bar(xp + 0.19, fdr_df.frac_null, 0.38, color=C_RAND, edgecolor="black", lw=0.5, label="Shifted null (1,000×)")
axJ.set_xticks(xp, [f"O/E>{t}" for t in fdr_df.threshold])
axJ.set_ylim(0, 0.88)
axJ.set_ylabel("Fraction of positions")
axJ.legend(frameon=False, loc="upper right", fontsize=6.1)
fdr_strs = [f"@{t}: {f:.3f}" if f < 0.1 else f"@{t}: {f:.2f}" for t, f in zip(fdr_df.threshold, fdr_df.fdr)]
axJ.set_title("Single-pixel FDR upper bound\n" + "; ".join(fdr_strs),
              loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axJ)

axK = fig.add_subplot(gs[2, 2])
axK.scatter(cs_apa.oe[okp], cs_apa.oe_ko[okp], s=20, color=C_WT, alpha=0.85, edgecolor="black", lw=0.3)
lim = max(cs_apa.oe[okp].max(), cs_apa.oe_ko[okp].max()) * 1.12
axK.plot([0, lim], [0, lim], "--", color="#777777", lw=0.85)
axK.axhline(1.0, color="#CCCCCC", lw=0.6)
axK.axvline(1.0, color="#CCCCCC", lw=0.6)
axK.set_xlabel("Anchor O/E in WT")
axK.set_ylabel("Anchor O/E in napM-KO")
axK.set_xlim(0, lim)
axK.set_ylim(0, lim)
axK.set_title(f"napM-KO attenuation\nWilcoxon P = {w_p:.1e}", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axK)

axL = fig.add_subplot(gs[2, 3])
bins_oe = np.arange(0.5, 3.35, 0.15)
axL.hist(real_oe, bins=bins_oe, density=True, alpha=0.72, color=C_WT, edgecolor="white", lw=0.3,
         label="WT loops")
axL.hist(shift_oe, bins=bins_oe, density=True, alpha=0.62, color=C_RAND, edgecolor="white", lw=0.3,
         label="Shifted null (1,000×)")
axL.axvline(1.0, color="#333333", ls="--", lw=0.8)
axL.set_ylim(0, axL.get_ylim()[1] * 1.24)
axL.set_xlabel("Anchor-pixel O/E (WT)")
axL.set_ylabel("Density")
axL.legend(frameon=False, loc="upper right", fontsize=6.0)
axL.set_title(f"Enrichment vs shifted null\nMann–Whitney P = {u_p:.1e}", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axL)

# Aligned figure-level panel letters A-L
row_axes = [
    [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")],
    [(ax_apa[0], "E"), (ax_apa[1], "F"), (ax_apa[2], "G"), (ax_apa[3], "H")],
    [(axI, "I"), (axJ, "J"), (axK, "K"), (axL, "L")],
]
for row in row_axes:
    for col_i, (ax, ltr) in enumerate(row):
        pos_ax = ax.get_position()
        x_l = 0.015 if col_i == 0 else pos_ax.x0 - 0.044
        fig.text(x_l, pos_ax.y1 + 0.045, ltr, fontsize=11.0, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_13.pdf"
png_p = OUT_FIG / "Response_Fig_13.png"
jpg_p = OUT_FIG / "Response_Fig_13.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_13.{pdf,png,jpg}")
