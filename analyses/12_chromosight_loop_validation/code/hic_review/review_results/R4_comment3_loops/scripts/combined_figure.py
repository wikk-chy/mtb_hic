#!/usr/bin/env python
"""Single combined figure for R4 comment 3 — Chromosight loop validation.

Logic flow (16 panels, 4x4):
  Row 1 — Why Chromosight: kernel template (annotated) | HiCCUPS call (no
          focal dot) | Chromosight call (focal dot) | algorithm explanation
  Row 2 — Where the loops are: positions on contact map | distance
          distributions | large-region view with zoom box | zoomed inset
  Row 3 — APA validation: WT | random | +100 kb shift | napM-KO
  Row 4 — Statistics: per-replicate APA | empirical FDR | WT vs KO per-loop |
          enrichment vs shifted null

Run: ../../.venv_cooltools/bin/python scripts/combined_figure.py
(from review_results/R4_comment3_loops/)
"""

import os
import numpy as np
import pandas as pd
import cooler
import cooltools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
HICCUPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")
CS_WT = os.path.join(OUT, "chromosight_out/wt.tsv")
REP_DIR = os.path.join(BASE, "review_data/rep_icedmatrix")
KERNEL_TXT = (os.path.join(BASE, ".venv_cooltools/lib/python3.9/site-packages/"
                           "chromosight/kernels/artificial_template_loops_type1.txt"))

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

C_WT, C_KO, C_RAND, C_HIC = "#0072B2", "#D55E00", "#999999", "#E69F00"
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list("oe", ["#2166ac", "#f7f7f7", "#b2182b"])

print("loading data ...")
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
        rows = np.arange(n - d); cols = rows + d
        vals = np.diag(mat_wt, k=d) / e
        oe_wt[rows, cols] = vals; oe_wt[cols, rows] = vals

cs = pd.read_csv(CS_WT, sep="\t")
cs["mid1"] = ((cs.start1 + cs.end1) // 2).astype(int)
cs["mid2"] = ((cs.start2 + cs.end2) // 2).astype(int)
cs["dist_kb"] = (cs.mid2 - cs.mid1) // 1000
cs["oe"] = [oe_wt[int(r.bin1), int(r.bin2)] if int(r.bin2) < n else np.nan
            for _, r in cs.iterrows()]

hic = pd.read_csv(HICCUPS, sep="\t", header=None,
                  names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                         "name", "score", "strand1", "strand2"])
hic["b1"] = (hic.start1 + hic.end1) // 2 // BINSIZE
hic["b2"] = (hic.start2 + hic.end2) // 2 // BINSIZE

ok = (cs.start1 - FLANK >= 0) & (cs.end2 + FLANK < GENOME)
cs_apa = cs[ok].reset_index(drop=True)

# ---------------- recompute validation stats (same as chromosight_validation.py) ----------------
def bedpe_of(df):
    return pd.DataFrame({"chrom1": "Chromosome", "start1": df.start1, "end1": df.end1,
                         "chrom2": "Chromosome", "start2": df.start2, "end2": df.end2})

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

print("pileup ...")
pile_wt = aggregate(cooltools.pileup(clr_wt, bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_rand = aggregate(cooltools.pileup(clr_wt, rand_bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_shift = aggregate(cooltools.pileup(clr_wt, shift_bp, expected_df=exp_wt, flank=FLANK, nproc=1))
pile_ko = aggregate(cooltools.pileup(clr_ko, bp, expected_df=exp_ko, flank=FLANK, nproc=1))

def center_pixel(m):
    c = m.shape[0] // 2
    return float(m[c, c])

# per-loop O/E + shifted null + KO
real_oe = cs_apa.oe.values[np.isfinite(cs_apa.oe.values)]
shift_oe = np.array([oe_wt[int(r.bin1) + 20, int(r.bin2) + 20]
                     for _, r in cs_apa.iterrows()
                     if int(r.bin2) + 20 < n and np.isfinite(oe_wt[int(r.bin1) + 20, int(r.bin2) + 20])])
ko_oe = []
for _, r in cs_apa.iterrows():
    b1, b2 = int(r.bin1), int(r.bin2)
    e = exp_ko_diag.get(abs(b2 - b1), np.nan)
    ko_oe.append(mat_ko[b1, b2] / e if e and e > 0 else np.nan)
cs_apa["oe_ko"] = ko_oe
okp = np.isfinite(cs_apa.oe) & np.isfinite(cs_apa.oe_ko)
w_stat, w_p = stats.wilcoxon(cs_apa.oe[okp], cs_apa.oe_ko[okp])
u_stat, u_p = stats.mannwhitneyu(real_oe, shift_oe, alternative="greater")

fdr_rows = []
for thr in [1.25, 1.5, 2.0]:
    fp = float((shift_oe > thr).mean())
    tp = int((real_oe > thr).sum())
    fdr_rows.append(dict(threshold=thr, frac_null=fp, frac_real=tp / len(real_oe),
                         fdr=fp * len(real_oe) / max(tp, 1)))
fdr_df = pd.DataFrame(fdr_rows)

# per-replicate APA
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
                    m[i, j] = v; m[j, i] = v
            except ValueError:
                continue
    return m

rep_mats = [load_rep_matrix(os.path.join(REP_DIR, f))
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
w_r_stat, w_r_p = stats.wilcoxon(rep_center, rep_center_rand)

kernel = np.loadtxt(KERNEL_TXT)

# ---------------- combined figure ----------------
print("rendering combined figure ...")
fig = plt.figure(figsize=(16.5, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.34,
                       left=0.055, right=0.985, top=0.92, bottom=0.06)

def panel_label(ax, letter, title, fs=10):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=fs, fontweight="bold")

# ================= ROW 1: Where the loops are =================
# strongest Chromosight loop (used by panels C/D)
best = cs_apa.dropna(subset=["oe"]).sort_values("oe", ascending=False).iloc[0]
bmid1, bmid2 = int(best.mid1), int(best.mid2)
half_w = 25000
sub = oe_wt[(bmid1 - half_w) // BINSIZE:(bmid1 + half_w) // BINSIZE + 1,
            (bmid2 - half_w) // BINSIZE:(bmid2 + half_w) // BINSIZE + 1]
vmax5 = np.nanpercentile(sub, 97)

# A: positions on contact map
ax = fig.add_subplot(gs[0, 0])
ax.imshow(mat_wt, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat_wt, 99),
          interpolation="nearest")
ax.scatter(cs.bin2, cs.bin1, s=24, facecolor="none", edgecolor=C_WT, lw=1.0,
           label=f"Chromosight (n={len(cs)})", alpha=0.95)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.0f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.0f}" for v in xt], fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=8.5)
ax.set_ylabel("Genomic position (Mb)", fontsize=8.5)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
panel_label(ax, "A", "Loop positions on WT contact map", 10)

# B: distance distributions (Chromosight only)
ax = fig.add_subplot(gs[0, 1])
bins_hist = np.arange(0, 1050, 50)
ax.hist(cs.dist_kb, bins=bins_hist, density=True, alpha=0.75,
        color=C_WT, label=f"Chromosight (n={len(cs)})")
ax.axvline(100, color="black", ls=":", lw=0.8)
ax.text(105, ax.get_ylim()[1] * 0.9, "100 kb", fontsize=7.5)
ax.set_xlabel("Anchor distance (kb)", fontsize=8.5)
ax.set_ylabel("Density", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
panel_label(ax, "B", "Anchor distance distribution\n(Chromosight loops span 30–995 kb)", 10)

# C: large-region view of the best loop with zoom box (O/E display)
reg_lo, reg_hi = 1400000, 2900000   # 1.4 - 2.9 Mb square
rb1, rb2 = reg_lo // BINSIZE, reg_hi // BINSIZE
sub_reg = oe_wt[rb1:rb2, rb1:rb2]
ax = fig.add_subplot(gs[0, 2])
vmax_reg = max(1.5, np.nanpercentile(sub_reg, 98))
im = ax.imshow(sub_reg, cmap=cmap_oe, vmin=0.5, vmax=vmax_reg,
               interpolation="nearest")
# anchor positions within the region
ga1 = (bmid1 - reg_lo) / (reg_hi - reg_lo) * sub_reg.shape[0]
ga2 = (bmid2 - reg_lo) / (reg_hi - reg_lo) * sub_reg.shape[0]
ax.axhline(ga1, color=C_WT, ls="--", lw=0.8)
ax.axvline(ga1, color=C_WT, ls="--", lw=0.8)
ax.axhline(ga2, color=C_WT, ls="--", lw=0.8)
ax.axvline(ga2, color=C_WT, ls="--", lw=0.8)
# zoom box around the anchor-crossing corner (bottom-left off-diagonal)
box_half = 80000  # +-80 kb box for visibility
bwp = 2 * box_half / (reg_hi - reg_lo) * sub_reg.shape[0]
ax.add_patch(Rectangle((ga1 - bwp / 2, ga2 - bwp / 2), bwp, bwp,
                       fill=False, edgecolor="black", lw=1.5))
ax.text(ga1 + bwp * 0.9, ga2 - bwp * 0.55, "shown in D", fontsize=8,
        fontweight="bold", ha="left", va="center")
# axis ticks in Mb
nb = sub_reg.shape[0]
tick_pos = np.linspace(0, nb, 4)
tick_lab = [f"{(reg_lo + p / nb * (reg_hi - reg_lo)) / 1e6:.1f}" for p in tick_pos]
ax.set_xticks(tick_pos, tick_lab, fontsize=8)
ax.set_yticks(tick_pos, tick_lab, fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=8.5)
panel_label(ax, "C", f"Loop context: {bmid1//1000} kb × {bmid2//1000} kb\n(dashes = anchors, box = region zoomed in D)", 10)
plt.colorbar(im, ax=ax, shrink=0.65, label="O/E")

# D: zoomed inset (anchor-centred O/E)
ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(sub, cmap=cmap_oe, vmin=1, vmax=vmax5, interpolation="nearest")
ax.add_patch(Circle((sub.shape[1] // 2, sub.shape[0] // 2), radius=2.2,
                    fill=False, edgecolor="black", lw=1.4))
ax.text(sub.shape[1] // 2 + 3, sub.shape[0] // 2, "loop pixel",
        fontsize=8, va="center", fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
panel_label(ax, "D", "Zoom-in of the boxed region (O/E)\n±25 kb around the anchor crossing", 10)
plt.colorbar(im, ax=ax, shrink=0.65, label="O/E")

# ================= ROW 2: APA =================
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50", "-25", "0", "+25", "+50"]
vmax_apa = max(center_pixel(pile_wt), 1.8)
apa_panels = [
    (pile_wt, "E", f"APA: loops on WT\n(n={len(bp)}, centre O/E={center_pixel(pile_wt):.2f})"),
    (pile_rand, "F", f"APA: random pairs on WT\n(n={len(rand_bp)}, centre O/E={center_pixel(pile_rand):.2f})"),
    (pile_shift, "G", f"APA: loops +100 kb shift\n(n={len(shift_bp)}, centre O/E={center_pixel(pile_shift):.2f})"),
    (pile_ko, "H", f"APA: loops on napM-KO\n(n={len(bp)}, centre O/E={center_pixel(pile_ko):.2f})"),
]
for i, (m, letter, title) in enumerate(apa_panels):
    ax = fig.add_subplot(gs[1, i])
    im = ax.imshow(m, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
    c = m.shape[0] // 2
    ax.add_patch(Rectangle((c - 2.5, c - 2.5), 5, 5, fill=False, edgecolor="black", lw=0.9))
    ax.set_xticks(ticks, ticklabels, fontsize=7.5)
    ax.set_yticks(ticks, ticklabels, fontsize=7.5)
    ax.set_xlabel("kb from anchor 1", fontsize=8)
    if i == 0:
        ax.set_ylabel("kb from anchor 2", fontsize=8)
    panel_label(ax, letter, title, 9.5)
    plt.colorbar(im, ax=ax, shrink=0.65, label="agg. O/E")

# ================= ROW 3: Statistics =================
# I: per-replicate APA
ax = fig.add_subplot(gs[2, 0])
xp = np.arange(len(rep_names))
w = 0.38
ax.bar(xp - w / 2, rep_center, w, color=C_WT, edgecolor="black", label="loops")
ax.bar(xp + w / 2, rep_center_rand, w, color=C_RAND, edgecolor="black", label="random")
ax.axhline(1.0, color="black", ls=":", lw=0.7)
ax.set_xticks(xp, rep_names, fontsize=8.5)
ax.set_ylabel("APA centre O/E", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
panel_label(ax, "I", f"Per-replicate APA (4 WT replicates)\nall 1.62–1.69 vs random 1.01–1.13", 9.5)

# J: empirical FDR
ax = fig.add_subplot(gs[2, 1])
xp = np.arange(len(fdr_df))
ax.bar(xp - 0.2, fdr_df.frac_real, 0.4, color=C_WT, edgecolor="black", label="loops")
ax.bar(xp + 0.2, fdr_df.frac_null, 0.4, color=C_RAND, edgecolor="black", label="shifted null")
ax.set_xticks(xp, [f"O/E>{t}" for t in fdr_df.threshold], fontsize=8.5)
ax.set_ylabel("Fraction of positions", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
panel_label(ax, "J", "Empirical FDR (shifted null)\n" +
            "; ".join(f"@{t}: {f:.2f}" for t, f in zip(fdr_df.threshold, fdr_df.fdr)), 9.5)

# K: WT vs KO per-loop
ax = fig.add_subplot(gs[2, 2])
ax.scatter(cs_apa.oe[okp], cs_apa.oe_ko[okp], s=26, color=C_WT, alpha=0.85,
           edgecolor="black", lw=0.3)
lim = max(cs_apa.oe[okp].max(), cs_apa.oe_ko[okp].max()) * 1.12
ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
ax.axhline(1.0, color="lightgrey", lw=0.6)
ax.axvline(1.0, color="lightgrey", lw=0.6)
ax.set_xlabel("Anchor O/E in WT", fontsize=8.5)
ax.set_ylabel("Anchor O/E in napM-KO", fontsize=8.5)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
panel_label(ax, "K", f"napM-KO attenuation\nWilcoxon p = {w_p:.1e}", 9.5)

# L: per-loop enrichment vs shifted null
ax = fig.add_subplot(gs[2, 3])
bins_oe = np.arange(0.8, 3.4, 0.15)
ax.hist(real_oe, bins=bins_oe, density=True, alpha=0.7, color=C_WT,
        label=f"loops (n={len(real_oe)})")
ax.hist(shift_oe, bins=bins_oe, density=True, alpha=0.7, color=C_RAND,
        label=f"+100 kb shifted (n={len(shift_oe)})")
ax.axvline(1.0, color="black", ls="--", lw=0.8)
ax.set_xlabel("Anchor-pixel O/E (WT)", fontsize=8.5)
ax.set_ylabel("Density", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
panel_label(ax, "L", f"Loop enrichment vs shifted null\nMann-Whitney p = {u_p:.1e}", 9.5)

fig.suptitle("Validation of Chromosight-called loops in M. tuberculosis Hi-C (response to Reviewer 4, comment 3)",
             fontsize=13.5, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig_combined_chromosight_validation.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig_combined_chromosight_validation.png"), dpi=300, bbox_inches="tight")
print("saved fig_combined_chromosight_validation")
