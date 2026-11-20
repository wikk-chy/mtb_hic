#!/usr/bin/env python
"""FINAL LRPC analysis — Response to Reviewer 4, comment 3.

Defines bacterial Long-range Preferential Chromosomal Contacts (LRPCs),
calls them de novo, and validates with two controls:
  - WT distance-matched random pairs (specificity control)
  - napM-KO matrix (biological control: NapM-dependent contacts should
    attenuate in the knockout)

LRPC operational definition (4 criteria, all computed on the WT matrix):
  1. Anchor distance 200 kb - 1.5 Mb (beyond the trivial near-diagonal regime)
  2. Center-pixel O/E >= 1.4 vs distance-matched expected (cooltools
     expected_cis on the balanced matrix)
  3. Focality: O/E(anchor) >= 1.25 x mean O/E of four diagonal-shifted
     positions (anchors +/-100 kb, +/-200 kb) -> focal, not regional
  4. Reproducibility: anchor pixel > 1.5x local donut background in >= 3 of
     4 WT replicates

Then: convergent overlap with HiCCUPS long-range calls and Chromosight
(bacteria-validated caller, Matthey-Doret et al. 2020) long-range calls.

Outputs one combined main figure + one supplementary Chromosight figure.

Run: ../.venv_cooltools/bin/python final_lrpc_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import cooler
import cooltools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
LOOPS_HICCUPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")
LOOPS_CHROMOSIGHT = os.path.join(OUT, "chromosight_out/wt.tsv")
REP_DIR = os.path.join(BASE, "review_data/wt_rep_icedmatrix")

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

# Okabe-Ito colorblind-safe palette
C_WT = "#0072B2"      # blue
C_KO = "#D55E00"      # vermillion
C_RAND = "#999999"    # grey
C_LRPC = "#009E73"    # bluish green
C_HIC = "#E69F00"     # orange
C_CS = "#CC79A7"      # reddish purple

# Hi-C style heatmap: white -> dark red
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])

print("=" * 60)
print("FINAL LRPC ANALYSIS")
print("=" * 60)

# ---------------- load ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)
exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)
exp_wt_diag = exp_wt.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()
exp_ko_diag = exp_ko.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()

mat_wt = clr_wt.matrix(sparse=False).fetch("Chromosome")
mat_ko = clr_ko.matrix(sparse=False).fetch("Chromosome")
n = mat_wt.shape[0]

def oe_matrix(mat, exp_diag):
    oe = np.full_like(mat, np.nan, dtype=float)
    for d in range(n):
        e = exp_diag.get(d, np.nan)
        if e and e > 0:
            rows = np.arange(n - d)
            cols = rows + d
            vals = np.diag(mat, k=d) / e
            oe[rows, cols] = vals
            oe[cols, rows] = vals
    return oe

print("building O/E matrices ...")
oe_wt = oe_matrix(mat_wt, exp_wt_diag)
oe_ko = oe_matrix(mat_ko, exp_ko_diag)

# ---------------- de novo LRPC calling ----------------
print("\n[1] de novo LRPC calling ...")
D_MIN, D_MAX = 40, 300          # bins: 200 kb .. 1.5 Mb
OE_THR = 2.0                    # anchor-pixel O/E vs distance-matched expected
FOCAL_THR = 1.5                 # focality vs +-100/+-200 kb diagonal shifts
NMS_RADIUS = 5                  # +-25 kb non-maximum suppression
MIN_REPS = 3                    # reproducible in >=3/4 WT replicates
SHIFTS = [(-20, -20), (20, 20), (-40, -40), (40, 40)]  # +-100kb, +-200kb in bins

cands = []
for d in range(D_MIN, D_MAX + 1):
    for i in range(n - d):
        j = i + d
        v = oe_wt[i, j]
        if not np.isfinite(v) or v < OE_THR:
            continue
        # focality vs diagonal shifts
        shift_vals = []
        for di, dj in SHIFTS:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and np.isfinite(oe_wt[ni, nj]):
                shift_vals.append(oe_wt[ni, nj])
        if len(shift_vals) < 3:
            continue
        focal = v / np.mean(shift_vals)
        if focal < FOCAL_THR:
            continue
        cands.append(dict(bin1=i, bin2=j, oe=v, focal=focal, dist_bins=d))

cands = pd.DataFrame(cands)
print(f"candidates after O/E>={OE_THR} & focality>={FOCAL_THR}: {len(cands)}")

# non-maximum suppression (+-3 bins)
def nonmax(df, radius=3):
    df = df.sort_values("oe", ascending=False).reset_index(drop=True)
    keep, used = [], np.zeros(len(df), dtype=bool)
    b1, b2 = df.bin1.values, df.bin2.values
    for k in range(len(df)):
        if used[k]:
            continue
        keep.append(k)
        used |= (np.abs(b1 - b1[k]) <= radius) & (np.abs(b2 - b2[k]) <= radius)
    return df.iloc[keep].reset_index(drop=True)

lrpc = nonmax(cands, radius=NMS_RADIUS)
lrpc["start1"] = lrpc.bin1 * BINSIZE
lrpc["start2"] = lrpc.bin2 * BINSIZE
lrpc["dist_kb"] = lrpc.dist_bins * BINSIZE // 1000
print(f"after NMS: {len(lrpc)}")

# ---------------- criterion 4: replicate reproducibility ----------------
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

rep_files = sorted([f for f in os.listdir(REP_DIR) if f.endswith(".matrix")])
rep_mats = [load_rep_matrix(os.path.join(REP_DIR, f)) for f in rep_files]
print(f"loaded {len(rep_mats)} replicates")

def rep_score(m, b1, b2):
    val = m[b1, b2]
    bg = []
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if abs(di) <= 1 and abs(dj) <= 1:
                continue
            ni, nj = b1 + di, b2 + dj
            if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
                bg.append(m[ni, nj])
    bgm = np.mean(bg) if bg else np.nan
    return val / bgm if bgm and bgm > 0 else np.nan

n_rep_pass = []
for _, r in lrpc.iterrows():
    cnt = 0
    for m in rep_mats:
        s = rep_score(m, int(r.bin1), int(r.bin2))
        if np.isfinite(s) and s > 1.5:
            cnt += 1
    n_rep_pass.append(cnt)
lrpc["n_reps_pass"] = n_rep_pass
lrpc["reproducible"] = lrpc.n_reps_pass >= MIN_REPS
lrpc_final = lrpc[lrpc.reproducible].reset_index(drop=True)
print(f"LRPCs passing >={MIN_REPS}/4 replicates: {len(lrpc_final)}")

# KO O/E for each LRPC
lrpc_final["oe_ko"] = [oe_ko[int(r.bin1), int(r.bin2)] for _, r in lrpc_final.iterrows()]
lrpc_final.to_csv(os.path.join(OUT, "tables/lrpc_final_set.csv"), index=False)

# ---------------- convergent overlap ----------------
hic = pd.read_csv(LOOPS_HICCUPS, sep="\t", header=None,
                  names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                         "name", "score", "strand1", "strand2"])
hic["c1"] = (hic.start1 + hic.end1) // 2 // BINSIZE
hic["c2"] = (hic.start2 + hic.end2) // 2 // BINSIZE
hic_long = hic[(hic.c2 - hic.c1) >= D_MIN]

cs = pd.read_csv(LOOPS_CHROMOSIGHT, sep="\t")
cs_long = cs[(cs.bin2 - cs.bin1) >= D_MIN]

def count_overlap(query_bins, ref_bins, tol=2):
    return sum(1 for q1, q2 in query_bins
               if any(abs(q1 - r1) <= tol and abs(q2 - r2) <= tol for r1, r2 in ref_bins))

lrpc_bins = list(zip(lrpc_final.bin1, lrpc_final.bin2))
hic_bins = list(zip(hic_long.c1, hic_long.c2))
cs_bins = list(zip(cs_long.bin1, cs_long.bin2))

overlap = {
    "LRPC_total": len(lrpc_final),
    "HiCCUPS_long_n": len(hic_long),
    "Chromosight_long_n": len(cs_long),
    "LRPC_overlap_HiCCUPS": count_overlap(lrpc_bins, hic_bins),
    "LRPC_overlap_Chromosight": count_overlap(lrpc_bins, cs_bins),
    "HiCCUPS_overlap_Chromosight": count_overlap(hic_bins, cs_bins),
}
print("overlap:", overlap)

# ---------------- APA ----------------
def bedpe_of_bins(df):
    return pd.DataFrame({
        "chrom1": "Chromosome", "start1": df.bin1 * BINSIZE, "end1": df.bin1 * BINSIZE + BINSIZE,
        "chrom2": "Chromosome", "start2": df.bin2 * BINSIZE, "end2": df.bin2 * BINSIZE + BINSIZE})

def aggregate(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    return np.nanmean(s, axis=0)

def center_pixel(m):
    c = m.shape[0] // 2
    return float(m[c, c])

def center5(m):
    c = m.shape[0] // 2
    return float(np.nanmean(m[c-2:c+3, c-2:c+3]))

# keep LRPCs whose +-50 kb window fits
ok = (lrpc_final.start1 - FLANK >= 0) & (lrpc_final.start2 + FLANK < GENOME)
lrpc_apa = lrpc_final[ok].reset_index(drop=True)
lrpc_bedpe = bedpe_of_bins(lrpc_apa)
print(f"LRPCs usable for APA: {len(lrpc_apa)}")

# random distance-matched control
rand_rows = []
for d in lrpc_apa.dist_bins.values:
    lo = FLANK // BINSIZE
    hi = (GENOME - FLANK) // BINSIZE - int(d) - 1
    if hi <= lo:
        continue
    b1 = int(rng.integers(lo, hi + 1))
    rand_rows.append(("Chromosome", b1 * BINSIZE, b1 * BINSIZE + BINSIZE,
                      "Chromosome", (b1 + int(d)) * BINSIZE, (b1 + int(d)) * BINSIZE + BINSIZE))
rand_bedpe = pd.DataFrame(rand_rows, columns=lrpc_bedpe.columns)

print("pileup ...")
st_lrpc_wt = cooltools.pileup(clr_wt, lrpc_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
st_rand_wt = cooltools.pileup(clr_wt, rand_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
st_lrpc_ko = cooltools.pileup(clr_ko, lrpc_bedpe, expected_df=exp_ko, flank=FLANK, nproc=1)

pile_lrpc_wt = aggregate(st_lrpc_wt)
pile_rand_wt = aggregate(st_rand_wt)
pile_lrpc_ko = aggregate(st_lrpc_ko)

np.save(os.path.join(OUT, "tables/apa_lrpc_wt.npy"), pile_lrpc_wt)
np.save(os.path.join(OUT, "tables/apa_lrpc_random.npy"), pile_rand_wt)
np.save(os.path.join(OUT, "tables/apa_lrpc_ko.npy"), pile_lrpc_ko)

apa_scores = pd.DataFrame([
    dict(condition="LRPCs on WT", n=len(lrpc_apa), center_pixel=center_pixel(pile_lrpc_wt), center_5x5=center5(pile_lrpc_wt)),
    dict(condition="random on WT", n=len(rand_bedpe), center_pixel=center_pixel(pile_rand_wt), center_5x5=center5(pile_rand_wt)),
    dict(condition="LRPCs on napM-KO", n=len(lrpc_apa), center_pixel=center_pixel(pile_lrpc_ko), center_5x5=center5(pile_lrpc_ko)),
])
apa_scores.to_csv(os.path.join(OUT, "tables/apa_lrpc_scores.csv"), index=False)
print(apa_scores.round(3))

# per-LRPC WT vs KO stats — use the DIRECT anchor-pixel O/E (center pixel),
# NOT the 5x5 window mean: focal contacts are diluted by window averaging.
pl_wt = lrpc_apa.oe.values.astype(float)
pl_ko = lrpc_apa.oe_ko.values.astype(float)
okp = np.isfinite(pl_wt) & np.isfinite(pl_ko)
w_stat, w_p = stats.wilcoxon(pl_wt[okp], pl_ko[okp])
print(f"per-LRPC anchor-pixel O/E: WT median={np.nanmedian(pl_wt):.3f}, "
      f"KO median={np.nanmedian(pl_ko):.3f}, "
      f"Wilcoxon p={w_p:.2e} (n={okp.sum()})")

# ---------------- MAIN COMBINED FIGURE ----------------
print("\nrendering main figure ...")
fig = plt.figure(figsize=(16, 8.5))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.06, right=0.97, top=0.90, bottom=0.10)

# --- A: O/E vs distance scatter, all callers ---
ax = fig.add_subplot(gs[0, 0])
# HiCCUPS all
hic_oe = []
for _, r in hic.iterrows():
    v = oe_wt[int(r.c1), int(r.c2)]
    hic_oe.append(v)
hic["oe"] = hic_oe
hic["dist_kb2"] = (hic.c2 - hic.c1) * BINSIZE / 1000
ax.scatter(hic.dist_kb2, hic.oe, s=22, c=C_HIC, alpha=0.75, edgecolor="none",
           label=f"HiCCUPS (n={len(hic)})", zorder=2)
# Chromosight all
cs["oe"] = [oe_wt[int(r.bin1), int(r.bin2)] if int(r.bin1) < n and int(r.bin2) < n else np.nan
            for _, r in cs.iterrows()]
cs["dist_kb2"] = (cs.bin2 - cs.bin1) * BINSIZE / 1000
ax.scatter(cs.dist_kb2, cs.oe, s=22, c=C_CS, alpha=0.75, edgecolor="none",
           label=f"Chromosight (n={len(cs)})", zorder=3)
# LRPC final
ax.scatter(lrpc_final.dist_kb, lrpc_final.oe, s=26, c=C_LRPC, alpha=0.95,
           edgecolor="black", lw=0.4, label=f"LRPC (n={len(lrpc_final)})", zorder=4)
ax.axhline(1.0, color="black", ls="--", lw=0.8, zorder=1)
ax.axhline(1.4, color="grey", ls=":", lw=0.8, zorder=1)
ax.axvline(200, color="grey", ls=":", lw=0.8, zorder=1)
ax.set_xlabel("Anchor distance (kb)", fontsize=9)
ax.set_ylabel("Center-pixel O/E (WT)", fontsize=9)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold", x=-0.18)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_ylim(0.5, 6.5)
ax.text(205, 6.0, "LRPC regime\n(>200 kb)", fontsize=7.5, color="dimgrey")

# --- B: APA LRPC on WT ---
vmax_apa = max(center_pixel(pile_lrpc_wt), 2.0)
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50", "-25", "0", "+25", "+50"]

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(pile_lrpc_wt, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
c = pile_lrpc_wt.shape[0] // 2
ax.add_patch(Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_xticks(ticks, ticklabels, fontsize=8)
ax.set_yticks(ticks, ticklabels, fontsize=8)
ax.set_title(f"LRPCs on WT\n(n={len(lrpc_apa)}, center O/E={center_pixel(pile_lrpc_wt):.2f})", fontsize=9)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold", x=-0.25)
ax.set_xlabel("kb from anchor 1", fontsize=8)
ax.set_ylabel("kb from anchor 2", fontsize=8)

# --- C: APA random on WT ---
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(pile_rand_wt, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
ax.add_patch(Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_xticks(ticks, ticklabels, fontsize=8)
ax.set_yticks(ticks, ticklabels, fontsize=8)
ax.set_title(f"Random pairs on WT\n(n={len(rand_bedpe)}, center O/E={center_pixel(pile_rand_wt):.2f})", fontsize=9)
ax.set_title("C", loc="left", fontsize=12, fontweight="bold", x=-0.25)
ax.set_xlabel("kb from anchor 1", fontsize=8)

# --- D: APA LRPC on KO ---
ax = fig.add_subplot(gs[0, 3])
im = ax.imshow(pile_lrpc_ko, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
ax.add_patch(Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_xticks(ticks, ticklabels, fontsize=8)
ax.set_yticks(ticks, ticklabels, fontsize=8)
ax.set_title(f"LRPCs on napM-KO\n(center O/E={center_pixel(pile_lrpc_ko):.2f})", fontsize=9)
ax.set_title("D", loc="left", fontsize=12, fontweight="bold", x=-0.25)
ax.set_xlabel("kb from anchor 1", fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Aggregate O/E", fontsize=8)

# --- E: zoomed map of strongest LRPC, WT vs KO ---
strong = lrpc_final.sort_values("oe", ascending=False).iloc[0]
mid1 = int(strong.start1) + BINSIZE // 2
mid2 = int(strong.start2) + BINSIZE // 2
span = mid2 - mid1
window = min(max(span // 2, 120000), 350000)
r1_s, r1_e = max(0, mid1 - window), mid1 + window
r2_s, r2_e = max(0, mid2 - window), mid2 + window

for k, (mat, cond, col) in enumerate([(mat_wt, "WT", C_WT), (mat_ko, "napM-KO", C_KO)]):
    ax = fig.add_subplot(gs[1, k])
    sub = mat[r1_s // BINSIZE:(r1_e // BINSIZE) + 1, r2_s // BINSIZE:(r2_e // BINSIZE) + 1]
    # fetch with fetch() for correct coords
    sub = clr_wt.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}",
                                            f"Chromosome:{r2_s}-{r2_e}") if cond == "WT" else \
          clr_ko.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}",
                                            f"Chromosome:{r2_s}-{r2_e}")
    im = ax.imshow(sub, cmap=cmap_hic,
                   vmin=np.nanpercentile(sub[sub > 0], 2),
                   vmax=np.nanpercentile(sub, 98), interpolation="nearest")
    ax.axhline((mid2 - r2_s) / (r2_e - r2_s) * sub.shape[0], color="#0072B2", ls="--", lw=1)
    ax.axvline((mid1 - r1_s) / (r1_e - r1_s) * sub.shape[1], color="#0072B2", ls="--", lw=1)
    oe_val = strong.oe if cond == "WT" else strong.oe_ko
    ax.set_title(f"{cond}\nO/E={oe_val:.2f}", fontsize=9)
    ax.set_title("E" if cond == "WT" else "F", loc="left", fontsize=12, fontweight="bold", x=-0.25)
    ax.set_xlabel(f"Anchor 1 region ({mid1//1000} kb)", fontsize=8)
    if cond == "WT":
        ax.set_ylabel(f"Anchor 2 region ({mid2//1000} kb)", fontsize=8)

# --- G: per-LRPC WT vs KO scatter ---
ax = fig.add_subplot(gs[1, 2])
ax.scatter(pl_wt[okp], pl_ko[okp], s=32, c=C_LRPC, alpha=0.9, edgecolor="black", lw=0.4)
lim = max(np.nanmax(pl_wt), np.nanmax(pl_ko)) * 1.15
ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
ax.axhline(1.0, color="lightgrey", lw=0.6)
ax.axvline(1.0, color="lightgrey", lw=0.6)
ax.set_xlabel("Center O/E in WT", fontsize=9)
ax.set_ylabel("Center O/E in napM-KO", fontsize=9)
ax.set_title(f"Wilcoxon p = {w_p:.1e}", fontsize=9)
ax.set_title("G", loc="left", fontsize=12, fontweight="bold", x=-0.25)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)

# --- H: replicate reproducibility + overlap summary ---
ax = fig.add_subplot(gs[1, 3])
labels = ["LRPC\n(this study)", "HiCCUPS\n>200 kb", "HiCCUPS\n<=100 kb"]
# compute recovery for each set
def set_recovery(bin_list):
    rates = []
    for m in rep_mats:
        cnt = 0
        for b1, b2 in bin_list:
            s = rep_score(m, int(b1), int(b2))
            if np.isfinite(s) and s > 1.5:
                cnt += 1
        rates.append(cnt / max(len(bin_list), 1))
    return np.mean(rates) * 100

hic_short = hic[(hic.c2 - hic.c1) < 20]  # <=100 kb
rec_lrpc = set_recovery(lrpc_bins)
rec_hic_long = set_recovery(hic_bins)
rec_hic_short = set_recovery(list(zip(hic_short.c1, hic_short.c2)))
vals = [rec_lrpc, rec_hic_long, rec_hic_short]
cols = [C_LRPC, C_HIC, "#dddddd"]
bars = ax.bar(labels, vals, color=cols, edgecolor="black", lw=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=8.5)
ax.set_ylabel("Mean replicate recovery (%)", fontsize=9)
ax.set_ylim(0, 110)
ax.set_title("Reproducibility (4 WT replicates)", fontsize=9)
ax.set_title("H", loc="left", fontsize=12, fontweight="bold", x=-0.25)
plt.setp(ax.get_xticklabels(), fontsize=8)

fig.suptitle("Validation of long-range preferential chromosomal contacts (LRPCs) in M. tuberculosis Hi-C",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig_main_LRPC_validation.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig_main_LRPC_validation.png"), dpi=300, bbox_inches="tight")

# ---------------- SUPPLEMENTARY: Chromosight evaluation ----------------
print("\nrendering chromosight supplementary figure ...")
# APA of chromosight loops (all + long-range) on WT, random, KO
cs_apa_ok = (cs.start1 - FLANK >= 0) & (cs.start2 + FLANK < GENOME)
cs_apa = cs[cs_apa_ok].reset_index(drop=True)
cs_bedpe = pd.DataFrame({
    "chrom1": "Chromosome", "start1": cs_apa.start1, "end1": cs_apa.end1,
    "chrom2": "Chromosome", "start2": cs_apa.start2, "end2": cs_apa.end2})
st_cs_wt = cooltools.pileup(clr_wt, cs_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
st_cs_ko = cooltools.pileup(clr_ko, cs_bedpe, expected_df=exp_ko, flank=FLANK, nproc=1)
pile_cs_wt = aggregate(st_cs_wt)
pile_cs_ko = aggregate(st_cs_ko)

# random for chromosight distances
cs_rand_rows = []
for d in ((cs_apa.bin2 - cs_apa.bin1)).values:
    lo = FLANK // BINSIZE
    hi = (GENOME - FLANK) // BINSIZE - int(d) - 1
    if hi <= lo:
        continue
    b1 = int(rng.integers(lo, hi + 1))
    cs_rand_rows.append(("Chromosome", b1 * BINSIZE, b1 * BINSIZE + BINSIZE,
                         "Chromosome", (b1 + int(d)) * BINSIZE, (b1 + int(d)) * BINSIZE + BINSIZE))
cs_rand_bedpe = pd.DataFrame(cs_rand_rows, columns=cs_bedpe.columns)
st_cs_rand = cooltools.pileup(clr_wt, cs_rand_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
pile_cs_rand = aggregate(st_cs_rand)

fig2 = plt.figure(figsize=(13, 4.2))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.3, left=0.06, right=0.95, top=0.82, bottom=0.14)
panels = [(pile_cs_wt, f"Chromosight loops on WT\n(n={len(cs_apa)}, center O/E={center_pixel(pile_cs_wt):.2f})"),
          (pile_cs_rand, f"Random pairs on WT\n(n={len(cs_rand_bedpe)}, center O/E={center_pixel(pile_cs_rand):.2f})"),
          (pile_cs_ko, f"Chromosight loops on napM-KO\n(center O/E={center_pixel(pile_cs_ko):.2f})")]
vmax2 = max(center_pixel(pile_cs_wt), 2.0)
for i, (m, title) in enumerate(panels):
    ax = fig2.add_subplot(gs2[0, i])
    im = ax.imshow(m, cmap=cmap_hic, vmin=0.9, vmax=vmax2, interpolation="nearest")
    c = m.shape[0] // 2
    ax.add_patch(Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
    ax.set_xticks(ticks, ticklabels, fontsize=8)
    ax.set_yticks(ticks, ticklabels, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.set_title(chr(ord("A") + i), loc="left", fontsize=12, fontweight="bold", x=-0.22)
fig2.colorbar(im, ax=fig2.get_axes(), fraction=0.02, pad=0.02, label="Aggregate O/E")
fig2.suptitle("Chromosight (bacteria-validated caller) loop validation by APA", fontsize=12, fontweight="bold")
fig2.savefig(os.path.join(OUT, "figures/fig_supp_chromosight_APA.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "figures/fig_supp_chromosight_APA.png"), dpi=300, bbox_inches="tight")

cs_scores = pd.DataFrame([
    dict(condition="Chromosight loops on WT", n=len(cs_apa), center_pixel=center_pixel(pile_cs_wt), center_5x5=center5(pile_cs_wt)),
    dict(condition="random on WT", n=len(cs_rand_bedpe), center_pixel=center_pixel(pile_cs_rand), center_5x5=center5(pile_cs_rand)),
    dict(condition="Chromosight loops on napM-KO", n=len(cs_apa), center_pixel=center_pixel(pile_cs_ko), center_5x5=center5(pile_cs_ko)),
])
cs_scores.to_csv(os.path.join(OUT, "tables/apa_chromosight_scores.csv"), index=False)
print(cs_scores.round(3))

# ---------------- genome-wide distribution of all loop sets ----------------
print("\nrendering genome-wide loop distribution figure ...")

fig3 = plt.figure(figsize=(16, 10))
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.32, wspace=0.25,
                        left=0.07, right=0.97, top=0.90, bottom=0.09)

# A: full contact map with loop positions overlaid
ax = fig3.add_subplot(gs3[0, 0])
im = ax.imshow(mat_wt, cmap=cmap_hic,
               vmin=0, vmax=np.nanpercentile(mat_wt, 99), interpolation="nearest")
# HiCCUPS all (small), Chromosight (medium), LRPC (large)
ax.scatter(hic.c2, hic.c1, s=8, facecolor="none", edgecolor=C_HIC, lw=0.7,
           label=f"HiCCUPS (n={len(hic)})", alpha=0.8)
ax.scatter(cs.bin2, cs.bin1, s=22, facecolor="none", edgecolor=C_CS, lw=1.0,
           label=f"Chromosight (n={len(cs)})", alpha=0.9)
ax.scatter(lrpc_final.bin2, lrpc_final.bin1, s=30, facecolor="none", edgecolor=C_LRPC, lw=1.2,
           label=f"LRPC (n={len(lrpc_final)})", alpha=0.95)
ax.set_xlabel("Genomic position (Mb)", fontsize=9)
ax.set_ylabel("Genomic position (Mb)", fontsize=9)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
ax.set_title("A", loc="left", fontsize=12, fontweight="bold", x=-0.15)
ax.set_title("Loop positions on the WT contact map", fontsize=10)

# B: loop distance distributions (KDE/hist)
ax = fig3.add_subplot(gs3[0, 1])
bins_hist = np.arange(0, 1050, 50)
ax.hist(hic.dist_kb2, bins=bins_hist, alpha=0.6, color=C_HIC, label=f"HiCCUPS (n={len(hic)})", density=True)
ax.hist(cs.dist_kb2, bins=bins_hist, alpha=0.6, color=C_CS, label=f"Chromosight (n={len(cs)})", density=True)
ax.hist(lrpc_final.dist_kb, bins=bins_hist, alpha=0.6, color=C_LRPC, label=f"LRPC (n={len(lrpc_final)})", density=True)
ax.axvline(200, color="black", ls="--", lw=0.8)
ax.text(210, ax.get_ylim()[1]*0.9, "200 kb", fontsize=8)
ax.set_xlabel("Anchor distance (kb)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("B", loc="left", fontsize=12, fontweight="bold", x=-0.15)
ax.set_title("Anchor distance distribution", fontsize=10)

# C: arc plot of all three sets (genome linear view)
ax = fig3.add_subplot(gs3[1, 0])
def draw_arcs(ax, df, c1_col, c2_col, color, y0, h=1.0, lw=0.7, alpha=0.6):
    for _, r in df.iterrows():
        x1, x2 = r[c1_col], r[c2_col]
        if x2 < x1:
            x1, x2 = x2, x1
        mid = (x1 + x2) / 2
        w = (x2 - x1) / 2
        theta = np.linspace(0, np.pi, 50)
        ax.plot(mid + w * np.cos(theta), y0 + h * np.sin(theta) * (w / 2e6),
                color=color, lw=lw, alpha=alpha)

# scale arcs by distance
ax.axhline(0, color="black", lw=1.5)
# genome axis in Mb
for _, r in hic.iterrows():
    x1, x2 = r.c1 * 5, r.c2 * 5
    t = np.linspace(0, np.pi, 40)
    ax.plot((x1+x2)/2 + (x2-x1)/2*np.cos(t), 0.15 + (x2-x1)/2*np.sin(t)/1800,
            color=C_HIC, lw=0.8, alpha=0.55)
for _, r in cs.iterrows():
    x1, x2 = r.bin1 * 5, r.bin2 * 5
    t = np.linspace(0, np.pi, 40)
    ax.plot((x1+x2)/2 + (x2-x1)/2*np.cos(t), 0.55 + (x2-x1)/2*np.sin(t)/1800,
            color=C_CS, lw=0.8, alpha=0.6)
for _, r in lrpc_final.iterrows():
    x1, x2 = r.bin1 * 5, r.bin2 * 5
    t = np.linspace(0, np.pi, 40)
    ax.plot((x1+x2)/2 + (x2-x1)/2*np.cos(t), 0.95 + (x2-x1)/2*np.sin(t)/1800,
            color=C_LRPC, lw=0.9, alpha=0.65)

ax.text(-150, 0.25, "HiCCUPS", fontsize=9, color=C_HIC, fontweight="bold")
ax.text(-150, 0.65, "Chromosight", fontsize=9, color=C_CS, fontweight="bold")
ax.text(-150, 1.05, "LRPC", fontsize=9, color=C_LRPC, fontweight="bold")
ax.set_xlim(-400, 4600)
ax.set_ylim(-0.05, 1.6)
ax.set_xlabel("Genomic position (kb)", fontsize=9)
ax.set_yticks([])
ax.set_title("C", loc="left", fontsize=12, fontweight="bold", x=-0.15)
ax.set_title("Genome-wide arc view of loop sets", fontsize=10)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

# D: LRPC position vs anti-diagonal (arm-arm check)
ax = fig3.add_subplot(gs3[1, 1])
# scatter of (bin1, bin2) for LRPCs, colored by oe; anti-diagonal = arm-arm axis
ax.scatter(lrpc_final.bin1 * 5 / 1000, lrpc_final.bin2 * 5 / 1000,
           c=lrpc_final.oe, cmap="viridis", s=28, edgecolor="black", lw=0.3)
# anti-diagonal reference: for genome linearized with ori at 0, arm-arm contacts
# satisfy bin1 + bin2 ~ const (ter-proximal symmetry)
xx = np.linspace(0, 4.4, 100)
ax.plot(xx, 4.4 - xx, color="red", ls="--", lw=0.8, label="arm-arm axis")
ax.set_xlabel("Anchor 1 position (Mb)", fontsize=9)
ax.set_ylabel("Anchor 2 position (Mb)", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("D", loc="left", fontsize=12, fontweight="bold", x=-0.15)
ax.set_title("LRPC anchor positions (color = O/E)", fontsize=10)
cb = fig3.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
cb.set_label("O/E", fontsize=8)

fig3.suptitle("Genome-wide distribution of Hi-C loop calls: HiCCUPS vs Chromosight vs LRPC",
              fontsize=13, fontweight="bold")
fig3.savefig(os.path.join(OUT, "figures/fig_genomewide_loop_distribution.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(OUT, "figures/fig_genomewide_loop_distribution.png"), dpi=300, bbox_inches="tight")

# ---------------- Chromosight in-situ zoomed maps ----------------
print("\nrendering Chromosight in-situ zoomed maps ...")
cs_top = cs.sort_values("score", ascending=False).head(6).reset_index(drop=True)

fig4 = plt.figure(figsize=(16, 10))
gs4 = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.38, wspace=0.3,
                        left=0.06, right=0.97, top=0.88, bottom=0.09)

for idx, (_, row) in enumerate(cs_top.iterrows()):
    ax = fig4.add_subplot(gs4[idx // 3, idx % 3])
    mid1 = int((row.start1 + row.end1) / 2)
    mid2 = int((row.start2 + row.end2) / 2)
    span = mid2 - mid1
    window = min(max(span // 2, 80000), 300000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    sub = clr_wt.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}",
                                            f"Chromosome:{r2_s}-{r2_e}")
    im = ax.imshow(sub, cmap=cmap_hic,
                   vmin=np.nanpercentile(sub[sub > 0], 2),
                   vmax=np.nanpercentile(sub, 98), interpolation="nearest")
    ax.axhline((mid2 - r2_s) / (r2_e - r2_s) * sub.shape[0], color="#0072B2", ls="--", lw=1)
    ax.axvline((mid1 - r1_s) / (r1_e - r1_s) * sub.shape[1], color="#0072B2", ls="--", lw=1)
    # O/E at anchor
    d_bins = abs(int(row.bin2) - int(row.bin1))
    e = exp_wt_diag.get(d_bins, np.nan)
    oe_px = mat_wt[int(row.bin1), int(row.bin2)] / e if e and e > 0 else np.nan
    ax.set_title(f"{mid1//1000} kb x {mid2//1000} kb ({span//1000} kb)\n"
                 f"score={row.score:.2f}, q={row.qvalue:.1e}, O/E={oe_px:.2f}", fontsize=8.5)
    ax.set_title(chr(ord("A") + idx), loc="left", fontsize=12, fontweight="bold", x=-0.16)
    ax.set_xlabel("Anchor 1 region", fontsize=8)
    if idx % 3 == 0:
        ax.set_ylabel("Anchor 2 region", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Contacts")

fig4.suptitle("Chromosight loops in situ: top 6 by kernel-match score (WT Hi-C, blue dashes = anchors)",
              fontsize=12.5, fontweight="bold")
fig4.savefig(os.path.join(OUT, "figures/fig_chromosight_insitu.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(OUT, "figures/fig_chromosight_insitu.png"), dpi=300, bbox_inches="tight")

print("new figures saved.")

# ---------------- summary ----------------
summary = {
    "LRPC_definition": {
        "distance_range_kb": [200, 1500],
        "center_OE_threshold": OE_THR,
        "focality_threshold": FOCAL_THR,
        "focality_shifts_kb": [-200, -100, 100, 200],
        "reproducibility": ">=3/4 replicates, >1.5x local donut background",
    },
    "LRPC_set": {
        "n_candidates_oe_focal": int(len(cands)),
        "n_after_nms": int(len(lrpc)),
        "n_final_reproducible": int(len(lrpc_final)),
        "median_OE_WT": float(np.nanmedian(lrpc_final.oe)),
        "median_OE_KO": float(np.nanmedian(lrpc_final.oe_ko)),
        "wilcoxon_WT_vs_KO_p": float(w_p),
    },
    "APA": apa_scores.to_dict("records"),
    "overlap": overlap,
    "chromosight_APA": cs_scores.to_dict("records"),
    "replicate_recovery": {"LRPC": float(rec_lrpc),
                           "HiCCUPS_long": float(rec_hic_long),
                           "HiCCUPS_short": float(rec_hic_short)},
}
with open(os.path.join(OUT, "tables/lrpc_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print(json.dumps(summary, indent=2))
print("=" * 60)
print("done.")
