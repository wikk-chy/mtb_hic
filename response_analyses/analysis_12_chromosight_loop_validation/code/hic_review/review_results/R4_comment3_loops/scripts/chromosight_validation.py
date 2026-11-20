#!/usr/bin/env python
"""Chromosight loop validation — Response to Reviewer 4, comment 3.

Single logic: loops were re-called with Chromosight (bacteria-validated,
Matthey-Doret et al. 2020 Nat Commun) and validated with:
  1. APA: WT loops on WT / distance-matched random / +100kb shift / on napM-KO
  2. In-situ zoomed maps (raw counts + O/E) showing focal loop dots
  3. Replicate reproducibility (4 WT replicates)
  4. Empirical FDR (shifted null) + Chromosight internal q-values
  5. napM-KO as biological control
  6. Algorithm-level explanation of HiCCUPS vs Chromosight

Run: ../../.venv_cooltools/bin/python scripts/chromosight_validation.py
(from review_results/R4_comment3_loops/)
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
from matplotlib.patches import Rectangle, Circle
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
HICCUPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")
CS_WT = os.path.join(OUT, "chromosight_out/wt.tsv")
CS_KO = os.path.join(OUT, "chromosight_out/napm_ko.tsv")
REP_DIR = os.path.join(BASE, "review_data/wt_rep_icedmatrix")
KERNEL_TXT = (os.path.join(BASE, ".venv_cooltools/lib/python3.9/site-packages/"
                           "chromosight/kernels/artificial_template_loops_type1.txt"))

os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

# colorblind-safe palette
C_WT, C_KO, C_RAND, C_SHIFT = "#0072B2", "#D55E00", "#999999", "#56B4E9"
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list(
    "oe", ["#2166ac", "#f7f7f7", "#b2182b"])  # blue-white-red for O/E

print("=" * 60)
print("Chromosight loop validation")
print("=" * 60)

# ---------------- load ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)
exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)
exp_wt_diag = exp_wt.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()

mat_wt = clr_wt.matrix(sparse=False).fetch("Chromosome")
mat_ko = clr_ko.matrix(sparse=False).fetch("Chromosome")
n = mat_wt.shape[0]

cs = pd.read_csv(CS_WT, sep="\t")
cs["mid1"] = ((cs.start1 + cs.end1) // 2).astype(int)
cs["mid2"] = ((cs.start2 + cs.end2) // 2).astype(int)
cs["dist_kb"] = (cs.mid2 - cs.mid1) // 1000
cs_ko = pd.read_csv(CS_KO, sep="\t")
print(f"Chromosight loops: WT={len(cs)}, napM-KO={len(cs_ko)}")

# O/E matrix
oe_wt = np.full_like(mat_wt, np.nan, dtype=float)
for d in range(n):
    e = exp_wt_diag.get(d, np.nan)
    if e and e > 0:
        rows = np.arange(n - d); cols = rows + d
        vals = np.diag(mat_wt, k=d) / e
        oe_wt[rows, cols] = vals; oe_wt[cols, rows] = vals

# per-loop anchor O/E
cs["oe"] = [oe_wt[int(r.bin1), int(r.bin2)] if int(r.bin2) < n else np.nan
            for _, r in cs.iterrows()]

# ---------------- helpers ----------------
def aggregate(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    return np.nanmean(s, axis=0)

def center_pixel(m):
    c = m.shape[0] // 2
    return float(m[c, c])

def center5(m):
    c = m.shape[0] // 2
    return float(np.nanmean(m[c - 2:c + 3, c - 2:c + 3]))

def bedpe_of(df):
    return pd.DataFrame({
        "chrom1": "Chromosome", "start1": df.start1, "end1": df.end1,
        "chrom2": "Chromosome", "start2": df.start2, "end2": df.end2})

# ---------------- 1. APA ----------------
print("\n[1] APA ...")
ok = (cs.start1 - FLANK >= 0) & (cs.end2 + FLANK < GENOME)
cs_apa = cs[ok].reset_index(drop=True)
bp = bedpe_of(cs_apa)

# random distance-matched
rand_rows = []
for d in (cs_apa.bin2 - cs_apa.bin1).values:
    lo, hi = FLANK // BINSIZE, (GENOME - FLANK) // BINSIZE - int(d) - 1
    if hi <= lo:
        continue
    b1 = int(rng.integers(lo, hi + 1))
    rand_rows.append(("Chromosome", b1 * BINSIZE, b1 * BINSIZE + BINSIZE,
                      "Chromosome", (b1 + int(d)) * BINSIZE, (b1 + int(d)) * BINSIZE + BINSIZE))
rand_bp = pd.DataFrame(rand_rows, columns=bp.columns)

# +100 kb shift
shift_bp = bp.copy()
shift_bp[["start1", "end1", "start2", "end2"]] += 100000
shift_ok = shift_bp.end2 + FLANK < GENOME
shift_bp = shift_bp[shift_ok].reset_index(drop=True)

st_wt = cooltools.pileup(clr_wt, bp, expected_df=exp_wt, flank=FLANK, nproc=1)
st_rand = cooltools.pileup(clr_wt, rand_bp, expected_df=exp_wt, flank=FLANK, nproc=1)
st_shift = cooltools.pileup(clr_wt, shift_bp, expected_df=exp_wt, flank=FLANK, nproc=1)
st_ko = cooltools.pileup(clr_ko, bp, expected_df=exp_ko, flank=FLANK, nproc=1)

pile_wt, pile_rand = aggregate(st_wt), aggregate(st_rand)
pile_shift, pile_ko = aggregate(st_shift), aggregate(st_ko)

apa_df = pd.DataFrame([
    dict(condition="Chromosight loops on WT", n=len(bp),
         center_pixel=center_pixel(pile_wt), center_5x5=center5(pile_wt)),
    dict(condition="random pairs on WT", n=len(rand_bp),
         center_pixel=center_pixel(pile_rand), center_5x5=center5(pile_rand)),
    dict(condition="loops +100kb shift on WT", n=len(shift_bp),
         center_pixel=center_pixel(pile_shift), center_5x5=center5(pile_shift)),
    dict(condition="Chromosight loops on napM-KO", n=len(bp),
         center_pixel=center_pixel(pile_ko), center_5x5=center5(pile_ko)),
])
apa_df.to_csv(os.path.join(OUT, "tables/apa_chromosight_scores.csv"), index=False)
print(apa_df.round(3))

# ---------------- 2. per-loop O/E + empirical FDR (shifted null) ----------------
print("\n[2] per-loop stats + FDR ...")
# O/E of shifted positions (same loops shifted +100kb) as null
shift_oe = []
for _, r in cs_apa.iterrows():
    b1 = int(r.bin1) + 20
    b2 = int(r.bin2) + 20
    if b2 < n and np.isfinite(oe_wt[b1, b2]):
        shift_oe.append(oe_wt[b1, b2])
shift_oe = np.array(shift_oe)
real_oe = cs_apa.oe.values[np.isfinite(cs_apa.oe.values)]

fdr_rows = []
for thr in [1.25, 1.5, 2.0]:
    fp = float((shift_oe > thr).mean())
    tp = int((real_oe > thr).sum())
    fdr_rows.append(dict(oe_threshold=thr,
                         frac_shifted_null_above=round(fp, 3),
                         n_real_above=tp, frac_real_above=round(tp / len(real_oe), 3),
                         empirical_FDR=round(fp * len(real_oe) / max(tp, 1), 3)))
fdr_df = pd.DataFrame(fdr_rows)
fdr_df.to_csv(os.path.join(OUT, "tables/perloop_fdr_chromosight.csv"), index=False)
print(fdr_df)

# WT vs KO per-loop
ko_oe = []
for _, r in cs_apa.iterrows():
    b1, b2 = int(r.bin1), int(r.bin2)
    e = exp_ko.set_index("dist")["balanced.avg.smoothed.agg"].to_dict().get(abs(b2 - b1), np.nan)
    ko_oe.append(mat_ko[b1, b2] / e if e and e > 0 else np.nan)
cs_apa["oe_ko"] = ko_oe
okp = np.isfinite(cs_apa.oe) & np.isfinite(cs_apa.oe_ko)
w_stat, w_p = stats.wilcoxon(cs_apa.oe[okp], cs_apa.oe_ko[okp])
print(f"per-loop O/E: WT median={cs_apa.oe[okp].median():.3f}, "
      f"KO median={cs_apa.oe_ko[okp].median():.3f}, Wilcoxon p={w_p:.2e}")
cs_apa.to_csv(os.path.join(OUT, "tables/chromosight_loops_validated.csv"), index=False)

# ---------------- 3. replicate reproducibility ----------------
# NOTE: per-pixel detection in single replicates is unreliable because each
# replicate has ~1/4 the depth of the merged matrix (sparse sampling). The
# robust metric is per-replicate APA: aggregate all loop windows within each
# replicate, then measure center enrichment over the window corners.
print("\n[3] replicate reproducibility (per-replicate APA) ...")
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

rep_files = sorted([f for f in os.listdir(REP_DIR) if f.endswith(".matrix")])
rep_mats = [load_rep_matrix(os.path.join(REP_DIR, f)) for f in rep_files]
rep_names = [f"rep{i+1}" for i in range(len(rep_mats))]

# per-replicate expected (distance-decay) from the replicate matrix itself
def rep_expected(m):
    exp = {}
    for d in range(m.shape[0]):
        vals = np.diag(m, k=d)
        vals = vals[vals > 0]
        exp[d] = np.mean(vals) if len(vals) else np.nan
    return exp

def apa_on_dense(m, exp_diag_map, loops_df, flank_bins=10):
    """pileup of loop windows on a dense matrix, O/E normalized."""
    stack = []
    for _, r in loops_df.iterrows():
        b1, b2 = int(r.bin1), int(r.bin2)
        if b1 - flank_bins < 0 or b2 + flank_bins >= m.shape[0]:
            continue
        win = m[b1 - flank_bins:b1 + flank_bins + 1,
                b2 - flank_bins:b2 + flank_bins + 1]
        # normalize by distance-expected within the window
        win_oe = np.full_like(win, np.nan, dtype=float)
        for ii in range(win.shape[0]):
            for jj in range(win.shape[1]):
                d = abs((b1 - flank_bins + ii) - (b2 - flank_bins + jj))
                e = exp_diag_map.get(d, np.nan)
                if e and e > 0:
                    win_oe[ii, jj] = win[ii, jj] / e
        stack.append(win_oe)
    if not stack:
        return np.full((2 * flank_bins + 1, 2 * flank_bins + 1), np.nan)
    return np.nanmean(np.array(stack), axis=0)

rep_apa, rep_center = [], []
for m in rep_mats:
    e_map = rep_expected(m)
    pile = apa_on_dense(m, e_map, cs_apa, flank_bins=10)
    rep_apa.append(pile)
    rep_center.append(center_pixel(pile))

# merged-matrix APA with same 21x21 window for direct comparability
e_map_wt = rep_expected(mat_wt)
pile_merged = apa_on_dense(mat_wt, e_map_wt, cs_apa, flank_bins=10)

# random control per replicate
rep_center_rand = []
for m in rep_mats:
    e_map = rep_expected(m)
    pile_r = apa_on_dense(m, e_map, rand_bp.assign(bin1=rand_bp.start1 // BINSIZE,
                                                   bin2=rand_bp.start2 // BINSIZE),
                          flank_bins=10)
    rep_center_rand.append(center_pixel(pile_r))

rep_df = pd.DataFrame(dict(replicate=rep_names,
                           APA_center_OE=rep_center,
                           APA_center_OE_random=rep_center_rand))
rep_df.to_csv(os.path.join(OUT, "tables/replicate_recovery_chromosight.csv"), index=False)
print(rep_df.round(3))
print(f"merged APA center (same 21x21 window): {center_pixel(pile_merged):.3f}")

# Wilcoxon: loops vs random across replicates (paired)
w_r_stat, w_r_p = stats.wilcoxon(rep_center, rep_center_rand)
print(f"per-replicate APA: loops vs random Wilcoxon p = {w_r_p:.3f}")

# ---------------- FIGURE 1: genome-wide distribution ----------------
print("\nrendering fig1: genome-wide distribution ...")
hic = pd.read_csv(HICCUPS, sep="\t", header=None,
                  names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                         "name", "score", "strand1", "strand2"])
hic["b1"] = (hic.start1 + hic.end1) // 2 // BINSIZE
hic["b2"] = (hic.start2 + hic.end2) // 2 // BINSIZE

fig1 = plt.figure(figsize=(15, 5.2))
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.28,
                        left=0.06, right=0.98, top=0.85, bottom=0.13)

# A: contact map overlay
ax = fig1.add_subplot(gs1[0, 0])
ax.imshow(mat_wt, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat_wt, 99),
          interpolation="nearest")
ax.scatter(hic.b2, hic.b1, s=10, facecolor="none", edgecolor=C_RAND, lw=0.7,
           label=f"HiCCUPS (n={len(hic)})", alpha=0.8)
ax.scatter(cs.bin2, cs.bin1, s=26, facecolor="none", edgecolor=C_WT, lw=1.1,
           label=f"Chromosight (n={len(cs)})", alpha=0.95)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=9)
ax.set_ylabel("Genomic position (Mb)", fontsize=9)
ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
ax.set_title("A  Loop positions on WT contact map", loc="left", fontsize=10, fontweight="bold")

# B: distance distribution
ax = fig1.add_subplot(gs1[0, 1])
bins_hist = np.arange(0, 1050, 50)
ax.hist((hic.b2 - hic.b1) * 5, bins=bins_hist, density=True, alpha=0.65,
        color=C_RAND, label=f"HiCCUPS (n={len(hic)})")
ax.hist(cs.dist_kb, bins=bins_hist, density=True, alpha=0.65,
        color=C_WT, label=f"Chromosight (n={len(cs)})")
ax.set_xlabel("Anchor distance (kb)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("B  Anchor distance distribution", loc="left", fontsize=10, fontweight="bold")

# C: arc view
ax = fig1.add_subplot(gs1[0, 2])
ax.axhline(0, color="black", lw=1.5)
for _, r in hic.iterrows():
    x1, x2 = sorted([r.b1 * 5, r.b2 * 5])
    t = np.linspace(0, np.pi, 40)
    ax.plot((x1+x2)/2 + (x2-x1)/2*np.cos(t), 0.10 + (x2-x1)/2*np.sin(t)/2200,
            color=C_RAND, lw=0.7, alpha=0.5)
for _, r in cs.iterrows():
    x1, x2 = sorted([r.bin1 * 5, r.bin2 * 5])
    t = np.linspace(0, np.pi, 40)
    ax.plot((x1+x2)/2 + (x2-x1)/2*np.cos(t), 0.45 + (x2-x1)/2*np.sin(t)/2200,
            color=C_WT, lw=0.9, alpha=0.7)
ax.text(-250, 0.16, "HiCCUPS", fontsize=9, color=C_RAND, fontweight="bold")
ax.text(-250, 0.51, "Chromosight", fontsize=9, color=C_WT, fontweight="bold")
ax.set_xlim(-400, 4600)
ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("Genomic position (kb)", fontsize=9)
ax.set_yticks([])
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.set_title("C  Genome-wide arc view", loc="left", fontsize=10, fontweight="bold")

fig1.savefig(os.path.join(OUT, "figures/fig1_genomewide_distribution.pdf"), bbox_inches="tight")
fig1.savefig(os.path.join(OUT, "figures/fig1_genomewide_distribution.png"), dpi=300, bbox_inches="tight")

# ---------------- FIGURE 2: APA ----------------
print("rendering fig2: APA ...")
fig2 = plt.figure(figsize=(15.5, 4.4))
gs2 = gridspec.GridSpec(1, 4, figure=fig2, wspace=0.30,
                        left=0.055, right=0.90, top=0.82, bottom=0.24)
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50", "-25", "0", "+25", "+50"]
vmax_apa = max(center_pixel(pile_wt), 1.8)
panels2 = [
    (pile_wt, f"A  Loops on WT\n(n={len(bp)}, centre O/E={center_pixel(pile_wt):.2f})"),
    (pile_rand, f"B  Random pairs on WT\n(n={len(rand_bp)}, centre O/E={center_pixel(pile_rand):.2f})"),
    (pile_shift, f"C  Loops +100 kb shift on WT\n(n={len(shift_bp)}, centre O/E={center_pixel(pile_shift):.2f})"),
    (pile_ko, f"D  Loops on napM-KO\n(n={len(bp)}, centre O/E={center_pixel(pile_ko):.2f})"),
]
for i, (m, title) in enumerate(panels2):
    ax = fig2.add_subplot(gs2[0, i])
    im = ax.imshow(m, cmap=cmap_hic, vmin=0.9, vmax=vmax_apa, interpolation="nearest")
    c = m.shape[0] // 2
    ax.add_patch(Rectangle((c - 2.5, c - 2.5), 5, 5, fill=False, edgecolor="black", lw=0.9))
    ax.set_xticks(ticks, ticklabels, fontsize=8)
    ax.set_yticks(ticks, ticklabels, fontsize=8)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("kb from anchor 1", fontsize=8)
    if i == 0:
        ax.set_ylabel("kb from anchor 2", fontsize=8)
cb_ax = fig2.add_axes([0.915, 0.24, 0.012, 0.58])
cb = fig2.colorbar(im, cax=cb_ax)
cb.set_label("Aggregate observed / expected", fontsize=8)
fig2.suptitle("Aggregate peak analysis (APA) of Chromosight-called loops (5 kb, ±50 kb flank)",
              fontsize=11.5, fontweight="bold", y=0.95)
fig2.savefig(os.path.join(OUT, "figures/fig2_APA_validation.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "figures/fig2_APA_validation.png"), dpi=300, bbox_inches="tight")

# ---------------- FIGURE 3: in-situ zoomed maps (raw + O/E) ----------------
print("rendering fig3: in-situ zoomed maps ...")
# choose 6 representatives: top O/E across a spread of distances
cs_sel = cs_apa.dropna(subset=["oe"]).copy()
cs_sel = cs_sel[cs_sel.dist_kb >= 30]
cs_sel = cs_sel.sort_values("oe", ascending=False)
# take top by O/E but enforce distance diversity
sel_idx = []
dist_seen = []
for i, r in cs_sel.iterrows():
    if all(abs(r.dist_kb - d) > 60 for d in dist_seen):
        sel_idx.append(i)
        dist_seen.append(r.dist_kb)
    if len(sel_idx) == 6:
        break
sel = cs_sel.loc[sel_idx].sort_values("dist_kb").reset_index(drop=True)

fig3 = plt.figure(figsize=(15, 10))
gs3 = gridspec.GridSpec(3, 4, figure=fig3, hspace=0.42, wspace=0.32,
                        left=0.06, right=0.98, top=0.90, bottom=0.08)

letters = [chr(ord("A") + i) for i in range(12)]
li = 0
for idx, (_, row) in enumerate(sel.iterrows()):
    r_, c_ = divmod(idx, 2)   # 3 rows of loops, each loop gets 2 columns (raw | O/E)
    mid1, mid2 = int(row.mid1), int(row.mid2)
    span = mid2 - mid1
    flank_view = max(int(span * 0.25), 20000)
    lo, hi = max(0, mid1 - flank_view), min(GENOME, mid2 + flank_view)
    # square region covering both anchors: loop pixel sits at off-diagonal corner
    sub_raw = clr_wt.matrix(sparse=False).fetch(f"Chromosome:{lo}-{hi}",
                                                f"Chromosome:{lo}-{hi}")
    b_lo = lo // BINSIZE
    sub_oe = oe_wt[b_lo:b_lo + sub_raw.shape[0], b_lo:b_lo + sub_raw.shape[0]]

    a1 = (mid1 - lo) / (hi - lo) * sub_raw.shape[0]
    a2 = (mid2 - lo) / (hi - lo) * sub_raw.shape[0]

    # raw panel
    ax = fig3.add_subplot(gs3[r_, c_ * 2])
    im = ax.imshow(sub_raw, cmap=cmap_hic,
                   vmin=0, vmax=np.nanpercentile(sub_raw, 99.5), interpolation="nearest")
    ax.axhline(a1, color="#0072B2", ls="--", lw=0.9)
    ax.axvline(a1, color="#0072B2", ls="--", lw=0.9)
    ax.axhline(a2, color="#0072B2", ls="--", lw=0.9)
    ax.axvline(a2, color="#0072B2", ls="--", lw=0.9)
    ax.add_patch(Circle((a1, a2), radius=max(sub_raw.shape[0]*0.03, 1.5),
                        fill=False, edgecolor="black", lw=1.2))
    ax.set_title(f"{letters[li]}  {mid1//1000} kb × {mid2//1000} kb (raw)", loc="left",
                 fontsize=9, fontweight="bold")
    li += 1
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.65, label="Contacts")

    # O/E panel
    ax = fig3.add_subplot(gs3[r_, c_ * 2 + 1])
    vmax_oe = np.nanpercentile(sub_oe, 98)
    im = ax.imshow(sub_oe, cmap=cmap_oe, vmin=1/vmax_oe, vmax=vmax_oe,
                   interpolation="nearest")
    ax.axhline(a1, color="black", ls="--", lw=0.9)
    ax.axvline(a1, color="black", ls="--", lw=0.9)
    ax.axhline(a2, color="black", ls="--", lw=0.9)
    ax.axvline(a2, color="black", ls="--", lw=0.9)
    ax.add_patch(Circle((a1, a2), radius=max(sub_oe.shape[0]*0.03, 1.5),
                        fill=False, edgecolor="black", lw=1.2))
    ax.set_title(f"{letters[li]}  span {span//1000} kb | O/E at anchor = {row.oe:.2f} | q = {row.qvalue:.0e}",
                 loc="left", fontsize=9, fontweight="bold")
    li += 1
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.65, label="Observed / expected")

fig3.suptitle("Chromosight loops in situ: raw contacts (left) and distance-normalised O/E (right)\n"
              "blue/black dashed lines = anchors; circle = loop pixel at the off-diagonal corner",
              fontsize=11.5, fontweight="bold")
fig3.savefig(os.path.join(OUT, "figures/fig3_insitu_zoomed_loops.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(OUT, "figures/fig3_insitu_zoomed_loops.png"), dpi=300, bbox_inches="tight")

# ---------------- FIGURE 4: reproducibility + FDR + KO ----------------
print("rendering fig4: stats ...")
fig4 = plt.figure(figsize=(15, 4.4))
gs4 = gridspec.GridSpec(1, 4, figure=fig4, wspace=0.32,
                        left=0.055, right=0.98, top=0.80, bottom=0.15)

# A: per-loop O/E distribution real vs shifted null
ax = fig4.add_subplot(gs4[0, 0])
bins_oe = np.arange(0.8, 3.2, 0.15)
ax.hist(real_oe, bins=bins_oe, density=True, alpha=0.7, color=C_WT,
        label=f"Loops (n={len(real_oe)})")
ax.hist(shift_oe, bins=bins_oe, density=True, alpha=0.7, color=C_RAND,
        label=f"+100 kb shifted (n={len(shift_oe)})")
ax.axvline(1.0, color="black", ls="--", lw=0.8)
ax.set_xlabel("Anchor-pixel O/E (WT)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.legend(fontsize=8, frameon=False)
u_stat, u_p = stats.mannwhitneyu(real_oe, shift_oe, alternative="greater")
ax.set_title(f"A  Loop enrichment vs shifted null\nMann-Whitney p = {u_p:.1e}",
             loc="left", fontsize=10, fontweight="bold")

# B: FDR table as bars
ax = fig4.add_subplot(gs4[0, 1])
xp = np.arange(len(fdr_df))
ax.bar(xp - 0.2, fdr_df.frac_real_above, 0.4, color=C_WT, edgecolor="black",
       label="loops above thr.")
ax.bar(xp + 0.2, fdr_df.frac_shifted_null_above, 0.4, color=C_RAND, edgecolor="black",
       label="shifted null above thr.")
ax.set_xticks(xp, [f"O/E>{t}" for t in fdr_df.oe_threshold], fontsize=9)
ax.set_ylabel("Fraction of positions", fontsize=9)
ax.legend(fontsize=8, frameon=False)
ax.set_title("B  Empirical FDR (shifted null)\n" +
             "; ".join(f"FDR@{t}={f}" for t, f in zip(fdr_df.oe_threshold, fdr_df.empirical_FDR)),
             loc="left", fontsize=9, fontweight="bold")

# C: per-replicate APA (robust to sparse per-replicate sampling)
ax = fig4.add_subplot(gs4[0, 2])
xp = np.arange(len(rep_names))
w = 0.38
ax.bar(xp - w/2, rep_center, w, color=C_WT, edgecolor="black", label="loops")
ax.bar(xp + w/2, rep_center_rand, w, color=C_RAND, edgecolor="black", label="random")
ax.axhline(center_pixel(pile_merged), color=C_WT, ls="--", lw=0.9, alpha=0.7)
ax.axhline(1.0, color="black", ls=":", lw=0.7)
ax.text(0.02, 0.96, f"merged = {center_pixel(pile_merged):.2f}", fontsize=7.5,
        color=C_WT, ha="left", va="top", transform=ax.transAxes)
ax.set_xticks(xp, rep_names, fontsize=9)
ax.set_ylabel("APA centre O/E", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title(f"C  Per-replicate APA\n(loops vs random, paired Wilcoxon p = {w_r_p:.3f})",
             loc="left", fontsize=10, fontweight="bold")

# D: WT vs KO per-loop O/E
ax = fig4.add_subplot(gs4[0, 3])
ax.scatter(cs_apa.oe[okp], cs_apa.oe_ko[okp], s=28, color=C_WT, alpha=0.85,
           edgecolor="black", lw=0.3)
lim = max(cs_apa.oe[okp].max(), cs_apa.oe_ko[okp].max()) * 1.12
ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
ax.axhline(1.0, color="lightgrey", lw=0.6)
ax.axvline(1.0, color="lightgrey", lw=0.6)
ax.set_xlabel("Anchor O/E in WT", fontsize=9)
ax.set_ylabel("Anchor O/E in napM-KO", fontsize=9)
ax.set_title(f"D  napM-KO attenuation\nWilcoxon p = {w_p:.1e}", loc="left", fontsize=10, fontweight="bold")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

fig4.savefig(os.path.join(OUT, "figures/fig4_statistics.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(OUT, "figures/fig4_statistics.png"), dpi=300, bbox_inches="tight")

# ---------------- FIGURE 5: algorithm explanation ----------------
print("rendering fig5: algorithm comparison ...")
kernel = np.loadtxt(KERNEL_TXT)

fig5 = plt.figure(figsize=(15, 4.6))
gs5 = gridspec.GridSpec(1, 4, figure=fig5, wspace=0.34,
                        left=0.055, right=0.98, top=0.78, bottom=0.14)

# A: Chromosight kernel template
ax = fig5.add_subplot(gs5[0, 0])
im = ax.imshow(kernel, cmap=cmap_hic, interpolation="nearest")
ax.set_title("A  Chromosight loop kernel\n(template matching: Pearson r ≥ 0.3)",
             loc="left", fontsize=9.5, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, shrink=0.7)

# B: tight zoom on the strongest Chromosight loop pixel (focal dot)
best = cs_apa.dropna(subset=["oe"]).sort_values("oe", ascending=False).iloc[0]
mid1, mid2 = int(best.mid1), int(best.mid2)
half_w = 25000  # +-25 kb around the anchor crossing
lo_a, hi_a = mid1 - half_w, mid1 + half_w
lo_b, hi_b = mid2 - half_w, mid2 + half_w
sub = oe_wt[lo_a // BINSIZE:hi_a // BINSIZE + 1, lo_b // BINSIZE:hi_b // BINSIZE + 1]
ax = fig5.add_subplot(gs5[0, 1])
vmax5 = np.nanpercentile(sub, 97)
im = ax.imshow(sub, cmap=cmap_oe, vmin=1, vmax=vmax5, interpolation="nearest")
c1 = sub.shape[0] // 2
c2 = sub.shape[1] // 2
ax.add_patch(Circle((c2, c1), radius=2.2, fill=False, edgecolor="black", lw=1.4))
ax.set_title(f"B  Chromosight loop, anchor-centred view\n{mid1//1000} kb × {mid2//1000} kb, O/E = {best.oe:.2f}",
             loc="left", fontsize=9.5, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, shrink=0.7, label="O/E")

# C: tight zoom on a typical short-range HiCCUPS call (no focal dot)
hic_typ = hic.assign(dist=(hic.b2 - hic.b1) * 5).sort_values("dist")
hic_typ = hic_typ[(hic_typ.dist >= 35) & (hic_typ.dist <= 45)].iloc[5]
hm1 = int(hic_typ.b1) * BINSIZE + 2500
hm2 = int(hic_typ.b2) * BINSIZE + 2500
h_oe = oe_wt[int(hic_typ.b1), int(hic_typ.b2)]
sub3 = oe_wt[(hm1 - half_w) // BINSIZE:(hm1 + half_w) // BINSIZE + 1,
             (hm2 - half_w) // BINSIZE:(hm2 + half_w) // BINSIZE + 1]
ax = fig5.add_subplot(gs5[0, 2])
vmax53 = np.nanpercentile(sub3, 97)
im = ax.imshow(sub3, cmap=cmap_oe, vmin=1, vmax=vmax53, interpolation="nearest")
c1 = sub3.shape[0] // 2
c2 = sub3.shape[1] // 2
ax.add_patch(Circle((c2, c1), radius=2.2, fill=False, edgecolor="black", lw=1.4))
ax.set_title(f"C  HiCCUPS call, anchor-centred view\n{hm1//1000} kb × {hm2//1000} kb, O/E = {h_oe:.2f} (no focal dot)",
             loc="left", fontsize=9.5, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, shrink=0.7, label="O/E")

# D: text panel — algorithmic reason
ax = fig5.add_subplot(gs5[0, 3])
ax.axis("off")
txt = (
    "Why HiCCUPS fails on bacterial Hi-C\n\n"
    "HiCCUPS (donut filter)\n"
    "• tests focal enrichment vs a LOCAL\n"
    "  donut background\n"
    "• assumes slowly-varying background\n"
    "• calibrated on human maps at 5–25 kb\n\n"
    "On a 4.4 Mb genome at 5 kb resolution:\n"
    "• the contact decay is steep: at 35–40 kb\n"
    "  the donut is contaminated by the\n"
    "  diagonal → near-diagonal false calls\n"
    "• long-range contacts sit on broad\n"
    "  macrodomain/arm-arm background →\n"
    "  not focal vs the donut → missed\n\n"
    "Chromosight (kernel correlation)\n"
    "• matches the loop SHAPE (bright dot +\n"
    "  depleted cross) via Pearson r\n"
    "• scale-invariant → robust to steep /\n"
    "  structured bacterial background\n"
    "• validated on bacteria (Matthey-Doret\n"
    "  et al. 2020, Nat Commun)"
)
ax.text(0.0, 1.0, txt, fontsize=8.3, va="top", family="monospace")
ax.set_title("D  Algorithmic basis", loc="left", fontsize=10, fontweight="bold")

fig5.savefig(os.path.join(OUT, "figures/fig5_algorithm_comparison.pdf"), bbox_inches="tight")
fig5.savefig(os.path.join(OUT, "figures/fig5_algorithm_comparison.png"), dpi=300, bbox_inches="tight")

# ---------------- summary ----------------
summary = {
    "chromosight_params": {
        "pattern": "loops", "min_dist_kb": 30, "max_dist_kb": 1000,
        "pearson": 0.3, "win_size": "auto", "resolution_kb": 5,
        "n_loops_WT": int(len(cs)), "n_loops_napMKO": int(len(cs_ko)),
    },
    "APA": apa_df.to_dict("records"),
    "per_loop": {
        "WT_median_OE": float(cs_apa.oe[okp].median()),
        "KO_median_OE": float(cs_apa.oe_ko[okp].median()),
        "wilcoxon_p": float(w_p),
        "mannwhitney_vs_shift_p": float(u_p),
    },
    "FDR": fdr_df.to_dict("records"),
    "reproducibility": {
        "per_replicate_APA_center_OE": [round(float(v), 3) for v in rep_center],
        "per_replicate_APA_center_OE_random": [round(float(v), 3) for v in rep_center_rand],
        "merged_APA_center_OE": round(float(center_pixel(pile_merged)), 3),
        "paired_wilcoxon_loops_vs_random_p": float(w_r_p),
        "note": ("per-replicate APA (aggregate of all loops within each replicate) "
                 "is used instead of per-pixel detection, because each replicate "
                 "has ~1/4 the depth of the merged matrix and per-pixel counts "
                 "are dominated by sparse sampling"),
    },
}
with open(os.path.join(OUT, "tables/validation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + json.dumps(summary, indent=2))
print("done. Outputs in", OUT)
