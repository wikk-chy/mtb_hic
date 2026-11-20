#!/usr/bin/env python
"""Refined prokaryotic interaction-peak calling + full validation.

Context (Reviewer 4, comment 3): HiCCUPS is tuned for eukaryotic Hi-C.
Our evaluation (loop_validation_extended.py) shows the 71 HiCCUPS calls are
only weakly enriched as a set (per-loop center O/E median = 1.02; only 6/71
loops with O/E > 1.5), so we:

  1. re-call peaks with a bacteria-appropriate local-enrichment z-score
     approach (strict z, non-maximum suppression, distance range matched to
     the HiCCUPS calls: 30-650 kb),
  2. estimate FDR with a distance-matched permutation null,
  3. validate the refined set: APA (with/without outliers), zoomed maps,
     replicate reproducibility, WT vs napM-KO comparison,
  4. compare refined peaks vs the 71 HiCCUPS loops.

Run: ../.venv_cooltools/bin/python refined_peaks.py
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
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment3_loops")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
KO_COOL = os.path.join(BASE, "review_data/cool_file/napm_ko_5k.cool") + "::/resolutions/5000"
LOOPS = os.path.join(BASE, "review_data/loops/wt-loops.bedpe")
REP_DIR = os.path.join(BASE, "review_data/wt_rep_icedmatrix")

os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 60)
print("Refined peak calling + validation")
print("=" * 60)

# ---------------- load data ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)
loops = pd.read_csv(LOOPS, sep="\t", header=None,
                    names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                           "name", "score", "strand1", "strand2"])
loops["center1"] = ((loops.start1 + loops.end1) // 2).astype(int)
loops["center2"] = ((loops.start2 + loops.end2) // 2).astype(int)
loops["dist"] = loops.center2 - loops.center1

exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)
exp_wt_diag = exp_wt.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()

# ---------------- build O/E matrix ----------------
print("building O/E matrix ...")
mat_wt = clr_wt.matrix(sparse=False).fetch("Chromosome")
n = mat_wt.shape[0]
oe = np.full_like(mat_wt, np.nan, dtype=float)
for d in range(n):
    e = exp_wt_diag.get(d, np.nan)
    if e and e > 0:
        rows = np.arange(n - d)
        cols = rows + d
        vals = np.diag(mat_wt, k=d) / e
        oe[rows, cols] = vals
        oe[cols, rows] = vals

# ---------------- local z-score with donut background ----------------
# For pixel (i,j): donut background = pixels within +-W that are > D_MIN bins
# away from (i,j) in both coordinates and on a similar genomic distance.
W = 5
MIN_GAP = 3  # exclude immediate neighbours (donut hole)

def pixel_z(i, j):
    val = oe[i, j]
    if not np.isfinite(val):
        return np.nan, np.nan, np.nan
    bg = []
    for di in range(-W, W + 1):
        for dj in range(-W, W + 1):
            if abs(di) <= MIN_GAP and abs(dj) <= MIN_GAP:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and np.isfinite(oe[ni, nj]):
                bg.append(oe[ni, nj])
    if len(bg) < 10:
        return np.nan, np.nan, np.nan
    m, s = np.mean(bg), np.std(bg)
    if s == 0:
        return np.nan, np.nan, np.nan
    return (val - m) / s, val, m

# ---------------- call peaks: z > Z_THR, distance 30-650 kb ----------------
Z_THR = 3.0
D_MIN_BINS, D_MAX_BINS = 6, 130   # 30 kb .. 650 kb, matches HiCCUPS loop span

cands = []
for d in range(D_MIN_BINS, D_MAX_BINS + 1):
    for i in range(n - d):
        j = i + d
        z, v, m = pixel_z(i, j)
        if np.isfinite(z) and z > Z_THR:
            cands.append(dict(bin1=i, bin2=j, z=z, oe=v, bg=m, dist_bins=d))
cands = pd.DataFrame(cands)
print(f"candidate pixels z>{Z_THR}: {len(cands)}")

# ---------------- non-maximum suppression ----------------
# keep the highest-z pixel within each 3-bin neighbourhood
def nonmax(df, radius=3):
    df = df.sort_values("z", ascending=False).reset_index(drop=True)
    keep_idx = []
    used = np.zeros(len(df), dtype=bool)
    b1 = df.bin1.values
    b2 = df.bin2.values
    for k in range(len(df)):
        if used[k]:
            continue
        keep_idx.append(k)
        near = (np.abs(b1 - b1[k]) <= radius) & (np.abs(b2 - b2[k]) <= radius)
        used |= near
    return df.iloc[keep_idx].reset_index(drop=True)

peaks = nonmax(cands, radius=3)
print(f"peaks after non-max suppression: {len(peaks)}")
peaks["start1"] = peaks.bin1 * BINSIZE
peaks["start2"] = peaks.bin2 * BINSIZE
peaks["dist_kb"] = peaks.dist_bins * BINSIZE // 1000
peaks.to_csv(os.path.join(OUT, "tables/refined_peaks.csv"), index=False)

# ---------------- permutation FDR ----------------
# null: for each distance bin, sample the same number of random pixels and
# compute their z against the donut background -> empirical FDR per z threshold
print("building permutation null ...")
null_z = []
for d in range(D_MIN_BINS, D_MAX_BINS + 1):
    idx = np.arange(n - d)
    rng.shuffle(idx)
    for i in idx[:15]:   # 15 random pixels per distance
        j = i + d
        z, _, _ = pixel_z(i, j)
        if np.isfinite(z):
            null_z.append(z)
null_z = np.array(null_z)
real_z = cands.z.values  # before NMS; rate is what matters

fdr_rows = []
for thr in [2.0, 2.5, 3.0, 3.5, 4.0]:
    n_real = int((real_z > thr).sum())
    n_null = int((null_z > thr).sum())
    # scale null to same number of tested pixels
    n_tested = sum((n - d) for d in range(D_MIN_BINS, D_MAX_BINS + 1))
    n_null_tested = len(null_z) * (n_tested / (15 * (D_MAX_BINS - D_MIN_BINS + 1)))
    fdr = (n_null * (n_tested / (15 * (D_MAX_BINS - D_MIN_BINS + 1)))) / max(n_real, 1)
    fdr_rows.append(dict(z_threshold=thr, n_real_pixels=n_real,
                         n_null_scaled=float(n_null * n_tested / (15 * (D_MAX_BINS - D_MIN_BINS + 1))),
                         empirical_FDR=float(fdr)))
fdr_df = pd.DataFrame(fdr_rows)
fdr_df.to_csv(os.path.join(OUT, "tables/refined_peaks_fdr.csv"), index=False)
print(fdr_df.round(3))

# ---------------- overlap with HiCCUPS ----------------
hiccups_bins = set((r.center1 // BINSIZE, r.center2 // BINSIZE) for _, r in loops.iterrows())
peak_bins = set((r.bin1, r.bin2) for _, r in peaks.iterrows())

def tolerant_overlap(a_set, b_set, tol=1):
    cnt = 0
    for a1, a2 in a_set:
        if any(abs(a1 - b1) <= tol and abs(a2 - b2) <= tol for b1, b2 in b_set):
            cnt += 1
    return cnt

ov_h_in_p = tolerant_overlap(hiccups_bins, peak_bins, tol=1)
ov_p_in_h = tolerant_overlap(peak_bins, hiccups_bins, tol=1)
overlap = {
    "HiCCUPS_loops": len(hiccups_bins),
    "refined_peaks": len(peak_bins),
    "HiCCUPS_recovered_by_refined": ov_h_in_p,
    "HiCCUPS_recovery_fraction": ov_h_in_p / len(hiccups_bins),
    "refined_overlapping_HiCCUPS": ov_p_in_h,
}
print(overlap)

# ---------------- helper: APA from a bedpe-like set ----------------
def bedpe_from_bins(df):
    return pd.DataFrame({
        "chrom1": "Chromosome",
        "start1": df.bin1 * BINSIZE,
        "end1": df.bin1 * BINSIZE + BINSIZE,
        "chrom2": "Chromosome",
        "start2": df.bin2 * BINSIZE,
        "end2": df.bin2 * BINSIZE + BINSIZE,
    })

def aggregate(stack):
    stack = np.where(np.isfinite(stack), stack, np.nan)
    return np.nanmean(stack, axis=0)

def enrich(m):
    c = m.shape[0] // 2
    center = np.nanmean(m[c - 2:c + 3, c - 2:c + 3])
    corners = np.nanmean(np.concatenate([
        m[:3, :3].ravel(), m[:3, -3:].ravel(),
        m[-3:, :3].ravel(), m[-3:, -3:].ravel()]))
    return dict(center_5x5=float(center), corners_3x3=float(corners),
                enrichment=float(center / corners), center_pixel=float(m[c, c]))

def per_loop_center(stack):
    s = np.where(np.isfinite(stack), stack, np.nan)
    c = s.shape[1] // 2
    return np.nanmean(s[:, c - 2:c + 3, c - 2:c + 3], axis=(1, 2))

# keep only peaks whose +-50kb windows fit
ok_peak = (peaks.start1 - FLANK >= 0) & (peaks.start2 + FLANK < GENOME)
peaks_f = peaks[ok_peak].reset_index(drop=True)
print(f"refined peaks usable for APA: {len(peaks_f)} of {len(peaks)}")

peak_bedpe = bedpe_from_bins(peaks_f)

print("pileup: refined peaks ...")
stack_pk_wt = cooltools.pileup(clr_wt, peak_bedpe, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_pk_ko = cooltools.pileup(clr_ko, peak_bedpe, expected_df=exp_ko, flank=FLANK, nproc=1)
pile_pk_wt = aggregate(stack_pk_wt)
pile_pk_ko = aggregate(stack_pk_ko)

# random distance-matched control for the refined peaks
rand_d = rng.choice(peaks_f.dist_bins.values, size=len(peaks_f), replace=True)
rand_rows = []
for d in rand_d:
    lo = FLANK // BINSIZE
    hi = (GENOME - FLANK) // BINSIZE - int(d) - 1
    if hi <= lo:
        continue
    b1 = int(rng.integers(lo, hi + 1))
    rand_rows.append(("Chromosome", b1 * BINSIZE, b1 * BINSIZE + BINSIZE,
                      "Chromosome", (b1 + int(d)) * BINSIZE, (b1 + int(d)) * BINSIZE + BINSIZE))
rand_pk = pd.DataFrame(rand_rows, columns=peak_bedpe.columns)
stack_pk_rand = cooltools.pileup(clr_wt, rand_pk, expected_df=exp_wt, flank=FLANK, nproc=1)
pile_pk_rand = aggregate(stack_pk_rand)

# ---------------- outlier-excluded APA for the 71 HiCCUPS loops ----------------
print("pileup: HiCCUPS loops with/without outliers ...")
pl_prev = pd.read_csv(os.path.join(OUT, "tables/per_loop_oe_wt_vs_ko.csv"))
loops_kept = loops.copy()
loops_kept["outlier"] = pl_prev.outlier_WT.values

hic_bedpe_all = loops_kept[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]]
hic_bedpe_no = loops_kept.loc[~loops_kept.outlier, ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]]

stack_h_all = cooltools.pileup(clr_wt, hic_bedpe_all, expected_df=exp_wt, flank=FLANK, nproc=1)
stack_h_no = cooltools.pileup(clr_wt, hic_bedpe_no, expected_df=exp_wt, flank=FLANK, nproc=1)
pile_h_all = aggregate(stack_h_all)
pile_h_no = aggregate(stack_h_no)

apa_results = {
    "HiCCUPS all 71 loops on WT": enrich(pile_h_all),
    "HiCCUPS 65 non-outlier loops on WT": enrich(pile_h_no),
    f"refined peaks (n={len(peaks_f)}) on WT": enrich(pile_pk_wt),
    f"refined peaks (n={len(peaks_f)}) on napM-KO": enrich(pile_pk_ko),
    f"refined-peak random control (n={len(rand_pk)})": enrich(pile_pk_rand),
}
apa_df = pd.DataFrame(apa_results).T
apa_df.index.name = "condition"
apa_df.to_csv(os.path.join(OUT, "tables/apa_refined_vs_hiccups.csv"))
print(apa_df.round(3))

# ---------------- per-peak WT vs KO ----------------
plp_wt = per_loop_center(stack_pk_wt)
plp_ko = per_loop_center(stack_pk_ko)
ok2 = np.isfinite(plp_wt) & np.isfinite(plp_ko)
w_stat, w_p = stats.wilcoxon(plp_wt[ok2], plp_ko[ok2])
print(f"refined peaks per-peak O/E: WT median={np.nanmedian(plp_wt):.3f}, "
      f"KO median={np.nanmedian(plp_ko):.3f}, Wilcoxon p={w_p:.2e} (n={ok2.sum()})")
peaks_f["OE_center5x5_WT"] = plp_wt
peaks_f["OE_center5x5_napMKO"] = plp_ko
peaks_f.to_csv(os.path.join(OUT, "tables/refined_peaks_validated.csv"), index=False)

# ---------------- replicate reproducibility for refined peaks ----------------
def load_rep_matrix(path, n_bins=883):
    mat = np.zeros((n_bins, n_bins))
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                v = float(parts[2])
                if 0 <= i < n_bins and 0 <= j < n_bins:
                    mat[i, j] = v
                    mat[j, i] = v
            except ValueError:
                continue
    return mat

rep_files = sorted([f for f in os.listdir(REP_DIR) if f.endswith(".matrix")])
rep_mats = {f.replace("_5000_iced.matrix", "").replace("wt_", "rep"): load_rep_matrix(os.path.join(REP_DIR, f))
            for f in rep_files}

rep_rows = []
for rep_id, m in rep_mats.items():
    rec = 0
    for _, p in peaks_f.iterrows():
        b1, b2 = int(p.bin1), int(p.bin2)
        val = m[b1, b2]
        bg = []
        for di in range(-3, 4):
            for dj in range(-3, 4):
                if abs(di) <= 1 and abs(dj) <= 1:
                    continue
                ni, nj = b1 + di, b2 + dj
                if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
                    bg.append(m[ni, nj])
        bgm = np.mean(bg) if bg else 1.0
        if bgm > 0 and val / bgm > 1.5:
            rec += 1
    rep_rows.append(dict(replicate=rep_id, recovered=rec,
                         recovery_rate=rec / len(peaks_f)))
rep_pk_df = pd.DataFrame(rep_rows)
rep_pk_df.to_csv(os.path.join(OUT, "tables/replicate_recovery_refined_peaks.csv"), index=False)
print(rep_pk_df)

# ---------------- figures ----------------
cmap = LinearSegmentedColormap.from_list("redwhite", ["#ffffff", "#d7301f"])

# fig7: refined peaks APA
panels = [
    (pile_h_all, f"HiCCUPS all (n=71)", apa_results["HiCCUPS all 71 loops on WT"]),
    (pile_h_no, f"HiCCUPS non-outlier (n=65)", apa_results["HiCCUPS 65 non-outlier loops on WT"]),
    (pile_pk_wt, f"Refined peaks (n={len(peaks_f)}) on WT", apa_results[f"refined peaks (n={len(peaks_f)}) on WT"]),
    (pile_pk_ko, f"Refined peaks on napM-KO", apa_results[f"refined peaks (n={len(peaks_f)}) on napM-KO"]),
]
vmax = max(np.nanmax(m) for m, *_ in panels)
vmin = min(np.nanmin(m) for m, *_ in panels)
ticks = np.array([0, 5, 10, 15, 20])
ticklabels = ["-50 kb", "-25 kb", "0", "+25 kb", "+50 kb"]

fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.1), constrained_layout=True)
for i, (ax, (m, title, enr)) in enumerate(zip(axes, panels)):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    c = m.shape[0] // 2
    ax.add_patch(plt.Rectangle((c - 2.5, c - 2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
    ax.set_xticks(ticks, ticklabels, fontsize=8)
    ax.set_yticks(ticks, ticklabels, fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.text(-0.18, 1.06, chr(ord("A") + i), transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.text(0.5, -0.34, f"center/corner = {enr['enrichment']:.2f}\ncenter pixel = {enr['center_pixel']:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015, shrink=0.85, label="Aggregate observed / expected")
fig.suptitle("APA: HiCCUPS loops vs refined local-enrichment peaks", fontsize=11)
fig.savefig(os.path.join(OUT, "figures/fig7_refined_peaks_APA.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig7_refined_peaks_APA.png"), dpi=300, bbox_inches="tight")

# fig8: zoomed maps for top refined peaks (this is what the reviewer asked for)
top_peaks = peaks_f.sort_values("z", ascending=False).head(6).reset_index(drop=True)
fig8 = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig8, hspace=0.35, wspace=0.3)
for idx, (_, row) in enumerate(top_peaks.iterrows()):
    ax = fig8.add_subplot(gs[idx // 3, idx % 3])
    mid1 = int(row.start1) + BINSIZE // 2
    mid2 = int(row.start2) + BINSIZE // 2
    span = abs(mid2 - mid1)
    window = min(max(span // 2, 50000), 300000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    try:
        m = clr_wt.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}", f"Chromosome:{r2_s}-{r2_e}")
        im = ax.imshow(m, cmap=cmap, vmin=np.nanpercentile(m[m > 0], 2), vmax=np.nanpercentile(m, 98))
        ax.axhline((mid2 - r2_s) / (r2_e - r2_s) * m.shape[0], color="blue", linestyle="--", lw=1)
        ax.axvline((mid1 - r1_s) / (r1_e - r1_s) * m.shape[1], color="blue", linestyle="--", lw=1)
        plt.colorbar(im, ax=ax, shrink=0.6, label="Contact freq")
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:60], ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"Peak {idx+1}: {mid1//1000}kb x {mid2//1000}kb\nz={row.z:.1f}, O/E={row.oe:.2f}", fontsize=9)
    ax.set_xlabel("Anchor 1 region (kb)", fontsize=8)
    ax.set_ylabel("Anchor 2 region (kb)", fontsize=8)
fig8.suptitle("Zoomed contact maps: top refined interaction peaks (blue lines = anchors)", fontsize=13)
fig8.savefig(os.path.join(OUT, "figures/fig8_zoomed_refined_peaks.pdf"), bbox_inches="tight")
fig8.savefig(os.path.join(OUT, "figures/fig8_zoomed_refined_peaks.png"), dpi=300, bbox_inches="tight")

# fig9: summary stats panel
fig9, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

# A: per-set median O/E comparison
sets = ["HiCCUPS\n(n=71)", "Refined\n(n=%d)" % len(peaks_f)]
med_wt = [np.nanmedian(pl_prev.OE_center5x5_WT), np.nanmedian(plp_wt)]
frac_11 = [float((pl_prev.OE_center5x5_WT > 1.1).mean()), float((plp_wt > 1.1).mean())]
axes[0].bar(sets, med_wt, color=["#999999", "#297270"], edgecolor="black")
axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.6)
axes[0].set_ylabel("Median per-loop center O/E (5x5)")
axes[0].set_title("A  Per-loop enrichment: HiCCUPS vs refined", fontsize=10)

# B: fraction of loops with O/E>1.1
axes[1].bar(sets, frac_11, color=["#999999", "#297270"], edgecolor="black")
axes[1].set_ylabel("Fraction of loops with O/E > 1.1")
axes[1].set_ylim(0, 1)
axes[1].set_title("B  Fraction of individually enriched loops", fontsize=10)

# C: replicate recovery comparison
axes[2].bar(["HiCCUPS", "Refined"],
            [45.4, rep_pk_df.recovery_rate.mean() * 100],
            color=["#999999", "#297270"], edgecolor="black")
axes[2].set_ylabel("Mean replicate recovery (%)")
axes[2].set_title("C  Replicate reproducibility", fontsize=10)

fig9.savefig(os.path.join(OUT, "figures/fig9_hiccups_vs_refined_summary.pdf"), bbox_inches="tight")
fig9.savefig(os.path.join(OUT, "figures/fig9_hiccups_vs_refined_summary.png"), dpi=300, bbox_inches="tight")

# ---------------- summary json ----------------
summary = {
    "HiCCUPS": {
        "n_loops": 71,
        "median_perloop_OE_WT": float(np.nanmedian(pl_prev.OE_center5x5_WT)),
        "frac_OE_gt_1.1": float((pl_prev.OE_center5x5_WT > 1.1).mean()),
        "APA_center_pixel": apa_results["HiCCUPS all 71 loops on WT"]["center_pixel"],
        "APA_center_pixel_nonoutlier": apa_results["HiCCUPS 65 non-outlier loops on WT"]["center_pixel"],
        "replicate_recovery_mean": 0.454,
    },
    "refined": {
        "n_peaks": int(len(peaks_f)),
        "z_threshold": Z_THR,
        "median_perloop_OE_WT": float(np.nanmedian(plp_wt)),
        "frac_OE_gt_1.1": float((plp_wt > 1.1).mean()),
        "APA_center_pixel": apa_results[f"refined peaks (n={len(peaks_f)}) on WT"]["center_pixel"],
        "APA_center_pixel_KO": apa_results[f"refined peaks (n={len(peaks_f)}) on napM-KO"]["center_pixel"],
        "replicate_recovery_mean": float(rep_pk_df.recovery_rate.mean()),
        "wilcoxon_WT_vs_KO_p": float(w_p),
    },
    "overlap": overlap,
    "FDR": fdr_df.to_dict("records"),
}
with open(os.path.join(OUT, "tables/refined_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("REFINED SUMMARY")
print(json.dumps(summary, indent=2))
print("=" * 60)
