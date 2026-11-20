#!/usr/bin/env python
"""Combined figure v3 for R4 comment 2 — merges fig_combined_CID_validation
(dropping the old C/D/E zoom + cooltools-pileup panels) with
fig5_whole_cid_rescaled_pileup, in a logical order:

  A  DI track: boundary definition (negative->positive transitions), CID shading
  B  WT contact map with continuous CID blocks along the diagonal
  C  Contact pileup at boundaries (coolpup; insulation cross)
  D  Within-CID enrichment vs 1000 random partitions (permutation)
  E  DI metaplot at boundaries vs random positions
  F  Insulation-score metaplot at boundaries vs random positions
  G  Per-replicate insulation-cross depletion (reproducibility)
  H  Whole-CID rescaled pileup (confirmed CIDs)
  I  Random same-size regions (3 sets)
  J  Difference (CID - random): enrichment concentrated inside the domain

Logic: (A,B) what the boundaries are -> (C,E,F) aggregate evidence at
boundaries -> (D,G) statistics and reproducibility -> (H,I,J) whole-domain
evidence with matched random controls.

Run: ../../.venv_cooltools/bin/python combined_figure_v3.py
"""

import os
import json
import numpy as np
import pandas as pd
import cooler
import cooltools
from coolpuppy import coolpup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter, zoom

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment2_CID")
WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
DI_WT = os.path.join(BASE, "review_data/DI_Score_results/wt.csv")
REP_DIR = os.path.join(BASE, "review_data/rep_icedmatrix")

BINSIZE = 5000
WINDOW_BINS = 20
NEG_THR, POS_THR = -2.0, 2.0
LOOKBACK, LOOKAHEAD = 10, 10
MIN_CID_BINS = 20
REP_TOL_BINS = 5
SEED = 42
rng = np.random.default_rng(SEED)

C_WT, C_RAND = "#0072B2", "#999999"
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])
cmap_oe = LinearSegmentedColormap.from_list("oe", ["#2166ac", "#f7f7f7", "#b2182b"])

print("loading data ...")
clr = cooler.Cooler(WT_COOL)
mat = np.nan_to_num(clr.matrix(sparse=False).fetch("Chromosome"), nan=0.0)
n = mat.shape[0]
di = pd.read_csv(DI_WT, header=None).values.flatten()

# ---------------- boundary calling ----------------
def call_boundaries(di):
    cands = []
    for i in range(1, n - 1):
        left = di[max(0, i - LOOKBACK):i + 1]
        right = di[i + 1:i + 1 + LOOKAHEAD]
        if (left <= NEG_THR).any() and (right >= POS_THR).any():
            cands.append((i, di[i + 1] - di[i]))
    cands.sort(key=lambda x: -x[1])
    chosen = []
    for i, jump in cands:
        if all(abs(i - c) >= MIN_CID_BINS for c, _ in chosen):
            chosen.append((i, jump))
    return sorted(chosen)

boundaries = call_boundaries(di)

# ---------------- WT replicate matrices (for confirmation + panel G) ----------------
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

def compute_di(m):
    m = np.nan_to_num(m, nan=0.0)
    S = gaussian_filter(m, sigma=2.0, mode="nearest")
    L = np.log1p(S)
    C = np.nan_to_num(np.corrcoef(L))
    out = np.full(n, np.nan)
    for i in range(n):
        k = min(i, n - 1 - i, WINDOW_BINS)
        if k < 5:
            continue
        t, _ = stats.ttest_rel(C[i, i + 1:i + 1 + k], C[i, i - k:i])
        out[i] = t
    return out

rep_files = sorted([f for f in os.listdir(REP_DIR)
                    if f.startswith("wt_") and f.endswith(".matrix")])
rep_mats = {rf.replace("_5000_iced.matrix", ""): load_rep_matrix(os.path.join(REP_DIR, rf))
            for rf in rep_files}
rep_dis = {rid: compute_di(m) for rid, m in rep_mats.items()}
rep_ids = sorted(rep_mats.keys())
print(f"WT replicates: {rep_ids}")

cons = {}
for i, _ in boundaries:
    ok_per = []
    for rid, rdi in rep_dis.items():
        ok = False
        for j in range(max(0, i - REP_TOL_BINS), min(n, i + REP_TOL_BINS + 1)):
            lft = rdi[max(0, j - LOOKBACK):j]
            rgt = rdi[j + 1:j + 1 + LOOKAHEAD]
            lft, rgt = lft[np.isfinite(lft)], rgt[np.isfinite(rgt)]
            if len(lft) and len(rgt) and lft.mean() < 0 and rgt.mean() > 0:
                ok = True
                break
        ok_per.append(ok)
    cons[i] = ok_per

confirmed_bins = sorted([i for i, _ in boundaries if sum(cons[i]) >= 3])
cand_only = [i for i, _ in boundaries if i not in confirmed_bins]
print(f"candidates: {len(boundaries)}, confirmed: {len(confirmed_bins)}, "
      f"candidate-only: {len(cand_only)} ({[b*5 for b in cand_only]} kb)")

# ---------------- O/E matrix ----------------
exp = cooltools.expected_cis(clr, nproc=1)
exp_diag = exp.set_index("dist")["balanced.avg.smoothed.agg"].to_dict()
oe = np.full_like(mat, np.nan, dtype=float)
for d in range(n):
    e = exp_diag.get(d, np.nan)
    if e and e > 0:
        rows = np.arange(n - d); cols = rows + d
        vals = np.diag(mat, k=d) / e
        oe[rows, cols] = vals; oe[cols, rows] = vals

# ---------------- helpers ----------------
FL = 30
rand_centers = []
while len(rand_centers) < len(confirmed_bins):
    c = int(rng.integers(FL + 5, n - FL - 5))
    if all(abs(c - b) > 10 for b in confirmed_bins):
        rand_centers.append(c)

def pileup_windows(centers, matrix, diag_mask=3):
    stack = []
    for c in centers:
        if c - FL < 0 or c + FL + 1 > matrix.shape[0]:
            continue
        w = matrix[c - FL:c + FL + 1, c - FL:c + FL + 1].copy()
        nd = w.shape[0]
        for d in range(-diag_mask, diag_mask + 1):
            idx = np.arange(max(0, -d), nd - max(0, d))
            w[idx, idx + d] = np.nan
        stack.append(w)
    return np.nanmean(np.array(stack), axis=0)

def cross_depletion(p, arm=5, corner=10):
    c = p.shape[0] // 2
    arms = np.concatenate([p[c-arm:c+arm+1, :c-corner].ravel(), p[c-arm:c+arm+1, c+corner+1:].ravel(),
                           p[:c-corner, c-arm:c+arm+1].ravel(), p[c+corner+1:, c-arm:c+arm+1].ravel()])
    corners = np.concatenate([p[:c-corner, :c-corner].ravel(), p[:c-corner, c+corner+1:].ravel(),
                              p[c+corner+1:, :c-corner].ravel(), p[c+corner+1:, c+corner+1:].ravel()])
    arms, corners = arms[np.isfinite(arms)], corners[np.isfinite(corners)]
    return float(np.nanmedian(arms) / np.nanmedian(corners))

def insulation_score(m, w=20):
    ins = np.full(n, np.nan)
    for i in range(n - 2 * w):
        v = m[i:i + w, i + w:i + 2 * w]
        v = v[np.isfinite(v)]
        if len(v) > w * w * 0.5:
            ins[i] = np.nanmedian(v)
    sm = np.copy(ins)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        sm[i] = np.nanmedian(ins[lo:hi])
    return sm

def meta_profile(centers, track, half=40):
    prof = []
    for c in centers:
        if c - half < 0 or c + half + 1 > len(track):
            continue
        prof.append(track[c - half:c + half + 1])
    return np.nanmean(np.array(prof), axis=0)

ins = insulation_score(oe, w=20)

# ---------------- pileup: coolpup ----------------
print("pileup with coolpup ...")
bd_bed = pd.DataFrame({
    "chrom": "Chromosome",
    "start": [b * BINSIZE for b in confirmed_bins],
    "end": [b * BINSIZE + BINSIZE for b in confirmed_bins]})
cpup = coolpup.pileup(clr, bd_bed, features_format="bed", local=True,
                      expected_df=exp, flank=150000, nshifts=0)
pile_cp = np.asarray(cpup["data"].iloc[0]).astype(float)
nd = pile_cp.shape[0]
for d in range(-3, 4):
    idx = np.arange(max(0, -d), nd - max(0, d))
    pile_cp[idx, idx + d] = np.nan
depl_cp = cross_depletion(pile_cp)
print(f"cross depletion (coolpup) = {depl_cp:.3f}")

# ---------------- metaplots + permutation ----------------
ins_bd, ins_rd = meta_profile(confirmed_bins, ins), meta_profile(rand_centers, ins)
di_bd, di_rd = meta_profile(confirmed_bins, di), meta_profile(rand_centers, di)

def within_vs_cross(bounds):
    bounds = sorted(bounds)
    within, cross = [], []
    for i in range(n):
        for d in range(6, 30):
            j = i + d
            if j >= n:
                continue
            v = oe[i, j]
            if not np.isfinite(v):
                continue
            nb = sum(1 for b in bounds if i < b <= j)
            (cross if nb > 0 else within).append(v)
    return np.array(within), np.array(cross)

w_in, w_cross = within_vs_cross(confirmed_bins)
obs_ratio = np.nanmedian(w_in) / np.nanmedian(w_cross)
sizes_seg = np.diff([0] + confirmed_bins + [n - 1])
null_ratios = []
for it in range(1000):
    shuffled = np.concatenate([[0], np.cumsum(rng.permutation(sizes_seg))[:-1]])
    rb = [int(x) for x in shuffled[1:] if 0 < x < n - 1]
    wi, wc = within_vs_cross(rb)
    if len(wc) > 100:
        null_ratios.append(np.nanmedian(wi) / np.nanmedian(wc))
null_ratios = np.array(null_ratios)
perm_p = (np.sum(null_ratios >= obs_ratio) + 1) / (len(null_ratios) + 1)

# ---------------- per-replicate quantitative validation ----------------
def rep_expected(m):
    e = {}
    for d in range(m.shape[0]):
        vals = np.diag(m, k=d)
        vals = vals[vals > 0]
        e[d] = np.mean(vals) if len(vals) else np.nan
    return e

def rep_oe(m):
    e = rep_expected(m)
    o = np.full_like(m, np.nan, dtype=float)
    for d in range(m.shape[0]):
        ev = e.get(d, np.nan)
        if ev and ev > 0:
            rows = np.arange(m.shape[0] - d); cols = rows + d
            vals = np.diag(m, k=d) / ev
            o[rows, cols] = vals; o[cols, rows] = vals
    return o

rep_depl = {}
for rid, m in rep_mats.items():
    o = rep_oe(np.nan_to_num(m, nan=0.0))
    p = pileup_windows(confirmed_bins, o)
    rep_depl[rid] = cross_depletion(p)
print("per-replicate cross depletion:", {k: round(v, 3) for k, v in rep_depl.items()})

# ---------------- whole-CID rescaled pileup (from fig5) ----------------
RES_SIZE = 61
verts = [0] + confirmed_bins + [n - 1]
cids = [(verts[k], verts[k + 1]) for k in range(len(verts) - 1)]
cid_sizes = np.array([b - a for a, b in cids])

def rescaled_snip(a, b, size=RES_SIZE, flank_frac=1.0):
    L = b - a
    fl = int(round(L * flank_frac))
    lo, hi = a - fl, b + fl
    if lo < 0 or hi > n:
        return None
    w = oe[lo:hi, lo:hi]
    if w.shape[0] < 4:
        return None
    zf = size / w.shape[0]
    return zoom(np.nan_to_num(w, nan=np.nanmedian(w)), zf, order=1)

def pileup_robust(regions, trim=0.2):
    from scipy.stats import trim_mean
    stack = []
    for a, b in regions:
        s = rescaled_snip(a, b)
        if s is not None:
            stack.append(s)
    return trim_mean(np.array(stack), trim, axis=0), len(stack)

pile_cid, n_used = pileup_robust(cids)
rand_piles = []
for rep in range(3):
    shuffled = np.concatenate([[0], np.cumsum(rng.permutation(cid_sizes))[:-1]])
    regs = [(int(shuffled[k]), int(shuffled[k] + cid_sizes[k]))
            for k in range(len(cid_sizes)) if shuffled[k] + cid_sizes[k] < n - 1]
    pile_r, _ = pileup_robust(regs)
    rand_piles.append(pile_r)
pile_rand = np.nanmean(np.array(rand_piles), axis=0)

def mask_diag(p, w=2):
    p = p.copy()
    nd = p.shape[0]
    for d in range(-w, w + 1):
        idx = np.arange(max(0, -d), nd - max(0, d))
        p[idx, idx + d] = np.nan
    return p

pc, pr = mask_diag(pile_cid), mask_diag(pile_rand)
diff = pc - pr
vals = np.concatenate([pc[np.isfinite(pc)], pr[np.isfinite(pr)]])
dev = np.nanpercentile(np.abs(vals - 1.0), 95)
oe_lo, oe_hi = 1.0 - dev, 1.0 + dev
dmax = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)

t = RES_SIZE // 3
def domain_score(p):
    inside = np.nanmedian(p[t:2 * t, t:2 * t])
    flanks = np.nanmedian(np.concatenate([p[:t, :].ravel(), p[2 * t:, :].ravel()]))
    return float(inside / flanks)
ratio_c, ratio_r = domain_score(pc), domain_score(pr)
print(f"whole-CID inside/flank: CID={ratio_c:.3f}, random={ratio_r:.3f}")

# ================= figure =================
print("rendering combined figure v3 ...")
fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[0.85, 1.1, 1.0, 1.05],
                       hspace=0.45, wspace=0.32,
                       left=0.06, right=0.98, top=0.945, bottom=0.05)

def plabel(ax, letter, title, fs=10.5):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=fs, fontweight="bold", pad=8)

# ---- A: DI track with alternating CID shading ----
ax = fig.add_subplot(gs[0, :])
for k in range(len(verts) - 1):
    if k % 2 == 0:
        ax.axvspan(verts[k] * 5, verts[k + 1] * 5, color="grey", alpha=0.10, zorder=0)
x = np.arange(n) * BINSIZE / 1000
ax.fill_between(x, di, 0, where=(di < 0), color="#2166ac", alpha=0.75, zorder=2)
ax.fill_between(x, di, 0, where=(di >= 0), color="#b2182b", alpha=0.75, zorder=2)
ax.axhline(NEG_THR, color="#2166ac", ls=":", lw=0.8)
ax.axhline(POS_THR, color="#b2182b", ls=":", lw=0.8)
ax.scatter([i * 5 for i in confirmed_bins], [2.12] * len(confirmed_bins),
           marker="v", s=55, color="black", zorder=5, clip_on=False)
ax.scatter([i * 5 for i in cand_only], [2.12] * len(cand_only),
           marker="v", s=45, facecolor="white", edgecolor="black", zorder=5, clip_on=False)
ax.set_xlim(0, n * 5)
ax.set_ylim(-2, 2)
ax.set_xlabel("Genomic position (kb)", fontsize=10)
ax.set_ylabel("Directional preference (DI)", fontsize=10)
ax.legend(handles=[
    Line2D([0], [0], marker="v", color="w", markerfacecolor="black", markersize=9,
           label=f"confirmed (n={len(confirmed_bins)})"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="white", markeredgecolor="black",
           markersize=9, label=f"candidate only (n={len(cand_only)})"),
    Line2D([0], [0], color="#2166ac", lw=5, alpha=0.75, label="upstream preference (DI<0)"),
    Line2D([0], [0], color="#b2182b", lw=5, alpha=0.75, label="downstream preference (DI>0)")],
    fontsize=8.5, loc="upper right", frameon=False, ncol=2)
plabel(ax, "A", "DI track: boundaries fall at negative→positive transitions")

# ---- B: contact map with continuous CID blocks ----
ax = fig.add_subplot(gs[1, 0])
ax.imshow(mat, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat, 99), interpolation="nearest")
cid_colors = ["#0072B2", "#8fd3f4"]
for k in range(len(verts) - 1):
    a, b = verts[k], verts[k + 1]
    ax.add_patch(Rectangle((a, a), b - a, b - a, fill=True,
                           facecolor=cid_colors[k % 2], alpha=0.30, edgecolor="none", zorder=2))
for i in confirmed_bins:
    ax.plot([i, i], [-6, 0], color="#003f63", lw=1.0, clip_on=False, zorder=3)
    ax.plot([n, n + 6], [i, i], color="#003f63", lw=1.0, clip_on=False, zorder=3)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.0f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.0f}" for v in xt], fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=9)
ax.set_ylabel("Genomic position (Mb)", fontsize=9)
plabel(ax, "B", f"WT contact map: {len(confirmed_bins)} confirmed boundaries\n"
                f"segment the diagonal into continuous CIDs (shaded blocks)")

# ---- C: boundary pileup (coolpup) ----
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(pile_cp, cmap=cmap_oe, vmin=0.7, vmax=1.3, interpolation="nearest")
cb2 = pile_cp.shape[0] // 2
ax.axhline(cb2, color="black", ls="--", lw=0.8)
ax.axvline(cb2, color="black", ls="--", lw=0.8)
ax.set_xticks([0, cb2, 2 * cb2], ["-150 kb", "boundary", "+150 kb"], fontsize=8)
ax.set_yticks([0, cb2, 2 * cb2], ["-150 kb", "boundary", "+150 kb"], fontsize=8)
plabel(ax, "C", f"Pileup at boundaries (coolpup)\n(insulation cross, arms/corners={depl_cp:.2f})")
plt.colorbar(im, ax=ax, shrink=0.6, label="agg. O/E")

# ---- D: permutation ----
ax = fig.add_subplot(gs[1, 2])
ax.hist(null_ratios, bins=40, color=C_RAND, edgecolor="black", alpha=0.8)
ax.axvline(obs_ratio, color=C_WT, lw=2)
ax.text(obs_ratio, ax.get_ylim()[1] * 0.8, f"observed={obs_ratio:.2f}\np={perm_p:.3f}",
        color=C_WT, fontsize=9, ha="center", fontweight="bold")
ax.set_xlabel("within-CID / cross-boundary median O/E ratio", fontsize=9)
ax.set_ylabel("Random partitions", fontsize=9)
plabel(ax, "D", "Within-CID contact enrichment exceeds\n1000 random partitions (permutation)")

# ---- E: DI metaplot ----
ax = fig.add_subplot(gs[2, 0])
xmeta = np.arange(-40, 41) * 5
ax.plot(xmeta, di_bd, color=C_WT, lw=1.6, label=f"boundaries (n={len(confirmed_bins)})")
ax.plot(xmeta, di_rd, color=C_RAND, lw=1.6, label=f"random (n={len(rand_centers)})")
ax.axvline(0, color="black", ls="--", lw=0.8)
ax.axhline(0, color="grey", ls=":", lw=0.7)
ax.set_xlabel("Distance from boundary (kb)", fontsize=9)
ax.set_ylabel("Mean DI", fontsize=9)
ax.legend(fontsize=8, frameon=False)
plabel(ax, "E", "DI metaplot: transition at boundaries,\noscillation only at random")

# ---- F: insulation metaplot ----
ax = fig.add_subplot(gs[2, 1])
ax.plot(xmeta, ins_bd, color=C_WT, lw=1.6, label=f"boundaries (n={len(confirmed_bins)})")
ax.plot(xmeta, ins_rd, color=C_RAND, lw=1.6, label=f"random (n={len(rand_centers)})")
ax.axvline(0, color="black", ls="--", lw=0.8)
ax.set_xlabel("Distance from boundary (kb)", fontsize=9)
ax.set_ylabel("Insulation score", fontsize=9)
ax.legend(fontsize=8, frameon=False)
plabel(ax, "F", "Insulation-score metaplot:\ndip at boundaries, none at random")

# ---- G: per-replicate quantitative validation ----
ax = fig.add_subplot(gs[2, 2])
rnames = [r.replace("wt_", "rep") for r in rep_ids]
rvals = [rep_depl[r] for r in rep_ids]
bars = ax.bar(rnames, rvals, color=C_WT, edgecolor="black", alpha=0.85, width=0.55)
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.axhline(depl_cp, color="#D55E00", ls=":", lw=1.2, label=f"merged = {depl_cp:.2f}")
for b_, v in zip(bars, rvals):
    ax.text(b_.get_x() + b_.get_width() / 2, v - 0.03, f"{v:.2f}",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold")
ax.set_ylabel("Insulation-cross depletion\n(arms/corners, <1 = insulation)", fontsize=9)
ax.set_ylim(min(rvals + [depl_cp]) * 0.85, 1.05)
ax.legend(fontsize=8, frameon=False)
plabel(ax, "G", "Per-replicate validation: insulation cross\nreproduces in every WT replicate")

# ---- H/I/J: whole-CID rescaled pileup ----
ticks = [0, t, 2 * t, RES_SIZE - 1]
ticklabels = ["-1×", "0", "CID", "+1×"]
panels_hij = [
    (pc, "H", f"Whole-CID rescaled pileup\n({n_used} confirmed CIDs, inside/flank={ratio_c:.2f})",
     oe_lo, oe_hi, "agg. O/E"),
    (pr, "I", f"Random same-size regions\n(3 sets, inside/flank={ratio_r:.2f})",
     oe_lo, oe_hi, "agg. O/E"),
    (diff, "J", "Difference (CID − random)\nenrichment concentrated inside the domain",
     -dmax, dmax, "Δ agg. O/E"),
]
for i, (m_, letter, title, vmin, vmax, clab) in enumerate(panels_hij):
    ax = fig.add_subplot(gs[3, i])
    im = ax.imshow(m_, cmap=cmap_oe, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.add_patch(Rectangle((t - 0.5, t - 0.5), t, t, fill=False,
                           edgecolor="black", lw=1.2))
    ax.set_xticks(ticks, ticklabels, fontsize=8)
    ax.set_yticks(ticks, ticklabels, fontsize=8)
    plabel(ax, letter, title)
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label(clab, fontsize=8)
    cb.ax.tick_params(labelsize=8)

fig.suptitle("Validation of re-called CID boundaries in M. tuberculosis Hi-C (response to Reviewer 4, comment 2)",
             fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig_CID_validation_merged.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig_CID_validation_merged.png"), dpi=300, bbox_inches="tight")

json.dump(dict(
    candidates=len(boundaries), confirmed=len(confirmed_bins),
    candidate_only_kb=[b * 5 for b in cand_only],
    depletion_coolpup=depl_cp,
    permutation_p=float(perm_p),
    per_replicate_depletion={k: float(v) for k, v in rep_depl.items()},
    whole_cid_inside_over_flank=ratio_c,
    random_inside_over_flank=ratio_r,
), open(os.path.join(OUT, "tables/merged_v3_summary.json"), "w"), indent=2)
print("saved fig_CID_validation_merged")
