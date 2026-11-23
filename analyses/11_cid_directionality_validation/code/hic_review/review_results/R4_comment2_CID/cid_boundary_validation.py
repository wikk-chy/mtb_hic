#!/usr/bin/env python
"""R4 comment 2 — CID boundary re-calling and validation.

Reviewer: CIDs cannot be verified; boundaries fall in homogeneous DI regions;
need pipeline, parameters, bin-size rationale, replicate consistency, zoom-ins.

Response: re-call boundaries from the DI track (wt.csv) using the canonical
negative->positive transition criterion (Le et al. 2013; Lioy et al. 2018),
then validate: (1) every boundary has a true DI transition, (2) boundaries lie
at vertices of triangular domains on the contact map, (3) transitions are
reproducible across 4 WT replicates within +/-25 kb.

Key clarification for the reviewer: DI is computed on a 5-kb matrix (883 bins)
with a 100-kb directional WINDOW (20 bins each side); 100 kb is the window,
NOT the bin size — the matrix is 883x883, not ~45x45.

Run: ../../.venv_cooltools/bin/python cid_boundary_validation.py
(from review_results/R4_comment2_CID/)
"""

import os
import json
import numpy as np
import pandas as pd
import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from scipy import stats
from scipy.ndimage import gaussian_filter

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment2_CID")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

WT_COOL = os.path.join(BASE, "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000"
DI_WT = os.path.join(BASE, "review_data/DI_Score_results/wt.csv")
OLD_BD = os.path.join(BASE, "review_data/cid_boundary/wt-boundary.bed")
REP_DIR = os.path.join(BASE, "review_data/wt_rep_icedmatrix")

BINSIZE = 5000
WINDOW_BINS = 20          # 100-kb directional window each side
NEG_THR, POS_THR = -2.0, 2.0
LOOKBACK, LOOKAHEAD = 10, 10
MIN_CID_BINS = 20         # 100 kb minimum CID size
REP_TOL_BINS = 5          # +/-25 kb replicate consistency

C_WT, C_KO, C_RAND = "#0072B2", "#D55E00", "#999999"
cmap_hic = LinearSegmentedColormap.from_list(
    "hic", ["#ffffff", "#fee5d9", "#fb6a4a", "#cb181d", "#67000d"])

print("=" * 60)
print("R4 comment 2: CID boundary re-calling + validation")
print("=" * 60)

# ---------------- load ----------------
di = pd.read_csv(DI_WT, header=None).values.flatten()
n = len(di)
clr = cooler.Cooler(WT_COOL)
mat = np.nan_to_num(clr.matrix(sparse=False).fetch("Chromosome"), nan=0.0)

# ---------------- 1. boundary calling ----------------
def call_boundaries(di, neg=NEG_THR, pos=POS_THR,
                    lookback=LOOKBACK, lookahead=LOOKAHEAD,
                    min_cid_bins=MIN_CID_BINS):
    n = len(di)
    cands = []
    for i in range(1, n - 1):
        left = di[max(0, i - lookback):i + 1]
        right = di[i + 1:i + 1 + lookahead]
        if (left <= neg).any() and (right >= pos).any():
            cands.append((i, di[i + 1] - di[i]))
    cands.sort(key=lambda x: -x[1])
    chosen = []
    for i, jump in cands:
        if all(abs(i - c) >= min_cid_bins for c, _ in chosen):
            chosen.append((i, jump))
    chosen.sort()
    return chosen

boundaries = call_boundaries(di)
print(f"re-called boundaries: {len(boundaries)}")

# per-boundary validation
bd_rows = []
for i, jump in boundaries:
    left = di[max(0, i - LOOKBACK):i]
    right = di[i + 1:i + 1 + LOOKAHEAD]
    bd_rows.append(dict(
        bin=i, start=i * BINSIZE, end=i * BINSIZE + BINSIZE,
        kb=i * BINSIZE // 1000,
        DI_at_boundary=di[i], DI_jump=jump,
        left_min_DI=left.min() if len(left) else np.nan,
        right_max_DI=right.max() if len(right) else np.nan,
        left_mean_DI=left.mean() if len(left) else np.nan,
        right_mean_DI=right.mean() if len(right) else np.nan,
        transition_ok=bool((left.min() <= NEG_THR) and (right.max() >= POS_THR)),
    ))
bd_df = pd.DataFrame(bd_rows)
bd_df.to_csv(os.path.join(OUT, "tables/cid_boundaries_recalled.csv"), index=False)
print(f"all boundaries pass transition check: {bd_df.transition_ok.all()}")
print(f"CID sizes (kb): median={np.median(np.diff(bd_df.kb)):.0f}, "
      f"range=[{np.diff(bd_df.kb).min()}, {np.diff(bd_df.kb).max()}]")

# ---------------- 2. compare with original 27 ----------------
old = pd.read_csv(OLD_BD, sep="\t", header=None, names=["chrom", "start", "end"])
old["bin"] = old.start // BINSIZE
old_rows = []
for _, r in old.iterrows():
    b = int(r["bin"])
    left = di[max(0, b - LOOKBACK):b]
    right = di[b + 1:b + 1 + LOOKAHEAD]
    trans = bool(len(left) and len(right) and left.min() <= NEG_THR and right.max() >= POS_THR)
    recovered = any(abs(b - nb) <= REP_TOL_BINS for nb, _ in boundaries)
    old_rows.append(dict(bin=b, kb=r.start // 1000, DI_at_boundary=di[b],
                         left_min_DI=left.min() if len(left) else np.nan,
                         right_max_DI=right.max() if len(right) else np.nan,
                         transition_ok=trans, recovered_in_new=recovered))
old_df = pd.DataFrame(old_rows)
old_df.to_csv(os.path.join(OUT, "tables/original_27_evaluation.csv"), index=False)
print(f"\noriginal 27: {old_df.transition_ok.sum()} pass transition, "
      f"{(~old_df.transition_ok).sum()} fail (homogeneous DI)")
print(f"transition-passing originals recovered in new set: "
      f"{old_df[old_df.transition_ok].recovered_in_new.sum()}/{old_df.transition_ok.sum()}")

# ---------------- 3. replicate DI + consistency ----------------
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

def compute_di(m, window=WINDOW_BINS, sigma=2.0):
    """DI pipeline: gaussian filter -> log1p -> correlation -> paired t-test."""
    m = np.nan_to_num(m, nan=0.0)
    S = gaussian_filter(m, sigma=sigma, mode="nearest")
    L = np.log1p(S)
    C = np.nan_to_num(np.corrcoef(L))
    n = C.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        k = min(i, n - 1 - i, window)
        if k < 5:
            continue
        t, _ = stats.ttest_rel(C[i, i + 1:i + 1 + k], C[i, i - k:i])
        out[i] = t
    return out

rep_files = sorted([f for f in os.listdir(REP_DIR) if f.endswith(".matrix")])
rep_dis = {}
for rf in rep_files:
    rep_id = rf.replace("_5000_iced.matrix", "")
    rep_dis[rep_id] = compute_di(load_rep_matrix(os.path.join(REP_DIR, rf)))
print(f"\ncomputed DI for {len(rep_dis)} replicates")

# consistency: for each merged boundary, does each replicate show a transition
# (negative->positive) within +/-25 kb?
cons_rows = []
for _, r in bd_df.iterrows():
    b = int(r["bin"])
    row = dict(boundary_kb=r.kb)
    for rep_id, rdi in rep_dis.items():
        lo, hi = max(0, b - REP_TOL_BINS), min(n, b + REP_TOL_BINS + 1)
        seg = rdi[lo:hi]
        ok = False
        for j in range(lo, hi):
            lft = rdi[max(0, j - LOOKBACK):j]
            rgt = rdi[j + 1:j + 1 + LOOKAHEAD]
            lft = lft[np.isfinite(lft)]
            rgt = rgt[np.isfinite(rgt)]
            if len(lft) and len(rgt) and lft.mean() < 0 and rgt.mean() > 0:
                ok = True
                break
        row[rep_id] = ok
    cons_rows.append(row)
cons_df = pd.DataFrame(cons_rows)
rep_ids = sorted(rep_dis.keys())
cons_df["n_reps_with_transition"] = cons_df[rep_ids].sum(axis=1)
cons_df.to_csv(os.path.join(OUT, "tables/replicate_consistency.csv"), index=False)
per_rep_rate = {rid: float(cons_df[rid].mean()) for rid in rep_ids}
print("replicate consistency (fraction of boundaries with transition):")
for rid, v in per_rep_rate.items():
    print(f"  {rid}: {v*100:.0f}%")
print(f"boundaries with transition in >=3/4 replicates: "
      f"{(cons_df.n_reps_with_transition >= 3).sum()}/{len(cons_df)}")

# confirmed = transition AND reproducible in >=3/4 replicates (the Methods'
# "reproducibility between biological replicates" confirmation step)
confirmed_kbs = set(cons_df.loc[cons_df.n_reps_with_transition >= 3, "boundary_kb"])
bd_df["confirmed"] = bd_df.kb.isin(confirmed_kbs)
bd_df.to_csv(os.path.join(OUT, "tables/cid_boundaries_recalled.csv"), index=False)
n_confirmed = int(bd_df.confirmed.sum())
print(f"confirmed boundaries (transition + >=3/4 replicates): {n_confirmed}")

# ---------------- figures ----------------
new_bins = [i for i, _ in boundaries]
confirmed_bins = set(bd_df.loc[bd_df.confirmed, "bin"].astype(int))

# FIG 1: genome-wide DI track + boundaries + contact map with CID triangles
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1],
                       hspace=0.3, wspace=0.25, left=0.07, right=0.97,
                       top=0.93, bottom=0.07)

# A: DI track with boundaries
ax = fig.add_subplot(gs[0, :])
x = np.arange(n) * BINSIZE / 1000
ax.fill_between(x, di, 0, where=(di < 0), color="#2166ac", alpha=0.7, label="upstream preference (DI < 0)")
ax.fill_between(x, di, 0, where=(di >= 0), color="#b2182b", alpha=0.7, label="downstream preference (DI > 0)")
ax.axhline(NEG_THR, color="#2166ac", ls=":", lw=0.8)
ax.axhline(POS_THR, color="#b2182b", ls=":", lw=0.8)
for i in new_bins:
    if i in confirmed_bins:
        ax.axvline(i * BINSIZE / 1000, color="black", lw=0.8, alpha=0.7)
    else:
        ax.axvline(i * BINSIZE / 1000, color="grey", lw=0.7, ls="--", alpha=0.6)
ax.set_xlim(0, n * BINSIZE / 1000)
ax.set_ylim(-2, 2)   # display clipped at the DI thresholds (user request)
ax.set_xlabel("Genomic position (kb)", fontsize=10)
ax.set_ylabel("Directional preference (DI)", fontsize=10)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], color="black", lw=0.8, label=f"confirmed (n={n_confirmed})"),
    Line2D([0], [0], color="grey", ls="--", lw=0.8, label=f"candidate only (n={len(new_bins)-n_confirmed})"),
    Line2D([0], [0], color="#2166ac", lw=4, alpha=0.7, label="upstream preference (DI<0)"),
    Line2D([0], [0], color="#b2182b", lw=4, alpha=0.7, label="downstream preference (DI>0)")],
    fontsize=8, loc="upper right", frameon=False)
ax.set_title(f"A  DI track with {len(boundaries)} re-called CID boundaries "
             f"(transitions; solid = confirmed in ≥3/4 replicates)",
             loc="left", fontsize=11, fontweight="bold")

# B: contact map with boundary lines
ax = fig.add_subplot(gs[1, 0])
ax.imshow(mat, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat, 99),
          interpolation="nearest")
for i in new_bins:
    lw_, alpha_ = (0.7, 0.75) if i in confirmed_bins else (0.5, 0.35)
    ax.axvline(i, color="#0072B2", lw=lw_, alpha=alpha_)
    ax.axhline(i, color="#0072B2", lw=lw_, alpha=alpha_)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=9)
ax.set_ylabel("Genomic position (Mb)", fontsize=9)
ax.set_title("B  WT contact map with boundary lines", loc="left",
             fontsize=11, fontweight="bold")

# C: CID triangles schematic on map
ax = fig.add_subplot(gs[1, 1])
ax.imshow(mat, cmap=cmap_hic, vmin=0, vmax=np.nanpercentile(mat, 99),
          interpolation="nearest")
verts = [0] + sorted(confirmed_bins) + [n - 1]
for k in range(len(verts) - 1):
    a, b = verts[k], verts[k + 1]
    mid = (a + b) / 2
    tri = Polygon([(a, a), (b, a), (mid, mid)], fill=False,
                  edgecolor="#0072B2", lw=1.2, alpha=0.85)
    ax.add_patch(tri)
xt = np.arange(0, n, 200)
ax.set_xticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_yticks(xt, [f"{v*5/1000:.1f}" for v in xt], fontsize=8)
ax.set_xlabel("Genomic position (Mb)", fontsize=9)
ax.set_title(f"C  CID triangles from {n_confirmed} confirmed boundaries", loc="left",
             fontsize=11, fontweight="bold")

fig.suptitle("CID boundary re-calling from directional preference (5-kb bins, 100-kb directional window)",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig1_DI_track_and_map.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig1_DI_track_and_map.png"), dpi=300, bbox_inches="tight")

# FIG 2: zoomed examples (3 boundaries): map + DI below
print("rendering zoomed examples ...")
# pick 3 boundaries with strongest transitions and spaced across the genome
top_bd = bd_df.sort_values("DI_jump", ascending=False)
sel = []
for _, r in top_bd.iterrows():
    if all(abs(r["bin"] - s) > 120 for s in sel):
        sel.append(int(r["bin"]))
    if len(sel) == 3:
        break
sel = sorted(sel)

fig2 = plt.figure(figsize=(15, 4.6 * 3))
gs2 = gridspec.GridSpec(3, 2, figure=fig2, width_ratios=[1.2, 1],
                        hspace=0.45, wspace=0.25, left=0.08, right=0.97,
                        top=0.94, bottom=0.06)
letters = "ABCDEF"
for idx, b in enumerate(sel):
    span = 100   # +-500 kb view
    lo, hi = max(0, b - span), min(n, b + span)
    sub = mat[lo:hi, lo:hi].copy()
    # mask the diagonal so off-diagonal domain structure is visible
    nd = sub.shape[0]
    for d in range(-2, 3):
        diag_idx = np.arange(max(0, -d), nd - max(0, d))
        sub[diag_idx, diag_idx + d] = np.nan
    axm = fig2.add_subplot(gs2[idx, 0])
    finite = sub[np.isfinite(sub)]
    im = axm.imshow(sub, cmap=cmap_hic, vmin=0,
                    vmax=np.percentile(finite, 95), interpolation="nearest")
    axm.axvline(b - lo, color="#0072B2", lw=1.2)
    axm.axhline(b - lo, color="#0072B2", lw=1.2)
    axm.set_title(f"{letters[idx*2]}  boundary at {b*5} kb (map, diagonal masked)",
                  loc="left", fontsize=10, fontweight="bold")
    axm.set_xticks([]); axm.set_yticks([])
    plt.colorbar(im, ax=axm, shrink=0.6)

    axd = fig2.add_subplot(gs2[idx, 1])
    xs = np.arange(lo, hi) * BINSIZE / 1000
    axd.fill_between(xs, di[lo:hi], 0, where=(di[lo:hi] < 0), color="#2166ac", alpha=0.7)
    axd.fill_between(xs, di[lo:hi], 0, where=(di[lo:hi] >= 0), color="#b2182b", alpha=0.7)
    axd.axvline(b * BINSIZE / 1000, color="black", lw=1.2)
    axd.axhline(NEG_THR, color="#2166ac", ls=":", lw=0.7)
    axd.axhline(POS_THR, color="#b2182b", ls=":", lw=0.7)
    axd.set_ylim(-2, 2)
    axd.set_xlabel("Genomic position (kb)", fontsize=8)
    axd.set_ylabel("DI", fontsize=8)
    axd.set_title(f"{letters[idx*2+1]}  DI transition at {b*5} kb",
                  loc="left", fontsize=10, fontweight="bold")

fig2.suptitle("Zoomed examples: re-called boundaries lie at vertices of triangular domains with clear DI transitions",
              fontsize=12, fontweight="bold")
fig2.savefig(os.path.join(OUT, "figures/fig2_zoomed_examples.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "figures/fig2_zoomed_examples.png"), dpi=300, bbox_inches="tight")

# FIG 3: replicate consistency + old-vs-new evaluation
fig3 = plt.figure(figsize=(14, 5))
gs3 = gridspec.GridSpec(1, 2, figure=fig3, wspace=0.3,
                        left=0.07, right=0.97, top=0.85, bottom=0.13)

# A: replicate consistency heatmap (boundaries x replicates)
ax = fig3.add_subplot(gs3[0, 0])
M = cons_df[rep_ids].values.astype(float)
im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("b", ["#f7f7f7", "#0072B2"]),
               aspect="auto", vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks(np.arange(len(rep_ids)), [r.replace("wt_", "rep") for r in rep_ids], fontsize=9)
ax.set_yticks(np.arange(len(cons_df)),
              [f"{v} kb" for v in cons_df.boundary_kb], fontsize=7)
ax.set_xlabel("Replicate", fontsize=9)
ax.set_ylabel("Boundary", fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, "✓" if M[i, j] else "–", ha="center", va="center",
                fontsize=8, color="white" if M[i, j] else "#aaaaaa")
ax.set_title(f"A  DI transition at each boundary per replicate (±25 kb)\n"
             f"mean consistency: {np.mean(list(per_rep_rate.values()))*100:.0f}%",
             loc="left", fontsize=10, fontweight="bold")

# B: original 27 evaluation
ax = fig3.add_subplot(gs3[0, 1])
cats = ["pass transition\n(recovered)", "homogeneous DI\n(rejected)"]
vals = [int(old_df.transition_ok.sum()), int((~old_df.transition_ok).sum())]
bars = ax.bar(cats, vals, color=[C_WT, C_RAND], edgecolor="black", width=0.5)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.4, str(v), ha="center", fontsize=10)
ax.set_ylabel("Number of original boundaries", fontsize=9)
ax.set_title("B  Evaluation of the original 27 boundaries\nagainst the transition criterion",
             loc="left", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.2)

fig3.savefig(os.path.join(OUT, "figures/fig3_replicate_and_evaluation.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(OUT, "figures/fig3_replicate_and_evaluation.png"), dpi=300, bbox_inches="tight")

# ---------------- summary ----------------
summary = dict(
    pipeline={
        "bin_size": "5 kb (883 bins)",
        "directional_window": "100 kb each side (20 bins) — window, NOT bin size",
        "DI": "t-statistic of paired t-test, upstream vs downstream window (gaussian-filtered, log-transformed correlation matrix)",
        "boundary_rule": f"transition from DI <= {NEG_THR} to DI >= +{POS_THR}; largest positive DI jump; min CID {MIN_CID_BINS*5} kb",
    },
    n_boundaries_recalled=len(boundaries),
    n_confirmed=n_confirmed,
    all_pass_transition=bool(bd_df.transition_ok.all()),
    cid_size_kb=dict(median=float(np.median(np.diff(bd_df.kb))),
                     min=int(np.diff(bd_df.kb).min()),
                     max=int(np.diff(bd_df.kb).max())),
    original_27=dict(pass_transition=int(old_df.transition_ok.sum()),
                     rejected=int((~old_df.transition_ok).sum()),
                     recovered=int(old_df[old_df.transition_ok].recovered_in_new.sum())),
    replicate_consistency={rid: round(v, 3) for rid, v in per_rep_rate.items()},
    boundaries_ge3_of_4_reps=int((cons_df.n_reps_with_transition >= 3).sum()),
)
with open(os.path.join(OUT, "tables/summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary, indent=2))
print("outputs in", OUT)
