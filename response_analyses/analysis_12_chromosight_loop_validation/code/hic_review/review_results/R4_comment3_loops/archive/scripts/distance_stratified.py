#!/usr/bin/env python
"""Distance-stratified validation of the 71 HiCCUPS loops.

Key finding: 50/71 HiCCUPS loops span <=100 kb, where bacterial Hi-C contacts
are intrinsically high; their O/E is ~1.0 (indistinguishable from distance
expectation). The 9 long-range loops (>200 kb) show genuine enrichment
(median O/E 1.43; 78% > 1.1) and NapM dependence.

This script:
  1. APA stratified by loop distance (short vs long-range)
  2. per-loop empirical FDR using a +100 kb shifted null, per distance stratum
  3. zoomed contact maps for all 9 long-range loops (WT + KO side by side)
  4. replicate reproducibility, short vs long
  5. combined final figure

Run: ../.venv_cooltools/bin/python distance_stratified.py
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

BINSIZE = 5000
FLANK = 50000
GENOME = 4411532
SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 60)
print("Distance-stratified loop validation")
print("=" * 60)

# ---------------- load ----------------
clr_wt = cooler.Cooler(WT_COOL)
clr_ko = cooler.Cooler(KO_COOL)
loops = pd.read_csv(LOOPS, sep="\t", header=None,
                    names=["chrom1", "start1", "end1", "chrom2", "start2", "end2",
                           "name", "score", "strand1", "strand2"])
loops["center1"] = ((loops.start1 + loops.end1) // 2).astype(int)
loops["center2"] = ((loops.start2 + loops.end2) // 2).astype(int)
loops["dist_kb"] = (loops.center2 - loops.center1) // 1000

exp_wt = cooltools.expected_cis(clr_wt, nproc=1)
exp_ko = cooltools.expected_cis(clr_ko, nproc=1)

# merge per-loop O/E from previous step
pl_prev = pd.read_csv(os.path.join(OUT, "tables/per_loop_oe_wt_vs_ko.csv"))
loops["OE_WT"] = pl_prev["OE_center5x5_WT"].values
loops["OE_KO"] = pl_prev["OE_center5x5_napMKO"].values

# strata
loops["stratum"] = pd.cut(loops.dist_kb,
                          bins=[0, 100, 200, 1000],
                          labels=["<=100 kb", "100-200 kb", ">200 kb"])
print(loops.groupby("stratum", observed=True).size())

# ---------------- helpers ----------------
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

def bedpe_of(df):
    return df[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].reset_index(drop=True)

def shifted_bedpe(df, shift=100000):
    sb = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    sb[["start1", "end1", "start2", "end2"]] += shift
    ok = sb.end2 + FLANK < GENOME
    return sb[ok].reset_index(drop=True)

# ---------------- 1. stratified APA ----------------
print("\n[1] stratified APA ...")
strata = {}
for name in ["<=100 kb", "100-200 kb", ">200 kb"]:
    sub = loops[loops.stratum == name]
    if len(sub) == 0:
        continue
    bp = bedpe_of(sub)
    stack = cooltools.pileup(clr_wt, bp, expected_df=exp_wt, flank=FLANK, nproc=1)
    # shifted null
    sbp = shifted_bedpe(sub)
    stack_sh = cooltools.pileup(clr_wt, sbp, expected_df=exp_wt, flank=FLANK, nproc=1)
    # KO
    stack_ko = cooltools.pileup(clr_ko, bp, expected_df=exp_ko, flank=FLANK, nproc=1)
    strata[name] = dict(
        n=len(sub),
        pile_wt=aggregate(stack),
        pile_shift=aggregate(stack_sh),
        pile_ko=aggregate(stack_ko),
        pl_wt=per_loop_center(stack),
        pl_shift=per_loop_center(stack_sh),
        pl_ko=per_loop_center(stack_ko),
        n_shift=len(sbp),
    )
    for k in ["pile_wt", "pile_shift", "pile_ko"]:
        np.save(os.path.join(OUT, f"tables/apa_stratum_{name.replace(' ','_').replace('<','lt').replace('>','gt')}_{k}.npy"),
                strata[name][k])

apa_strata_rows = []
for name, d in strata.items():
    for cond, pile in [("WT", d["pile_wt"]), ("shift+100kb", d["pile_shift"]), ("napM-KO", d["pile_ko"])]:
        e = enrich(pile)
        apa_strata_rows.append(dict(stratum=name, condition=cond, n=d["n"], **e))
apa_strata_df = pd.DataFrame(apa_strata_rows)
apa_strata_df.to_csv(os.path.join(OUT, "tables/apa_by_distance.csv"), index=False)
print(apa_strata_df[["stratum", "condition", "n", "center_5x5", "center_pixel", "enrichment"]].round(3))

# ---------------- 2. per-loop empirical FDR (shifted null) ----------------
print("\n[2] per-loop FDR from shifted null ...")
fdr_rows = []
for name, d in strata.items():
    null_vals = d["pl_shift"][np.isfinite(d["pl_shift"])]
    real_vals = d["pl_wt"][np.isfinite(d["pl_wt"])]
    for thr in [1.1, 1.25, 1.5]:
        fp = float((null_vals > thr).mean()) if len(null_vals) else np.nan
        tp = int((real_vals > thr).sum())
        fdr = fp * len(real_vals) / tp if tp > 0 else np.nan
        fdr_rows.append(dict(stratum=name, threshold=thr,
                             frac_null_above=fp, n_real_above=tp,
                             empirical_FDR=fdr))
fdr_df = pd.DataFrame(fdr_rows)
fdr_df.to_csv(os.path.join(OUT, "tables/perloop_fdr_shifted_null.csv"), index=False)
print(fdr_df.round(3))

# ---------------- 3. zoomed maps: all 9 long-range loops, WT vs KO ----------------
print("\n[3] zoomed maps for >200 kb loops ...")
long_loops = loops[loops.stratum == ">200 kb"].sort_values("OE_WT", ascending=False).reset_index(drop=True)
cmap = LinearSegmentedColormap.from_list("redwhite", ["#ffffff", "#d7301f"])

fig = plt.figure(figsize=(16, 4 * len(long_loops)))
gs = gridspec.GridSpec(len(long_loops), 2, figure=fig, hspace=0.4, wspace=0.25)
for idx, (_, row) in enumerate(long_loops.iterrows()):
    mid1, mid2 = int(row.center1), int(row.center2)
    span = mid2 - mid1
    window = min(max(span // 2, 100000), 400000)
    r1_s, r1_e = max(0, mid1 - window), mid1 + window
    r2_s, r2_e = max(0, mid2 - window), mid2 + window
    for j, (clr, cond) in enumerate([(clr_wt, "WT"), (clr_ko, "napM-KO")]):
        ax = fig.add_subplot(gs[idx, j])
        m = clr.matrix(sparse=False).fetch(f"Chromosome:{r1_s}-{r1_e}", f"Chromosome:{r2_s}-{r2_e}")
        im = ax.imshow(m, cmap=cmap,
                       vmin=np.nanpercentile(m[m > 0], 2),
                       vmax=np.nanpercentile(m, 98))
        ax.axhline((mid2 - r2_s) / (r2_e - r2_s) * m.shape[0], color="blue", ls="--", lw=1)
        ax.axvline((mid1 - r1_s) / (r1_e - r1_s) * m.shape[1], color="blue", ls="--", lw=1)
        oe_val = row.OE_WT if cond == "WT" else row.OE_KO
        ax.set_title(f"{mid1//1000}kb x {mid2//1000}kb ({cond})\nO/E = {oe_val:.2f}", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle("Long-range loops (>200 kb): zoomed contact maps, WT vs napM-KO", fontsize=13, y=1.0)
fig.savefig(os.path.join(OUT, "figures/fig10_zoomed_longrange_loops.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig10_zoomed_longrange_loops.png"), dpi=300, bbox_inches="tight")

# ---------------- 4. replicate reproducibility by stratum ----------------
print("\n[4] replicate reproducibility by stratum ...")
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
for name, d in strata.items():
    sub = loops[loops.stratum == name]
    for rep_id, m in rep_mats.items():
        rec = 0
        tot = 0
        for _, row in sub.iterrows():
            b1, b2 = int(row.center1) // BINSIZE, int(row.center2) // BINSIZE
            if b1 >= m.shape[0] or b2 >= m.shape[1]:
                continue
            tot += 1
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
        rep_rows.append(dict(stratum=name, replicate=rep_id,
                             recovered=rec, total=tot,
                             recovery_rate=rec / tot if tot else np.nan))
rep_strata_df = pd.DataFrame(rep_rows)
rep_strata_df.to_csv(os.path.join(OUT, "tables/replicate_recovery_by_distance.csv"), index=False)
print(rep_strata_df.pivot(index="replicate", columns="stratum", values="recovery_rate").round(3))

# ---------------- 5. long-range loops: WT vs KO statistics ----------------
print("\n[5] long-range loops WT vs KO ...")
lr = loops[loops.stratum == ">200 kb"]
ok = np.isfinite(lr.OE_WT) & np.isfinite(lr.OE_KO)
w_stat, w_p = stats.wilcoxon(lr.OE_WT[ok], lr.OE_KO[ok])
print(f"long-range loops (n={ok.sum()}): WT median O/E={lr.OE_WT.median():.3f}, "
      f"KO median O/E={lr.OE_KO.median():.3f}, Wilcoxon p={w_p:.3e}")

# ---------------- combined final figure ----------------
print("\n[6] combined figure ...")
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

# A: per-loop O/E vs distance (the money plot)
ax = fig.add_subplot(gs[0, 0])
colors = np.where(loops.OE_WT > 1.1, "#d7301f", "#999999")
ax.scatter(loops.dist_kb, loops.OE_WT, c=colors, s=30, alpha=0.8, edgecolor="black", lw=0.3)
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.axhline(1.1, color="grey", ls=":", lw=0.8)
ax.axvline(200, color="blue", ls="--", lw=0.8)
ax.set_xlabel("Loop distance (kb)", fontsize=9)
ax.set_ylabel("Per-loop center O/E (5x5, WT)", fontsize=9)
ax.set_title("A  Enrichment vs loop distance\nred: O/E>1.1; blue line: 200 kb", fontsize=9.5)

# B-E: APA heatmaps, shared color scale clipped at 2.2 for visibility
vmax_apa = 2.2

# B: APA short loops
ax = fig.add_subplot(gs[0, 1])
m = strata["<=100 kb"]["pile_wt"]
im = ax.imshow(m, cmap=cmap, vmin=0.9, vmax=vmax_apa)
c = m.shape[0] // 2
ax.add_patch(plt.Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_title(f"B  APA: short loops (<=100kb, n={strata['<=100 kb']['n']})\ncenter px = {enrich(m)['center_pixel']:.2f}", fontsize=9.5)
plt.colorbar(im, ax=ax, shrink=0.7)

# C: APA long loops
ax = fig.add_subplot(gs[0, 2])
m = strata[">200 kb"]["pile_wt"]
im = ax.imshow(m, cmap=cmap, vmin=0.9, vmax=vmax_apa)
ax.add_patch(plt.Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_title(f"C  APA: long-range loops (>200kb, n={strata['>200 kb']['n']})\ncenter px = {enrich(m)['center_pixel']:.2f}", fontsize=9.5)
plt.colorbar(im, ax=ax, shrink=0.7)

# D: APA long loops on KO
ax = fig.add_subplot(gs[0, 3])
m = strata[">200 kb"]["pile_ko"]
im = ax.imshow(m, cmap=cmap, vmin=0.9, vmax=vmax_apa)
ax.add_patch(plt.Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_title(f"D  Same long loops on napM-KO\ncenter px = {enrich(m)['center_pixel']:.2f}", fontsize=9.5)
plt.colorbar(im, ax=ax, shrink=0.7)

# E: shifted null APA (long)
ax = fig.add_subplot(gs[1, 0])
m = strata[">200 kb"]["pile_shift"]
im = ax.imshow(m, cmap=cmap, vmin=0.9, vmax=vmax_apa)
ax.add_patch(plt.Rectangle((c-2.5, c-2.5), 5, 5, fill=False, edgecolor="black", lw=0.8))
ax.set_title(f"E  Long loops +100 kb shift (null)\ncenter px = {enrich(m)['center_pixel']:.2f}", fontsize=9.5)
plt.colorbar(im, ax=ax, shrink=0.7)

# F: APA center pixel: real vs shifted null, per stratum
ax = fig.add_subplot(gs[1, 1])
strata_names = ["<=100 kb", "100-200 kb", ">200 kb"]
real_px = [enrich(strata[s]["pile_wt"])["center_pixel"] for s in strata_names]
shift_px = [enrich(strata[s]["pile_shift"])["center_pixel"] for s in strata_names]
xpos = np.arange(3)
wbar = 0.35
ax.bar(xpos - wbar/2, real_px, wbar, label="real loops", color="#d7301f", edgecolor="black")
ax.bar(xpos + wbar/2, shift_px, wbar, label="+100 kb shift (null)", color="#999999", edgecolor="black")
ax.axhline(1.0, color="black", ls="--", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(["<=100 kb", "100-200 kb", ">200 kb"], fontsize=9)
ax.set_ylabel("APA center pixel O/E", fontsize=9)
ax.set_title("F  Focal enrichment: real vs shifted\nper distance stratum", fontsize=9.5)
ax.legend(fontsize=8)
for i, (rv, sv) in enumerate(zip(real_px, shift_px)):
    ax.text(i - wbar/2, rv + 0.1, f"{rv:.2f}", ha="center", fontsize=8)
    ax.text(i + wbar/2, sv + 0.1, f"{sv:.2f}", ha="center", fontsize=8)

# G: replicate recovery by stratum
ax = fig.add_subplot(gs[1, 2])
piv = rep_strata_df.pivot(index="replicate", columns="stratum", values="recovery_rate") * 100
piv.plot(kind="bar", ax=ax, color=["#999999", "#8ab07c", "#297270"], edgecolor="black")
ax.set_ylabel("Recovery (%)", fontsize=9)
ax.set_title("G  Replicate recovery by distance", fontsize=9.5)
ax.legend(fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=0)

# H: long loops WT vs KO scatter
ax = fig.add_subplot(gs[1, 3])
ax.scatter(lr.OE_WT, lr.OE_KO, s=40, color="#d7301f", alpha=0.8, edgecolor="black", lw=0.3)
lim = max(lr.OE_WT.max(), lr.OE_KO.max()) * 1.1
ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
ax.set_xlabel("O/E in WT", fontsize=9)
ax.set_ylabel("O/E in napM-KO", fontsize=9)
ax.set_title(f"H  Long-range loops: WT vs KO\nWilcoxon p = {w_p:.1e}", fontsize=9.5)

fig.suptitle("Figure: Distance-stratified validation of Hi-C loops — only long-range contacts (>200 kb) are bona fide",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/fig11_distance_stratified_summary.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig11_distance_stratified_summary.png"), dpi=300, bbox_inches="tight")

# ---------------- summary ----------------
summary = {
    "strata": {name: dict(n=d["n"],
                          median_OE_WT=float(np.nanmedian(d["pl_wt"])),
                          median_OE_shift=float(np.nanmedian(d["pl_shift"])),
                          median_OE_KO=float(np.nanmedian(d["pl_ko"])),
                          APA_center_pixel_WT=enrich(d["pile_wt"])["center_pixel"],
                          APA_center_pixel_shift=enrich(d["pile_shift"])["center_pixel"],
                          APA_center_pixel_KO=enrich(d["pile_ko"])["center_pixel"])
               for name, d in strata.items()},
    "longrange_WT_vs_KO_wilcoxon_p": float(w_p),
    "fdr_table": fdr_df.to_dict("records"),
}
with open(os.path.join(OUT, "tables/distance_stratified_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + json.dumps(summary["strata"], indent=2))
print("done.")
