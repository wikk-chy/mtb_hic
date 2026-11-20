#!/usr/bin/env python3
"""R2 Comment 6 - binding_distribution, part 5.

Positional annotation of the 634 curated NapM peaks relative to ENA gene
annotation (ASM19595v2): gene body / promoter (300 bp upstream of TSS,
strand-aware) / intergenic. Null model: circular shift of the actual peak
centers (preserves spacing structure), 1,000 shifts, seed=42.

Outputs: tables/position_annotation.csv,
         figures/fig5_position_annotation.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
PEAKS = BASE + "hic_review/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
GTF = BASE + "hic_review/annotation.gtf"
OUT_T = BASE + "hic_review/review_results/R2_comments/binding_distribution/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/binding_distribution/figures/"
GENOME_LEN = 4_411_532
SEED = 42
N_SHIFT = 1000
PROM = 300

# ---- load genes -------------------------------------------------------------
genes = []
with open(GTF) as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        gid = f[8].split('gene_id "')[1].split('"')[0]
        genes.append((gid, int(f[3]) - 1, int(f[4]), f[6]))  # 0-based half-open
gdf = pd.DataFrame(genes, columns=["gene", "start", "end", "strand"])

def _stabber(intervals):
    """Precompute sorted-start + running-max-end arrays for interval stabbing."""
    iv = sorted(intervals)
    st = np.array([s for s, e in iv])
    rmax = np.maximum.accumulate(np.array([e for s, e in iv]))
    return st, rmax

def _hit(st, rmax, c):
    i = np.searchsorted(st, c, side="right") - 1
    return (i >= 0) & (rmax[np.maximum(i, 0)] > c)

gene_st, gene_rmax = _stabber([(s, e) for s, e in zip(gdf.start, gdf.end)])
tss = np.where(gdf.strand == "+", gdf.start, gdf.end)
prom_iv = [(t - PROM, t) if s == "+" else (t, t + PROM)
           for t, s in zip(tss, gdf.strand)]
prom_st, prom_rmax = _stabber(prom_iv)

def annotate(centers):
    """Return counts of gene-body / promoter / intergenic for an array of centers."""
    centers = np.asarray(centers)
    hit_gb = _hit(gene_st, gene_rmax, centers)
    hit_pr = _hit(prom_st, prom_rmax, centers)
    n_gb = int(hit_gb.sum())
    n_pr = int((~hit_gb & hit_pr).sum())
    n_ig = int((~hit_gb & ~hit_pr).sum())
    return n_gb, n_pr, n_ig

peaks = pd.read_csv(PEAKS, sep="\t")
centers = np.sort(peaks.center.values)
obs = annotate(centers)

rng = np.random.default_rng(SEED)
null = np.empty((N_SHIFT, 3), dtype=int)
for i in range(N_SHIFT):
    shift = rng.integers(0, GENOME_LEN)
    shifted = (centers + shift) % GENOME_LEN
    null[i] = annotate(shifted)

rows = []
for j, cat in enumerate(["gene_body", "promoter_300bp", "intergenic"]):
    o = obs[j]
    nm = null[:, j].mean()
    p_enr = (null[:, j] >= o).mean()
    p_dep = (null[:, j] <= o).mean()
    rows.append(dict(category=cat, observed=o,
                     observed_frac=round(o / len(centers), 4),
                     null_mean=round(nm, 1),
                     null_q025=int(np.quantile(null[:, j], 0.025)),
                     null_q975=int(np.quantile(null[:, j], 0.975)),
                     fold=round(o / nm, 3) if nm > 0 else np.nan,
                     p_two_sided=round(2 * min(p_enr, p_dep), 4)))
out = pd.DataFrame(rows)
out.to_csv(OUT_T + "position_annotation.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))
ax = axes[0]
fr = [o / len(centers) for o in obs]
ax.bar(["gene\nbody", "promoter\n(-300 bp)", "inter-\ngenic"], fr,
       color=["#6a51a3", "#9e9ac8", "#bdbdbd"], width=0.6)
for k, v in enumerate(fr):
    ax.text(k, v + 0.01, f"{obs[k]}\n({v*100:.1f}%)", ha="center", fontsize=7)
ax.set_ylabel("fraction of 634 peaks"); ax.set_ylim(0, max(fr) * 1.35)
ax.set_title("A  observed annotation", loc="left", fontweight="bold")

ax = axes[1]
labels = ["gene body", "promoter", "intergenic"]
x = np.arange(3)
ax.bar(x - 0.18, out.observed_frac, width=0.36, color="#6a51a3",
       label="observed")
ax.bar(x + 0.18, out.null_mean / 634, width=0.36, color="lightgrey",
       label="circular-shift null")
for k, r in out.iterrows():
    ax.text(k - 0.18, r.observed_frac + 0.01, f"p={r.p_two_sided:.3f}",
            ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("fraction"); ax.legend(frameon=False, fontsize=7)
ax.set_title("B  observed vs null", loc="left", fontweight="bold")
ax.set_ylim(0, 1.0)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig5_position_annotation.png", dpi=300)
fig.savefig(OUT_F + "fig5_position_annotation.pdf")
print(out.to_string(index=False))
print("OK")
