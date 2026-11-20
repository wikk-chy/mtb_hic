#!/usr/bin/env python3
"""Combined figure for the NapM binding-distribution analyses (R2 comment 6).

Panels:
  A  genome-wide continuous ChIP-seq signal track + curated-peak stems
  B  peak-width histogram (median 363 bp)
  C  inter-peak gap histogram
  D  spatial clustering: mean nearest-neighbour distance vs random null
  E  merged coverage vs score threshold (sensitivity curve)
  F  GC fraction of peak regions vs length-matched null
  G  top enriched 8-mers (AT-rich, no strong consensus)
  H  positional annotation: observed vs circular-shift null

Data: R1_mechanism_reanalysis curated peak table, bedgraph signal, tables/
of binding_distribution. Seed 42. Output: figures/fig_binding_distribution.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review"
BASE = f"{ROOT}/hic_review"
OUT = f"{BASE}/review_results/R2_comments/binding_distribution"
PK = f"{BASE}/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
BG = f"{BASE}/review_data/chip_seq_data/Rv0047c.bedgraph"
GENOME_LEN = 4_411_532
SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.6})
C_PK, C_SIG, C_NULL = "#665A7A", "#3D8BC3", "#BBBBBB"

pk = pd.read_csv(PK, sep="\t")
widths = (pk.end - pk.start).values
gaps = np.diff(np.sort(pk.center.values))
sc = pd.read_csv(f"{OUT}/tables/coverage_threshold_sensitivity.csv")
gc_tab = pd.read_csv(f"{OUT}/tables/gc_composition.csv")
kmer = pd.read_csv(f"{OUT}/tables/kmer8_enrichment.csv")
pos = pd.read_csv(f"{OUT}/tables/position_annotation.csv")

fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4))
gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.75, wspace=0.55,
                       left=0.065, right=0.985, top=0.90, bottom=0.09)

def plabel(ax, letter):
    ax.text(-0.12, 1.10, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")

# ---- A: signal track + peak stems (full width) ----
gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, :],
                                       height_ratios=[2.2, 1.0], hspace=0.05)
axA1 = fig.add_subplot(gsA[0])
bg = pd.read_csv(BG, sep="\t", header=None,
                 names=["chrom", "start", "end", "sig"])
bincen = (bg.start.values + bg.end.values) / 2 / 1e6
axA1.fill_between(bincen, bg.sig.values, color=C_SIG, alpha=0.5, lw=0)
axA1.plot(bincen, bg.sig.values, color=C_SIG, lw=0.3)
axA1.set_ylabel("ChIP signal", fontsize=7)
axA1.set_xlim(0, GENOME_LEN / 1e6)
axA1.set_xticks([])
q95 = np.quantile(bg.sig.values, 0.95)
axA1.axhline(q95, color="grey", ls="--", lw=0.7)
axA1.text(GENOME_LEN / 1e6 * 0.995, q95 * 0.75, "q95", fontsize=6, ha="right")
axA1.set_title("Rv0047c continuous ChIP-seq signal (whole genome)",
               loc="left", fontsize=8, fontweight="bold", pad=2)
plabel(axA1, "A")
axA2 = fig.add_subplot(gsA[1], sharex=axA1)
axA2.vlines(pk.center.values / 1e6, 0, pk.score.values, color=C_PK, lw=0.6)
axA2.set_ylabel("peak score", fontsize=7)
axA2.set_xlabel("genomic position (Mb)", fontsize=7)
axA2.set_ylim(0, 1.1)
axA2.set_title("634 curated peaks", loc="left", fontsize=8, pad=2)

# ---- B: peak width ----
axB = fig.add_subplot(gs[1, 0:3])
axB.hist(widths, bins=40, color=C_PK, edgecolor="white", lw=0.3)
axB.axvline(np.median(widths), color="black", ls="--", lw=0.8)
axB.set_xlabel("peak width (bp)")
axB.set_ylabel("count")
axB.set_title(f"B  peak width (median {int(np.median(widths))} bp)",
              loc="left", fontsize=8, fontweight="bold")

# ---- C: inter-peak gaps ----
axC = fig.add_subplot(gs[1, 3:6])
axC.hist(np.log10(gaps), bins=40, color=C_PK, edgecolor="white", lw=0.3)
axC.set_xlabel("log10 inter-peak gap (bp)")
axC.set_title(f"C  gaps (median {int(np.median(gaps))} bp)", loc="left",
              fontsize=8, fontweight="bold")

# ---- D: spatial clustering vs null ----
axD = fig.add_subplot(gs[1, 6:9])
centers = np.sort(pk.center.values)
# nearest-neighbour = min(forward, backward) gap, circular
fwd = np.diff(np.r_[centers, centers[0] + GENOME_LEN])
bwd = np.diff(np.r_[centers[-1] - GENOME_LEN, centers])
nn_obs = np.minimum(fwd, bwd)
obs_nn = nn_obs.mean()
null_nn = np.empty(10000)
for i in range(10000):
    rp = np.sort(rng.uniform(0, GENOME_LEN, len(centers)))
    f = np.diff(np.r_[rp, rp[0] + GENOME_LEN])
    b = np.diff(np.r_[rp[-1] - GENOME_LEN, rp])
    null_nn[i] = np.minimum(f, b).mean()
axD.hist(null_nn, bins=60, color=C_NULL, edgecolor="white", lw=0.2)
axD.axvline(obs_nn, color=C_PK, lw=1.5)
p_nn = (np.sum(np.abs(null_nn - null_nn.mean())
               >= abs(obs_nn - null_nn.mean())) + 1) / 10001
axD.set_xlabel("mean nearest-neighbour distance (bp)")
axD.set_ylabel("count (10k sims)")
axD.set_title(f"D  spacing vs null (p = {p_nn:.2f})",
              loc="left", fontsize=8, fontweight="bold")

# ---- E: coverage sensitivity ----
axE = fig.add_subplot(gs[1, 9:12])
sub = sc[sc.definition == "curated634_score>=thr"]
axE.plot(sub.threshold, sub.coverage * 100, "o-", ms=3, color=C_PK, lw=1)
axE.set_xlabel("score threshold (634-peak table)")
axE.set_ylabel("merged coverage (% genome)")
axE.set_title("E  coverage vs threshold", loc="left", fontsize=8,
              fontweight="bold")

# ---- F: GC vs null ----
axF = fig.add_subplot(gs[2, 0:4])
gv = gc_tab.set_index("metric")["value"]
lo, hi = gv["null_gc_lenmatched_q025"], gv["null_gc_lenmatched_q975"]
xn = np.linspace(lo - 0.001, hi + 0.001, 200)
sd_n = (hi - lo) / 3.92
mu_n = gv["null_gc_lenmatched_mean"]
axF.fill_between(xn, np.exp(-0.5 * ((xn - mu_n) / sd_n) ** 2), color=C_NULL,
                 alpha=0.8)
axF.axvline(gv["peak_gc_bp_weighted"], color=C_PK, lw=1.5, label="peaks")
axF.axvline(gv["genome_gc"], color="black", ls="--", lw=0.8, label="genome")
axF.set_xlabel("GC fraction")
axF.set_ylabel("density (null)")
axF.legend(fontsize=6.5, frameon=False)
axF.set_title(f"F  GC: peaks {gv['peak_gc_bp_weighted']:.3f} vs "
              f"genome {gv['genome_gc']:.3f}", loc="left", fontsize=8,
              fontweight="bold")

# ---- G: top enriched 8-mers ----
axG = fig.add_subplot(gs[2, 4:8])
top = kmer.head(10).iloc[::-1]
yc = [c for c in kmer.columns if "log2" in c or "enrich" in c][0]
axG.barh(range(len(top)), top[yc].values, color=C_PK)
axG.set_yticks(range(len(top)))
axG.set_yticklabels(top.iloc[:, 0].values, fontsize=6, family="monospace")
axG.set_xlabel("log2 enrichment vs GC/len-matched bg")
axG.set_title("G  top enriched 8-mers (AT-rich)", loc="left", fontsize=8,
              fontweight="bold")

# ---- H: position annotation ----
axH = fig.add_subplot(gs[2, 8:12])
cats = pos["category"].values
x = np.arange(len(cats))
w = 0.38
axH.bar(x - w / 2, pos.observed_frac.values, w, color=C_PK,
        label="observed")
axH.bar(x + w / 2, pos.null_mean.values / 634, w, color=C_NULL,
        label="circular-shift null")
lab = [c.replace("promoter_300bp", "promoter\n(−300 bp)").replace("_", "\n")
       for c in cats]
axH.set_xticks(x)
axH.set_xticklabels(lab, fontsize=6.5)
axH.set_ylabel("fraction of 634 peaks")
axH.legend(fontsize=6.5, frameon=False)
for i, r in pos.iterrows():
    star = "***" if r.p_two_sided < 0.001 else ("ns" if r.p_two_sided > 0.05 else "*")
    axH.text(i - w / 2, r.observed_frac + 0.03, star, ha="center", fontsize=8)
axH.set_title("H  positional annotation vs null", loc="left", fontsize=8,
              fontweight="bold")

fig.suptitle("NapM binding distribution: discrete, AT-biased loci widely "
             "scattered across the chromosome", fontsize=11,
             fontweight="bold")
fig.savefig(f"{OUT}/figures/fig_binding_distribution.png", dpi=300,
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/fig_binding_distribution.pdf",
            bbox_inches="tight")
print("saved figures/fig_binding_distribution.{png,pdf}")
