#!/usr/bin/env python3
"""R2C6 final combined figure: NapM binding, expression integration and
interaction evidence (Reviewer 2, Comment 6).

Panels:
  A  genome-wide continuous ChIP-seq signal + curated-peak stems
  B  merged coverage vs score threshold
  C  peak spacing vs random null (nearest neighbour, circular)
  D  GC fraction of peaks vs length-matched null
  E  top enriched 8-mers (AT-rich, no strong consensus)
  F  positional annotation vs circular-shift null
  G  DEG fraction by binding tier (OR ~0.88, no enrichment)
  H  peak score vs |log2FC| (Spearman rho = -0.01)
  I  NapM ego network (14 PPI neighbours + DNA-binding edges)

Inputs: R1 curated peak table, bedgraph, R2_comments tables (existing
analyses). Seed 42. Output: R2C6_final/figures/Fig_R2C6_combined.{png,pdf}
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review"
BASE = f"{ROOT}/hic_review"
OUT = f"{BASE}/review_results/R2_comments/R2C6_final"
os.makedirs(f"{OUT}/figures", exist_ok=True)
PK = f"{BASE}/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
BG = f"{BASE}/review_data/chip_seq_data/Rv0047c.bedgraph"
EDGES = f"{BASE}/review_data/Interaction_network/full_edges.csv"
BD_TAB = f"{BASE}/review_results/R2_comments/binding_distribution/tables"
CR_TAB = f"{BASE}/review_results/R2_comments/chip_rna_integration/tables"
PPI_TAB = f"{BASE}/review_results/R2_comments/ppi_evidence/tables"
GENOME_LEN = 4_411_532
SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.6})
C_PK, C_SIG, C_NULL = "#665A7A", "#3D8BC3", "#BBBBBB"

pk = pd.read_csv(PK, sep="\t")

fig = plt.figure(figsize=(180 / 25.4, 210 / 25.4))
gs = gridspec.GridSpec(4, 12, figure=fig, hspace=0.85, wspace=0.6,
                       left=0.065, right=0.985, top=0.925, bottom=0.06,
                       height_ratios=[1.1, 1, 1, 1.6])

def plabel(ax, letter, x=-0.12):
    ax.text(x, 1.08, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")

# ---------------- A: signal + peaks ----------------
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
axA1.text(GENOME_LEN / 1e6 * 0.98, q95 * 0.72, "q95", fontsize=6, ha="right")
axA1.set_title("Rv0047c continuous ChIP-seq signal (whole genome)",
               loc="left", fontsize=8, fontweight="bold", pad=2)
plabel(axA1, "A")
axA2 = fig.add_subplot(gsA[1], sharex=axA1)
axA2.vlines(pk.center.values / 1e6, 0, pk.score.values, color=C_PK, lw=0.6)
axA2.set_ylabel("peak score", fontsize=7)
axA2.set_xlabel("genomic position (Mb)", fontsize=7)
axA2.set_ylim(0, 1.1)
axA2.set_title("634 curated peaks (merged span 5.30% of genome)", loc="left",
               fontsize=8, pad=3)

# ---------------- B: coverage vs threshold ----------------
sc = pd.read_csv(f"{BD_TAB}/coverage_threshold_sensitivity.csv")
axB = fig.add_subplot(gs[1, 0:3])
sub = sc[sc.definition == "curated634_score>=thr"]
axB.plot(sub.threshold, sub.coverage * 100, "o-", ms=3, color=C_PK, lw=1)
axB.set_xlabel("score threshold (634-peak table)")
axB.set_ylabel("merged coverage (% genome)")
axB.set_title("B  coverage vs threshold", loc="left", fontsize=8,
              fontweight="bold")

# ---------------- C: spacing vs null ----------------
axC = fig.add_subplot(gs[1, 3:6])
centers = np.sort(pk.center.values)
fwd = np.diff(np.r_[centers, centers[0] + GENOME_LEN])
bwd = np.diff(np.r_[centers[-1] - GENOME_LEN, centers])
obs_nn = np.minimum(fwd, bwd).mean()
null_nn = np.empty(10000)
for i in range(10000):
    rp = np.sort(rng.uniform(0, GENOME_LEN, len(centers)))
    f = np.diff(np.r_[rp, rp[0] + GENOME_LEN])
    b = np.diff(np.r_[rp[-1] - GENOME_LEN, rp])
    null_nn[i] = np.minimum(f, b).mean()
axC.hist(null_nn, bins=60, color=C_NULL, edgecolor="white", lw=0.2)
axC.axvline(obs_nn, color=C_PK, lw=1.5)
p_nn = (np.sum(np.abs(null_nn - null_nn.mean())
               >= abs(obs_nn - null_nn.mean())) + 1) / 10001
axC.set_xlabel("mean nearest-neighbour distance (bp)")
axC.set_ylabel("count (10k sims)")
axC.set_title(f"C  spacing vs null (p = {p_nn:.2f})", loc="left",
              fontsize=8, fontweight="bold")

# ---------------- D: GC vs null ----------------
gc_tab = pd.read_csv(f"{BD_TAB}/gc_composition.csv")
gv = gc_tab.set_index("metric")["value"]
axD = fig.add_subplot(gs[1, 6:9])
lo, hi = gv["null_gc_lenmatched_q025"], gv["null_gc_lenmatched_q975"]
xn = np.linspace(lo - 0.001, hi + 0.001, 200)
sd_n = (hi - lo) / 3.92
axD.fill_between(xn, np.exp(-0.5 * ((xn - gv["null_gc_lenmatched_mean"]) / sd_n) ** 2),
                 color=C_NULL, alpha=0.8)
axD.axvline(gv["peak_gc_bp_weighted"], color=C_PK, lw=1.5, label="peaks")
axD.axvline(gv["genome_gc"], color="black", ls="--", lw=0.8, label="genome")
axD.set_xlabel("GC fraction")
axD.set_yticks([])
axD.legend(fontsize=6.5, frameon=False)
axD.set_title(f"D  GC {gv['peak_gc_bp_weighted']:.3f} vs {gv['genome_gc']:.3f}", loc="left", fontsize=8,
              fontweight="bold")

# ---------------- E: top enriched 8-mers ----------------
kmer = pd.read_csv(f"{BD_TAB}/kmer8_enrichment.csv")
axE = fig.add_subplot(gs[1, 9:12])
top = kmer.head(10).iloc[::-1]
yc = [c for c in kmer.columns if "log2" in c or "enrich" in c][0]
axE.barh(range(len(top)), top[yc].values, color=C_PK)
axE.set_yticks(range(len(top)))
axE.set_yticklabels(top.iloc[:, 0].values, fontsize=6, family="monospace")
axE.set_xlabel("log2 enrichment vs GC/len-matched bg")
axE.set_title("E  enriched 8-mers (AT-rich)", loc="left", fontsize=8,
              fontweight="bold")

# ---------------- F: position annotation ----------------
pos = pd.read_csv(f"{BD_TAB}/position_annotation.csv")
axF = fig.add_subplot(gs[2, 0:4])
cats = pos["category"].values
x = np.arange(len(cats))
w = 0.38
axF.bar(x - w / 2, pos.observed_frac.values, w, color=C_PK, label="observed")
axF.bar(x + w / 2, pos.null_mean.values / 634, w, color=C_NULL,
        label="circular-shift null")
lab = [c.replace("promoter_300bp", "promoter\n(−300 bp)").replace("_", "\n")
       for c in cats]
axF.set_xticks(x)
axF.set_xticklabels(lab, fontsize=6.5)
axF.set_ylabel("fraction of 634 peaks")
axF.legend(fontsize=6.5, frameon=False)
for i, r in pos.iterrows():
    star = "***" if r.p_two_sided < 0.001 else ("ns" if r.p_two_sided > 0.05 else "*")
    axF.text(i - w / 2, r.observed_frac + 0.03, star, ha="center", fontsize=8)
axF.set_title("F  positional annotation vs null", loc="left", fontsize=8,
              fontweight="bold")

# ---------------- G: DEG fraction by binding tier ----------------
tier = pd.read_csv(f"{CR_TAB}/deg_overlap_enrichment.csv")
axG = fig.add_subplot(gs[2, 4:8])
tlab = {"direct_promoter": "promoter", "direct_promoter_or_body": "prom+body",
        "operon_extended_only": "operon\next.", "any_binding_associated": "any"}
x = np.arange(len(tier))
bg_frac = tier.n_deg_total_in_universe.iloc[0] / tier.universe_size.iloc[0]
axG.axhline(bg_frac, color="grey", ls="--", lw=0.8)
axG.text(-0.45, bg_frac + 0.008,
         f"background {bg_frac*100:.1f}%", fontsize=6, ha="left", color="grey")
axG.bar(x, tier.frac_deg.values, color=[C_PK, "#8A7FA8", "#B5AEC9", "#D0CADF"])
for xi, r in enumerate(tier.itertuples()):
    axG.text(xi, r.frac_deg + 0.006, f"{r.n_deg}/{r.n_genes} p={r.fisher_p:.2f}",
             ha="center", fontsize=5.5)
axG.set_xticks(x)
axG.set_xticklabels([tlab[t] for t in tier.tier], fontsize=6.5)
axG.set_ylabel("fraction DEG (KO vs WT)")
axG.set_ylim(0, 0.28)
axG.set_title("G  DEG fraction by binding tier", loc="left", fontsize=8,
              fontweight="bold")

# ---------------- H: peak score vs |log2FC| ----------------
gstat = pd.read_csv(f"{CR_TAB}/gene_binding_status.csv")
corr = pd.read_csv(f"{CR_TAB}/binding_vs_expression_corr.csv")
axH = fig.add_subplot(gs[2, 8:12])
bound = gstat[gstat.binding_tier.isin(["direct_body", "direct_promoter"])]
axH.scatter(bound.gene_peak_score_max, np.abs(bound.log2FC_KOvsWT), s=5,
            color=C_PK, alpha=0.4, lw=0)
rho = corr[(corr.x == "gene_peak_score_max")].iloc[0]
axH.set_xlabel("peak score (max per gene)")
axH.set_ylabel("|log2FC| (KO vs WT)")
axH.set_title(f"H  score vs |log2FC| (rho = {rho.spearman_rho:.2f}, "
              f"p = {rho.p:.2f})", loc="left", fontsize=8, fontweight="bold")

# ---------------- I: NapM ego network ----------------
axI = fig.add_subplot(gs[3, 2:10])
axI.axis("off")
edges = pd.read_csv(EDGES)
ppi = edges[(edges.interaction_type == "Protein-Protein") &
            ((edges.source == "Rv0047c") | (edges.target == "Rv0047c"))]
nbrs = sorted(set(ppi.source) | set(ppi.target) - {"Rv0047c"})
nb_tab = pd.read_csv(f"{PPI_TAB}/napm_ppi_neighbours.csv")
nb_deg = dict(zip(nb_tab.neighbour, nb_tab.degree_full))
nb_fc = dict(zip(nb_tab.neighbour, nb_tab.log2FC_KOvsWT))
nb_up = dict(zip(nb_tab.neighbour, nb_tab.akr_regulate_KOvsWT))
cx, cy = 0, 0.2
r0 = 1.6
ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(nbrs), endpoint=False)
axI.scatter([cx], [cy], s=1500, color=C_PK, edgecolor="black", zorder=3)
axI.text(cx, cy, "NapM", color="white", fontsize=9, fontweight="bold",
         ha="center", va="center", zorder=4)
for a, nb in zip(ang, nbrs):
    x1, y1 = r0 * np.cos(a), r0 * np.sin(a)
    axI.plot([cx, x1], [cy, y1], color="#7A6A94", lw=1.2, zorder=1)
    fc = nb_fc.get(nb, 0)
    col = "#e78a7a" if nb_up.get(nb) == "Up" else \
          "#7aa6d6" if nb_up.get(nb) == "Down" else "#CFCFCF"
    axI.scatter([x1], [y1], s=60 + 30 * nb_deg.get(nb, 1), color=col,
                edgecolor="black", lw=0.5, zorder=2)
    axI.text(x1 * 1.16, y1 * 1.16, nb, fontsize=6, ha="center", va="center")
# DNA edges summary node
axI.plot([cx, cx], [cy, -2.3], color="#2a9d8f", lw=1.4, ls=":", zorder=1)
axI.scatter([cx], [-2.3], s=1400, color="#2a9d8f", edgecolor="black", zorder=2)
axI.text(cx, -2.3, "427 DNA bins\n(5 kb, ChIP-seq)", color="white",
         fontsize=6.5, ha="center", va="center", zorder=3)
axI.text(0, 2.15, "NapM ego network: 14 PPI neighbours "
         "(no NAP/SMC/hub; no self-loop) + DNA-binding",
         fontsize=8, fontweight="bold", ha="center")
axI.text(0, -2.95, "PPI edges = author-built PPI network (Methods); "
         "red = up in KO, blue = down", fontsize=6, ha="center",
         color="grey")
axI.set_xlim(-2.4, 2.4)
axI.set_ylim(-3.2, 2.3)
axI.set_aspect("equal")
plabel(axI, "I", x=-0.02)

fig.suptitle("NapM binding, expression and interaction evidence (Reviewer 2, Comment 6)",
             fontsize=10.5, fontweight="bold", y=0.985)
fig.savefig(f"{OUT}/figures/Fig_R2C6_combined.png", dpi=300,
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/Fig_R2C6_combined.pdf", bbox_inches="tight")
print("saved", f"{OUT}/figures/Fig_R2C6_combined.{{png,pdf}}")
