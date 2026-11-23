#!/usr/bin/env python
"""Fig 2f remake with the new calls: NapM (Rv0047c) vs the average of
DNA-binding proteins across four network layers.

Metric definitions (2026-10-23, unified; the original notebook hardcoded the
old values and its metric definitions could not be recovered — these are the
re-implemented definitions, documented honestly):

  CID boundary : number of the 13 confirmed CID boundary bins bound
                 (bin-level DNA-Protein edge; NapM: interval-overlap bins)
  Loop         : number of the 41 validated Chromosight loops with BOTH
                 anchors bound by the protein
  Operon       : number of the 749 annotated operons with >=1 bound bin
  PPI          : protein's degree in the PPI network (Protein-Protein edges)

Average = mean across the proteins having >=1 DNA-Protein edge
("proteome-wide average of DNA-binding proteins", as in the manuscript).

Outputs: figures/fig2f_napm_importance.{pdf,png}, tables/fig2f_metrics.csv
"""
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment6")
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

EDGES = os.path.join(BASE, "review_data/Interaction_network/full_edges.csv")
BD_CSV = os.path.join(BASE, "review_results/R4_comment2_CID/tables/cid_boundaries_recalled.csv")
LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")
OPERONS = os.path.join(BASE, "review_data/operon_metadata/operon_list_updated_with_bin_coverage.csv")
NAPM_INTERVAL = os.path.join(BASE, "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_bound_bins_interval.csv")

RV = "Rv0047c"
SEED = 42

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 9, "axes.linewidth": 0.8})

# ---------------- data ----------------
edges = pd.read_csv(EDGES)
dp = edges[edges.interaction_type == "DNA-Protein"]
pp = edges[edges.interaction_type == "Protein-Protein"]

# bin (0-based) -> proteins; NapM overridden with the interval definition
bin2prot = defaultdict(set)
for _, r in dp.iterrows():
    bin2prot[int(str(r.target)[3:]) - 1].add(str(r.source))
for b in list(bin2prot):
    bin2prot[b].discard(RV)
napm_bins = set(pd.read_csv(NAPM_INTERVAL)["bin"].astype(int))
for b in napm_bins:
    bin2prot[b].add(RV)

proteins = sorted({p for ps in bin2prot.values() for p in ps})
print(f"DNA-binding proteins: {len(proteins)}")

# PPI degree
ppi_deg = defaultdict(int)
for _, r in pp.iterrows():
    ppi_deg[str(r.source)] += 1
    ppi_deg[str(r.target)] += 1

# confirmed CID boundary bins (0-based)
bd = pd.read_csv(BD_CSV)
bd_bins = sorted(bd.loc[bd.confirmed, "bin"].astype(int).tolist())

# 41 validated loops (0-based anchor bins)
cs = pd.read_csv(LOOPS, sep="\t")
loops = list(zip(cs.bin1.astype(int), cs.bin2.astype(int)))

# operons -> bins (parse the "bin1,bin2" column)
ops = pd.read_csv(OPERONS)
op_bins = []
for s in ops["bin"]:
    op_bins.append({int(x.strip().strip("'").replace("bin", "")) - 1
                    for x in str(s).split(",")})  # network bins are 1-based
print(f"operons: {len(ops)}")

# ---------------- per-protein metrics ----------------
rows = []
for p in proteins:
    bound_bins = {b for b, ps in bin2prot.items() if p in ps}
    n_cid = sum(b in bound_bins for b in bd_bins)
    n_loop = sum(b1 in bound_bins and b2 in bound_bins for b1, b2 in loops)
    n_op = sum(len(obs & bound_bins) > 0 for obs in op_bins)
    rows.append(dict(protein=p, cid_boundary=n_cid, loop=n_loop,
                     operon=n_op, ppi=ppi_deg.get(p, 0)))
met = pd.DataFrame(rows)
met.to_csv(os.path.join(OUT, "tables/fig2f_metrics.csv"), index=False)

rv = met[met.protein == RV].iloc[0]
avg = met[met.protein != RV].mean(numeric_only=True)

cats = ["Loop", "Operon", "PPI"]   # 2026-10-23: CID boundary layer dropped (user decision)
rv_vals = [rv.loop, rv.operon, rv.ppi]
avg_vals = [avg.loop, avg.operon, avg.ppi]
print("Rv0047c:", rv_vals)
print("average:", [round(v, 2) for v in avg_vals])

# ranks / percentiles (NapM vs other DNA-binding proteins)
pct = {}
for k, c in zip(cats, ["loop", "operon", "ppi"]):
    v = met[c]
    pct[k] = float((v < v[met.protein == RV].iloc[0]).mean() * 100)
print("percentile of NapM among DNA-binding proteins:", pct)

# ---------------- figure ----------------
labels = ["Loop\n(41 validated)", "Operon\n(749 annotated)",
          "PPI\n(author-built network)"]
folds = [r / a for r, a in zip(rv_vals, avg_vals)]
pcts = [pct[c] for c in cats]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
# left: raw counts, log y
x = np.arange(len(cats))
width = 0.38
r1 = axl.bar(x - width / 2, rv_vals, width, label="Rv0047c",
             color="#3B6FB6", edgecolor="black", linewidth=0.6)
r2 = axl.bar(x + width / 2, avg_vals, width,
             label=f"average of {len(proteins) - 1} DNA-binding proteins",
             color="#E07A2B", edgecolor="black", linewidth=0.6)
axl.set_yscale("log")
for rects, vals, fmt in ((r1, rv_vals, "{:.0f}"), (r2, avg_vals, "{:.2f}")):
    for rect, v in zip(rects, vals):
        axl.annotate(fmt.format(v), xy=(rect.get_x() + rect.get_width() / 2,
                     rect.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5)
axl.set_ylabel("Count (log scale)")
axl.set_xticks(x)
axl.set_xticklabels(labels, fontsize=9)
axl.legend(frameon=False, fontsize=8.5, loc="upper left")
axl.set_ylim(0.5, max(rv_vals) * 4)
axl.spines["top"].set_visible(False)
axl.spines["right"].set_visible(False)
axl.set_title("raw counts (log scale)", fontsize=9.5, loc="left")
axl.text(-0.18, 1.02, "A", transform=axl.transAxes, fontsize=12,
         fontweight="bold", va="bottom")

# right: fold over average + percentile
rb = axr.bar(x, folds, 0.55, color="#3B6FB6", edgecolor="black", linewidth=0.6)
axr.axhline(1.0, color="black", lw=0.8, ls="--")
for rect, f, pc in zip(rb, folds, pcts):
    axr.annotate(f"{f:.1f}×\n(top {100 - pc:.0f}%)",
                 xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center",
                 fontsize=8.5)
axr.set_ylabel("fold over the average of DNA-binding proteins")
axr.set_xticks(x)
axr.set_xticklabels(labels, fontsize=9)
axr.set_ylim(0, max(folds) * 1.35)
axr.spines["top"].set_visible(False)
axr.spines["right"].set_visible(False)
axr.set_title("enrichment over average (percentile = rank among "
              "DNA-binding proteins)", fontsize=9.5, loc="left")
axr.text(-0.14, 1.02, "B", transform=axr.transAxes, fontsize=12,
         fontweight="bold", va="bottom")

fig.suptitle("NapM (Rv0047c) across network layers "
             "(41 validated loops; NapM binding = interval-overlap)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/fig2f_napm_importance.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/fig2f_napm_importance.png"), dpi=300,
            bbox_inches="tight")

json.dump(dict(
    definitions=dict(
        cid_boundary="bound of 13 confirmed CID boundary bins",
        loop="loops with BOTH anchors bound (41 validated Chromosight loops)",
        operon="operons with >=1 bound bin (749 annotated)",
        ppi="degree in the author-built PPI network"),
    napm=dict(zip(cats, [float(v) for v in rv_vals])),
    average_dna_binding_proteins=dict(zip(cats, [float(v) for v in avg_vals])),
    napm_percentile_vs_dna_binding_proteins=pct,
    n_dna_binding_proteins=len(proteins),
    note="Old notebook values (21/10/7/14 vs 3.65/1.31/2.88/2.02) were "
         "hardcoded; metric definitions re-implemented and documented here.",
), open(os.path.join(OUT, "tables/fig2f_summary.json"), "w"), indent=2)
print("saved fig2f_napm_importance.{pdf,png} + tables")
