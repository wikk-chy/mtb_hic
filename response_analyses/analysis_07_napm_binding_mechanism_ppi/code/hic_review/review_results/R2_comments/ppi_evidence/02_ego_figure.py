#!/usr/bin/env python3
"""R2 Comment 6 - ppi_evidence, part 2.

Evidence-typed ego network of NapM (Rv0047c).
  - centre node: NapM
  - PPI neighbours (edges coloured by interaction type; node fill =
    functional category; node border = DEG status in KO vs WT)
  - DNA-binding associations are summarised as a glyph (NapM -> DNA bins
    count), not drawn as 400+ edges.

PPI edges are labelled "source to be confirmed" (full_edges.csv has no
evidence-type column).

Depends on tables from 01_napm_network.py.

Outputs: figures/fig1_ego_network.{png,pdf}
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
OUT_T = BASE + "hic_review/review_results/R2_comments/ppi_evidence/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/ppi_evidence/figures/"

nb = pd.read_csv(OUT_T + "napm_ppi_neighbours.csv")
summ = pd.read_csv(OUT_T + "napm_network_summary.csv").set_index("metric").value
n_dna = int(summ["napm_degree_full"] - summ["napm_degree_ppi"])

CAT_COLORS = {
    "conserved hypothetical": "#bdbdbd", "unknown": "#e0e0e0",
    "TA system": "#fb9a99", "lipid metabolism": "#a6cee3",
    "insertion seqs/phages": "#b2df8a", "intermediary metabolism": "#fdbf6f",
    "cell wall/cell processes": "#cab2d6", "DNA repair": "#ffff99",
}
DEG_EDGE = {"Up": "#d73027", "Down": "#4575b4"}

fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.axis("off")

# layout: NapM at centre, neighbours on a circle, DNA glyph at bottom
# (half-step angular offset so no node collides with the bottom glyph)
n = len(nb)
ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2 + np.pi / n
R = 1.0
pos = {row.neighbour: (R * np.cos(a), R * np.sin(a))
       for row, a in zip(nb.itertuples(), ang)}
pos["Rv0047c"] = (0, 0)

for row in nb.itertuples():
    x, y = pos[row.neighbour]
    ax.plot([0, x], [0, y], color="#7570b3", lw=1.2, zorder=1)

# DNA-association glyph
gy = -1.55
ax.plot([0, 0], [pos["Rv0047c"][1] - 0.13, gy + 0.13], color="#1b9e77", lw=1.2,
        ls=":")
dna = plt.Circle((0, gy), 0.13, color="#1b9e77", alpha=0.85, zorder=2)
ax.add_patch(dna)
ax.text(0.18, gy, f"{n_dna} DNA bins (5 kb)\nDNA-Protein edges\n(ChIP-seq)",
        fontsize=7, va="center")

for row in nb.itertuples():
    x, y = pos[row.neighbour]
    fc = CAT_COLORS.get(row.functional_category, "#d9d9d9")
    ec = DEG_EDGE.get(row.akr_regulate_KOvsWT, "#666666")
    lw = 1.6 if row.akr_regulate_KOvsWT in DEG_EDGE else 0.6
    s = 60 + 4 * row.degree_full
    ax.scatter(x, y, s=s, c=fc, edgecolors=ec, linewidths=lw, zorder=3)
    lab = row.gene_name.split(" /")[0]
    ax.text(x, y + 0.09 + 0.004 * np.sqrt(s), lab, ha="center", fontsize=6.5,
            zorder=4)
ax.scatter(0, 0, s=700, c="#6a51a3", edgecolors="black", linewidths=1.2,
           zorder=3)
ax.text(0, 0, "NapM", ha="center", va="center", color="white", fontsize=8,
        fontweight="bold", zorder=4)

edge_leg = [Line2D([0], [0], color="#7570b3", lw=1.2,
                   label="Protein-Protein (source to be confirmed:\nexperimental / database / predicted not annotated)"),
            Line2D([0], [0], color="#1b9e77", lw=1.2, ls=":",
                   label="DNA-Protein (ChIP-seq association)")]
cat_leg = [Patch(facecolor=c, edgecolor="none", label=k)
           for k, c in CAT_COLORS.items() if k in set(nb.functional_category)]
deg_leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                  markeredgecolor="#d73027", markersize=7,
                  label="DEG up in KO (border)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                  markeredgecolor="#4575b4", markersize=7,
                  label="DEG down in KO (border)")]
leg1 = ax.legend(handles=edge_leg, loc="upper left", bbox_to_anchor=(-0.02, 1.02),
                 frameon=False, fontsize=6.5, title="edges", title_fontsize=7)
ax.add_artist(leg1)
leg2 = ax.legend(handles=cat_leg, loc="upper left", bbox_to_anchor=(-0.30, 0.62),
                 frameon=False, fontsize=6.5, title="functional category",
                 title_fontsize=7)
ax.add_artist(leg2)
ax.legend(handles=deg_leg, loc="lower left", bbox_to_anchor=(-0.02, -0.06),
          frameon=False, fontsize=6.5)
ax.set_xlim(-2.3, 1.7); ax.set_ylim(-1.9, 1.5)
ax.set_title("NapM (Rv0047c) ego network: 14 PPI neighbours + DNA-binding\n"
             "(no self-loop in data; PPI edge evidence type not annotated)",
             fontsize=8)
fig.tight_layout()
fig.savefig(OUT_F + "fig1_ego_network.png", dpi=300)
fig.savefig(OUT_F + "fig1_ego_network.pdf")
print("OK")
