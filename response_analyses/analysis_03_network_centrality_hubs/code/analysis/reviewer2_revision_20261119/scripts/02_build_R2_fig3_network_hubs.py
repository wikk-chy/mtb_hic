#!/usr/bin/env python3
"""
Build Response Fig. 3 (Reviewer 2, Comment 2 / Audit Comment 8):
NapM connectivity within the integrated DNA-protein-PPI network (non-unique hub positioning).
  Panel A: Degree vs betweenness centrality across all protein nodes in the integrated network
           (highlighting NapM alongside the five higher-degree hubs: Rv0081, csoR, Rv1353c, Rv0678, Rv1816).
  Panel B: Top 10 highest-degree proteins compared by total integrated degree and
           double-anchor loop occupancy (out of 41 validated WT loops).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
EDGES = ROOT / "hic_review/review_data/Interaction_network/full_edges.csv"
LOOP_RANK = ROOT / "analysis/loop_augmented_network_20261105/loop_augmented_degree_ranking.csv"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_TAB = ROOT / "analysis/reviewer2_revision_20261119/tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 8.0,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

# Load cached centrality table if already computed, else compute
cache_path = OUT_TAB / "full_network_centrality.csv"
if cache_path.exists():
    net = pd.read_csv(cache_path)
else:
    edges = pd.read_csv(EDGES)
    edges = edges[edges["source"] != "Protein1"]
    G = nx.Graph()
    G.add_edges_from(edges[["source", "target"]].itertuples(index=False))

    deg = pd.Series(dict(G.degree()), name="degree")
    bet = pd.Series(nx.betweenness_centrality(G, normalized=True), name="betweenness")
    net = pd.concat([deg, bet], axis=1)
    net.index.name = "node"
    net = net.reset_index()
    net["degree_rank"] = net["degree"].rank(ascending=False, method="min").astype(int)
    net["betweenness_rank"] = net["betweenness"].rank(ascending=False, method="min").astype(int)
    net = net.sort_values(["degree", "betweenness", "node"], ascending=[False, False, True]).reset_index(drop=True)
    net.to_csv(cache_path, index=False)

# Merge with loop double-anchor ranking
loop_df = pd.read_csv(LOOP_RANK)
top10 = loop_df.head(10).merge(net[["node", "betweenness", "betweenness_rank"]], left_on="protein", right_on="node", how="left")

# Plot
C_NAPM = "#6A51A3"
C_TOP5 = "#D55E00"
C_OTHER = "#A7B3BC"
C_BLUE = "#2B6CB0"
C_TEXT = "#1F2933"

fig = plt.figure(figsize=(8.27, 4.15))
gs = GridSpec(1, 2, figure=fig, left=0.09, right=0.97, top=0.84, bottom=0.15, wspace=0.36, width_ratios=[1.12, 1.0])
fig.suptitle("NapM (Rv0047c) is one of several highly connected nodes in the integrated network",
             x=0.05, y=0.955, ha="left", fontsize=10.2, fontweight="bold", color=C_TEXT)

# Panel A: Degree vs Betweenness scatter
axA = fig.add_subplot(gs[0, 0])
top5_names = {"Rv0081", "csoR", "Rv1353c", "Rv0678", "Rv1816"}
is_napm = net["node"] == "Rv0047c"
is_top5 = net["node"].isin(top5_names)
is_rest = ~(is_napm | is_top5)

# Plot background (exclude zero-betweenness clutter only visually via small alpha; plot all!)
axA.scatter(net.loc[is_rest, "degree"], net.loc[is_rest, "betweenness"],
            s=10, color=C_OTHER, alpha=0.45, linewidths=0, zorder=1, label=f"Other nodes (n = {is_rest.sum():,})")
axA.scatter(net.loc[is_top5, "degree"], net.loc[is_top5, "betweenness"],
            s=42, color=C_TOP5, edgecolor="black", linewidth=0.6, zorder=3, label="Top-5 by degree")
axA.scatter(net.loc[is_napm, "degree"], net.loc[is_napm, "betweenness"],
            s=65, color=C_NAPM, edgecolor="black", linewidth=0.8, marker="D", zorder=4, label="NapM (Rv0047c)")

# Careful non-overlapping annotations
offsets = {
    "Rv0081": (-58, -4),
    "csoR": (6, 2),
    "Rv1353c": (-56, 4),
    "Rv0678": (-52, -13),
    "Rv1816": (-56, 3),
    "Rv0047c": (-78, 8),
}
for _, r in net[is_top5 | is_napm].iterrows():
    nm = "NapM (Rv0047c)" if r["node"] == "Rv0047c" else r["node"]
    dx, dy = offsets[r["node"]]
    col = C_NAPM if r["node"] == "Rv0047c" else C_TEXT
    fw = "bold" if r["node"] == "Rv0047c" else "normal"
    axA.annotate(
        nm, (r["degree"], r["betweenness"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=7.1, fontweight=fw, color=col,
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.55, shrinkA=0, shrinkB=3),
    )

# Also annotate top-betweenness node cleanly below/right so it never touches legend
for _, r in net[is_rest & (net["betweenness"] > 0.045)].iterrows():
    axA.annotate(r["node"], (r["degree"], r["betweenness"]),
                 xytext=(8, -10), textcoords="offset points", fontsize=6.5, color="#555555",
                 arrowprops=dict(arrowstyle="-", color="#999999", lw=0.45, shrinkA=0, shrinkB=2))

axA.set_xlabel("Network degree (integrated DNA–protein–PPI)")
axA.set_ylabel("Betweenness centrality")
axA.set_xlim(-20, 610)
axA.set_ylim(-0.004, 0.105)
axA.legend(loc="center left", bbox_to_anchor=(0.01, 0.56), frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=6.8, handletextpad=0.3)
axA.set_title("Integrated network centrality (all nodes)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axA.text(-0.14, 1.09, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="top")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# Panel B: Top 10 proteins — base degree vs double-anchor loops (dual aligned horizontal bars)
axB = fig.add_subplot(gs[0, 1])
top10_plot = top10.iloc[::-1].reset_index(drop=True)
y = np.arange(len(top10_plot))
labels = [
    f"NapM (Rv0047c)" if p == "Rv0047c" else p
    for p in top10_plot["protein"]
]
bar_colors = [
    C_NAPM if p == "Rv0047c" else (C_TOP5 if p in top5_names else "#7B9EB8")
    for p in top10_plot["protein"]
]

# Plot horizontal bars for integrated degree
bars = axB.barh(y, top10_plot["base_degree"], height=0.62, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.88)
axB.set_yticks(y, labels)
for tick, p in zip(axB.get_yticklabels(), top10_plot["protein"]):
    if p == "Rv0047c":
        tick.set_fontweight("bold")
        tick.set_color(C_NAPM)

# Annotate degree + double-anchor loops + betweenness rank cleanly on the right
for i, r in top10_plot.iterrows():
    txt = f"deg {int(r['base_degree'])} | {int(r['loops_with_both_anchors_bound'])}/41 loops | bet #{int(r['betweenness_rank'])}"
    fw = "bold" if r["protein"] == "Rv0047c" else "normal"
    col = C_NAPM if r["protein"] == "Rv0047c" else C_TEXT
    axB.text(r["base_degree"] + 10, i, txt, va="center", ha="left", fontsize=6.4, fontweight=fw, color=col)

axB.set_xlim(0, 960)
axB.set_xlabel("Integrated network degree")
axB.set_title("Top-10 hubs: degree, loop anchors & betweenness", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axB.text(-0.22, 1.09, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="top")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

pdf_path = OUT_FIG / "Response_Fig_3.pdf"
png_path = OUT_FIG / "Response_Fig_3.png"
jpg_path = OUT_FIG / "Response_Fig_3.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved Response_Fig_3.{pdf,png,jpg}")
