#!/usr/bin/env python3
"""Draw Fig. 2a/b at final manuscript dimensions for PDF assembly."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "hic_review"
WORK = ROOT / "tmp/manuscript_update"
WORK.mkdir(parents=True, exist_ok=True)
NET_DIR = BASE / "review_data/Interaction_network"
CS_LOOPS = BASE / "review_results/R4_comment3_loops/chromosight_out/wt.tsv"
BD_CSV = BASE / "review_results/R4_comment2_CID/tables/cid_boundaries_recalled.csv"
DNA_COLOR, PROTEIN_COLOR, LOOP_COLOR = "#529AC9", "#8F7289", "#29A15C"
SEED = 42

nodes = pd.read_csv(NET_DIR / "full_nodes.csv")
edges = pd.read_csv(NET_DIR / "full_edges.csv")
loops_table = pd.read_csv(CS_LOOPS, sep="\t")
boundaries = pd.read_csv(BD_CSV)
boundary_bins = sorted(boundaries.loc[
    boundaries.transition_ok.astype(str).str.lower().eq("true"), "bin"].astype(int))
assert len(boundary_bins) == 18
boundary_nodes = [f"bin{value + 1}" for value in boundary_bins]
loop_edges = [(f"bin{int(row.bin1) + 1}", f"bin{int(row.bin2) + 1}")
              for _, row in loops_table.iterrows()]
loop_anchors = sorted(set(node for edge in loop_edges for node in edge))
assert len(loop_edges) == 41 and len(loop_anchors) == 75

graph = nx.Graph()
for _, row in nodes.iterrows():
    graph.add_node(row.id, label=row.label, type=row.get("type", "Unknown"))
for _, row in edges.iterrows():
    if row.interaction_type != "DNA-DNA":
        graph.add_edge(row.source, row.target,
                       interaction_type=row.interaction_type, weight=row.get("weight", 1))
for left, right in loop_edges:
    graph.add_edge(left, right, interaction_type="DNA-DNA", weight=1)
loop_graph = nx.Graph()
for node in loop_anchors:
    loop_graph.add_node(node, **graph.nodes[node])
loop_graph.add_edges_from(loop_edges)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def highlight(graph_object, positions, axis, node_list, color, size, width, zorder=5, face="none"):
    node_list = [node for node in node_list if node in positions]
    if node_list:
        coll = nx.draw_networkx_nodes(graph_object, positions, nodelist=node_list,
                                      node_color=face, edgecolors=color,
                                      linewidths=width, node_size=size, ax=axis)
        coll.set_zorder(zorder)


print("Drawing final-size Fig. 2a ...", flush=True)
pos_a = nx.spring_layout(graph, k=0.5, iterations=50, seed=SEED)
pure_bd = [n for n in boundary_nodes if n not in set(loop_anchors)]
dual_bd = [n for n in boundary_nodes if n in set(loop_anchors)]
pure_anc = [n for n in loop_anchors if n not in set(boundary_nodes)]
fig, axis = plt.subplots(figsize=(247 / 72, 250 / 72))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
node_list = list(graph.nodes)
colors = [DNA_COLOR if graph.nodes[node].get("type") == "DNA" else PROTEIN_COLOR
          for node in node_list]
nonloops = [(left, right) for left, right, data in graph.edges(data=True)
            if data["interaction_type"] != "DNA-DNA"]
ec = nx.draw_networkx_edges(graph, pos_a, edgelist=nonloops, edge_color="lightgray",
                            width=0.16, alpha=0.40, ax=axis)
if ec is not None:
    ec.set_zorder(1)
nc = nx.draw_networkx_nodes(graph, pos_a, nodelist=node_list, node_color=colors,
                            node_size=8, alpha=0.70, ax=axis)
nc.set_zorder(2)
for left, right in loop_edges:
    x1, y1 = pos_a[left]
    x2, y2 = pos_a[right]
    axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                   connectionstyle="arc3,rad=0.20", arrowstyle="-",
                                   linewidth=0.55, color=LOOP_COLOR, alpha=0.82, zorder=3))
highlight(graph, pos_a, axis, pure_anc, "#C0392B", 15, 0.50, zorder=4)
highlight(graph, pos_a, axis, pure_bd, "black", 24, 0.80, zorder=6, face=DNA_COLOR)
highlight(graph, pos_a, axis, dual_bd, "#C0392B", 14, 0.55, zorder=5, face=DNA_COLOR)
highlight(graph, pos_a, axis, dual_bd, "black", 30, 0.85, zorder=6, face="none")
axis.set_title("a  Integrated DNA–protein–PPI network", loc="left",
               fontsize=8.5, fontweight="bold", pad=4)
axis.axis("off")
fig.savefig(WORK / "Fig2a_final_size.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

print("Drawing final-size Fig. 2b ...", flush=True)
pos_b = nx.spring_layout(loop_graph, k=0.8, iterations=100, seed=SEED)
fig, axis = plt.subplots(figsize=(100 / 72, 112 / 72))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.78)
nx.draw_networkx_edges(loop_graph, pos_b, edge_color=LOOP_COLOR,
                       width=0.55, alpha=0.82, ax=axis)
nx.draw_networkx_nodes(loop_graph, pos_b, node_color=DNA_COLOR,
                       node_size=12, alpha=0.85, ax=axis)
axis.set_title("b  DNA–DNA network\n75 loop-anchor bins; 41 loops",
               loc="left", fontsize=6.5, fontweight="bold", pad=3)
axis.axis("off")
fig.savefig(WORK / "Fig2b_final_size.pdf", bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print("Saved final-size Fig. 2a/b panels", flush=True)
