#!/usr/bin/env python3
"""Redraw manuscript Fig. 2a and Fig. 2b with 18 CID boundaries and 41 loops.

DNA-DNA edges are exclusively the validated Chromosight loop calls. CID
boundaries are node annotations, not additional DNA-DNA edges. PPI and
DNA-protein inputs remain unchanged. Outputs are standalone and do not alter
the assembled manuscript figure or text.
"""

from collections import defaultdict
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "hic_review"
OUT = ROOT / "docx_latex_conversion/generated"
OUT.mkdir(parents=True, exist_ok=True)
NET_DIR = BASE / "review_data/Interaction_network"
CS_LOOPS = BASE / "review_results/R4_comment3_loops/chromosight_out/wt.tsv"
BD_CSV = BASE / "review_results/R4_comment2_CID/tables/cid_boundaries_recalled.csv"
NAPM_INTERVAL = BASE / "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_bound_bins_interval.csv"
RV = "Rv0047c"
SEED = 42
DNA_COLOR, PROTEIN_COLOR, LOOP_COLOR = "#529AC9", "#8F7289", "#29A15C"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

nodes_df = pd.read_csv(NET_DIR / "full_nodes.csv")
edges_df = pd.read_csv(NET_DIR / "full_edges.csv")
chromosight = pd.read_csv(CS_LOOPS, sep="\t")
boundaries = pd.read_csv(BD_CSV)
boundary_bins = sorted(boundaries.loc[
    boundaries.transition_ok.astype(str).str.lower().eq("true"), "bin"].astype(int).tolist())
assert len(boundary_bins) == 18
boundary_nodes = [f"bin{value + 1}" for value in boundary_bins]
loop_edges = [(f"bin{int(row.bin1) + 1}", f"bin{int(row.bin2) + 1}")
              for _, row in chromosight.iterrows()]
assert len(loop_edges) == 41
loop_anchor_nodes = sorted(set(node for edge in loop_edges for node in edge))
assert len(loop_anchor_nodes) == 75

# Rebuild the integrated graph while removing every legacy DNA-DNA edge.
graph = nx.Graph()
for _, row in nodes_df.iterrows():
    graph.add_node(row.id, label=row.label, type=row.get("type", "Unknown"))
for _, row in edges_df.iterrows():
    if row.interaction_type != "DNA-DNA":
        graph.add_edge(row.source, row.target,
                       interaction_type=row.interaction_type, weight=row.get("weight", 1))
for left, right in loop_edges:
    assert left in graph and right in graph
    graph.add_edge(left, right, interaction_type="DNA-DNA", weight=1)

dna_dna_edges = [(u, v) for u, v, data in graph.edges(data=True)
                 if data["interaction_type"] == "DNA-DNA"]
dna_protein_edges = [(u, v) for u, v, data in graph.edges(data=True)
                     if data["interaction_type"] == "DNA-Protein"]
ppi_edges = [(u, v) for u, v, data in graph.edges(data=True)
             if data["interaction_type"] == "Protein-Protein"]
input_ppi_count = int((edges_df.interaction_type == "Protein-Protein").sum())
input_dna_protein_count = int((edges_df.interaction_type == "DNA-Protein").sum())
assert len(dna_dna_edges) == 41
assert len(ppi_edges) <= input_ppi_count
assert len(dna_protein_edges) <= input_dna_protein_count

# Classify loop-associated shared occupancy without implying physical bridging.
bin_to_protein = defaultdict(set)
for left, right in dna_protein_edges:
    if graph.nodes[left]["type"] == "DNA":
        bin_to_protein[left].add(right)
    else:
        bin_to_protein[right].add(left)
for node in bin_to_protein:
    bin_to_protein[node].discard(RV)
napm_bins = pd.read_csv(NAPM_INTERVAL)["bin"].astype(int)
for value in napm_bins:
    bin_to_protein[f"bin{value + 1}"].add(RV)

loop_records = []
for left, right in loop_edges:
    shared = sorted(bin_to_protein.get(left, set()) & bin_to_protein.get(right, set()))
    loop_records.append({
        "source": left,
        "target": right,
        "source_bin_0based": int(left.replace("bin", "")) - 1,
        "target_bin_0based": int(right.replace("bin", "")) - 1,
        "source_start_kb": (int(left.replace("bin", "")) - 1) * 5,
        "target_start_kb": (int(right.replace("bin", "")) - 1) * 5,
        "shared_anchor_protein_count": len(shared),
        "shared_anchor_proteins": ";".join(shared),
    })
loop_table = pd.DataFrame(loop_records)
assert (loop_table.shared_anchor_protein_count > 0).all()
loop_table.to_csv(OUT / "Fig2ab_DNA_DNA_41_loop_edges.csv", index=False)

boundary_anchor_nodes = sorted(set(boundary_nodes) & set(loop_anchor_nodes))
ppi_graph = nx.Graph()
ppi_graph.add_edges_from(ppi_edges)
dna_protein_graph = nx.Graph()
dna_protein_graph.add_edges_from(dna_protein_edges)
for node in dna_protein_graph.nodes:
    dna_protein_graph.add_node(node, **graph.nodes[node])
dna_dna_graph = nx.Graph()
for node in loop_anchor_nodes:
    dna_dna_graph.add_node(node, **graph.nodes[node])
dna_dna_graph.add_edges_from(dna_dna_edges)

summary = {
    "integrated_nodes": graph.number_of_nodes(),
    "integrated_unique_edges": graph.number_of_edges(),
    "integrated_input_edges": input_ppi_count + input_dna_protein_count + len(dna_dna_edges),
    "protein_nodes": sum(data.get("type") == "Protein" for _, data in graph.nodes(data=True)),
    "dna_bins": sum(data.get("type") == "DNA" for _, data in graph.nodes(data=True)),
    "ppi_input_edges": input_ppi_count,
    "ppi_unique_edges": len(ppi_edges),
    "dna_protein_input_edges": input_dna_protein_count,
    "dna_protein_unique_edges": len(dna_protein_edges),
    "dna_dna_loop_edges": len(dna_dna_edges),
    "loop_anchor_bins": len(loop_anchor_nodes),
    "cid_boundaries": len(boundary_nodes),
    "cid_boundary_loop_anchor_overlap": len(boundary_anchor_nodes),
    "boundary_anchor_nodes": boundary_anchor_nodes,
    "loops_with_at_least_one_shared_anchor_protein": int((loop_table.shared_anchor_protein_count > 0).sum()),
    "shared_proteins_per_loop_min": int(loop_table.shared_anchor_protein_count.min()),
    "shared_proteins_per_loop_median": float(loop_table.shared_anchor_protein_count.median()),
    "shared_proteins_per_loop_max": int(loop_table.shared_anchor_protein_count.max()),
    "layout_seed": SEED,
    "note": "CID boundaries are highlighted nodes; only the 41 Chromosight loops are DNA-DNA edges.",
}
with open(OUT / "Fig2ab_networks_18_boundaries_summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2), flush=True)


def node_colors(target_graph, nodes):
    return [DNA_COLOR if target_graph.nodes[node].get("type") == "DNA" else PROTEIN_COLOR
            for node in nodes]


def highlight_nodes(axis, positions, nodes, color, size, width, zorder=5, face="none"):
    nodes = [node for node in nodes if node in positions]
    if nodes:
        coll = nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_color=face,
            edgecolors=color, linewidths=width, node_size=size, ax=axis)
        coll.set_zorder(zorder)


def curved_edges(axis, positions, edges, color=LOOP_COLOR, rad=0.20, width=1.2):
    for left, right in edges:
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        axis.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", linestyle="-", linewidth=width,
            color=color, alpha=0.82, zorder=3))


# ---------------- Fig. 2a: full integrated network ----------------
print("Drawing Fig. 2a ...", flush=True)
positions_a = nx.spring_layout(graph, k=0.5, iterations=50, seed=SEED)
pure_boundary_nodes = [n for n in boundary_nodes if n not in set(loop_anchor_nodes)]
dual_boundary_nodes = [n for n in boundary_nodes if n in set(loop_anchor_nodes)]
pure_anchor_nodes = [n for n in loop_anchor_nodes if n not in set(boundary_nodes)]

def render_fig2a(out_pdf=None, out_png=None, show_title=True, dpi=200):
    fig, axis = plt.subplots(figsize=(11, 11))
    all_nodes = list(graph.nodes)
    non_loop_edges = [(u, v) for u, v, data in graph.edges(data=True)
                      if data["interaction_type"] != "DNA-DNA"]
    e_coll = nx.draw_networkx_edges(
        graph, positions_a, edgelist=non_loop_edges, edge_color="lightgray",
        width=0.4, alpha=0.42, ax=axis)
    if e_coll is not None:
        e_coll.set_zorder(1)
    n_coll = nx.draw_networkx_nodes(
        graph, positions_a, nodelist=all_nodes, node_color=node_colors(graph, all_nodes),
        node_size=30, alpha=0.70, ax=axis)
    n_coll.set_zorder(2)
    curved_edges(axis, positions_a, dna_dna_edges)
    # Loop anchors (zorder=4) and all 18 CID boundaries clearly on top (zorder=5-6)
    highlight_nodes(axis, positions_a, pure_anchor_nodes, "#C0392B", 68, 1.15, zorder=4)
    highlight_nodes(axis, positions_a, pure_boundary_nodes, "black", 108, 1.85, zorder=6, face=DNA_COLOR)
    highlight_nodes(axis, positions_a, dual_boundary_nodes, "#C0392B", 62, 1.25, zorder=5, face=DNA_COLOR)
    highlight_nodes(axis, positions_a, dual_boundary_nodes, "black", 132, 1.95, zorder=6, face="none")
    if show_title:
        axis.set_title("a  Integrated DNA–protein–PPI network", loc="left",
                       fontsize=14, fontweight="bold", pad=10)
    axis.axis("off")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

render_fig2a(
    out_pdf=OUT / "Fig2a_integrated_network_18_boundaries.pdf",
    out_png=OUT / "Fig2a_integrated_network_18_boundaries.png",
    show_title=True, dpi=200,
)
render_fig2a(
    out_png=OUT / "Fig2a_integrated_network_18_boundaries_notitle.png",
    show_title=False, dpi=260,
)

# ---------------- Fig. 2b: DNA-DNA network only ----------------
print("Drawing Fig. 2b (DNA-DNA only) ...", flush=True)
positions_dd = nx.spring_layout(dna_dna_graph, k=0.8, iterations=100, seed=SEED)
fig, axis = plt.subplots(figsize=(7, 7))
nx.draw_networkx_edges(dna_dna_graph, positions_dd, edge_color=LOOP_COLOR,
                       width=1.3, alpha=0.82, ax=axis)
nx.draw_networkx_nodes(dna_dna_graph, positions_dd, node_color=DNA_COLOR,
                       node_size=70, alpha=0.85, ax=axis)
axis.set_title(f"b  DNA–DNA network\n{len(loop_anchor_nodes)} loop-anchor bins; "
               f"{len(dna_dna_edges)} loops",
               loc="left", fontsize=12, fontweight="bold", pad=10)
axis.axis("off")
fig.savefig(OUT / "Fig2b_subnetworks_18_boundaries.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig2b_subnetworks_18_boundaries.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("Saved standalone Fig. 2a and Fig. 2b", flush=True)
