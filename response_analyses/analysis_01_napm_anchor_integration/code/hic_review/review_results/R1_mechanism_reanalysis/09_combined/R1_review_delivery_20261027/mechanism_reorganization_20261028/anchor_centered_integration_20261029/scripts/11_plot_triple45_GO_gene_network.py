#!/usr/bin/env python3
"""Publication-style bipartite network of 45-gene GO terms and member genes."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
STEM = 'R1_triple45_GO_pathway_gene_network'

r = pd.read_csv(TAB / 'triple45_GO_KEGG_enrichment.csv')
z = r[r.database.str.startswith('GO_') & r.p_value.lt(.05)].copy()
assert len(z) == 17 and not z.q_BH_all_tested_terms.lt(.05).any()
q = pd.read_csv(TAB / 'triple45_query_genes.csv').set_index('gene_id')
assert len(q) == 45

# Build and audit the complete pathway-gene membership graph.
graph = nx.Graph()
for row in z.itertuples():
    graph.add_node(row.term, kind='GO', name=row.name, p=row.p_value,
                   database=row.database)
    for gene in row.genes.split(','):
        graph.add_node(gene, kind='gene', name=gene)
        graph.add_edge(row.term, gene)
genes = sorted(n for n, d in graph.nodes(data=True) if d['kind'] == 'gene')
assert len(graph) == 31 and len(genes) == 14 and graph.number_of_edges() == int(z.overlap.sum()) == 25

# Manual component-aware layout: no inferred pathway clusters or causal directions.
# Term positions and text offsets are specified in a common 160 x 100 canvas.
term_pos = {
    'GO:0016740': (14, 83), 'GO:0031177': (8, 57),
    'GO:0006633': (50, 52), 'GO:0071770': (51, 84),
    'GO:0006538': (96, 82), 'GO:0052170': (121, 83),
    'GO:0052027': (143, 73), 'GO:0030682': (143, 53),
    'GO:0042802': (99, 47), 'GO:0009927': (80, 49),
    'GO:0042780': (8, 31), 'GO:0042781': (29, 32),
    'GO:0010126': (48, 23), 'GO:0034355': (73, 23),
    'GO:0008948': (98, 23), 'GO:0004322': (123, 23),
    'GO:0080120': (148, 23),
}
gene_pos = {
    'Rv3800c': (38, 65), 'Rv2381c': (22, 68), 'Rv0704': (10, 77),
    'Rv2950c': (37, 48), 'Rv2952': (48, 77),
    'Rv0410c': (111, 64), 'Rv0902c': (84, 64), 'Rv3343c': (140, 86),
    'Rv3923c': (19, 25),
    'Rv1082': (48, 11), 'Rv0573c': (73, 11), 'Rv2332': (98, 11),
    'Rv0846c': (123, 11), 'Rv0658c': (148, 11),
}
pos = {**term_pos, **gene_pos}
assert set(pos) == set(graph)
label_pos = {
    'GO:0016740': (14, 88, 'center', 'top'),
    'GO:0031177': (8, 53, 'center', 'top'),
    'GO:0006633': (51, 47, 'center', 'top'),
    'GO:0071770': (53, 89, 'center', 'top'),
    'GO:0006538': (92, 85, 'right', 'bottom'),
    'GO:0052170': (120, 89, 'center', 'bottom'),
    'GO:0052027': (147, 76, 'left', 'center'),
    'GO:0030682': (147, 54, 'left', 'center'),
    'GO:0042802': (102, 45, 'left', 'top'),
    'GO:0009927': (78, 44, 'center', 'top'),
    'GO:0042780': (8, 35, 'center', 'bottom'),
    'GO:0042781': (29, 36, 'center', 'bottom'),
    'GO:0010126': (48, 27, 'center', 'bottom'),
    'GO:0034355': (73, 27, 'center', 'bottom'),
    'GO:0008948': (98, 27, 'center', 'bottom'),
    'GO:0004322': (123, 27, 'center', 'bottom'),
    'GO:0080120': (148, 27, 'center', 'bottom'),
    'Rv3800c': (40, 63, 'left', 'top'), 'Rv2381c': (20, 65, 'right', 'top'),
    'Rv0704': (8, 74, 'right', 'top'), 'Rv2950c': (35, 44, 'right', 'top'),
    'Rv2952': (51, 77, 'left', 'center'), 'Rv0410c': (111, 59, 'center', 'top'),
    'Rv0902c': (82, 61, 'right', 'top'), 'Rv3343c': (142, 84, 'left', 'top'),
    'Rv3923c': (19, 21, 'center', 'top'),
    'Rv1082': (48, 7, 'center', 'top'), 'Rv0573c': (73, 7, 'center', 'top'),
    'Rv2332': (98, 7, 'center', 'top'), 'Rv0846c': (123, 7, 'center', 'top'),
    'Rv0658c': (148, 7, 'center', 'top'),
}
assert set(label_pos) == set(graph)

INK = '#20262B'
MUTED = '#68757D'
EDGE = '#B5C0C6'
GENE_FILL = '#F0EBE4'
GENE_EDGE = '#9A8067'
CMAP = LinearSegmentedColormap.from_list(
    'GO_p', ['#294F66', '#5F8194', '#A9BEC9', '#DCE5E9'])
norm = Normalize(vmin=z.p_value.min(), vmax=z.p_value.max())
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
fig = plt.figure(figsize=(17.5, 11.0))
ax = fig.add_axes([.035, .125, .835, .745])
ax.set_xlim(0, 170)
ax.set_ylim(0, 96)
ax.axis('off')
fig.text(.042, .945, 'GO pathway–gene network', fontsize=20, color=INK,
         ha='left', va='top')
fig.text(.043, .906,
         '45 NapM-bound, insulation-changed DEGs  ·  17 GO terms  ·  14 member genes',
         fontsize=10.5, color=MUTED, ha='left', va='top')

# Group headings describe graph components, not enrichment significance or regulation.
for x, y, text in [
    (3, 94, 'LIPID / CELL ENVELOPE'),
    (76, 94, 'SIGNALING / HOST INTERACTION'),
    (3, 42, 'RNA PROCESSING'),
    (45, 42, 'SINGLE-GENE GO TERMS'),
]:
    ax.text(x, y, text, fontsize=9.4, color=MUTED, weight='bold',
            ha='left', va='top')

# Curved membership edges stay behind nodes and labels.
for i, (term, gene) in enumerate(graph.edges()):
    rad = .06 if i % 2 else -.06
    ax.add_patch(FancyArrowPatch(pos[term], pos[gene], arrowstyle='-',
                                connectionstyle=f'arc3,rad={rad}',
                                linewidth=1.0, color=EDGE, alpha=.78,
                                zorder=1, shrinkA=8, shrinkB=8))

# Equal area-degree mapping for both node types.
def node_size(degree):
    return 190 + 230 * (degree - 1)

for term in term_pos:
    d = graph.nodes[term]
    degree = graph.degree(term)
    ax.scatter(*pos[term], s=node_size(degree), marker='o',
               color=CMAP(norm(d['p'])), edgecolor='white', linewidth=1.25,
               zorder=3)
    ax.text(*pos[term], str(degree), fontsize=7.2, ha='center', va='center',
            color='white' if norm(d['p']) < .58 else INK, weight='bold', zorder=4)
for gene in gene_pos:
    degree = graph.degree(gene)
    ax.scatter(*pos[gene], s=node_size(degree), marker='s',
               facecolor=GENE_FILL, edgecolor=GENE_EDGE, linewidth=1.25,
               zorder=3)

for node, (x, y, ha, va) in label_pos.items():
    d = graph.nodes[node]
    if d['kind'] == 'GO':
        width = 25 if node not in {'GO:0052027', 'GO:0052170'} else 22
        if node in {'GO:0034355', 'GO:0008948', 'GO:0080120'}:
            width = 19
        label = textwrap.fill(d['name'], width=width)
        ax.text(x, y, label, fontsize=8.7, color=INK, ha=ha, va=va,
                linespacing=1.28, zorder=5,
                path_effects=[patheffects.withStroke(linewidth=2.8, foreground='white')])
    else:
        name = q.at[node, 'gene_name']
        label = node if pd.isna(name) or not str(name).strip() else str(name)
        ax.text(x, y, label, fontsize=9.6, color=INK, ha=ha, va=va,
                style='italic', weight='bold', zorder=5,
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground='white')])

# Separate legend panel: quantitative scales are not mixed with gene-node color.
legend_ax = fig.add_axes([.888, .20, .09, .62])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis('off')
legend_ax.text(0, 1.0, 'NETWORK KEY', fontsize=9.5, color=MUTED,
               weight='bold', ha='left', va='top')
legend_ax.text(0, .92, 'GO pathway', fontsize=10, color=INK, ha='left')
legend_ax.text(0, .865, 'Color: nominal P', fontsize=9, color=MUTED, ha='left')
cax = fig.add_axes([.891, .585, .022, .120])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                  ticks=[.02, .03, .04])
cb.ax.set_yticklabels(['0.02', '0.03', '0.04'])
cb.ax.tick_params(labelsize=8.5, length=3, color=MUTED, labelcolor=INK, pad=5)
cb.outline.set_visible(False)
legend_ax.text(0, .52, 'Size: connections', fontsize=9, color=MUTED, ha='left')
for i, degree in enumerate([1, 3, 5]):
    y = .445 - i * .085
    legend_ax.scatter(.09, y, s=node_size(degree) * .58, marker='o',
                      facecolor='#A9BEC9', edgecolor='white', linewidth=1)
    legend_ax.text(.29, y, str(degree), fontsize=9.5, color=INK,
                   ha='left', va='center')
legend_ax.text(0, .185, 'Node types', fontsize=9, color=MUTED, ha='left')
handles = [
    Line2D([0], [0], marker='o', linestyle='none', markersize=9,
           markerfacecolor='#7D99AB', markeredgecolor='white', label='GO pathway'),
    Line2D([0], [0], marker='s', linestyle='none', markersize=9,
           markerfacecolor=GENE_FILL, markeredgecolor=GENE_EDGE, label='Gene'),
]
legend_ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.08, .145),
                 frameon=False, fontsize=9.3, handletextpad=.8,
                 labelspacing=1.2, borderaxespad=0)
legend_ax.text(0, .005, 'Line: annotated membership\nNumber: genes per GO term',
               fontsize=8.6, color=MUTED, ha='left', va='top', linespacing=1.5)

fig.text(.043, .071,
         'One-sided hypergeometric test  ·  4,060 RNA-tested genes  ·  nominal P < 0.05  ·  no displayed term passes BH FDR < 0.05',
         fontsize=9.1, color=MUTED, ha='left')
fig.text(.043, .040,
         'Edges indicate annotation only; they do not establish direct NapM regulation or pathway-to-gene causality.',
         fontsize=9.1, color=MUTED, ha='left')
fig.savefig(FIG / f'{STEM}.pdf', bbox_inches='tight')
fig.savefig(FIG / f'{STEM}.png', dpi=300, bbox_inches='tight')
plt.close(fig)

nodes = []
for node, d in graph.nodes(data=True):
    nodes.append(dict(node_id=node, node_type=d['kind'],
                      term_name=d.get('name', ''), gene_name='' if d['kind'] == 'GO' else q.at[node, 'gene_name'],
                      database=d.get('database', ''), nominal_P=d.get('p', np.nan),
                      degree=graph.degree(node), x=pos[node][0], y=pos[node][1]))
pd.DataFrame(nodes).to_csv(TAB / f'{STEM}_nodes.csv', index=False)
edge_rows = []
for a, b in graph.edges():
    # NetworkX may return either orientation for an undirected edge.
    term, gene = (a, b) if graph.nodes[a]['kind'] == 'GO' else (b, a)
    assert graph.nodes[term]['kind'] == 'GO' and graph.nodes[gene]['kind'] == 'gene'
    edge_rows.append(dict(GO_term=term, gene_id=gene, relation='annotated_membership'))
edges = pd.DataFrame(edge_rows)
edges.to_csv(TAB / f'{STEM}_edges.csv', index=False)
print(f'{STEM}: {len(term_pos)} GO terms, {len(gene_pos)} genes, {len(edges)} edges')
print('Connected components:', sorted(map(len, nx.connected_components(graph)), reverse=True))
print('GO degrees:', sorted(set(graph.degree(t) for t in term_pos)))
print('Gene degrees:', sorted(set(graph.degree(g) for g in gene_pos)))
