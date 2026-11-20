#!/usr/bin/env python3
"""Rank highly connected nodes in the integrated DNA–protein–PPI network."""
from pathlib import Path

import networkx as nx
import pandas as pd

base = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
edges_path = base / 'hic_review/review_data/Interaction_network/full_edges.csv'
out_path = base / 'hic_review/review_results/R2_comments/ppi_evidence/tables/network_top20_hubs.csv'

edges = pd.read_csv(edges_path)
edges = edges[edges.source != 'Protein1']
graph = nx.Graph()
graph.add_edges_from(edges[['source', 'target']].itertuples(index=False))

degree = pd.Series(dict(graph.degree()), name='degree')
betweenness = pd.Series(nx.betweenness_centrality(graph, normalized=True), name='betweenness')
result = pd.concat([degree, betweenness], axis=1)
result['degree_rank'] = result['degree'].rank(ascending=False, method='min').astype(int)
result['betweenness_rank'] = result['betweenness'].rank(ascending=False, method='min').astype(int)
result.index.name = 'node'
result = result.reset_index().sort_values(
    ['degree', 'betweenness', 'node'], ascending=[False, False, True]
)
result.head(20).to_csv(out_path, index=False)
print(f'Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
print(result.head(20).to_string(index=False))
print(f'Wrote {out_path}')
