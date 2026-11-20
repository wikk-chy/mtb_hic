#!/usr/bin/env python3
"""R2 Comment 6 - ppi_evidence, part 1.

NapM (Rv0047c) in the integrated interaction network
(review_data/Interaction_network/full_edges.csv):
  - degree / rank in the full DNA-protein-PPI network and in the
    Protein-Protein subgraph
  - betweenness rank (exact, Brandes; unweighted)
  - self-loop check (oligomerization evidence)
  - identity and functional annotation of the 14 PPI neighbours
    (MycoBrowser, accessed 2026-10-07; gene names in network mapped to Rv ids
    via ENA GTF: esxI=Rv1037c, mmaA2=Rv0644c, ogt=Rv1316c)
  - how many PPI neighbours are themselves hubs (top-100 degree)
  - neighbour DEG status (KO vs WT, AKR limma)

CAVEAT honoured throughout: full_edges.csv carries NO evidence-type /
source column for PPI edges (experimental vs database vs prediction cannot
be determined from the file) -- stated in every output.

Outputs: tables/napm_network_summary.csv, tables/napm_ppi_neighbours.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import networkx as nx

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
EDGES = BASE + "hic_review/review_data/Interaction_network/full_edges.csv"
DEG = BASE + "hic_review/review_data/rnaseq/KOvsWT_supo_DEG_limma.csv"
GENE = BASE + "hic_review/review_results/R1_mechanism_reanalysis/04_rna_integration/gene_level_rna_v2.csv"
OUT_T = BASE + "hic_review/review_results/R2_comments/ppi_evidence/tables/"

e = pd.read_csv(EDGES)
e = e[e.source != "Protein1"]          # one embedded header row in the file
assert set(e.interaction_type) == {"DNA-Protein", "Protein-Protein", "DNA-DNA"}
ppi = e[e.interaction_type == "Protein-Protein"]

G = nx.Graph()
G.add_edges_from(e[["source", "target"]].itertuples(index=False))
Gp = nx.Graph()
Gp.add_edges_from(ppi[["source", "target"]].itertuples(index=False))

# ---- degrees ------------------------------------------------------------------
deg_full = dict(G.degree())
deg_ppi = dict(Gp.degree())
rk_full = pd.Series(deg_full).rank(ascending=False, method="min")
rk_ppi = pd.Series(deg_ppi).rank(ascending=False, method="min")

# ---- betweenness (exact, unweighted) ------------------------------------------
btw_full = nx.betweenness_centrality(G, normalized=True)
rk_btw_full = pd.Series(btw_full).rank(ascending=False, method="min")
btw_ppi = nx.betweenness_centrality(Gp, normalized=True)
rk_btw_ppi = pd.Series(btw_ppi).rank(ascending=False, method="min")

self_loop = int(((e.source == "Rv0047c") & (e.target == "Rv0047c")).sum())
nb = sorted(set(ppi.loc[ppi.source == "Rv0047c", "target"]) |
            set(ppi.loc[ppi.target == "Rv0047c", "source"]))

summary = pd.DataFrame([
    ("full_network_nodes", G.number_of_nodes()),
    ("full_network_edges", G.number_of_edges()),
    ("napm_degree_full", deg_full["Rv0047c"]),
    ("napm_degree_rank_full", int(rk_full["Rv0047c"])),
    ("napm_degree_full_of", G.number_of_nodes()),
    ("napm_betweenness_rank_full", int(rk_btw_full["Rv0047c"])),
    ("ppi_subgraph_nodes", Gp.number_of_nodes()),
    ("ppi_subgraph_edges", Gp.number_of_edges()),
    ("napm_degree_ppi", deg_ppi["Rv0047c"]),
    ("napm_degree_rank_ppi", int(rk_ppi["Rv0047c"])),
    ("napm_degree_ppi_of", Gp.number_of_nodes()),
    ("napm_betweenness_rank_ppi", int(rk_btw_ppi["Rv0047c"])),
    ("napm_ppi_neighbours_n", len(nb)),
    ("napm_self_loop_edges", self_loop),
], columns=["metric", "value"])
summary.to_csv(OUT_T + "napm_network_summary.csv", index=False)

# ---- neighbour annotation ------------------------------------------------------
# MycoBrowser (mycobrowser.epfl.ch), accessed 2026-10-07
ANN = {
    "Rv0791c": ("Rv0791c", "conserved protein (function unknown; weak similarity to F420-dependent dehydrogenases)", "conserved hypothetical"),
    "Rv2267c": ("Rv2267c", "conserved hypothetical protein", "conserved hypothetical"),
    "Rv2474c": ("Rv2474c", "hypothetical protein", "unknown"),
    "Rv2595": ("vapB40", "possible antitoxin VapB40 (toxin-antitoxin system with Rv2596)", "TA system"),
    "Rv3222c": ("Rv3222c", "hypothetical protein", "unknown"),
    "Rv3234c": ("tgs3", "putative triacylglycerol synthase (diacylglycerol acyltransferase)", "lipid metabolism"),
    "Rv3387": ("Rv3387", "possible transposase (IS1560)", "insertion seqs/phages"),
    "Rv3467": ("Rv3467", "conserved hypothetical protein (REP13E12 repeat)", "insertion seqs/phages"),
    "Rv3530c": ("Rv3530c", "possible oxidoreductase (short-chain dehydrogenase/reductase family)", "intermediary metabolism"),
    "Rv3753c": ("Rv3753c", "conserved protein, core mycobacterial gene (function unknown)", "conserved hypothetical"),
    "Rv3837c": ("Rv3837c", "probable phosphoglycerate mutase (glycolysis)", "intermediary metabolism"),
    "esxI": ("esxI / Rv1037c", "ESAT-6-like secreted protein EsxI (ESX secretion)", "cell wall/cell processes"),
    "mmaA2": ("mmaA2 / Rv0644c", "methoxy mycolic acid synthase 2", "lipid metabolism"),
    "ogt": ("ogt / Rv1316c", "O6-methylguanine-DNA methyltransferase (DNA repair)", "DNA repair"),
}
KNOWN_STRUCTURAL = {  # none of these appear among the neighbours; kept explicit
    "NAPs (HupB/Rv2986c, Lsr2/Rv3597c, MDP1/Rv1219c, NapM itself)": [],
    "condensin/cohesion (SMC/Rv2922c, ScpA/Rv2921c, ScpB)": [],
    "segregation (ParA/Rv3918c, ParB/Rv3917c)": [],
    "topoisomerases (TopA, GyrA, GyrB)": [],
}
deg = pd.read_csv(DEG, index_col=0)
gv = pd.read_csv(GENE)[["gene_id", "gene_name"]]
name2rv = dict(zip(gv.gene_name, gv.gene_id)) | dict(zip(gv.gene_id, gv.gene_id))

top100_full = set(pd.Series(deg_full).nlargest(100).index)
rows = []
for n in nb:
    rv = name2rv.get(n, n)
    nm, desc, cat = ANN.get(n, (n, "no annotation", "unknown"))
    rows.append(dict(
        neighbour=n, rv_equivalent=rv, gene_name=nm, description=desc,
        functional_category=cat,
        degree_full=deg_full.get(n, 0),
        in_top100_hubs_full=n in top100_full,
        akr_regulate_KOvsWT=deg.regulate.get(rv, "not_in_table"),
        log2FC_KOvsWT=round(deg.avg_log2FC.get(rv, np.nan), 3)
        if rv in deg.index else np.nan))
nbdf = pd.DataFrame(rows)
nbdf.to_csv(OUT_T + "napm_ppi_neighbours.csv", index=False)

n_hub = int(nbdf.in_top100_hubs_full.sum())
frac_hub_neigh = n_hub / len(nbdf)
frac_hub_all = 100 / G.number_of_nodes()
from scipy.stats import fisher_exact
od, p_hub = fisher_exact([[n_hub, len(nbdf) - n_hub],
                          [100 - n_hub, G.number_of_nodes() - 100 - (len(nbdf) - n_hub)]])
extra = pd.DataFrame([
    ("neighbours_in_top100_degree", n_hub),
    ("neighbours_n", len(nbdf)),
    ("expected_frac_if_random", round(frac_hub_all, 4)),
    ("fisher_p_hub_enrichment", p_hub),
    ("neighbours_that_are_known_NAP_SMC_Par_topo", 0),
], columns=["metric", "value"])
extra.to_csv(OUT_T + "napm_neighbour_hub_test.csv", index=False)

print(summary.to_string(index=False))
print(nbdf[["neighbour", "gene_name", "functional_category", "degree_full",
            "in_top100_hubs_full", "akr_regulate_KOvsWT"]].to_string(index=False))
print(extra.to_string(index=False))
print("CAVEAT: PPI edge evidence type (experimental/database/predicted) is not "
      "annotated in full_edges.csv -- source to be confirmed by authors.")
print("OK")
