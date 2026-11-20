#!/usr/bin/env python3
"""GO + KEGG enrichment for M. tuberculosis H37Rv NapM-KO vs WT DEGs.

Gene sets: Up DEGs, Down DEGs, All DEGs (p_val < 0.05 / regulate != Stable).
Test: one-sided hypergeometric (equivalent to Fisher exact), BH FDR.
Background: genes present in the DEG table AND annotated in the respective database.
"""
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "annotation_cache")
TABLES = os.path.join(BASE, "tables")
DEG_FILE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_data/rnaseq/KOvsWT_supo_DEG_limma.csv"

os.makedirs(TABLES, exist_ok=True)

# ---------------- load DEGs ----------------
deg = pd.read_csv(DEG_FILE, index_col=0)
deg.index = deg.index.astype(str)
deg = deg[deg["p_val"] < 0.05]
up = set(deg.index[deg["regulate"] == "Up"])
down = set(deg.index[deg["regulate"] == "Down"])
universe = set(deg.index)  # actually full table universe = all tested genes
all_genes_tested = set(pd.read_csv(DEG_FILE, index_col=0).index.astype(str))
print(f"Up={len(up)} Down={len(down)} All={len(up | down)} tested={len(all_genes_tested)}")

GENE_SETS = {"Up": up, "Down": down, "All": up | down}

# ---------------- load KEGG ----------------
k2p = pd.read_csv(os.path.join(CACHE, "kegg_gene2pathway.tsv"), sep="\t",
                  header=None, names=["gene", "pathway"])
k2p["gene"] = k2p["gene"].str.replace("mtu:", "", regex=False)
k2p["pathway"] = k2p["pathway"].str.replace("path:", "", regex=False)
pnames = pd.read_csv(os.path.join(CACHE, "kegg_pathway_names.tsv"), sep="\t",
                     header=None, names=["pathway", "name"])
pnames["name"] = pnames["name"].str.replace(
    " - Mycobacterium tuberculosis H37Rv", "", regex=False)
pname_map = dict(zip(pnames["pathway"], pnames["name"]))

kegg_term2genes = k2p.groupby("pathway")["gene"].apply(set).to_dict()
print(f"KEGG pathways: {len(kegg_term2genes)}, annotated genes: {len(set(k2p['gene']))}")

# ---------------- load GO (GAF) ----------------
rv_pat = re.compile(r"Rv\d{3,4}[A-Za-z]?(?![A-Za-z0-9])")
go_rows = []
with open(os.path.join(CACHE, "30.M_tuberculosis_ATCC_25618.goa")) as fh:
    for line in fh:
        if line.startswith("!"):
            continue
        f = line.rstrip("\n").split("\t")
        go_id, aspect, syns = f[4], f[8], f[10]
        genes = set(rv_pat.findall(syns))
        for g in genes:
            go_rows.append((g, go_id, aspect))
gaf = pd.DataFrame(go_rows, columns=["gene", "go", "aspect"]).drop_duplicates()
go_term2genes = gaf.groupby("go")["gene"].apply(set).to_dict()
go_aspect = gaf.drop_duplicates("go").groupby("go")["aspect"].first().to_dict()
print(f"GO terms: {len(go_term2genes)}, annotated genes: {gaf['gene'].nunique()}")

# ---------------- GO names from go-basic.obo ----------------
go_name = {}
go_ns = {"biological_process": "BP", "cellular_component": "CC",
         "molecular_function": "MF"}
cur_id = cur_name = cur_ns = None
obsolete = set()
with open(os.path.join(CACHE, "go-basic.obo")) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line == "[Term]":
            if cur_id and cur_name:
                go_name[cur_id] = cur_name
            cur_id = cur_name = cur_ns = None
        elif line.startswith("id: "):
            cur_id = line[4:]
        elif line.startswith("name: "):
            cur_name = line[6:]
        elif line.startswith("is_obsolete: true"):
            if cur_id:
                obsolete.add(cur_id)
    if cur_id and cur_name:
        go_name[cur_id] = cur_name
print(f"GO names loaded: {len(go_name)}")

# ---------------- enrichment engine ----------------
def enrich(gene_set, term2genes, background, min_term_size=3):
    """One-sided hypergeometric enrichment."""
    M = len(background)
    gs = gene_set & background
    N = len(gs)
    rows = []
    for term, genes in term2genes.items():
        tg = genes & background
        n = len(tg)
        if n < min_term_size:
            continue
        k = len(gs & tg)
        if k == 0:
            continue
        # P(X >= k)
        p = hypergeom.sf(k - 1, M, n, N)
        fold = (k / N) / (n / M) if N else np.nan
        rows.append({"term": term, "overlap": k, "term_size": n,
                     "query_size": N, "bg_size": M,
                     "fold_enrichment": fold, "p_val": p,
                     "genes": ",".join(sorted(gs & tg))})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_val"] = multipletests(df["p_val"], method="fdr_bh")[1]
        df = df.sort_values("p_val").reset_index(drop=True)
    return df

def run_all(term2genes, name_map, aspect_map, db_name, annotated_genes):
    background = all_genes_tested & set(annotated_genes)
    print(f"\n{db_name}: background = {len(background)} genes")
    out = {}
    for set_name, gs in GENE_SETS.items():
        df = enrich(gs, term2genes, background)
        df.insert(1, "name", df["term"].map(name_map).fillna(df["term"]))
        if aspect_map:
            df.insert(2, "namespace", df["term"].map(aspect_map))
        df["gene_set"] = set_name
        df["database"] = db_name
        df.to_csv(os.path.join(TABLES, f"{db_name}_enrichment_{set_name}.csv"),
                  index=False)
        sig = df[df["q_val"] < 0.05]
        print(f"  {set_name}: {len(sig)} terms q<0.05 "
              f"({(df['p_val'] < 0.05).sum()} terms p<0.05)")
        out[set_name] = df
    return out

kegg_res = run_all(kegg_term2genes, pname_map, None, "KEGG", set(k2p["gene"]))
go_res = run_all(go_term2genes, go_name, go_aspect, "GO", set(gaf["gene"]))

# save background lists for reproducibility
pd.Series(sorted(all_genes_tested & set(k2p["gene"]))).to_csv(
    os.path.join(TABLES, "background_KEGG_genes.txt"), index=False, header=False)
pd.Series(sorted(all_genes_tested & set(gaf["gene"]))).to_csv(
    os.path.join(TABLES, "background_GO_genes.txt"), index=False, header=False)
print("\nDone.")
