from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
OUT = ROOT / "analysis/loop_augmented_network_20261105"
OUT.mkdir(exist_ok=True)
per_loop = pd.read_csv(ROOT / "hic_review/review_results/R4_comment6/loop_network_analysis/tables/per_loop_proteins_full.csv")
edges = pd.read_csv(ROOT / "hic_review/review_data/Interaction_network/full_edges.csv")
edges = edges[edges["source"] != "Protein1"].copy()

assert len(per_loop) == 41
shared_counts = per_loop["n_shared"].astype(int)
summary = pd.DataFrame([
    ("loops_total", len(per_loop)),
    ("loops_with_at_least_one_shared_protein", int((shared_counts > 0).sum())),
    ("minimum_shared_proteins_per_loop", int(shared_counts.min())),
    ("median_shared_proteins_per_loop", float(shared_counts.median())),
    ("maximum_shared_proteins_per_loop", int(shared_counts.max())),
], columns=["metric", "value"])
summary.to_csv(OUT / "shared_protein_summary.csv", index=False)
per_loop[["loop", "anchor1", "anchor2", "n_shared", "shared_proteins"]].to_csv(
    OUT / "per_loop_shared_proteins.csv", index=False)

# Degree among protein nodes in the existing integrated network.
degree = Counter()
for source, target in edges[["source", "target"]].itertuples(index=False):
    degree[source] += 1
    degree[target] += 1
proteins = set(edges.loc[edges["interaction_type"].eq("DNA-Protein"), "source"])

loop_degree = Counter()
for shared in per_loop["shared_proteins"]:
    if pd.isna(shared):
        continue
    for protein in str(shared).split(";"):
        loop_degree[protein] += 1

rows = []
for protein in sorted(proteins):
    base = int(degree[protein])
    loops = int(loop_degree[protein])
    rows.append({
        "protein": protein,
        "base_degree": base,
        "loops_with_both_anchors_bound": loops,
        "loop_augmented_degree": base + loops,
    })
ranking = pd.DataFrame(rows)
ranking["augmented_rank"] = ranking["loop_augmented_degree"].rank(ascending=False, method="min").astype(int)
ranking["base_rank"] = ranking["base_degree"].rank(ascending=False, method="min").astype(int)
ranking = ranking.sort_values(["loop_augmented_degree", "protein"], ascending=[False, True])
ranking.to_csv(OUT / "loop_augmented_degree_ranking.csv", index=False)
napm = ranking.loc[ranking["protein"].eq("Rv0047c")].iloc[0]

print(summary.to_string(index=False))
print("\nNapM:")
print(napm.to_string())
print("\nTop 15 by loop-augmented degree:")
print(ranking.head(15).to_string(index=False))
