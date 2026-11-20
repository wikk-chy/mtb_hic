from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
RNA = ROOT / "hic_review/review_data/rnaseq"
OUT = ROOT / "figures/stress_response_analysis_20261105"
OUT.mkdir(exist_ok=True)

# Curated from the M. tuberculosis LexA/SOS literature and the existing GO annotation.
CORE_SOS = {
    "Rv2720": "lexA", "Rv3370c": "dnaE2", "Rv3395c": "imuA", "Rv3394c": "imuB",
    "Rv2737c": "recA", "Rv2593c": "ruvA", "Rv2594c": "ruvC",
}
LEXA_DAMAGE = [
    "Rv3395c", "Rv0515", "Rv3776", "Rv0336", "Rv3370c", "Rv3260c", "Rv2720",
    "Rv1378c", "Rv3074", "Rv2737c", "Rv1702c", "Rv1000c", "Rv1057", "Rv2719c",
    "Rv2100", "Rv0071", "Rv2594c", "Rv0095c", "Rv2578c", "Rv2579", "Rv1588c", "Rv2593c",
]
GO_DNA_REPAIR = ["Rv0670", "Rv0937c", "Rv0944", "Rv1317c", "Rv1638", "Rv3370c"]
GENE_SETS = {
    "core_SOS_LexA": list(CORE_SOS),
    "LexA_associated_DNA_damage": LEXA_DAMAGE,
    "GO_DNA_repair": GO_DNA_REPAIR,
}

# Gene names from the primary annotation.
name_map = {}
with (ROOT / "hic_review/annotation.gtf").open() as handle:
    for line in handle:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9 or fields[2] != "gene":
            continue
        attrs = fields[8]
        gid_match = re.search(r'gene_id "([^"]+)"', attrs)
        name_match = re.search(r'gene_name "([^"]+)"', attrs)
        if gid_match:
            name_map[gid_match.group(1)] = name_match.group(1) if name_match else ""

def load_deg(filename, prefix):
    frame = pd.read_csv(RNA / filename, index_col=0)
    frame.index.name = "gene"
    frame = frame[~frame.index.duplicated(keep="first")].copy()
    frame["q"] = stats.false_discovery_control(frame["p_val"].to_numpy(), method="bh")
    return frame[["avg_log2FC", "p_val", "q", "regulate"]].rename(columns={
        "avg_log2FC": f"{prefix}_log2FC", "p_val": f"{prefix}_p",
        "q": f"{prefix}_q", "regulate": f"{prefix}_regulate",
    })

hyp = load_deg("hypoxiaWTvsWT_supo_DEG_limma.csv", "hypoxia")
lat = load_deg("LatentvsWT_supo_DEG_limma.csv", "latent")
merged = hyp.join(lat, how="outer")
merged["gene_name"] = [name_map.get(g, "") for g in merged.index]
merged.to_csv(OUT / "stress_gene_level.csv")

# Summaries of DNA-damage-related transcriptional responses.
summary_rows = []
for set_name, genes in GENE_SETS.items():
    for condition in ("hypoxia", "latent"):
        observed = merged.loc[merged.index.intersection(genes)].dropna(subset=[f"{condition}_log2FC"]).copy()
        fc = observed[f"{condition}_log2FC"].to_numpy(float)
        q = observed[f"{condition}_q"].to_numpy(float)
        calls = observed[f"{condition}_regulate"].astype(str)
        n = len(observed)
        up = int((fc > 0).sum())
        down = int((fc < 0).sum())
        deg = int((calls.isin(["Up", "Down"])).sum())
        wilcoxon_p = stats.wilcoxon(fc).pvalue if n and np.any(fc != 0) else np.nan
        background = merged[f"{condition}_regulate"].dropna().astype(str)
        bg_deg = int(background.isin(["Up", "Down"]).sum())
        bg_total = len(background)
        table = [[deg, n - deg], [bg_deg - deg, bg_total - bg_deg - (n - deg)]]
        odds, fisher_p = stats.fisher_exact(table)
        summary_rows.append({
            "gene_set": set_name, "condition": condition, "n_tested": n,
            "n_up": up, "n_down": down, "n_DEG_by_original_call": deg,
            "fraction_up": up / n if n else np.nan,
            "median_log2FC": float(np.median(fc)) if n else np.nan,
            "median_q": float(np.median(q)) if n else np.nan,
            "wilcoxon_signed_rank_P_vs_zero": wilcoxon_p,
            "background_DEG_fraction": bg_deg / bg_total,
            "DEG_enrichment_OR": odds, "DEG_enrichment_Fisher_P": fisher_p,
        })
pd.DataFrame(summary_rows).to_csv(OUT / "dna_damage_response_summary.csv", index=False)

# Core SOS gene-level results for inspection and reporting.
core_rows = []
for gene, label in CORE_SOS.items():
    row = {"gene": gene, "annotation": label, "gene_name_in_GTF": name_map.get(gene, "")}
    for condition in ("hypoxia", "latent"):
        if gene in merged.index:
            row[f"{condition}_log2FC"] = merged.loc[gene, f"{condition}_log2FC"]
            row[f"{condition}_q"] = merged.loc[gene, f"{condition}_q"]
            row[f"{condition}_regulate"] = merged.loc[gene, f"{condition}_regulate"]
    core_rows.append(row)
pd.DataFrame(core_rows).to_csv(OUT / "core_SOS_gene_results.csv", index=False)

# Shared genome-wide and DEG responses between hypoxia and latency.
both = merged.dropna(subset=["hypoxia_log2FC", "latent_log2FC"]).copy()
rho, rho_p = stats.spearmanr(both["hypoxia_log2FC"], both["latent_log2FC"])
r, pearson_p = stats.pearsonr(both["hypoxia_log2FC"], both["latent_log2FC"])
concordant = int((np.sign(both["hypoxia_log2FC"]) == np.sign(both["latent_log2FC"])).sum())
nonzero = both[(both["hypoxia_log2FC"] != 0) & (both["latent_log2FC"] != 0)]
concordant_nonzero = int((np.sign(nonzero["hypoxia_log2FC"]) == np.sign(nonzero["latent_log2FC"])).sum())

hyp_degs = set(merged.index[merged["hypoxia_regulate"].isin(["Up", "Down"])])
lat_degs = set(merged.index[merged["latent_regulate"].isin(["Up", "Down"])])
shared_degs = hyp_degs & lat_degs
union_degs = hyp_degs | lat_degs
shared_frame = merged.loc[sorted(shared_degs)].copy()
shared_same = int((np.sign(shared_frame["hypoxia_log2FC"]) == np.sign(shared_frame["latent_log2FC"])).sum())
N = len(merged)
K = len(hyp_degs)
n = len(lat_degs)
x = len(shared_degs)
hypergeom_p = stats.hypergeom.sf(x - 1, N, K, n)
shared_metrics = [
    ("genes_tested_in_both", len(both)),
    ("Spearman_rho_log2FC", rho),
    ("Spearman_P", rho_p),
    ("Pearson_r_log2FC", r),
    ("Pearson_P", pearson_p),
    ("direction_concordant_genes", concordant),
    ("direction_concordant_percent", 100 * concordant / len(both)),
    ("hypoxia_DEGs_original_call", len(hyp_degs)),
    ("latent_DEGs_original_call", len(lat_degs)),
    ("shared_DEGs", x),
    ("Jaccard_DEGs", x / len(union_degs)),
    ("shared_DEG_hypergeometric_P", hypergeom_p),
    ("shared_DEGs_same_direction", shared_same),
    ("shared_DEGs_same_direction_percent", 100 * shared_same / x if x else np.nan),
]
pd.DataFrame(shared_metrics, columns=["metric", "value"]).to_csv(OUT / "shared_response_summary.csv", index=False)
shared_frame.to_csv(OUT / "shared_DEG_gene_level.csv")

print("\nDNA-damage response:")
print(pd.DataFrame(summary_rows).to_string(index=False))
print("\nCore SOS genes:")
print(pd.DataFrame(core_rows).to_string(index=False))
print("\nShared response:")
for key, value in shared_metrics:
    print(f"{key}: {value}")
print(f"\nOutputs: {OUT}")
