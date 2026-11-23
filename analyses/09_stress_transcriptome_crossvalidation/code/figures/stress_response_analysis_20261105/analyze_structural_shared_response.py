from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
OUT = ROOT / "figures/stress_response_analysis_20261105"
BASE = ROOT / "hic_review/review_results/hypoxia_latency_integration_20261030"
hyp = pd.read_csv(BASE / "source_data/Hypoxia_bin_level.csv", usecols=["bin", "d_ins_hyp_WT"])
lat = pd.read_csv(BASE / "source_data/Latency_bin_level.csv", usecols=["bin", "d_ins_latent_WT"])
merged = hyp.merge(lat, on="bin", validate="1:1").dropna()
rho, _ = stats.spearmanr(merged["d_ins_hyp_WT"], merged["d_ins_latent_WT"])
r, _ = stats.pearsonr(merged["d_ins_hyp_WT"], merged["d_ins_latent_WT"])
concordant = (np.sign(merged["d_ins_hyp_WT"]) == np.sign(merged["d_ins_latent_WT"])).sum()
summary = pd.DataFrame([
    ("bins_tested", len(merged)),
    ("Spearman_rho_hypoxia_vs_latent_insulation", rho),
    ("Pearson_r_hypoxia_vs_latent_insulation", r),
    ("same_direction_bins", int(concordant)),
    ("same_direction_percent", 100 * concordant / len(merged)),
], columns=["metric", "value"])
summary.to_csv(OUT / "shared_structural_response_summary.csv", index=False)
print(summary.to_string(index=False))

metrics = pd.read_csv(ROOT / "hic_review/review_results/R4_comment5/tables/hypoxia_vs_latent_signature.csv")
print(f"Global metrics with same direction: {metrics['same_direction'].sum()}/{len(metrics)}")
print(metrics[["metric", "hypoxia_fold", "latent_fold", "same_direction", "latent_stronger"]].to_string(index=False))
