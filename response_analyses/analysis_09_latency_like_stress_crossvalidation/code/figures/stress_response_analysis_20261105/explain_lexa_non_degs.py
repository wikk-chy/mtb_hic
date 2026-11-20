from pathlib import Path
import pandas as pd

out = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/figures/stress_response_analysis_20261105")
genes = pd.read_csv(out / "stress_gene_level.csv", index_col=0)
lexa = ["Rv3395c", "Rv0515", "Rv3776", "Rv0336", "Rv3370c", "Rv3260c", "Rv2720",
        "Rv1378c", "Rv3074", "Rv2737c", "Rv1702c", "Rv1000c", "Rv1057", "Rv2719c",
        "Rv2100", "Rv0071", "Rv2594c", "Rv0095c", "Rv2578c", "Rv2579", "Rv1588c", "Rv2593c"]
frame = genes.loc[lexa].copy()
frame["abs_log2FC"] = frame["latent_log2FC"].abs()
frame["explanation"] = "other"
frame.loc[(frame["latent_q"] < 0.05) & (frame["abs_log2FC"] <= 1), "explanation"] = "q < 0.05 but |log2FC| <= 1"
frame.loc[frame["latent_q"] >= 0.05, "explanation"] = "q >= 0.05"
non_degs = frame[~frame["latent_regulate"].isin(["Up", "Down"])].copy()
non_degs.to_csv(out / "latent_LexA_nonDEG_details.csv")
print(f"Non-DEG count: {len(non_degs)}")
print(non_degs["explanation"].value_counts().to_string())
print(non_degs[["latent_log2FC", "latent_q", "latent_regulate", "explanation"]].to_string())
