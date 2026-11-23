#!/usr/bin/env python3
"""R4C5 - hypoxia vs latent signature table.

Side-by-side fold changes (vs WT) and Welch/MW p values for all replicate
metrics, plus direction-consistency and relative-strength columns.
Output: tables/hypoxia_vs_latent_signature.csv
"""
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
R4C4 = f"{BASE}/review_results/R4_comment4"
OUT = f"{BASE}/review_results/R4_comment5"

met = pd.read_csv(f"{R4C4}/tables/replicate_metrics.csv")
stat = pd.read_csv(f"{R4C4}/tables/stats_tests.csv")

METRICS = ["frac_short_lt100kb", "frac_mid_100kb_1Mb", "frac_long_gt1Mb",
           "sec_diag_OE", "ori_ter_OE", "compaction_slope"]

rows = []
for m in METRICS:
    wt = met.loc[met.condition == "WT", m].values.mean()
    row = {"metric": m, "WT_mean": wt}
    for cond in ["hypoxia", "latent"]:
        x = met.loc[met.condition == cond, m].values
        p = stat.query("comparison == @cond + ' vs WT' and metric == @m")
        row[f"{cond}_mean"] = x.mean()
        row[f"{cond}_fold"] = x.mean() / wt
        row[f"{cond}_welch_p"] = p["welch_p"].iloc[0]
        row[f"{cond}_mw_p"] = p["mw_p"].iloc[0]
    fh, fl = row["hypoxia_fold"], row["latent_fold"]
    row["same_direction"] = bool((fh - 1) * (fl - 1) > 0)
    row["latent_stronger"] = bool(abs(fl - 1) > abs(fh - 1))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/hypoxia_vs_latent_signature.csv", index=False)
cols = ["metric", "hypoxia_fold", "hypoxia_welch_p", "latent_fold",
        "latent_welch_p", "same_direction", "latent_stronger"]
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nSaved tables/hypoxia_vs_latent_signature.csv")
