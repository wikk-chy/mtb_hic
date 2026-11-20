#!/usr/bin/env python3
"""R4C5 - Hypothesis predictions vs observed values.

Three explanations for increased ori-ter contact under stress:
  S1 global condensation  - whole chromosome compacts; ALL long-range
                            contacts increase and both arms come closer.
  S2 ori-ter tethering    - a specific physical link between ori and ter;
                            only the ori-ter contact changes.
  S3 ring/toroid folding  - chromosome folds into a ring/toroid; ori and ter
                            approach, arms fold around the toroid (arm-arm
                            juxtaposition does NOT increase), long-range
                            contacts increase globally.

Outputs:
  tables/hypothesis_predictions.csv    prediction direction per metric
  tables/observed_vs_predictions.csv   observed fold changes + p values
                                       (latent/WT, hypoxia/WT) vs predictions
"""
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
R4C4 = f"{BASE}/review_results/R4_comment4"
OUT = f"{BASE}/review_results/R4_comment5"
PDB = f"{BASE}/review_data/3d_model_pdbfile"
TER_POS = 2_030_000

METRICS = ["ori_ter_OE", "sec_diag_OE", "frac_short_lt100kb",
           "frac_long_gt1Mb", "compaction_slope", "ori_ter_3D_dist"]

# ---------------- 1. prediction table ----------------
# arrows: up / down / flat (interpretation in README)
pred = {
    "S1_global_condensation": {
        "ori_ter_OE": "up",
        "sec_diag_OE": "up",        # arms also come closer
        "frac_short_lt100kb": "down",
        "frac_long_gt1Mb": "up",
        "compaction_slope": "flatter",
        "ori_ter_3D_dist": "down",
    },
    "S2_oriter_tethering": {
        "ori_ter_OE": "up",
        "sec_diag_OE": "flat",      # specific link only
        "frac_short_lt100kb": "flat",
        "frac_long_gt1Mb": "flat",  # except a bump at ~2 Mb
        "compaction_slope": "flat",
        "ori_ter_3D_dist": "down",
    },
    "S3_ring_toroid": {
        "ori_ter_OE": "up",
        "sec_diag_OE": "down_or_flat",  # arms fold around toroid, not tighter
        "frac_short_lt100kb": "down",
        "frac_long_gt1Mb": "up",
        "compaction_slope": "flatter",
        "ori_ter_3D_dist": "down",
    },
}
df_pred = pd.DataFrame(pred).T
df_pred.index.name = "hypothesis"
df_pred.to_csv(f"{OUT}/tables/hypothesis_predictions.csv")

# ---------------- 2. observed values ----------------
met = pd.read_csv(f"{R4C4}/tables/replicate_metrics.csv")
stat = pd.read_csv(f"{R4C4}/tables/stats_tests.csv")


def load_pdb(path):
    c = []
    for line in open(path):
        if line.startswith("ATOM"):
            c.append([float(line[30:38]), float(line[38:46]),
                      float(line[46:54])])
    return np.array(c)


PDBS = {"WT": "wt_5k_1701486311672.pdb",
        "napM-KO": "napm_ko_5k_1704965399048.pdb",
        "hypoxia": "wt-hypoxia_5k_1704962916426.pdb",
        "latent": "wt-latent_5k_1704965200849.pdb"}
dist3d = {}
for cond, fn in PDBS.items():
    xyz = load_pdb(f"{PDB}/{fn}")
    ori = xyz[0:10].mean(axis=0)
    ter = xyz[400:413].mean(axis=0)
    dist3d[cond] = np.linalg.norm(ori - ter)

rows = []
for m in METRICS:
    row = {"metric": m,
           "pred_S1": pred["S1_global_condensation"][m],
           "pred_S2": pred["S2_oriter_tethering"][m],
           "pred_S3": pred["S3_ring_toroid"][m]}
    for cond in ["latent", "hypoxia", "napM-KO"]:
        if m == "ori_ter_3D_dist":
            row[f"{cond}_fold"] = dist3d[cond] / dist3d["WT"]
            row[f"{cond}_p"] = np.nan     # single model, no test
        else:
            x = met.loc[met.condition == cond, m].values
            y = met.loc[met.condition == "WT", m].values
            row[f"{cond}_fold"] = x.mean() / y.mean()
            p = stat.query("comparison == @cond + ' vs WT' and metric == @m")
            row[f"{cond}_p"] = p["welch_p"].iloc[0] if len(p) else np.nan
    rows.append(row)
df_obs = pd.DataFrame(rows)
df_obs.to_csv(f"{OUT}/tables/observed_vs_predictions.csv", index=False)
print(df_pred.to_string())
print()
print(df_obs.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nSaved tables/hypothesis_predictions.csv, observed_vs_predictions.csv")
