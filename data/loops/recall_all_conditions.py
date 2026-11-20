#!/usr/bin/env python
"""Convert validated Chromosight loops + DI-transition CID boundaries to bedpe/bed,
overwriting the previous incorrect results for all conditions.

Loops: Chromosight (already run for WT + 4 conditions). CID boundaries: DI
transition rule (DI <= -2 -> DI >= +2, largest positive jump, min CID 100 kb)
applied to each condition's DI score track.

Conditions: wt, hypoxia, latent, napm_ko, napmhypoxia (ko-hypoxia)

Outputs (overwrite):
  review_data/loops/{wt-loops,hypoxia-loops,latent-loops,napm_ko-loops,napM_ko_hypoxia_loops}.bedpe
  review_data/cid_boundary/{wt-boundary,hypoxia-boundary,Latent-boundary,NapM-Ko-boundary,NapM-Ko-hypoxia-boundary}.bed
"""

import os
import json
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
RD = os.path.join(BASE, "review_data")
CS_DIR = os.path.join(RD, "loops/chromosight_runs")
DI_DIR = os.path.join(RD, "DI_Score_results")

BINSIZE = 5000
NEG_THR, POS_THR = -2.0, 2.0
LOOKBACK, LOOKAHEAD = 10, 10
MIN_CID_BINS = 20   # 100 kb

# loop file name mapping: condition -> (chromosight_tsv, output_bedpe)
LOOP_MAP = {
    "wt": (os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv"),
           os.path.join(RD, "loops/wt-loops.bedpe")),
    "hypoxia": (os.path.join(CS_DIR, "hypoxia.tsv"),
                os.path.join(RD, "loops/hypoxia-loops.bedpe")),
    "latent": (os.path.join(CS_DIR, "latent.tsv"),
               os.path.join(RD, "loops/latent-loops.bedpe")),
    "napm_ko": (os.path.join(CS_DIR, "napm_ko.tsv"),
                os.path.join(RD, "loops/napm_ko-loops.bedpe")),
    "napmhypoxia": (os.path.join(CS_DIR, "napmhypoxia.tsv"),
                    os.path.join(RD, "loops/napM_ko_hypoxia_loops.bedpe")),
}

# boundary file name mapping
BD_MAP = {
    "wt": (os.path.join(DI_DIR, "wt.csv"), os.path.join(RD, "cid_boundary/wt-boundary.bed")),
    "hypoxia": (os.path.join(DI_DIR, "hypoxia.csv"), os.path.join(RD, "cid_boundary/hypoxia-boundary.bed")),
    "latent": (os.path.join(DI_DIR, "latent.csv"), os.path.join(RD, "cid_boundary/Latent-boundary.bed")),
    "napm_ko": (os.path.join(DI_DIR, "napm-ko.csv"), os.path.join(RD, "cid_boundary/NapM-Ko-boundary.bed")),
    "napmhypoxia": (os.path.join(DI_DIR, "napm-ko-hypoxia.csv"), os.path.join(RD, "cid_boundary/NapM-Ko-hypoxia-boundary.bed")),
}

summary = {"loops": {}, "boundaries": {}}

# ---------------- loops: chromosight tsv -> bedpe ----------------
print("=== loops (Chromosight -> bedpe) ===")
for cond, (tsv, outbed) in LOOP_MAP.items():
    df = pd.read_csv(tsv, sep="\t")
    bed = pd.DataFrame({
        0: df.chrom1, 1: df.start1, 2: df.end1,
        3: df.chrom2, 4: df.start2, 5: df.end2,
        6: ".", 7: 1, 8: "+", 9: "+",
    })
    bed.to_csv(outbed, sep="\t", header=False, index=False)
    summary["loops"][cond] = len(df)
    print(f"  {cond}: {len(df)} loops -> {os.path.basename(outbed)}")

# ---------------- CID boundaries: DI transition rule ----------------
def call_boundaries(di):
    n = len(di)
    cands = []
    for i in range(1, n - 1):
        left = di[max(0, i - LOOKBACK):i + 1]
        right = di[i + 1:i + 1 + LOOKAHEAD]
        if (left <= NEG_THR).any() and (right >= POS_THR).any():
            cands.append((i, di[i + 1] - di[i]))
    cands.sort(key=lambda x: -x[1])
    chosen = []
    for i, jump in cands:
        if all(abs(i - c) >= MIN_CID_BINS for c, _ in chosen):
            chosen.append((i, jump))
    return sorted(chosen)

print("\n=== CID boundaries (DI transition rule) ===")
for cond, (dicsv, outbed) in BD_MAP.items():
    di = pd.read_csv(dicsv, header=None).values.flatten()
    boundaries = call_boundaries(di)
    df = pd.DataFrame({
        0: "Chromosome",
        1: [b * BINSIZE for b, _ in boundaries],
        2: [b * BINSIZE + BINSIZE for b, _ in boundaries],
    })
    df.to_csv(outbed, sep="\t", header=False, index=False)
    summary["boundaries"][cond] = len(boundaries)
    print(f"  {cond}: {len(boundaries)} boundaries -> {os.path.basename(outbed)}")

with open(os.path.join(RD, "loops/recalling_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nsummary:", json.dumps(summary))
