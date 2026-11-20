#!/usr/bin/env python3
"""
Map <-> 3D-model reconciliation metrics for R4 comment 4 (WT vs napM-KO).

For each LorDG reconstruction (bead = 5 kb, 883 beads), compute the 3D
counterparts of the three map features discussed in the response:
  1. secondary diagonal (arm-arm juxtaposition):
     mean 3D distance of mirror-symmetric bead pairs (x, L-x) with circular
     distance > 500 kb (same definition as the sec-diag O/E metric),
     plus the 20-window arm profile;
  2. ori-ter interaction: 3D distance between ori (bins 0-9, 0-50 kb) and
     ter (bins 400-412, ~2.00-2.06 Mb) bead centroids;
  3. overall decay/compaction: radius of gyration (already in
     model_3d_metrics.csv, repeated here for one-table convenience).

Output: tables/model_map_reconciliation.csv
"""
import numpy as np
import pandas as pd

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{BASE}/review_results/R4_comment4"
PDB = f"{BASE}/review_data/3d_model_pdbfile"
BIN = 5000
N_BINS = 883
GENOME_LEN = 4_411_532


def load_pdb(path):
    c = []
    for line in open(path):
        if line.startswith("ATOM"):
            c.append([float(line[30:38]), float(line[38:46]),
                      float(line[46:54])])
    return np.array(c)


def mirror_pairs(min_mb=0.25, max_mb=2.0, min_circ_kb=500):
    """Bead index pairs (i, j) with pos_j = L - pos_i, arm coordinate in
    [min_mb, max_mb] Mb, circular distance > min_circ_kb."""
    pos = (np.arange(N_BINS) + 0.5) * BIN
    pairs = []
    for i in range(N_BINS):
        x = pos[i]
        xm = GENOME_LEN - x
        j = int(round((xm - BIN / 2) / BIN)) % N_BINS
        dp = abs(pos[i] - pos[j])
        circ = min(dp, GENOME_LEN - dp)
        mb = min(x, xm) / 1e6 if False else x / 1e6
        # arm coordinate: use the left-arm position x in Mb
        if min_mb * 1e6 <= x <= max_mb * 1e6 and circ > min_circ_kb * 1e3 \
                and i < j:
            pairs.append((i, j))
    return pairs


rows = []
profiles = {}
for cond, fn in [("WT", "wt_5k_1701486311672.pdb"),
                 ("napM-KO", "napm_ko_5k_1704965399048.pdb")]:
    xyz = load_pdb(f"{PDB}/{fn}")
    cen = xyz.mean(axis=0)
    rg = float(np.sqrt(((xyz - cen) ** 2).sum(axis=1).mean()))
    ori_c = xyz[0:10].mean(axis=0)
    ter_c = xyz[400:413].mean(axis=0)
    ori_ter_3d = float(np.linalg.norm(ori_c - ter_c))

    pairs = mirror_pairs()
    d = np.array([np.linalg.norm(xyz[i] - xyz[j]) for i, j in pairs])
    # 20-window arm profile (same windows as Panel D of the v3 figure)
    xs = np.array([(i + 0.5) * BIN for i, j in pairs]) / 1e6
    edges = np.linspace(0.25, 2.0, 21)
    prof = [d[(xs >= edges[k]) & (xs < edges[k + 1])].mean()
            for k in range(20)]
    profiles[cond] = prof
    rows.append(dict(
        condition=cond, Rg=rg, ori_ter_3D=ori_ter_3d,
        mirror_arm_3D_mean=float(d.mean()),
        mirror_arm_3D_median=float(np.median(d)),
        n_mirror_pairs=len(pairs),
    ))
    print(f"{cond}: Rg={rg:.3f}, ori-ter 3D={ori_ter_3d:.3f}, "
          f"mirror arm-arm 3D mean={d.mean():.3f} (n={len(pairs)})")

df = pd.DataFrame(rows)
ko, wt = df.iloc[1], df.iloc[0]
print(f"\nKO/WT: Rg {ko.Rg/wt.Rg:.3f}x; ori-ter 3D {ko.ori_ter_3D/wt.ori_ter_3D:.3f}x; "
      f"mirror arm-arm 3D {ko.mirror_arm_3D_mean/wt.mirror_arm_3D_mean:.3f}x")
print(f"=> ori-ter juxtaposition (1/dist) KO/WT = {wt.ori_ter_3D/ko.ori_ter_3D:.3f}x "
      f"(Hi-C ori-ter O/E KO/WT = 1.107)")
print(f"=> arm juxtaposition (1/dist) KO/WT = {wt.mirror_arm_3D_mean/ko.mirror_arm_3D_mean:.3f}x "
      f"(Hi-C sec-diag O/E KO/WT = 0.896)")

df.to_csv(f"{OUT}/tables/model_map_reconciliation.csv", index=False)
pd.DataFrame({"arm_pos_mb": (edges[:-1] + edges[1:]) / 2,
              "WT": profiles["WT"], "napM-KO": profiles["napM-KO"]}).to_csv(
    f"{OUT}/tables/model_mirror_arm_profile.csv", index=False)
print(f"\nwrote tables/model_map_reconciliation.csv, model_mirror_arm_profile.csv")
