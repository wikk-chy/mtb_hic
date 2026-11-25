#!/usr/bin/env python
"""Loop-anchor protein binding analysis.

Questions:
  1. How many of the 41 Chromosight loops have BOTH anchors bound by the SAME
     protein (direct bridging)?
  2. How many loops are INDIRECTLY bridged (anchor1 and anchor2 bound by two
     different proteins that interact in the PPI network)?
  3. How many loop anchors does Rv0047c (NapM) bind (one end = 1 anchor)?
  4. How many loops does Rv0047c bridge (BOTH ends = 1 loop)?

All counts are compared against distance-matched random bin pairs
(1000 permutations) to assess whether the observed counts exceed background,
because Rv0047c binds ~48% of all bins.

Bin numbering: network bins are 1-based (bin_k = genomic bin k-1, 5 kb bins).
Verified: all 71 network DNA-DNA edges match the HiCCUPS bedpe with +1 offset.

Run: python loop_network_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R4_comment6/loop_network_analysis")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

NET_DIR = os.path.join(BASE, "review_data/Interaction_network")
CS_LOOPS = os.path.join(BASE, "review_results/R4_comment3_loops/chromosight_out/wt.tsv")
# 2026-10-06: NapM binding = interval-overlap definition (unified; see
# R1_mechanism_reanalysis/scripts/s08_loop_bridging_unified.py). All other
# proteins keep network-edge binding (no curated peak tables available).
NAPM_INTERVAL = os.path.join(BASE, "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_bound_bins_interval.csv")

BINSIZE = 5000
GENOME = 4411532
N_BINS = 883
SEED = 42
rng = np.random.default_rng(SEED)

C_OBS, C_RAND = "#0072B2", "#999999"

print("=" * 60)
print("Loop-anchor protein binding analysis")
print("=" * 60)

# ---------------- load network ----------------
nodes = pd.read_csv(os.path.join(NET_DIR, "full_nodes.csv"))
edges = pd.read_csv(os.path.join(NET_DIR, "full_edges.csv"))

dp = edges[edges.interaction_type == "DNA-Protein"]
pp = edges[edges.interaction_type == "Protein-Protein"]

# bin -> set of proteins  (network bin "binK" -> 0-based bin K-1)
# 2026-10-06: Rv0047c's bin set is overridden with the unified
# interval-overlap definition (see NAPM_INTERVAL below); all other proteins
# keep network-edge binding.
bin2prot = {}
for _, r in dp.iterrows():
    b = int(str(r.target)[3:]) - 1
    bin2prot.setdefault(b, set()).add(str(r.source))

# PPI adjacency
ppi = set()
for _, r in pp.iterrows():
    ppi.add((str(r.source), str(r.target)))
    ppi.add((str(r.target), str(r.source)))

print(f"DNA-Protein edges: {len(dp)}, proteins with binding: "
      f"{len(set(dp.source))}, PPI edges: {len(pp)}")

rv = "Rv0047c"
rv_bins = set(pd.read_csv(NAPM_INTERVAL)["bin"].astype(int))
# apply the override to bin2prot so shared-protein/bridging counts also use
# the interval definition for NapM
for b in bin2prot:
    bin2prot[b].discard(rv)
for b in rv_bins:
    bin2prot.setdefault(b, set()).add(rv)
print(f"Rv0047c binds {len(rv_bins)}/{N_BINS} bins ({len(rv_bins)/N_BINS*100:.1f}%) "
      f"[interval-overlap definition]")

# ---------------- load Chromosight loops ----------------
cs = pd.read_csv(CS_LOOPS, sep="\t")
loops = list(zip(cs.bin1.astype(int), cs.bin2.astype(int)))
print(f"Chromosight loops: {len(loops)}")

# ---------------- per-loop analysis ----------------
def analyze_loop(b1, b2):
    p1 = bin2prot.get(b1, set())
    p2 = bin2prot.get(b2, set())
    shared = p1 & p2
    # indirect: p1 protein and p2 protein connected by PPI (excluding shared proteins)
    indirect_pairs = [(a, b) for a in (p1 - shared) for b in (p2 - shared)
                      if (a, b) in ppi]
    rv_b1 = b1 in rv_bins
    rv_b2 = b2 in rv_bins
    return dict(
        shared_proteins=sorted(shared),
        n_shared=len(shared),
        direct_bridge=len(shared) > 0,
        n_indirect_pairs=len(indirect_pairs),
        indirect_bridge=len(indirect_pairs) > 0,
        indirect_pairs=indirect_pairs[:10],   # cap for table size
        rv_binds_anchor1=rv_b1,
        rv_binds_anchor2=rv_b2,
        rv_n_anchors=int(rv_b1) + int(rv_b2),
        rv_bridges_loop=rv_b1 and rv_b2,
    )

rows = []
for b1, b2 in loops:
    d = analyze_loop(b1, b2)
    d.update(dict(bin1=b1, bin2=b2,
                  pos1_kb=b1 * BINSIZE // 1000, pos2_kb=b2 * BINSIZE // 1000,
                  dist_kb=(b2 - b1) * BINSIZE // 1000))
    rows.append(d)
res = pd.DataFrame(rows)

# ---------------- summary counts ----------------
n_direct = int(res.direct_bridge.sum())
n_indirect = int(res.indirect_bridge.sum())
n_rv_anchor = int(res.rv_n_anchors.sum())
n_rv_anchor_loops = int((res.rv_n_anchors > 0).sum())
n_rv_bridge = int(res.rv_bridges_loop.sum())

print(f"\n--- observed counts (n={len(loops)} loops) ---")
print(f"direct bridge (same protein both anchors): {n_direct}")
print(f"indirect bridge (PPI-connected proteins):  {n_indirect}")
print(f"Rv0047c-bound anchors: {n_rv_anchor} / {2*len(loops)}")
print(f"loops with >=1 Rv0047c-bound anchor: {n_rv_anchor_loops}")
print(f"loops bridged by Rv0047c (both anchors): {n_rv_bridge}")

# ---------------- permutation background ----------------
print("\nrunning 1000 permutations ...")
dists = [(b2 - b1) for b1, b2 in loops]
perm_stats = []
for it in range(1000):
    pd_ = dict(direct=0, indirect=0, rv_anchor=0, rv_anchor_loop=0, rv_bridge=0)
    for d in dists:
        lo = 1
        hi = N_BINS - d - 1
        if hi <= lo:
            continue
        b1 = int(rng.integers(lo, hi + 1))
        b2 = b1 + d
        r = analyze_loop(b1, b2)
        pd_["direct"] += int(r["direct_bridge"])
        pd_["indirect"] += int(r["indirect_bridge"])
        pd_["rv_anchor"] += r["rv_n_anchors"]
        pd_["rv_anchor_loop"] += int(r["rv_n_anchors"] > 0)
        pd_["rv_bridge"] += int(r["rv_bridges_loop"])
    perm_stats.append(pd_)
perm = pd.DataFrame(perm_stats)

obs = dict(direct=n_direct, indirect=n_indirect, rv_anchor=n_rv_anchor,
           rv_anchor_loop=n_rv_anchor_loops, rv_bridge=n_rv_bridge)

summary_rows = []
for k, o in obs.items():
    null = perm[k].values
    mu, sd = float(np.mean(null)), float(np.std(null))
    p = float((np.sum(null >= o) + 1) / (len(null) + 1))
    z = (o - mu) / sd if sd > 0 else np.nan
    summary_rows.append(dict(metric=k, observed=o,
                             null_mean=round(mu, 2), null_sd=round(sd, 2),
                             z_score=round(z, 2), empirical_p=round(p, 4)))
    print(f"{k:>16}: observed={o:>4}, null={mu:6.1f}±{sd:5.1f}, z={z:6.2f}, p={p:.4f}")

summary = pd.DataFrame(summary_rows)
summary.to_csv(os.path.join(OUT, "tables/binding_summary.csv"), index=False)

# ---------------- detailed tables ----------------
res_out = res.drop(columns=["shared_proteins", "indirect_pairs"])
res_out["shared_proteins"] = res.shared_proteins.apply(lambda x: ";".join(x))
res_out["indirect_pairs"] = res.indirect_pairs.apply(lambda x: ";".join(f"{a}-{b}" for a, b in x))
res_out.to_csv(os.path.join(OUT, "tables/per_loop_binding.csv"), index=False)

# most frequent bridging proteins
from collections import Counter
shared_counter = Counter()
for s in res.shared_proteins:
    shared_counter.update(s)
shared_df = pd.DataFrame(shared_counter.most_common(),
                         columns=["protein", "n_loops_bridged"])
shared_df.to_csv(os.path.join(OUT, "tables/bridging_proteins.csv"), index=False)
print("\ntop bridging proteins:")
print(shared_df.head(10).to_string(index=False))

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# A: observed vs null for the 5 metrics
labels = {"direct": "direct bridge",
          "indirect": "indirect bridge",
          "rv_anchor": "Rv0047c\nanchors",
          "rv_anchor_loop": "loops ≥1\nRv0047c anchor",
          "rv_bridge": "Rv0047c\nbridged loops"}
ks = list(labels.keys())
x = np.arange(len(ks))
obs_vals = [obs[k] for k in ks]
null_means = [summary.loc[summary.metric == k, "null_mean"].iloc[0] for k in ks]
null_sds = [summary.loc[summary.metric == k, "null_sd"].iloc[0] for k in ks]
ps = [summary.loc[summary.metric == k, "empirical_p"].iloc[0] for k in ks]

w = 0.38
axes[0].bar(x - w/2, obs_vals, w, color=C_OBS, edgecolor="black", label="observed (41 loops)")
axes[0].bar(x + w/2, null_means, w, color=C_RAND, edgecolor="black",
            yerr=null_sds, capsize=3, label="random pairs (1000 perm.)")
axes[0].set_xticks(x, [labels[k] for k in ks], fontsize=8)
axes[0].set_ylabel("Count", fontsize=9)
axes[0].set_ylim(0, max(obs_vals) * 1.28)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
for xi, (o, p) in enumerate(zip(obs_vals, ps)):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    axes[0].text(xi - w/2, o + max(obs_vals) * 0.03, f"{o} {star}", ha="center", fontsize=8.5)
axes[0].set_title("A  Protein bridging of Chromosight loops\nvs distance-matched random pairs",
                  loc="left", fontsize=10, fontweight="bold")

# B: permutation null distribution for Rv0047c-bridged loops
axes[1].hist(perm.rv_bridge, bins=20, color=C_RAND, edgecolor="black", alpha=0.8)
axes[1].axvline(n_rv_bridge, color=C_OBS, lw=2)
axes[1].text(n_rv_bridge, axes[1].get_ylim()[1] * 0.9,
             f"observed = {n_rv_bridge}\np = {ps[-1]:.3f}", color=C_OBS,
             fontsize=9, ha="center", fontweight="bold")
axes[1].set_xlabel("Rv0047c-bridged loops in random pairs", fontsize=9)
axes[1].set_ylabel("Permutations", fontsize=9)
axes[1].set_title("B  Permutation null: Rv0047c bridging", loc="left",
                  fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures/binding_analysis.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/binding_analysis.png"), dpi=300, bbox_inches="tight")

with open(os.path.join(OUT, "tables/summary.json"), "w") as f:
    json.dump(dict(observed=obs, summary=summary_rows), f, indent=2)

print("\noutputs in", OUT)
print("done.")
