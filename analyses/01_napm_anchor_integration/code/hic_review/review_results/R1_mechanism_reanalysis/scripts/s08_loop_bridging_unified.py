#!/usr/bin/env python3
"""s08_loop_bridging_unified.py — unified loop-bridging statistics.

Single source of truth for "NapM bridges loop anchors". Three NapM binding
definitions x three hit metrics, all tested against the same circular-shift
null (whole loop set shifted by s bins, (bin+s)%883, 10,000 shifts, seed 42).

Definitions:
  primary     = INTERVAL OVERLAP (2026-10-06 unified definition):
                a peak binds a 5-kb bin if its [start, end] interval
                overlaps the bin -> avoids the arbitrary centre-bin
                assignment that caused the 15-vs-17 discrepancy
                (network edges had used the peak end / rounding, shifting
                edge-crossing peaks by one bin)
  curated     = Excel/curated ChIP peaks (Minch et al. 2015):
                unique center_bin of NapM_binding_curated.tsv -> 433 bins
  network     = interaction-network DNA-Protein edges for Rv0047c
                (full_edges.csv; network "binK" 1-based -> 0-based K-1) -> 427 bins

Metrics (per 41 Chromosight WT loops):
  both         = both anchors in bound bins
  any          = at least one anchor bound
  exactly_one  = exactly one anchor bound

Outputs (R1_mechanism_reanalysis/06_permutation_tests/):
  loop_bridging_unified.csv
  null_loop_bridging_{definition}_{both,any,exactly_one}.npy
"""
import numpy as np
import pandas as pd

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review"
RA = f"{ROOT}/hic_review/review_results/R1_mechanism_reanalysis"
OUT = f"{RA}/06_permutation_tests"
N_BINS = 883
BIN = 5000
NPERM = 10000
SEED = 42

# ---- loops (identical input for all definitions) ----
cs = pd.read_csv(f"{ROOT}/hic_review/review_results/R4_comment3_loops/"
                 f"chromosight_out/wt.tsv", sep="\t")
b1 = cs["bin1"].values.astype(int)
b2 = cs["bin2"].values.astype(int)
assert len(cs) == 41

# ---- binding definitions ----
pk = pd.read_csv(f"{RA}/01_chip_curated/NapM_binding_curated.tsv", sep="\t")

# interval overlap: peak [start,end] overlaps bin [b*5000, (b+1)*5000)
bins_interval = np.zeros(N_BINS, dtype=bool)
for _, r in pk.iterrows():
    b_lo = int(r.start // BIN)
    b_hi = int(min(r.end - 1, N_BINS * BIN - 1) // BIN)
    bins_interval[b_lo:b_hi + 1] = True

bins_curated = np.sort(pk["center_bin"].unique().astype(int))

edges = pd.read_csv(f"{ROOT}/hic_review/review_data/Interaction_network/full_edges.csv")
dp = edges[edges.interaction_type == "DNA-Protein"]
bins_network = np.sort(np.array(
    [int(str(t)[3:]) - 1 for s, t in zip(dp.source, dp.target)
     if str(s) == "Rv0047c"], dtype=int))
bins_network = np.unique(bins_network)

def mask(bins):
    m = np.zeros(N_BINS, dtype=bool)
    m[bins] = True
    return m

masks = dict(interval=bins_interval,
             curated=mask(bins_curated),
             network=mask(bins_network))
print(f"primary  (interval overlap):  {int(bins_interval.sum())} bins")
print(f"sens.cur (curated center bin): {len(bins_curated)} bins")
print(f"sens.net (network edges):      {len(bins_network)} bins")

def count(bound, x1, x2):
    h1, h2 = bound[x1], bound[x2]
    return dict(both=int((h1 & h2).sum()),
                any=int((h1 | h2).sum()),
                exactly_one=int((h1 ^ h2).sum()))

# ---- circular-shift null: shift the WHOLE loop set, same shift array for all ----
rng = np.random.default_rng(SEED)
shifts = rng.integers(0, N_BINS, NPERM)

rows = []
for defname, bound in masks.items():
    obs = count(bound, b1, b2)
    nulls = {k: np.empty(NPERM) for k in obs}
    for i, s in enumerate(shifts):
        ns = count(bound, (b1 + s) % N_BINS, (b2 + s) % N_BINS)
        for k in nulls:
            nulls[k][i] = ns[k]
    for k, null in nulls.items():
        np.save(f"{OUT}/null_loop_bridging_{defname}_{k}.npy", null)
        mu, sd = float(null.mean()), float(null.std())
        lo, hi = np.percentile(null, [2.5, 97.5])
        p = float((np.sum(null >= obs[k]) + 1) / (NPERM + 1))
        z = (obs[k] - mu) / sd if sd > 0 else np.nan
        rows.append(dict(definition=defname, metric=k, observed=obs[k],
                         null_mean=round(mu, 3), null_sd=round(sd, 3),
                         null_ci95_lo=float(lo), null_ci95_hi=float(hi),
                         fold_enrichment=round(obs[k] / mu, 3),
                         z_score=round(z, 3), p_empirical=round(p, 5)))
        print(f"{defname:>11} {k:>12}: obs={obs[k]:>3}  null={mu:5.2f}±{sd:4.2f} "
              f"[{lo:.0f},{hi:.0f}]  fold={obs[k]/mu:4.2f}  z={z:5.2f}  p={p:.5f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/loop_bridging_unified.csv", index=False)
print("wrote", f"{OUT}/loop_bridging_unified.csv")

# cross-check: curated (centre-bin) both-null must reproduce the existing
# Fig1Dc null from s06
old = np.load(f"{OUT}/null_fig1dc_loops_both.npy")
new = np.load(f"{OUT}/null_loop_bridging_curated_both.npy")
print("curated both-null identical to s06 null_fig1dc_loops_both.npy:",
      bool(np.array_equal(old, new)))
