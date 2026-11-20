#!/usr/bin/env python3
"""Recompute Fig. 3C matched gene/operon effects from the actual feature pairs.

The legacy gene-level circular-shift statistics indexed gene-level arrays with
5-kb bin coordinates. Those statistics are not used here. The operon-level
observed effects were consistent with their source table, but its spatial null
is also not carried forward. This script therefore reports observed matched
effects and ordinary matched-pair percentile-bootstrap intervals only. These
intervals are descriptive and are not spatial-autocorrelation-corrected tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NBOOT = 10_000
OUT = Path(__file__).resolve().parents[1]
DEL = OUT.parent
RA = DEL.parent.parent
TAB = OUT / "tables"
SRC = TAB / "corrected_figure_sources"
TAB.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

rows = []
boot_distributions = {}


def add_result(level, metric, bound, matched):
    bound = np.asarray(bound, dtype=float)
    matched = np.asarray(matched, dtype=float)
    ok = np.isfinite(bound) & np.isfinite(matched)
    diffs = bound[ok] - matched[ok]
    n = len(diffs)
    if n == 0:
        raise ValueError(f"No finite matched pairs for {level} {metric}")
    observed = float(np.mean(diffs))
    boot = np.mean(
        rng.choice(diffs, size=(NBOOT, n), replace=True), axis=1
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    key = f"{level}_{'abslog2FC' if 'log2FC' in metric else 'DEG'}"
    boot_distributions[key] = boot
    rows.append({
        "level": level,
        "metric": metric,
        "n_pairs": n,
        "observed": observed,
        "delta": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": np.nan,
        "test": "not reported",
        "interval_method": (
            f"ordinary matched-pair percentile bootstrap 95% interval; "
            f"{NBOOT} resamples; seed {SEED}"
        ),
        "spatial_p_status": (
            "not reported; legacy feature-level spatial null was not retained"
        ),
    })
    return diffs


# Gene level: use the gene identifiers in the matched-pair table, never bin
# coordinates as indices into the gene array.
gene = pd.read_csv(RA / "04_rna_integration/gene_level_rna_v2.csv")
pairs = pd.read_csv(RA / "05_matched_controls/matched_pairs_genes.csv")
g = gene.set_index("gene_id")
if not pairs.bound_gene.is_unique or not pairs.matched_gene.is_unique:
    raise ValueError("Gene matching is not one-to-one")
gb = g.loc[pairs.bound_gene].reset_index()
gm = g.loc[pairs.matched_gene].reset_index()
detected = {"detected_increase", "detected_decrease"}
gene_detail = pd.DataFrame({
    "pair_id": np.arange(1, len(pairs) + 1),
    "bound_gene": pairs.bound_gene,
    "matched_gene": pairs.matched_gene,
    "bound_tss_bin": gb.bin_tss.to_numpy(),
    "matched_tss_bin": gm.bin_tss.to_numpy(),
    "bound_abslog2FC": gb.log2FC_KOvsWT.abs().to_numpy(),
    "matched_abslog2FC": gm.log2FC_KOvsWT.abs().to_numpy(),
    "bound_deg": gb.class_KOvsWT.isin(detected).astype(float),
    "matched_deg": gm.class_KOvsWT.isin(detected).astype(float),
    "match_distance": pairs.match_distance,
    "caliper_exceeded": pairs.caliper_exceeded,
})
gene_detail.to_csv(SRC / "fig3c_gene_pairs_recomputed.csv", index=False)
add_result(
    "gene", "|RNA log2FC|",
    gene_detail.bound_abslog2FC, gene_detail.matched_abslog2FC,
)
add_result("gene", "P(DEG)", gene_detail.bound_deg, gene_detail.matched_deg)

# Operon level: the observed values in fig6B_source are indexed by operon,
# and reproduce the legacy observed effects. Only those observed values are
# reused; its nearest-mid-filled circular-shift null is not reused.
operon = pd.read_csv(RA / "08_extended/fig6_operon/tables/fig6B_source.csv")
operon_detail = operon.copy()
operon_detail.insert(0, "pair_id", np.arange(1, len(operon_detail) + 1))
operon_detail.to_csv(SRC / "fig3c_operon_pairs_recomputed.csv", index=False)
add_result(
    "operon", "|RNA log2FC|",
    operon.bound_abslog2FC, operon.matched_abslog2FC,
)
add_result(
    "operon", "DEG fraction",
    operon.bound_deg_frac, operon.matched_deg_frac,
)

summary = pd.DataFrame(rows)
summary.to_csv(TAB / "fig3c_matched_effects_recomputed.csv", index=False)
summary.to_csv(SRC / "fig3_final_panelC_matched_summary.csv", index=False)
np.savez_compressed(TAB / "fig3c_matched_bootstrap.npz", **boot_distributions)

# The existing ECDF source already uses gene indices and should agree with the
# reconstructed gene-pair values.
ecdf = pd.read_csv(SRC / "fig3_final_panelC_gene_ecdf.csv")
if len(ecdf) != len(gene_detail):
    raise AssertionError("ECDF and reconstructed gene-pair row counts differ")
if not np.allclose(
    ecdf.bound_abslog2FC, gene_detail.bound_abslog2FC, equal_nan=True
) or not np.allclose(
    ecdf.matched_abslog2FC, gene_detail.matched_abslog2FC, equal_nan=True
):
    raise AssertionError("ECDF and reconstructed gene-pair values differ")

print(summary[["level", "metric", "n_pairs", "observed", "ci_lo", "ci_hi"]].to_string(index=False))
print("No feature-level spatial P values are reported; intervals are descriptive.")
print("ECDF source agrees with the reconstructed gene pairs.")
