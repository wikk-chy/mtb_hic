#!/usr/bin/env python3
"""Create new Response Fig. 2C, D and E using response modes, not binding classes."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
SRC = ROOT / "response_figure_reproducibility/figure_02_hypoxia_factorial_integration/results/hic_review/review_results/hypoxia_latency_integration_20261030/source_data"
SAMPLES = ROOT / "response_figure_reproducibility/figure_01_napm_anchor_integration/results/hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/source_data/Fig2A_independent_samples.csv"
OUT = ROOT / "docx_latex_conversion/generated"
DET = ["detected_increase", "detected_decrease"]
RNG = np.random.default_rng(42)

bins = pd.read_csv(SRC / "Mechanism_bin_evidence_classes.csv")
full = pd.read_csv(SRC / "Hypoxia_bin_level.csv")
assert len(bins) == len(full) == 883
full = full[["bin", "RNA_WT_hypoxia_n_tested_genes", "RNA_KO_hypoxia_n_tested_genes",
             "RNA_interaction_n_tested_genes"]]
bins = bins.merge(full, on="bin", how="left", validate="one_to_one")
bound = bins.NapM_bound.fillna(False).astype(bool)
structure = bins.interaction_ins_class.isin(DET)
rna = bins.RNA_interaction_any_DEG.fillna(False).astype(bool)
any_rna = bins[["RNA_WT_hypoxia_any_DEG", "RNA_KO_hypoxia_any_DEG",
                "RNA_interaction_any_DEG"]].fillna(False).astype(bool).any(axis=1)
tested = bins[["RNA_WT_hypoxia_n_tested_genes", "RNA_KO_hypoxia_n_tested_genes",
               "RNA_interaction_n_tested_genes"]].min(axis=1).fillna(0).gt(0)
small = bins[["RNA_WT_hypoxia_mean_abs_log2FC", "RNA_KO_hypoxia_mean_abs_log2FC",
              "RNA_interaction_mean_abs_log2FC"]].max(axis=1).lt(0.5)
stringent = structure & tested & small & ~any_rna

masks = {
    "RNA only": rna & ~structure,
    "Structure only": structure & ~rna,
    "Joint response": rna & structure,
    "Other / unresolved": ~rna & ~structure,
}
expected = {"RNA only": 292, "Structure only": 144, "Joint response": 171, "Other / unresolved": 276}
assert {key: int(value.sum()) for key, value in masks.items()} == expected
assert int(stringent.sum()) == 33
assert int((stringent & bound).sum()) == 17
bins["response_mode"] = "Other / unresolved"
for key in ["RNA only", "Structure only", "Joint response"]:
    bins.loc[masks[key], "response_mode"] = key
bins["stringent_no_local_RNA_response"] = stringent
bins.to_csv(OUT / "Response_Fig2E_response_mode_bins.csv", index=False)

summary = []
for key, mask in masks.items():
    summary.append(dict(response_mode=key, n_bins=int(mask.sum()),
                        percent_all=100 * mask.sum() / len(bins),
                        percent_responsive=100 * mask.sum() / int((rna | structure).sum()),
                        napm_bound=int((mask & bound).sum()), unbound=int((mask & ~bound).sum())))
pd.DataFrame(summary).to_csv(OUT / "Response_Fig2D_response_mode_counts.csv", index=False)

# NapM binding enrichment is assessed after classification, not used to define classes.
enrichment = []
for key, mask in masks.items():
    a = int((mask & bound).sum())
    c = int((mask & ~bound).sum())
    table = [[a, c], [int(bound.sum() - a), int((~bound).sum() - c)]]
    odds, p = fisher_exact(table, alternative="two-sided")
    enrichment.append(dict(response_mode=key, odds_ratio=odds, p_value=p))
enrichment = pd.DataFrame(enrichment)
order = enrichment.p_value.argsort().to_numpy()
q = np.empty(len(enrichment))
running = 1.0
for rank in range(len(enrichment), 0, -1):
    idx = order[rank - 1]
    running = min(running, enrichment.p_value.iloc[idx] * len(enrichment) / rank)
    q[idx] = running
enrichment["BH_q"] = q
enrichment.to_csv(OUT / "Response_Fig2D_NapM_binding_enrichment.csv", index=False)
assert np.allclose(enrichment.BH_q, 0.943, atol=0.002)

# ---------------- C: hypoxia-associated insulation response ----------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.labelsize": 7.4, "axes.titlesize": 9.0,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
wt = bins.d_ins_hyp_WT.dropna().to_numpy(float)
ko = bins.d_ins_hyp_KO.dropna().to_numpy(float)
fig, ax = plt.subplots(figsize=(3.45, 4.20))
fig.subplots_adjust(left=0.16, right=0.98, top=0.83, bottom=0.20)
parts = ax.violinplot([wt, ko], positions=[1, 2], widths=0.68,
                      showmeans=False, showmedians=False, showextrema=False)
for body, color in zip(parts["bodies"], ["#3B6FB6", "#D97925"]):
    body.set_facecolor(color)
    body.set_alpha(0.26)
    body.set_edgecolor("none")
for i, (values, color) in enumerate([(wt, "#3B6FB6"), (ko, "#D97925")], 1):
    x = i + RNG.uniform(-0.09, 0.09, len(values))
    ax.scatter(x, values, s=2.2, color=color, alpha=0.10, edgecolors="none")
    mean = np.mean(values)
    median = np.median(values)
    ax.scatter(i, mean, marker="D", s=28, color=color, edgecolors="white", linewidths=0.6, zorder=5)
    ax.plot([i - 0.24, i + 0.24], [median, median], color="#263238", lw=1.0, zorder=4)
    ax.text(i, mean + 0.10, f"mean {mean:+.3f}", ha="center", fontsize=6.2, color=color)
ax.axhline(0, color="#999999", lw=0.55, ls="--")
ax.set_xticks([1, 2], ["WT", "ΔnapM"])
ax.set_ylabel("Δ insulation\n(hypoxia − normoxia)")
ax.set_title("Hypoxia-associated insulation change", loc="left", pad=8, fontweight="bold")
ax.text(-0.14, 1.10, "C", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
ax.text(0.03, 0.015,
        "Independent-sample genome means: WT +0.041; KO −0.067\n"
        "Genotype contrast = −0.108; exact oxygen-label P = 4.08 × 10$^{-4}$",
        transform=ax.transAxes, fontsize=5.7, va="bottom", color="#263238")
fig.savefig(OUT / "Response_Fig_02C_insulation_response.pdf", bbox_inches="tight")
fig.savefig(OUT / "Response_Fig_02C_insulation_response.png", dpi=450, bbox_inches="tight")
plt.close(fig)
bins[["bin", "d_ins_hyp_WT", "d_ins_hyp_KO"]].to_csv(OUT / "Response_Fig2C_insulation_response.csv", index=False)

samples = pd.read_csv(SAMPLES)
means = samples.groupby("condition").genome_mean_insulation.mean()
wt_response = means["HYP"] - means["WT"]
ko_response = means["KO-HYP"] - means["KO"]
pd.DataFrame([
    dict(contrast="WT hypoxia − WT normoxia", response=wt_response),
    dict(contrast="KO hypoxia − KO normoxia", response=ko_response),
    dict(contrast="KO response − WT response", response=ko_response - wt_response,
         exact_oxygen_label_P=4.08e-4),
]).to_csv(OUT / "Response_Fig2C_independent_sample_summary.csv", index=False)
print("Independent responses:", wt_response, ko_response, ko_response - wt_response)

# ---------------- D: response-mode counts and proportions ----------------
colors = {"RNA only": "#2A9D8F", "Structure only": "#6F4E9C",
          "Joint response": "#D97925", "Other / unresolved": "#C8CDD1"}
fig, ax = plt.subplots(figsize=(3.35, 3.90))
fig.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.12)
keys = list(masks)
y = np.arange(len(keys))[::-1]
for yi, key in zip(y, keys):
    n = int(masks[key].sum())
    ax.barh(yi, n, height=0.58, color=colors[key], alpha=0.82)
    ax.text(n + 8, yi, f"{n} ({100*n/len(bins):.1f}%)", va="center", fontsize=6.7, color="#263238")
ax.set_yticks(y, ["RNA only", "Structure only", "Joint response", "Other / unresolved"])
ax.set_xlim(0, 350)
ax.set_xlabel("Bins, n")
ax.set_title("Major response modes", loc="left", pad=8, fontweight="bold")
ax.text(-0.29, 1.08, "D", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
fig.savefig(OUT / "Response_Fig_02D_response_modes.pdf", bbox_inches="tight")
fig.savefig(OUT / "Response_Fig_02D_response_modes.png", dpi=450, bbox_inches="tight")
plt.close(fig)

# ---------------- E: matching four-class scatter ----------------
fig, ax = plt.subplots(figsize=(4.85, 3.25))
fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.18)
for key in ["Other / unresolved", "RNA only", "Structure only", "Joint response"]:
    z = bins[bins.response_mode.eq(key)]
    size = 5 if key == "Other / unresolved" else 9 if key == "RNA only" else 15
    alpha = 0.40 if key == "Other / unresolved" else 0.68
    ax.scatter(z.RNA_interaction_mean_log2FC, z.d_ins_interaction,
               s=size, color=colors[key], alpha=alpha, edgecolors="none",
               label=f"{key} (n={len(z)})")
ax.axhline(0, color="#AAAAAA", lw=0.5)
ax.axvline(0, color="#AAAAAA", lw=0.5)
for bin_id, name, dx, dy in [(0, "oriC", 5, 4), (874, "ESX-2", 5, -9),
                             (272, "sigE", 5, 4), (699, "DosR/S", 5, 4)]:
    row = bins[bins["bin"].eq(bin_id)].iloc[0]
    ax.annotate(name, (row.RNA_interaction_mean_log2FC, row.d_ins_interaction),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.8,
                fontweight="bold", color="#263238")
ax.text(0.025, 0.975, "Spearman ρ = 0.028; circular-shift P = 0.419",
        transform=ax.transAxes, va="top", fontsize=6.7, color="#263238")
ax.text(-0.105, 1.10, "E", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
ax.set_title("RNA–structure response modes", loc="left", pad=8, fontweight="bold")
ax.set_xlabel("Bin-mean RNA interaction log$_2$FC")
ax.set_ylabel("Insulation interaction (Δ log$_2$)")
ax.legend(frameon=True, framealpha=0.92, edgecolor="none", loc="lower left",
          fontsize=5.9, handletextpad=0.35, borderpad=0.35, labelspacing=0.32)
ax.margins(x=0.055, y=0.10)
fig.savefig(OUT / "Response_Fig_02E_response_modes.pdf", bbox_inches="tight")
fig.savefig(OUT / "Response_Fig_02E_response_modes.png", dpi=450, bbox_inches="tight")
plt.close(fig)
print("Wrote new C, D and E panels")
