#!/usr/bin/env python
"""Response: NapM expression across conditions + NapM-dependence of genome
organization in hypoxia/latent (Reviewer question).

Part 1: NapM (Rv0047c) expression across WT conditions.
Part 2: how much does genome organization in hypoxia/latent depend on NapM.

Run: python napm_expression_dependence.py
"""

import os, json
import numpy as np, pandas as pd, cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = os.path.join(BASE, "review_results/R_napm_expression_dependence")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

C_WT, C_HYP, C_LAT, C_KO = "#0072B2", "#56B4E9", "#8ab07c", "#D55E00"
matplotlib.rcParams["font.family"] = "Arial"

# ---------------- Part 1: NapM expression (CPM-normalized) ----------------
rna = pd.read_csv(BASE + "/review_data/rnaseq/AKR_RNA_seq_merged.csv").set_index("gene")
lib = rna.sum(axis=0)   # library size per sample
rv = rna.loc["Rv0047c"]
cpm = {c: rv[c] / lib[c] * 1e6 for c in rna.columns}
expr = {
    "WT": np.mean([cpm["WT-rep1"], cpm["WT-rep2"]]),
    "Hypoxia": np.mean([cpm["Hypoxia-rep1"], cpm["Hypoxia-rep2"]]),
    "Latent": np.mean([cpm["Latent-rep1"], cpm["Latent-rep2"]]),
}
# per-replicate CPM for error bars
expr_reps = {
    "WT": [cpm["WT-rep1"], cpm["WT-rep2"]],
    "Hypoxia": [cpm["Hypoxia-rep1"], cpm["Hypoxia-rep2"]],
    "Latent": [cpm["Latent-rep1"], cpm["Latent-rep2"]],
}
# DEG stats (limma, TMM-normalized)
deg_hyp = pd.read_csv(BASE + "/review_data/rnaseq/hypoxiaWTvsWT_supo_DEG_limma.csv", index_col=0)
deg_lat = pd.read_csv(BASE + "/review_data/rnaseq/LatentvsWT_supo_DEG_limma.csv", index_col=0)
hyp_fc, hyp_p = deg_hyp.loc["Rv0047c", "avg_log2FC"], deg_hyp.loc["Rv0047c", "p_val"]
lat_fc, lat_p = deg_lat.loc["Rv0047c", "avg_log2FC"], deg_lat.loc["Rv0047c", "p_val"]

print("NapM expression (CPM):", {k: round(v) for k, v in expr.items()})
print(f"  fold vs WT: hypoxia {expr['Hypoxia']/expr['WT']:.2f}x, latent {expr['Latent']/expr['WT']:.2f}x")
print(f"  limma (TMM): hypoxia log2FC={hyp_fc:.2f} (p={hyp_p:.1e}), latent log2FC={lat_fc:.2f} (p={lat_p:.1e})")

# ---------------- Part 2: structural dependence ----------------
clr = {k: cooler.Cooler(BASE + "/review_data/cool_file/" + v) for k, v in {
    "wt": "wt_5k.cool::/resolutions/5000", "ko": "napm_ko_5k.cool::/resolutions/5000",
    "hyp": "hypoxia_5k.mcool::/resolutions/5000", "kohyp": "napmhypoxia_5k.mcool::/resolutions/5000",
    "lat": "latent_5k.mcool::/resolutions/5000"}.items()}
mats = {k: np.nan_to_num(c.matrix(sparse=False).fetch("Chromosome"), nan=0) for k, c in clr.items()}
n = mats["wt"].shape[0]

def lr_frac(m, mind=10):
    tot = m.sum(axis=1)
    lr = np.array([m[i, :max(0, i - mind)].sum() + m[i, i + mind + 1:].sum() for i in range(m.shape[0])])
    return lr / np.maximum(tot, 1e-9)

fl = {k: lr_frac(m) for k, m in mats.items()}
resp_wt = fl["hyp"] - fl["wt"]
resp_ko = fl["kohyp"] - fl["ko"]
resp_lat = fl["lat"] - fl["wt"]
u, p_hyp = stats.mannwhitneyu(resp_wt, resp_ko)
rho_resp, _ = stats.spearmanr(resp_wt, resp_ko)

def gdist(a, b):
    va = mats[a] / max(mats[a].sum(), 1e-9)
    vb = mats[b] / max(mats[b].sum(), 1e-9)
    iu = np.triu_indices(n, 5)
    return 1 - stats.pearsonr(va[iu], vb[iu])[0]

dists = {"WT vs WT-hypoxia": gdist("wt", "hyp"), "WT vs WT-latent": gdist("wt", "lat"),
         "WT vs KO": gdist("wt", "ko"), "KO vs KO-hypoxia": gdist("ko", "kohyp")}

print("\nStructural distances:", {k: round(v, 4) for k, v in dists.items()})
print(f"Latent response (WT): {resp_lat.mean():.4f}")

# ---------------- figure ----------------
fig = plt.figure(figsize=(14, 4.5))
g = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32, left=0.06, right=0.98, top=0.80, bottom=0.14)

# A: NapM expression (CPM, with per-replicate points)
ax = fig.add_subplot(g[0, 0])
conds = ["WT", "Hypoxia", "Latent"]
vals = [expr[c] for c in conds]
bars = ax.bar(conds, vals, color=[C_WT, C_HYP, C_LAT], edgecolor="black", alpha=0.85)
ax.axhline(expr["WT"], color="black", ls="--", lw=0.8)
rng_plot = np.random.default_rng(0)
for xi, c in enumerate(conds):
    ax.scatter([xi] * 2 + rng_plot.normal(0, 0.04, 2), expr_reps[c],
               color="black", s=18, zorder=3)
    ax.text(xi, vals[xi] * 1.02, f"{vals[xi]:.0f}", ha="center", fontsize=9)
ax.set_ylabel("NapM (Rv0047c) expression\n(CPM, library-size normalized)", fontsize=9)
ax.set_title(f"A  NapM expression (CPM-normalized)\nhypoxia {expr['Hypoxia']/expr['WT']:.2f}x, latent {expr['Latent']/expr['WT']:.2f}x vs WT",
             loc="left", fontsize=10, fontweight="bold")

# B: structural distances
ax = fig.add_subplot(g[0, 1])
pairs = list(dists.keys())
dvals = [dists[p] for p in pairs]
bars = ax.bar(range(len(pairs)), dvals,
              color=[C_HYP, C_LAT, C_KO, "#E69F00"], edgecolor="black", alpha=0.85)
for b, v in zip(bars, dvals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(pairs)), pairs, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Genome-organization distance\n(1 − Pearson r)", fontsize=9)
ax.set_title("B  Structural reorganization between conditions",
             loc="left", fontsize=10, fontweight="bold")

# C: hypoxia response WT vs KO (NapM-dependence)
ax = fig.add_subplot(g[0, 2])
bp = ax.boxplot([resp_wt, resp_ko, resp_lat],
                labels=["hypoxia\nresponse\n(WT)", "hypoxia\nresponse\n(KO)", "latent\nresponse\n(WT)"],
                patch_artist=True, showfliers=False, widths=0.55,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], [C_HYP, C_KO, C_LAT]):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.axhline(0, color="grey", ls=":", lw=0.8)
ax.set_ylabel("Δ long-range fraction", fontsize=9)
ax.set_title(f"C  NapM-dependence of the structural response\nhypoxia WT vs KO: p = {p_hyp:.0e}",
             loc="left", fontsize=10, fontweight="bold")

fig.suptitle("NapM expression and the NapM-dependence of genome organization across conditions",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(OUT, "figures/napm_expression_dependence.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figures/napm_expression_dependence.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

json.dump(dict(
    napM_expression_CPM={k: float(v) for k, v in expr.items()},
    napM_expression_fold_vs_WT={"hypoxia": float(expr["Hypoxia"] / expr["WT"]),
                                 "latent": float(expr["Latent"] / expr["WT"])},
    napM_limma_hypoxia_log2FC=float(hyp_fc), napM_limma_hypoxia_p=float(hyp_p),
    napM_limma_latent_log2FC=float(lat_fc), napM_limma_latent_p=float(lat_p),
    structural_distances=dists,
    hypoxia_response=dict(WT=float(resp_wt.mean()), KO=float(resp_ko.mean()), p=float(p_hyp),
                          response_correlation=float(rho_resp)),
    latent_response_WT=float(resp_lat.mean()),
), open(os.path.join(OUT, "tables/summary.json"), "w"), indent=2)
print("saved to", OUT)
