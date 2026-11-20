#!/usr/bin/env python3
"""R2 Comment 6 - chip_rna_integration, part 3.

Binding x expression integration (KO vs WT):
  (1) overlap of binding tiers with DEG calls, two DEG definitions:
      (a) AKR limma tables (regulate = Up/Down; 822 DEGs in KOvsWT) --
          reproduces the previously reported "33/822 promoter" count;
      (b) R1 reanalysis classes (detected_increase / detected_decrease).
      Fisher exact enrichment per tier; direction split (up / down).
  (2) binding strength vs expression change: Spearman correlation of
      gene_peak_score_max / gene_peak_vpm_max with |log2FC_KOvsWT|
      (directly bound genes only).

Wording constraint honoured downstream: "binding + differential expression"
= candidate direct regulatory relationships only.

Outputs: tables/deg_overlap_enrichment.csv,
         tables/binding_vs_expression_corr.csv,
         figures/fig2_chip_rna_integration.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, spearmanr

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
OUT_T = BASE + "hic_review/review_results/R2_comments/chip_rna_integration/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/chip_rna_integration/figures/"
STATUS = OUT_T + "gene_binding_status.csv"
DEG = BASE + "hic_review/review_data/rnaseq/KOvsWT_supo_DEG_limma.csv"

g = pd.read_csv(STATUS)
deg = pd.read_csv(DEG, index_col=0)
g["akr_regulate"] = g.gene_id.map(deg.regulate)   # Up / Down / Stable / NaN

rows = []
universe = g[g.akr_regulate.notna()].copy()        # genes present in DEG table
nU = len(universe)
n_deg = int((universe.akr_regulate != "Stable").sum())

def overlap_report(mask, tier_name):
    sub = universe[mask]
    n = len(sub)
    d = int((sub.akr_regulate != "Stable").sum())
    up = int((sub.akr_regulate == "Up").sum())
    dn = int((sub.akr_regulate == "Down").sum())
    rest = universe[~mask]
    d_rest = int((rest.akr_regulate != "Stable").sum())
    od, p = fisher_exact([[d, n - d], [d_rest, len(rest) - d_rest]])
    return dict(tier=tier_name, n_genes=n, n_deg=d, frac_deg=round(d / n, 4),
                n_up=up, n_down=dn, fisher_OR=round(od, 3), fisher_p=p)

m_prom = universe.binding_tier == "direct_promoter"
m_body = universe.binding_tier == "direct_body"
m_direct = m_prom | m_body
m_operon = universe.binding_tier == "operon_extended"
m_any = universe.binding_tier != "unbound"

rows.append(overlap_report(m_prom, "direct_promoter"))
rows.append(overlap_report(m_direct, "direct_promoter_or_body"))
rows.append(overlap_report(m_operon, "operon_extended_only"))
rows.append(overlap_report(m_any, "any_binding_associated"))
ov = pd.DataFrame(rows)
ov["n_deg_total_in_universe"] = n_deg
ov["universe_size"] = nU
ov.to_csv(OUT_T + "deg_overlap_enrichment.csv", index=False)

# sanity check vs previously reported 33/822 (promoter-only, AKR DEGs)
print("check 33/822: promoter-bound DEGs =",
      ov.loc[ov.tier == "direct_promoter", "n_deg"].iloc[0],
      "| total AKR DEGs =", n_deg)

# ---- R1-class definition -----------------------------------------------------
cls_rows = []
for tier_name, mask in [("direct_promoter_or_body", g.binding_tier.isin(
        ["direct_promoter", "direct_body"])),
        ("any_binding_associated", g.binding_tier != "unbound")]:
    sub, rest = g[mask], g[~mask]
    for cls, lab in [("detected_increase", "up"), ("detected_decrease", "down")]:
        a = int((sub.class_KOvsWT == cls).sum())
        b_ = int((rest.class_KOvsWT == cls).sum())
        od, p = fisher_exact([[a, len(sub) - a], [b_, len(rest) - b_]])
        cls_rows.append(dict(tier=tier_name, direction=lab, n_in_tier=len(sub),
                             n_deg_class=a, frac=round(a / len(sub), 4),
                             fisher_OR=round(od, 3), fisher_p=p))
pd.DataFrame(cls_rows).to_csv(OUT_T + "deg_overlap_enrichment_R1classes.csv",
                              index=False)

# ---- binding strength vs |log2FC| -------------------------------------------
corr_rows = []
bd = g[g.direct_tier != "unbound"].dropna(
    subset=["gene_peak_score_max", "log2FC_KOvsWT"])
absfc = bd.log2FC_KOvsWT.abs()
for col in ["gene_peak_score_max", "gene_peak_vpm_max"]:
    rho, p = spearmanr(bd[col], absfc)
    corr_rows.append(dict(x=col, y="|log2FC_KOvsWT|", n=len(bd),
                          spearman_rho=round(rho, 4), p=p))
# across ALL genes (score=NaN treated as 0 -> unbound as baseline tier)
allg = g.dropna(subset=["log2FC_KOvsWT"]).copy()
allg["score0"] = allg.gene_peak_score_max.fillna(0)
rho, p = spearmanr(allg.score0, allg.log2FC_KOvsWT.abs())
corr_rows.append(dict(x="gene_peak_score_max(0 for unbound)",
                      y="|log2FC_KOvsWT| (all genes)", n=len(allg),
                      spearman_rho=round(rho, 4), p=p))
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT_T + "binding_vs_expression_corr.csv", index=False)

# ---- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

ax = axes[0]
labels = ["prom", "prom\n+body", "operon\next.", "any"]
fracs = ov.frac_deg.values
base = n_deg / nU
ax.bar(labels, fracs, color=["#6a51a3", "#9e9ac8", "#cbc9e2", "#dadaeb"],
       width=0.6)
ax.axhline(base, ls="--", lw=0.8, color="black")
ax.text(0.98, base + 0.008, f"background {base*100:.1f}%", fontsize=6,
        va="bottom", ha="right", transform=ax.get_yaxis_transform())
for k, r in ov.iterrows():
    ax.text(k, r.frac_deg - 0.012, f"{r.n_deg}/{r.n_genes}\np={r.fisher_p:.2g}",
            ha="center", va="top", fontsize=6,
            color="white" if k < 2 else "#444444")
ax.set_ylabel("fraction DEG (KO vs WT)")
ax.set_ylim(0, max(fracs) * 1.45)
ax.set_title("A  DEG fraction by binding tier", loc="left", fontweight="bold")

ax = axes[1]
cmap = {"direct_promoter": "#6a51a3", "direct_body": "#9e9ac8"}
for t, c in cmap.items():
    s = bd[bd.direct_tier == t]
    ax.scatter(s.gene_peak_score_max, s.log2FC_KOvsWT, s=4, alpha=0.5,
               color=c, label=t, lw=0)
r0 = corr.iloc[0]
ax.set_xlabel("peak score (max per gene)"); ax.set_ylabel("log2FC (KO vs WT)")
ax.axhline(0, lw=0.4, color="grey")
ax.legend(frameon=False, fontsize=6, markerscale=2)
ax.set_title(f"B  score vs log2FC (rho={r0.spearman_rho:.2f}, "
             f"p={r0.p:.2g})", loc="left", fontweight="bold")

ax = axes[2]
for t, c in [("unbound", "#d9d9d9"), ("operon_extended", "#cbc9e2"),
             ("direct_body", "#9e9ac8"), ("direct_promoter", "#6a51a3")]:
    s = g[g.binding_tier == t].dropna(subset=["log2FC_KOvsWT", "p_KOvsWT"])
    ax.scatter(s.log2FC_KOvsWT, -np.log10(s.p_KOvsWT.clip(lower=1e-300)),
               s=3, alpha=0.5, color=c, lw=0, label=t)
ax.set_xlabel("log2FC (KO vs WT)"); ax.set_ylabel("-log10 p")
ax.legend(frameon=False, fontsize=6, markerscale=2)
ax.set_title("C  volcano by binding tier", loc="left", fontweight="bold")

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT_F + "fig2_chip_rna_integration.png", dpi=300)
fig.savefig(OUT_F + "fig2_chip_rna_integration.pdf")
print(ov.to_string(index=False))
print(pd.DataFrame(cls_rows).to_string(index=False))
print(corr.to_string(index=False))
print("OK")
