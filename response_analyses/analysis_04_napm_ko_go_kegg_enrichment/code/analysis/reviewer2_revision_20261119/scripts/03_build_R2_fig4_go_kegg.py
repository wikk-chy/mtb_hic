#!/usr/bin/env python3
"""
Build Response Fig. 4 (Reviewer 2, Comment 3 / Audit Comment 9):
Redrawn GO and KEGG functional enrichment of 822 DEGs in napM-KO vs WT (460 up, 362 down).
Fixes all text collisions from original Response Fig. 3, removes duplicate panel letters,
and visually separates BH-significant KEGG pathways (q < 0.05) from nominal/non-significant pathways.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
TAB = ROOT / "response_figure_reproducibility/figure_03_baseline_functional_enrichment/results/hic_review/review_results/R2_comments/go_kegg_enrichment/tables"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7.6,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.65,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

go_up = pd.read_csv(TAB / "GO_enrichment_Up.csv")
go_dn = pd.read_csv(TAB / "GO_enrichment_Down.csv")
kegg_up = pd.read_csv(TAB / "KEGG_enrichment_Up.csv")
kegg_dn = pd.read_csv(TAB / "KEGG_enrichment_Down.csv")

# Select significant GO terms (q < 0.05), top 8
go_up_sig = go_up[go_up["q_val"] < 0.05].sort_values("q_val").head(8).iloc[::-1].reset_index(drop=True)
go_dn_sig = go_dn[go_dn["q_val"] < 0.05].sort_values("q_val").head(8).iloc[::-1].reset_index(drop=True)

# For KEGG: keep top pathways by p_val, excluding non-bacterial eukaryotic ISR term?
# Keep genuine top 6 microbial KEGG pathways (or show top 6 excluding eukaryotic ISR artifact, wait - let's keep raw top 6 microbial or show all with q >= 0.05 clearly greyed!)
# Actually, let's check all top 6 in KEGG_enrichment_Up: keeping the exact raw table sorted by p_val and greying q >= 0.05 is 100% transparent to the underlying CSV!
kegg_up_top = kegg_up.sort_values("p_val").head(6).iloc[::-1].reset_index(drop=True)
kegg_dn_top = kegg_dn.sort_values("p_val").head(5).iloc[::-1].reset_index(drop=True)


def wrap_label(s, max_len=34):
    if len(s) <= max_len:
        return s
    # wrap at space near midpoint
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2])


def fmt_go(row):
    ns_map = {
        "biological_process": "BP", "molecular_function": "MF", "cellular_component": "CC",
        "P": "BP", "F": "MF", "C": "CC",
    }
    ns = ns_map.get(str(row.get("namespace", "")).strip(), str(row.get("namespace", ""))[:2])
    return wrap_label(f"[{ns}] {row['name']}", 36)


C_UP = "#9E3B42"
C_DN = "#3B5B85"
C_NS = "#C2C7CC"
C_TEXT = "#1F2933"

fig = plt.figure(figsize=(8.27, 6.4))
gs = GridSpec(2, 2, figure=fig, left=0.255, right=0.965, top=0.89, bottom=0.085, wspace=0.72, hspace=0.40, height_ratios=[1.18, 0.92])
fig.suptitle("GO and KEGG enrichment of differentially expressed genes in napM-KO vs WT (n = 822 DEGs)",
             x=0.04, y=0.962, ha="left", fontsize=9.8, fontweight="bold", color=C_TEXT)

# A: GO Up
axA = fig.add_subplot(gs[0, 0])
yA = np.arange(len(go_up_sig))
xA = -np.log10(go_up_sig["q_val"])
sA = 35 + 14 * go_up_sig["overlap"]
axA.grid(axis="x", linestyle=":", color="#E0E4E8", lw=0.6, zorder=0)
axA.scatter(xA, yA, s=sA, color=C_UP, alpha=0.88, edgecolor="black", linewidth=0.5, zorder=3)
for i, r in go_up_sig.iterrows():
    axA.text(xA[i] + 0.16, i, f"{int(r['overlap'])}/{int(r['term_size'])}", va="center", ha="left", fontsize=6.5, color=C_TEXT)
axA.set_yticks(yA, [fmt_go(r) for _, r in go_up_sig.iterrows()])
axA.set_xlabel(r"$-\log_{10}$(BH FDR $q$-value)")
axA.set_xlim(1.9, max(xA) + 0.78)
axA.set_title("GO: upregulated in napM-KO (n = 460)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axA.text(-0.55, 1.09, "A", transform=axA.transAxes, fontsize=11, fontweight="bold", va="top")
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)

# B: GO Down
axB = fig.add_subplot(gs[0, 1])
yB = np.arange(len(go_dn_sig))
xB = -np.log10(go_dn_sig["q_val"])
sB = 35 + 14 * go_dn_sig["overlap"]
axB.grid(axis="x", linestyle=":", color="#E0E4E8", lw=0.6, zorder=0)
axB.scatter(xB, yB, s=sB, color=C_DN, alpha=0.88, edgecolor="black", linewidth=0.5, zorder=3)
for i, r in go_dn_sig.iterrows():
    axB.text(xB[i] + 0.14, i, f"{int(r['overlap'])}/{int(r['term_size'])}", va="center", ha="left", fontsize=6.5, color=C_TEXT)
axB.set_yticks(yB, [fmt_go(r) for _, r in go_dn_sig.iterrows()])
axB.set_xlabel(r"$-\log_{10}$(BH FDR $q$-value)")
axB.set_xlim(1.8, max(xB) + 0.68)
axB.set_title("GO: downregulated in napM-KO (n = 362)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axB.text(-0.55, 1.09, "B", transform=axB.transAxes, fontsize=11, fontweight="bold", va="top")
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

# C: KEGG Up
axC = fig.add_subplot(gs[1, 0])
yC = np.arange(len(kegg_up_top))
xC = -np.log10(kegg_up_top["p_val"])
colsC = [C_UP if q < 0.05 else C_NS for q in kegg_up_top["q_val"]]
axC.grid(axis="x", linestyle=":", color="#E0E4E8", lw=0.6, zorder=0)
axC.barh(yC, xC, height=0.60, color=colsC, edgecolor="black", linewidth=0.5, zorder=3)
for i, r in kegg_up_top.iterrows():
    sig_mark = "*" if r["q_val"] < 0.05 else ""
    qtxt = f"q={r['q_val']:.2g}{sig_mark}"
    axC.text(xC[i] + 0.09, i, f"{int(r['overlap'])}/{int(r['term_size'])} ({qtxt})",
             va="center", ha="left", fontsize=6.4, fontweight="bold" if r["q_val"] < 0.05 else "normal", color=C_TEXT)
axC.set_yticks(yC, [wrap_label(n, 32) for n in kegg_up_top["name"]])
axC.set_xlabel(r"$-\log_{10}$(raw hypergeometric $P$)")
axC.set_xlim(0, max(xC) + 2.55)
axC.set_title("KEGG: upregulated (top 6 by raw P)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axC.legend(handles=[Patch(facecolor=C_UP, edgecolor="black", lw=0.5, label="BH q < 0.05 (*)"),
                    Patch(facecolor=C_NS, edgecolor="black", lw=0.5, label="q ≥ 0.05 (n.s.)")],
           loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=6.2)
axC.text(-0.55, 1.11, "C", transform=axC.transAxes, fontsize=11, fontweight="bold", va="top")
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)

# D: KEGG Down
axD = fig.add_subplot(gs[1, 1])
yD = np.arange(len(kegg_dn_top))
xD = -np.log10(kegg_dn_top["p_val"])
colsD = [C_DN if q < 0.05 else C_NS for q in kegg_dn_top["q_val"]]
axD.grid(axis="x", linestyle=":", color="#E0E4E8", lw=0.6, zorder=0)
axD.barh(yD, xD, height=0.60, color=colsD, edgecolor="black", linewidth=0.5, zorder=3)
for i, r in kegg_dn_top.iterrows():
    sig_mark = "*" if r["q_val"] < 0.05 else ""
    qtxt = f"q={r['q_val']:.2g}{sig_mark}"
    axD.text(xD[i] + 0.12, i, f"{int(r['overlap'])}/{int(r['term_size'])} ({qtxt})",
             va="center", ha="left", fontsize=6.4, fontweight="bold" if r["q_val"] < 0.05 else "normal", color=C_TEXT)
axD.set_yticks(yD, [wrap_label(n, 32) for n in kegg_dn_top["name"]])
axD.set_xlabel(r"$-\log_{10}$(raw hypergeometric $P$)")
axD.set_xlim(0, max(xD) + 2.8)
axD.set_title("KEGG: downregulated (top 5 by raw P)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axD.legend(handles=[Patch(facecolor=C_DN, edgecolor="black", lw=0.5, label="BH q < 0.05 (*)"),
                    Patch(facecolor=C_NS, edgecolor="black", lw=0.5, label="q ≥ 0.05 (n.s.)")],
           loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=6.3)
axD.text(-0.55, 1.11, "D", transform=axD.transAxes, fontsize=11, fontweight="bold", va="top")
axD.spines["top"].set_visible(False)
axD.spines["right"].set_visible(False)

pdf_path = OUT_FIG / "Response_Fig_4.pdf"
png_path = OUT_FIG / "Response_Fig_4.png"
jpg_path = OUT_FIG / "Response_Fig_4.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved Response_Fig_4.{pdf,png,jpg}")
