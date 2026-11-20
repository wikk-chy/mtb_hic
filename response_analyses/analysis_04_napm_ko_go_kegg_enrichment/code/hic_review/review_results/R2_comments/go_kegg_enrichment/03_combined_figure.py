#!/usr/bin/env python3
"""R2C3 combined figure: GO + KEGG enrichment of NapM-KO DEGs.

  A  GO enrichment, upregulated in KO (dot plot, BH FDR)
  B  GO enrichment, downregulated in KO (dot plot)
  C  KEGG enrichment, upregulated (bar plot, -log10 p)
  D  KEGG enrichment, downregulated (bar plot)

Data: tables/GO_enrichment_{Up,Down}.csv, KEGG_enrichment_{Up,Down}.csv
Output: figures/Fig_R2C3_combined.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
OUT = f"{BASE}/review_results/R2_comments/go_kegg_enrichment"
TAB = f"{OUT}/tables"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.6})
C_UP, C_DN = "#8C2F39", "#3D5A80"
NS_LABEL = {"P": "BP", "C": "CC", "F": "MF"}

go_up = pd.read_csv(f"{TAB}/GO_enrichment_Up.csv")
go_dn = pd.read_csv(f"{TAB}/GO_enrichment_Down.csv")
kg_up = pd.read_csv(f"{TAB}/KEGG_enrichment_Up.csv")
kg_dn = pd.read_csv(f"{TAB}/KEGG_enrichment_Down.csv")

fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.9, wspace=0.55,
                       left=0.10, right=0.97, top=0.90, bottom=0.12,
                       width_ratios=[1, 1])


def plabel(ax, letter):
    ax.text(-0.35, 1.05, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def go_panel(ax, df, title, letter, color):
    sig = df[df.q_val < 0.05].copy()
    sig["mlog10q"] = -np.log10(sig.q_val)
    sig = sig.sort_values("mlog10q", ascending=True).tail(8)
    y = np.arange(len(sig))
    sc = ax.scatter(sig.mlog10q, y, s=sig.overlap * 22, c=sig.mlog10q,
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        "c", ["#d9d0dd", color]), vmin=0,
                    vmax=max(4, sig.mlog10q.max()), edgecolor="grey",
                    linewidth=0.3, zorder=3)
    ax.set_yticks(y)
    def _short(nm):
        return nm if len(nm) <= 44 else nm[:41] + "..."
    ax.set_yticklabels(
        [f"[{NS_LABEL.get(ns, ns)}] {_short(n)}" for ns, n in
         zip(sig.namespace, sig.name)], fontsize=6)
    ax.set_xlabel("−log10 (FDR q-value)", fontsize=7)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, lw=0.3)
    plabel(ax, letter)
    return sc


def kegg_panel(ax, df, title, letter, color, n_show=7):
    d = df.sort_values("p_val").head(n_show).iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, -np.log10(d.p_val), color=color, edgecolor="black",
            linewidth=0.4, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(d.name, fontsize=6.5)
    for yi, r in zip(y, d.itertuples()):
        star = "*" if r.q_val < 0.05 else ""
        ax.text(-np.log10(r.p_val) + 0.05, yi,
                f"{r.overlap}/{r.term_size} q={r.q_val:.1g}{star}",
                va="center", fontsize=5.5)
    ax.set_xlabel("−log10 (p-value)", fontsize=7)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    plabel(ax, letter)


sc1 = go_panel(fig.add_subplot(gs[0, 0]), go_up,
               "A  GO: upregulated in napM-KO (n=460)", "A", C_UP)
go_panel(fig.add_subplot(gs[0, 1]), go_dn,
         "B  GO: downregulated in napM-KO (n=362)", "B", C_DN)
kegg_panel(fig.add_subplot(gs[1, 0]), kg_up,
           "C  KEGG: upregulated", "C", C_UP)
kegg_panel(fig.add_subplot(gs[1, 1]), kg_dn,
           "D  KEGG: downregulated", "D", C_DN, n_show=3)

fig.suptitle("Functional enrichment of differentially expressed genes in "
             "napM-KO (one-sided hypergeometric, BH FDR)", fontsize=10,
             fontweight="bold")
fig.savefig(f"{OUT}/figures/Fig_R2C3_combined.png", dpi=300,
            bbox_inches="tight")
fig.savefig(f"{OUT}/figures/Fig_R2C3_combined.pdf", bbox_inches="tight")
print("saved figures/Fig_R2C3_combined.{png,pdf}")
