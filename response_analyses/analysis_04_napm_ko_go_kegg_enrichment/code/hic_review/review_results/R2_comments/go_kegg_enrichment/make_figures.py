#!/usr/bin/env python3
"""Publication-quality plots for GO + KEGG enrichment results."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, "tables")
FIGS = os.path.join(BASE, "figures")
os.makedirs(FIGS, exist_ok=True)

UP_C = "#8C2F39"   # deep red
DOWN_C = "#3D5A80" # slate blue
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def wrap(s, width=42):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def savefig(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------------- Figure 1: KEGG barplot ----------------
SKIP_KEGG = {"mtu01100", "mtu01110", "mtu01120"}  # global overview maps

def kegg_panel(ax, csv, color, title, topn=10):
    d = pd.read_csv(os.path.join(TABLES, csv))
    d = d[~d["term"].isin(SKIP_KEGG)]
    d = d[d["p_val"] < 0.05].head(topn).copy()
    d["mlog10p"] = -np.log10(d["p_val"])
    d = d.iloc[::-1]
    labels = [wrap(n) for n in d["name"]]
    bars = ax.barh(labels, d["mlog10p"], color=color, alpha=0.9, height=0.65)
    for b, (_, r) in zip(bars, d.iterrows()):
        sig = "*" if r["q_val"] < 0.05 else ""
        ax.text(b.get_width() + 0.06, b.get_y() + b.get_height() / 2,
                f'{int(r["overlap"])}/{int(r["term_size"])}  q={r["q_val"]:.2g}{sig}',
                va="center", fontsize=7.5, color="#333333")
    ax.set_xlabel(r"$-\log_{10}$(p-value)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, d["mlog10p"].max() * 1.45)
    ax.tick_params(axis="y", labelsize=8)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
kegg_panel(axes[0], "KEGG_enrichment_Up.csv", UP_C, "Up-regulated in NapM-KO (n=460)")
kegg_panel(axes[1], "KEGG_enrichment_Down.csv", DOWN_C, "Down-regulated in NapM-KO (n=362)")
fig.suptitle("KEGG pathway enrichment (M. tuberculosis H37Rv)", fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "Fig1_KEGG_enrichment_barplot")

# ---------------- Figure 2: GO dotplot (Up vs Down, by namespace) ----------------
NS_ORDER = ["P", "C", "F"]
NS_LABEL = {"P": "Biological Process", "C": "Cellular Component", "F": "Molecular Function"}
cmap_up = LinearSegmentedColormap.from_list("up", ["#E8C9CC", UP_C])
cmap_down = LinearSegmentedColormap.from_list("down", ["#C7D2E0", DOWN_C])

def go_top(csv, per_ns=4):
    d = pd.read_csv(os.path.join(TABLES, csv))
    d = d[d["q_val"] < 0.05]
    out = []
    for ns in NS_ORDER:
        sub = d[d["namespace"] == ns].head(per_ns)
        out.append(sub)
    return pd.concat(out) if out else pd.DataFrame()

up_go = go_top("GO_enrichment_Up.csv")
down_go = go_top("GO_enrichment_Down.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=False)
for ax, d, cmap, color, title in [
        (axes[0], up_go, cmap_up, UP_C, "Up-regulated in NapM-KO"),
        (axes[1], down_go, cmap_down, DOWN_C, "Down-regulated in NapM-KO")]:
    d = d.copy()
    d["mlog10q"] = -np.log10(d["q_val"])
    # order: P top, then C, then F; within ns by q
    d["ns_rank"] = d["namespace"].map({n: i for i, n in enumerate(NS_ORDER)})
    d = d.sort_values(["ns_rank", "q_val"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    sc = ax.scatter(d["mlog10q"], y, s=d["overlap"] * 22,
                    c=d["mlog10q"], cmap=cmap, edgecolors="#333333",
                    linewidths=0.6, zorder=3,
                    vmin=0, vmax=max(d["mlog10q"].max() * 1.15, 2))
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(f"[{ns}] {n}", 40) for ns, n in
                        zip(d["namespace"], d["name"])], fontsize=8)
    ax.set_xlabel(r"$-\log_{10}$(FDR q-value)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", lw=0.5, color="#BBBBBB", zorder=0)
    ax.set_xlim(left=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$-\log_{10}$(q)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # size legend
    sizes = [5, 10, 20]
    handles = [plt.scatter([], [], s=s * 22, facecolors="white",
                           edgecolors="#333333", linewidths=0.6) for s in sizes]
    ax.legend(handles, [str(s) for s in sizes], title="Gene count",
              loc="lower right", fontsize=7.5, title_fontsize=8, frameon=False)
fig.suptitle("GO enrichment (FDR q < 0.05, top terms per category; "
             "BP=Biological Process, CC=Cellular Component, MF=Molecular Function)",
             fontsize=10, fontweight="bold", y=1.03)
fig.tight_layout()
savefig(fig, "Fig2_GO_enrichment_dotplot")

# ---------------- Figure 3: combined KEGG + GO key terms diverging bar (gene ratio) ----------------
def top_terms(csv, p_col, p_cut, n=8):
    d = pd.read_csv(os.path.join(TABLES, csv))
    d = d[~d["term"].isin(SKIP_KEGG)] if "KEGG" in csv else d
    d = d[d[p_col] < p_cut].head(n)
    return d

fig, ax = plt.subplots(figsize=(8, 5))
ku = top_terms("KEGG_enrichment_Up.csv", "p_val", 0.05, 6)
kd = top_terms("KEGG_enrichment_Down.csv", "p_val", 0.05, 6)
kd["neg"] = -(-np.log10(kd["p_val"]))
ypos = np.arange(len(ku) + len(kd))
labels = [wrap(x, 40) for x in list(ku["name"]) + list(kd["name"])]
vals = list(-np.log10(ku["p_val"])) + list(kd["neg"])
colors = [UP_C] * len(ku) + [DOWN_C] * len(kd)
ax.barh(ypos[::-1], vals, color=colors, alpha=0.9, height=0.65)
ax.set_yticks(ypos[::-1])
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="#333333", lw=0.8)
ax.axvline(-np.log10(0.05), color="#999999", ls="--", lw=0.7)
ax.axvline(np.log10(0.05), color="#999999", ls="--", lw=0.7)
ax.set_xlabel(r"$-\log_{10}$(p-value)   (Up: right, Down: left)")
ax.set_title("KEGG enrichment summary — NapM-KO vs WT", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
leg = [Line2D([0], [0], color=UP_C, lw=6, label="Up in KO"),
       Line2D([0], [0], color=DOWN_C, lw=6, label="Down in KO")]
ax.legend(handles=leg, frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
savefig(fig, "Fig3_KEGG_diverging_summary")

print("All figures done.")
