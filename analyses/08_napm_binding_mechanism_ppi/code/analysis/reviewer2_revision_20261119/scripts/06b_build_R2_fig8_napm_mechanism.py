#!/usr/bin/env python3
"""
Build finalized 4-row 8-panel Response Fig. 8 (Reviewer 2, Comment 6),
strictly ordered along Reviewer #2's questions:
  Row 1 (A, B, C): Q1 discrete vs continuous (Option 2: 3 core tiers + peak-score scatter) & Q2 high abundance of small napM.
  Row 2 (D):       Q3 genome-wide strongly enriched NapM ChIP-seq occupancy (634 peaks across 454/883 5-kb bins, median 42.7x).
  Row 3 (E):       Q3 three representative high-resolution ChIP-seq binding tracks (E1-E3) with gene annotations.
  Row 4 (F, G, H): Q4 global contact shift (F) + candidate loop anchor loci (G) & Q5 NapM PPI ego network (14 B2H partners + DNA-binding layer, H).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
HIC = ROOT / "hic_review"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CR_TAB = HIC / "review_results/R2_comments/chip_rna_integration/tables"
PPI_TAB = HIC / "review_results/R2_comments/ppi_evidence/tables"

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 7.3,
    "axes.titlesize": 8.1,
    "axes.labelsize": 7.3,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.68,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

C_PK = "#6A51A3"
C_PK_FILL = "#B4A4D8"
C_WT = "#1F77B4"
C_KO = "#D62728"
C_TEAL = "#1B9E77"
C_ORANGE = "#D95F02"
C_TEXT = "#1F2933"

# ---------------- Load & verify all source data ----------------
deg_tab_all = pd.read_csv(CR_TAB / "deg_overlap_enrichment.csv")
deg_tab = deg_tab_all[deg_tab_all["tier"].isin(
    ["direct_promoter", "direct_promoter_or_body", "any_binding_associated"]
)].reset_index(drop=True)

gstat = pd.read_csv(CR_TAB / "gene_binding_status.csv")
corr = pd.read_csv(CR_TAB / "binding_vs_expression_corr.csv")

genes_dict = {}
genes_list = []
with open(HIC / "annotation.gtf") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 9 or p[2] != "gene":
            continue
        s, e, st, attr = int(p[3]), int(p[4]), p[6], p[8]
        gid = gname = None
        for kv in attr.split(";"):
            kv = kv.strip()
            if kv.startswith("gene_id"):
                gid = kv.split('"')[1]
            elif kv.startswith("gene_name"):
                gname = kv.split('"')[1]
        if gid:
            gn = gname or gid
            genes_dict[gid] = (s, e, e - s + 1, st, gn)
            genes_list.append((s, e, st, gid, gn))

rna = pd.read_csv(HIC / "review_data/rnaseq/AKR_RNA_seq_merged.csv").set_index("gene")
common = [g for g in rna.index if g in genes_dict]
lengths_kb = pd.Series({g: genes_dict[g][2] / 1000.0 for g in common})
rpk = rna.loc[common].div(lengths_kb, axis=0)
tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
wt_tpm = tpm[["WT-rep1", "WT-rep2"]].mean(axis=1)
hyp_tpm = tpm[["Hypoxia-rep1", "Hypoxia-rep2"]].mean(axis=1)

napm_wt_tpm = float(wt_tpm["Rv0047c"])
napm_hyp_tpm = float(hyp_tpm["Rv0047c"])
wt_med_tpm = float(wt_tpm.median())
napm_wt_top_pct = 100.0 * float((wt_tpm > napm_wt_tpm).mean())
napm_hyp_top_pct = 100.0 * float((hyp_tpm > napm_hyp_tpm).mean())

bg = pd.read_csv(HIC / "review_data/chip_seq_data/Rv0047c.bedgraph", sep="\t", header=None, names=["chrom", "start", "end", "val"])
bg_starts = bg["start"].to_numpy()
bg_ends = bg["end"].to_numpy()
bg_vals = bg["val"].to_numpy()
med_bg = float(np.median(bg_vals))

pk = pd.read_csv(HIC / "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv", sep="\t")
summits = []
for _, r in pk.iterrows():
    i0 = np.searchsorted(bg_starts, r["start"], side="left")
    i1 = np.searchsorted(bg_starts, r["end"], side="right")
    seg = bg_vals[max(0, i0 - 1):min(len(bg_vals), i1 + 1)]
    summits.append(float(seg.max()) if len(seg) else np.nan)
pk["fold_vs_med"] = np.array(summits) / med_bg
med_peak_fold = float(pk["fold_vs_med"].median())

cum_df = pd.read_csv(HIC / "review_results/R4_comment4/tables/g_cumulative_curves.csv")

edges = pd.read_csv(HIC / "review_data/Interaction_network/full_edges.csv")
ppi = edges[(edges["interaction_type"] == "Protein-Protein") & ((edges["source"] == "Rv0047c") | (edges["target"] == "Rv0047c"))]
nbrs = sorted((set(ppi["source"]) | set(ppi["target"])) - {"Rv0047c"})
assert "Rv0047c" not in nbrs and len(nbrs) == 14
nb_tab = pd.read_csv(PPI_TAB / "napm_ppi_neighbours.csv")

# ---------------- Build crisp 4-row figure (no bottom footnote) ----------------
fig = plt.figure(figsize=(10.0, 10.5))
outer = GridSpec(
    4, 1, figure=fig,
    left=0.068, right=0.978, top=0.960, bottom=0.048,
    hspace=0.40,
    height_ratios=[0.96, 0.86, 1.16, 1.24],
)

gs_r1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.30, width_ratios=[0.95, 1.0, 1.18])
gs_r2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
gs_r3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.24)
gs_r4 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[3], wspace=0.30, width_ratios=[0.95, 0.70, 2.05])


def style_ax(ax, title):
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color=C_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


# ==================== Row 1: Q1 (A, B) & Q2 (C) ====================
axA = fig.add_subplot(gs_r1[0, 0])
tier_lbl = {
    "direct_promoter": "Promoter\n(−300 bp)",
    "direct_promoter_or_body": "Prom +\nbody",
    "any_binding_associated": "Any\nbound",
}
xA = np.arange(len(deg_tab))
colsA = ["#54278F", "#756BB1", "#B6B0D5"]
axA.axhline(822 / 4060, color="#444444", ls="--", lw=0.9)
axA.bar(xA, deg_tab["frac_deg"], width=0.52, color=colsA, edgecolor="black", lw=0.5)
for i, r in deg_tab.iterrows():
    p_str = f"P={r['fisher_p']:.3f}" if r["fisher_p"] < 0.1 else f"P={r['fisher_p']:.2f}"
    axA.text(
        i, max(r["frac_deg"], 822 / 4060) + 0.008,
        f"{int(r['n_deg'])}/{int(r['n_genes'])}\nOR={r['fisher_OR']:.2f}, {p_str}",
        ha="center", va="bottom", fontsize=5.75, linespacing=1.18, color=C_TEXT,
    )
axA.set_xticks(xA, [tier_lbl.get(t, t) for t in deg_tab["tier"]])
axA.set_xlim(-0.55, 2.55)
axA.set_ylim(0, 0.325)
axA.set_ylabel("DEG fraction (KO vs WT)")
axA.text(
    0.97, 0.96, "Genome bg: 20.2%\nAll OR < 1 (no enrichment)",
    transform=axA.transAxes, ha="right", va="top", fontsize=6.0, color="#444444", linespacing=1.28,
)
style_ax(axA, "No DEG enrichment by binding tier")

axB = fig.add_subplot(gs_r1[0, 1])
bound = gstat[gstat["binding_tier"].isin(["direct_body", "direct_promoter"])].dropna(subset=["gene_peak_score_max", "log2FC_KOvsWT"])
axB.scatter(bound["gene_peak_score_max"], np.abs(bound["log2FC_KOvsWT"]), s=6.0, color=C_PK, alpha=0.35, linewidths=0)
rho_row = corr[corr["x"] == "gene_peak_score_max"].iloc[0]
axB.set_xlabel("Max peak score per bound gene")
axB.set_ylabel(r"$|\log_2\mathrm{FC}|$ (KO vs WT)")
axB.set_ylim(-0.3, 6.2)
axB.text(
    0.04, 0.96,
    f"n = {int(rho_row['n'])} bound genes\nSpearman ρ = {rho_row['spearman_rho']:.3f}\nP = {rho_row['p']:.3f} (no gradient)",
    transform=axB.transAxes, va="top", ha="left", fontsize=6.0, linespacing=1.32,
)
style_ax(axB, "Peak strength vs expression shift")

axC = fig.add_subplot(gs_r1[0, 2])
log_tpm = np.log10(wt_tpm.to_numpy() + 1.0)
axC.hist(log_tpm, bins=np.linspace(0, 4.4, 38), color="#CBD5E1", edgecolor="white", lw=0.35)
x_med = np.log10(wt_med_tpm + 1.0)
x_wt = np.log10(napm_wt_tpm + 1.0)
x_hyp = np.log10(napm_hyp_tpm + 1.0)
axC.axvline(x_med, color="#64748B", ls="--", lw=0.95)
axC.axvline(x_wt, color=C_PK, ls="-", lw=1.75)
axC.axvline(x_hyp, color=C_ORANGE, ls=":", lw=1.35)
axC.text(x_med - 0.14, 345, f"Median\n{wt_med_tpm:.0f}", ha="right", va="top", fontsize=5.9, color="#475569")
axC.annotate(
    f"napM (WT)\n{napm_wt_tpm:.0f} TPM\nTop {napm_wt_top_pct:.1f}%",
    xy=(x_wt, 375), xytext=(x_wt + 0.30, 405),
    ha="left", va="center", fontsize=6.0, fontweight="bold", color=C_PK,
    arrowprops=dict(arrowstyle="-|>", color=C_PK, lw=0.95),
)
axC.annotate(
    f"Hyp: {napm_hyp_tpm:.0f}\n(Top {napm_hyp_top_pct:.1f}%)",
    xy=(x_hyp, 225), xytext=(x_hyp + 0.25, 225),
    ha="left", va="center", fontsize=5.8, color=C_ORANGE,
    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=0.9),
)
axC.set_xlim(0.0, 4.55)
axC.set_ylim(0, 520)
axC.set_xticks([0, 1, 2, 3, 4], ["0", "10", "100", "1k", "10k"])
axC.set_xlabel("WT transcript abundance (TPM)")
axC.set_ylabel("Mtb genes (n = 4,109)")
axC.text(
    0.03, 0.96, "Small NAP (180 aa, ~19.5 kDa)\n6.1× genome median TPM",
    transform=axC.transAxes, va="top", ha="left", fontsize=5.85, color=C_TEXT,
    bbox=dict(boxstyle="round,pad=0.20", facecolor="#F7F5FA", edgecolor="#DCD6EB", lw=0.5),
)
style_ax(axC, r"High abundance of small $\mathit{napM}$")

# ==================== Row 2: Genome-wide strongly enriched NapM binding (D) ====================
axD = fig.add_subplot(gs_r2[0, 0])
mids_mb = ((pk["start"] + pk["end"]) / 2.0) / 1e6
folds = pk["fold_vs_med"].to_numpy()
axD.vlines(mids_mb, 1.0, folds, color=C_PK, linewidth=0.65, alpha=0.82)
axD.axhline(1.0, color="#94A3B8", ls="-", lw=0.6)
axD.axhline(med_peak_fold, color=C_ORANGE, ls="--", lw=1.05, zorder=4)
axD.text(
    4.38, med_peak_fold * 1.18, f"Median peak enrichment: {med_peak_fold:.1f}×",
    ha="right", va="bottom", fontsize=6.2, fontweight="bold", color=C_ORANGE,
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.92, edgecolor=C_ORANGE, lw=0.55),
    zorder=6,
)

for xm, lbl, col, xoff in [(0.0075, "E1", C_TEAL, 0.09), (3.5625, "E2", C_TEAL, 0.0), (3.8455, "E3", C_ORANGE, 0.0)]:
    axD.scatter([xm], [2500], marker="v", s=24, color=col, zorder=5)
    axD.text(xm + xoff, 3300, lbl, ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=col)

axD.set_yscale("log")
axD.set_xlim(-0.06, 4.46)
axD.set_ylim(1.0, 42000)
axD.set_xlabel("Genome position (Mb)")
axD.set_ylabel("ChIP-seq enrichment\n(fold / median)")
axD.text(
    0.015, 0.96,
    "634 curated peaks across 454/883 (51.4%) 5-kb bins | random spacing P = 0.88 | Peak GC 0.623 vs genome 0.656 | 2.55× promoter enrichment",
    transform=axD.transAxes, va="top", ha="left", fontsize=6.2, color=C_TEXT,
)
style_ax(axD, "Dense, strongly enriched NapM binding across the entire chromosome")

# ==================== Row 3: Representative high-resolution ChIP-seq binding tracks (E) ====================
tracks_spec = [
    {
        "title": r"E1: $\mathit{oriC}$ / $\mathit{dnaA}$–$\mathit{gyrBA}$ (0–15 kb)",
        "sub": "10 peaks (up to 261×) | L00/L01 anchor",
        "a": 0, "b": 15000, "ymax": 365,
        "anchor": (0, 5000, "L00/L01 anchor\n(lost in KO)", 2.5, 345, "center"),
        "show_genes": {"dnaA", "dnaN", "recF", "gyrB", "gyrA", "ppiA", "Rv0012"},
    },
    {
        "title": "E2: KO-lost loop L32 anchor (3,555–3,570 kb)",
        "sub": "6 peaks (up to 797×) | L32 O/E: 1.59 → 1.03",
        "a": 3555000, "b": 3570000, "ymax": 980,
        "anchor": (3560000, 3565000, "L32 anchor 1\n(lost in KO)", 3563.0, 745, "left"),
        "show_genes": {"Rv3190c", "Rv3192", "Rv3193c", "Rv3195", "Rv3197", "whiB7"},
    },
    {
        "title": r"E3: $\mathit{PPE57}$–$\mathit{PPE59}$ cluster (3,838–3,853 kb)",
        "sub": "7 peaks (up to 615×) | clustered sites",
        "a": 3838000, "b": 3853000, "ymax": 760,
        "anchor": None,
        "show_genes": {"alr", "PPE57", "PPE58", "PPE59", "Rv3430c", "gadB"},
    },
]

axE_list = []
for idx_t, spec in enumerate(tracks_spec):
    sub_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_r3[0, idx_t], height_ratios=[3.6, 1.0], hspace=0.06)
    ax_sig = fig.add_subplot(sub_gs[0])
    ax_gen = fig.add_subplot(sub_gs[1], sharex=ax_sig)
    axE_list.append(ax_sig)

    a, b = spec["a"], spec["b"]
    m_bg = (bg_ends > a) & (bg_starts < b)
    xs = ((bg_starts[m_bg] + bg_ends[m_bg]) / 2.0) / 1000.0
    ys = bg_vals[m_bg] / med_bg

    if spec["anchor"] is not None:
        anc_a, anc_b, anc_lbl, anc_tx, anc_ty, anc_ha = spec["anchor"]
        ax_sig.axvspan(anc_a / 1000.0, anc_b / 1000.0, color="#D6F5EB", alpha=0.65, zorder=0)
        ax_sig.text(anc_tx, anc_ty, anc_lbl, ha=anc_ha, va="top",
                    fontsize=5.7, fontweight="bold", color="#0F766E", linespacing=1.15)

    sp = pk[(pk["end"] > a) & (pk["start"] < b)]
    for _, r in sp.iterrows():
        ax_sig.axvspan(r["start"] / 1000.0, r["end"] / 1000.0, color=C_PK, alpha=0.14, zorder=1)
        ax_sig.plot([r["start"] / 1000.0, r["end"] / 1000.0], [spec["ymax"] * 0.025, spec["ymax"] * 0.025],
                    color=C_PK, lw=2.0, solid_capstyle="butt", zorder=4)

    ax_sig.fill_between(xs, 0, ys, color=C_PK_FILL, alpha=0.65, zorder=2)
    ax_sig.plot(xs, ys, color=C_PK, lw=0.95, zorder=3)
    ax_sig.axhline(1.0, color="#94A3B8", ls="--", lw=0.55, zorder=1)
    ax_sig.set_xlim(a / 1000.0, b / 1000.0)
    ax_sig.set_ylim(0, spec["ymax"])
    ax_sig.tick_params(labelbottom=False, direction="out", length=2.5)
    ax_sig.spines["top"].set_visible(False)
    ax_sig.spines["right"].set_visible(False)
    ax_sig.set_title(spec["title"], loc="left", pad=5, fontsize=7.6, fontweight="bold", color=C_TEXT)
    ax_sig.text(0.98, 0.96, spec["sub"], transform=ax_sig.transAxes, ha="right", va="top",
                fontsize=5.75, color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.88, edgecolor="#E2E8F0", lw=0.45))
    if idx_t == 0:
        ax_sig.set_ylabel("NapM ChIP-seq\n(fold / median)")

    ax_gen.axhline(0.5, color="#CBD5E1", lw=0.55, zorder=0)
    row_g = [(s, e, st, gid, gn) for (s, e, st, gid, gn) in genes_list if e > a and s < b]
    labeled_x = []
    for s, e, st, gid, gn in row_g:
        gs_kb, ge_kb = max(s, a) / 1000.0, min(e, b) / 1000.0
        y_bar = 0.66 if st == "+" else 0.34
        col_g = "#3B82F6" if st == "+" else "#64748B"
        rect = FancyBboxPatch(
            (gs_kb, y_bar - 0.11), max(ge_kb - gs_kb, 0.08), 0.22,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            facecolor=col_g, edgecolor="none", alpha=0.82, zorder=2,
        )
        ax_gen.add_patch(rect)
        if gn in spec["show_genes"]:
            xm = (gs_kb + ge_kb) / 2.0
            if all(abs(xm - lx) > 1.35 for lx in labeled_x):
                labeled_x.append(xm)
                ax_gen.text(xm, 0.84 if st == "+" else 0.14, gn,
                            ha="center", va="bottom" if st == "+" else "top",
                            fontsize=5.55, style="italic", color=C_TEXT)
    ax_gen.set_ylim(-0.35, 1.35)
    ax_gen.set_yticks([])
    for spn in ("top", "right", "left"):
        ax_gen.spines[spn].set_visible(False)
    ax_gen.tick_params(direction="out", length=2.5)
    ax_gen.set_xlabel("Genomic coordinate (kb)")

# ==================== Row 4: F, G, H on ONE row ====================
axF = fig.add_subplot(gs_r4[0, 0])
for cond_name, col in [("WT", C_WT), ("napM-KO", C_KO)]:
    sub = cum_df[cum_df["condition"] == cond_name]
    piv = sub.pivot(index="distance_bp", columns="replicate", values="cum_fraction")
    mu, sd = piv.mean(axis=1), piv.std(axis=1)
    axF.plot(mu.index / 1e3, mu.values, color=col, lw=1.4, label=cond_name)
    axF.fill_between(mu.index / 1e3, mu.values - sd.values, mu.values + sd.values, color=col, alpha=0.18)
axF.axhline(0.5, color="#777777", ls=":", lw=0.75)
axF.axvline(276, color=C_WT, ls="--", lw=0.8)
axF.axvline(515, color=C_KO, ls="--", lw=0.8)
axF.set_xscale("log")
axF.set_xlabel("Genomic distance s (kb)")
axF.set_ylabel("Cumulative contact fraction")
axF.set_ylim(0, 1.05)
axF.legend(loc="upper left", frameon=False, fontsize=6.3)
axF.text(
    0.97, 0.05,
    r"Median: 276 $\rightarrow$ 515 kb ($P$ = 1.5$\times 10^{-9}$)" "\n"
    "<100 kb: 0.377 → 0.242\n"
    "Peak-centred |Δins|: P = 0.97",
    transform=axF.transAxes, ha="right", va="bottom", fontsize=5.8, linespacing=1.28,
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D8DEE4", lw=0.55),
)
style_ax(axF, r"Global shift in $\mathit{napM}$-KO")

axG = fig.add_subplot(gs_r4[0, 1])
axG.bar([0, 1], [64.7, 41.7], width=0.52, color=[C_PK, "#8C9BA5"], edgecolor="black", lw=0.55)
axG.text(0, 64.7 + 2.5, "64.7%\n(11/17)", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=C_PK)
axG.text(1, 41.7 + 2.5, "41.7%\n(10/24)", ha="center", va="bottom", fontsize=6.4, color=C_TEXT)
axG.set_xticks([0, 1], ["Both anchors\nNapM-bound", "Other WT\nloops"])
axG.set_xlim(-0.55, 1.55)
axG.set_ylim(0, 112)
axG.set_ylabel("WT loops lost in KO (%)")
axG.text(
    0.04, 0.96,
    "WT double-anchor:\n17/41 vs null 10.9\n(1.56×, P = 0.033)",
    transform=axG.transAxes, va="top", ha="left", fontsize=5.8, linespacing=1.28, color=C_TEXT,
)
style_ax(axG, "Candidate loop loci")

axH = fig.add_subplot(gs_r4[0, 2])
axH.axis("off")

CAT_COLORS = {
    "conserved hypothetical": "#B8C2CC",
    "unknown": "#DCE1E7",
    "TA system": "#F6AD55",
    "lipid metabolism": "#63B3ED",
    "insertion seqs/phages": "#68D391",
    "intermediary metabolism": "#FBD38D",
    "cell wall/cell processes": "#B794F4",
    "DNA repair": "#FC8181",
}
DEG_EDGE = {"Up": "#C53030", "Down": "#2B6CB0"}

nb_idx = nb_tab.set_index("neighbour")
cx, cy = -2.25, 0.05
rx, ry = 1.38, 1.34
angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(nbrs), endpoint=False)

for a, nb in zip(angles, nbrs):
    row = nb_idx.loc[nb]
    x1 = cx + rx * np.cos(a)
    y1 = cy + ry * np.sin(a)
    axH.plot([cx, x1], [cy, y1], color="#7E6B9C", lw=1.1, zorder=1)
    fc = CAT_COLORS.get(row["functional_category"], "#CBD5E0")
    reg = str(row["akr_regulate_KOvsWT"])
    ec = DEG_EDGE.get(reg, "#4A5568")
    lw = 1.75 if reg in DEG_EDGE else 0.7
    sz = 82 + 4.2 * min(float(row["degree_full"]), 60)
    axH.scatter([x1], [y1], s=sz, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    lab = str(row["gene_name"]).split(" /")[0]
    lx = cx + (rx + 0.33) * np.cos(a)
    ly = cy + (ry + 0.29) * np.sin(a)
    ha = "left" if np.cos(a) > 0.25 else ("right" if np.cos(a) < -0.25 else "center")
    va = "bottom" if np.sin(a) > 0.35 else ("top" if np.sin(a) < -0.35 else "center")
    axH.text(lx, ly, lab, ha=ha, va=va, fontsize=6.05, fontweight="bold" if reg in DEG_EDGE else "normal", color=C_TEXT)

axH.scatter([cx], [cy], s=740, facecolor=C_PK, edgecolor="black", linewidth=1.1, zorder=4)
axH.text(cx, cy, "NapM\n(Rv0047c)", color="white", fontsize=6.35, fontweight="bold", ha="center", va="center", zorder=5)

dx, dy = 1.40, 1.30
axH.plot([cx + 0.45, dx - 0.32], [cy + 0.20, dy], color=C_TEAL, lw=1.25, ls=":", zorder=1)
axH.scatter([dx], [dy], s=500, facecolor=C_TEAL, edgecolor="black", linewidth=0.85, zorder=3)
axH.text(dx, dy, "454\nbins", color="white", fontsize=6.0, fontweight="bold", ha="center", va="center", zorder=4)
axH.text(dx + 0.40, dy, "ChIP-seq layer\n(634 peaks;\n100% co-bound)",
         va="center", ha="left", fontsize=5.95, color=C_TEXT, linespacing=1.22)

cat_handles = [Patch(facecolor=CAT_COLORS[k], edgecolor="#555555", lw=0.5, label=k)
               for k in CAT_COLORS if k in set(nb_tab["functional_category"])]
deg_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#C53030", markeredgewidth=1.6, markersize=5.6, label="DEG up in KO"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2B6CB0", markeredgewidth=1.6, markersize=5.6, label="DEG down in KO"),
    Line2D([0], [0], color="#7E6B9C", lw=1.15, label="B2H PPI (Wang et al.)"),
]
leg1 = axH.legend(handles=cat_handles, loc="center left", bbox_to_anchor=(0.61, 0.44),
                  ncol=1, frameon=False, fontsize=5.7, title="PPI category (14; no self-loop)",
                  title_fontsize=5.95, handletextpad=0.35, labelspacing=0.26)
axH.add_artist(leg1)
axH.legend(handles=deg_handles, loc="lower left", bbox_to_anchor=(0.61, -0.02),
           ncol=1, frameon=False, fontsize=5.7, handletextpad=0.35, labelspacing=0.26)

axH.set_xlim(-4.38, 4.35)
axH.set_ylim(-1.95, 2.02)
axH.set_title("NapM PPI network (14 B2H partners) & DNA-binding layer",
              loc="left", pad=6, fontweight="bold", color=C_TEXT)

# Perfectly aligned panel letters A-H
pA, pB, pC = axA.get_position(), axB.get_position(), axC.get_position()
pD = axD.get_position()
pE = axE_list[0].get_position()
pF, pG, pH = axF.get_position(), axG.get_position(), axH.get_position()

fig.text(0.016, pA.y1 + 0.022, "A", fontsize=11.5, fontweight="bold", va="top")
fig.text(pB.x0 - 0.048, pB.y1 + 0.022, "B", fontsize=11.5, fontweight="bold", va="top")
fig.text(pC.x0 - 0.050, pC.y1 + 0.022, "C", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.016, pD.y1 + 0.022, "D", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.016, pE.y1 + 0.022, "E", fontsize=11.5, fontweight="bold", va="top")
fig.text(0.016, pF.y1 + 0.022, "F", fontsize=11.5, fontweight="bold", va="top")
fig.text(pG.x0 - 0.046, pG.y1 + 0.022, "G", fontsize=11.5, fontweight="bold", va="top")
fig.text(pH.x0 - 0.024, pH.y1 + 0.022, "H", fontsize=11.5, fontweight="bold", va="top")

pdf_path = OUT_FIG / "Response_Fig_8_mechanism.pdf"
png_path = OUT_FIG / "Response_Fig_8_mechanism.png"
jpg_path = OUT_FIG / "Response_Fig_8_mechanism.jpg"
fig.savefig(pdf_path, dpi=300)
fig.savefig(png_path, dpi=300)
plt.close(fig)
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(300, 300))
print("Saved final:", pdf_path, png_path, jpg_path)
