#!/usr/bin/env python3
"""
Build Response Fig. 17 (Reviewer 4, Comment 6 — Re-evaluation of downstream biological claims
after corrected compartment, CID and loop calling):
  Row 1 (A-B): Re-evaluated chromosome-scale segregation (Extended Data Fig. 6c-f)
  Row 2 (C-E): Transcriptional coordination of validated loops (Fig. 1e) & CIDs (Fig. 1f) +
               top double-anchor loop proteins across 41 validated loops
  Row 3 (F-H): NapM double-anchor enrichment controls, loop-anchor bipartite network rank
               (no Rv0081 label), and multilayer connectivity across all 229 DNA-binding proteins
100% vector-editable TrueType Arial (pdf.fonttype=42), full-width print scale (10.6 x 9.2 in), ZERO overlaps.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
R4C1 = BASE / "review_results/R4_comment1_AB_compartment"
R4C6 = BASE / "review_results/R4_comment6"
R1P = BASE / "review_results/R1_mechanism_reanalysis/06_permutation_tests"
FIG1F_CSV = ROOT / "docx_latex_conversion/generated/Fig1f_CID_coexpression_boxplot_data.csv"
OUT_FIG = ROOT / "analysis/reviewer4_fig_polish_20261124/figures"
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.9,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.72,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})

C_NAPM = "#C0392B"
C_BLUE = "#2B6CB0"
C_TEAL = "#1F7A8C"
C_SLATE = "#64748B"
C_TEXT = "#1F2933"

# ---------------- Load verified data ----------------
# Row 1 (A, B): Extended Data Fig. 6c-f saddle matrices, strength & WT similarity
sad_df = pd.read_csv(R4C1 / "tables/control9_final_saddle.csv").set_index("condition")
sim_df = pd.read_csv(R4C1 / "tables/control9_saddle_similarity.csv").set_index("condition")
SAD_ORDER = [
    ("WT", "WT", "1.30", "#2B6CB0"),
    ("WT-hypoxia", "WT-Hyp", "0.84", "#2CA02C"),
    ("WT-latent", "WT-Lat", "0.60", "#8E44AD"),
    ("napM-KO", "KO", "1.06", "#C0392B"),
    ("napM-KO-hypoxia", "KO-Hyp", "0.90", "#D97706"),
]
sad_mats = {k: np.load(R4C1 / f"data/saddle20q_{k}.npy") for k, _, _, _ in SAD_ORDER}
wt_ceiling = float(sim_df.loc["WT replicate ceiling (mean pairwise r)", "pearson_r_vs_WT"])

# Row 2 (C): Loop-anchor expression divergence (exact seed=42 reproduction of Fig. 1e)
rna = pd.read_csv(BASE / "review_data/rnaseq/AKR_RNA_seq_merged.csv")
rna["WT"] = (rna["WT-rep1"] + rna["WT-rep2"]) / 2.0
gtf = pd.read_csv(
    BASE / "annotation.gtf", sep="\t", comment="#", header=None,
    names=["chr", "src", "feat", "start", "end", "score", "strand", "frame", "attr"],
)
gtf = gtf[gtf.feat == "gene"].copy()
gtf["gene_id"] = gtf.attr.str.extract(r'gene_id "?([^";]+)"?')
m_df = gtf.merge(rna[["gene", "WT"]], left_on="gene_id", right_on="gene").sort_values("start").reset_index(drop=True)
m_df["expr_norm"] = m_df.WT / (m_df.end - m_df.start)
m_df = m_df[m_df.expr_norm > 0].reset_index(drop=True)
n_genes = len(m_df)


def pos_to_gene(pos):
    inside = m_df[(m_df.start <= pos) & (m_df.end >= pos)]
    if len(inside):
        return inside.index[0]
    nxt = m_df[m_df.start > pos]
    return nxt.index[0] if len(nxt) else m_df.index[-1]


def div_g(gi, gj):
    a, b = m_df.loc[gi, "expr_norm"], m_df.loc[gj, "expr_norm"]
    return max(a, b) / min(a, b) - 1.0


cs = pd.read_csv(BASE / "review_results/R4_comment3_loops/chromosight_out/wt.tsv", sep="\t")
cs["m1"] = (cs.start1 + cs.end1) // 2
cs["m2"] = (cs.start2 + cs.end2) // 2
cs["d"] = cs.m2 - cs.m1
loop_pairs = [(pos_to_gene(r.m1), pos_to_gene(r.m2), int(r.d)) for _, r in cs.iterrows()]
loops_div = np.array([div_g(gi, gj) for gi, gj, _ in loop_pairs])
wt_e = m_df.expr_norm.values

rng = np.random.default_rng(42)
r1, r2, r3 = [], [], []
for gi, gj, d in loop_pairs:
    for _ in range(25):
        h = int(rng.integers(0, n_genes))
        if h != gi:
            r1.append(div_g(gi, h))
    got2 = got3 = 0
    tries = 0
    while (got2 < 25 or got3 < 10) and tries < 3000:
        tries += 1
        p1 = int(rng.integers(1, 4411532 - d - 1))
        hi, hj = pos_to_gene(p1), pos_to_gene(p1 + d)
        if hi == hj:
            continue
        if got2 < 25:
            r2.append(div_g(hi, hj))
            got2 += 1
        if got3 < 10 and 0.5 <= wt_e[hi] / wt_e[gi] <= 2.0:
            r3.append(div_g(hi, hj))
            got3 += 1
r1, r2, r3 = map(np.array, (r1, r2, r3))
p_r1 = stats.mannwhitneyu(loops_div, r1, alternative="less").pvalue
p_r2 = stats.mannwhitneyu(loops_div, r2, alternative="less").pvalue
p_r3 = stats.mannwhitneyu(loops_div, r3, alternative="less").pvalue

# Row 2 (D): CID gene-set transcriptional coordination (Fig. 1f)
cid_df = pd.read_csv(FIG1F_CSV)
cid_obs = cid_df.loc[cid_df.group == "CID", "mean_pair_r"].values
cid_rnd = cid_df.loc[cid_df.group == "same_size_random", "mean_pair_r"].values

# Row 2 (E) & Row 3 (F, G, H): Loop network & NapM connectivity (kept from previous B, C, D, E)
bridging = pd.read_csv(R4C6 / "loop_network_analysis/tables/bridging_proteins.csv")
bind_sum = pd.read_csv(R4C6 / "loop_network_analysis/tables/binding_summary.csv").set_index("metric")
null_int = np.load(R1P / "null_loop_bridging_interval_both.npy")
uni_both = pd.read_csv(R1P / "loop_bridging_unified.csv")
uni_both = uni_both[uni_both.metric == "both"].set_index("definition")

nodes_df = pd.read_csv(BASE / "review_data/Interaction_network/full_nodes.csv")
edges_df = pd.read_csv(BASE / "review_data/Interaction_network/full_edges.csv")
new_loops = [(f"bin{int(r.bin1)+1}", f"bin{int(r.bin2)+1}") for _, r in cs.iterrows()]
rv_bins = set(
    f"bin{int(b)+1}"
    for b in pd.read_csv(BASE / "review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_bound_bins_interval.csv")["bin"]
)

G = nx.Graph()
for _, r in nodes_df.iterrows():
    G.add_node(r["id"], label=r["label"], type=r["type"])
for _, r in edges_df.iterrows():
    if r["interaction_type"] != "DNA-DNA":
        G.add_edge(r["source"], r["target"], interaction_type=r["interaction_type"])
for u, v in new_loops:
    if u in G and v in G:
        G.add_edge(u, v, interaction_type="DNA-DNA")

dna_protein_edges = [(u, v) for u, v, d in G.edges(data=True) if d["interaction_type"] == "DNA-Protein"]
bin2prot = defaultdict(set)
for u, v in dna_protein_edges:
    if G.nodes[u]["type"] == "DNA":
        bin2prot[u].add(v)
    else:
        bin2prot[v].add(u)
for b in list(bin2prot):
    bin2prot[b].discard("Rv0047c")
for b in rv_bins:
    bin2prot[b].add("Rv0047c")

dna_nodes, protein_nodes = set(), set()
for u, v in new_loops:
    common = bin2prot[u] & bin2prot[v]
    if common:
        dna_nodes.update([u, v])
        protein_nodes.update(common)

sub_edges = [
    (u, v) for u, v in dna_protein_edges
    if (u in dna_nodes and v in protein_nodes) or (v in dna_nodes and u in protein_nodes)
]
G_sub = nx.Graph()
for nd in dna_nodes:
    G_sub.add_node(nd, type="DNA")
for nd in protein_nodes:
    G_sub.add_node(nd, type="Protein")
G_sub.add_edges_from(sub_edges)
G_sub.add_edges_from(new_loops)
prot_deg = sorted([(p, G_sub.degree(p)) for p in protein_nodes], key=lambda x: (-x[1], x[0]))

f2f = pd.read_csv(R4C6 / "tables/fig2f_metrics.csv")
napm_row = f2f[f2f.protein == "Rv0047c"].iloc[0]
other_rows = f2f[f2f.protein != "Rv0047c"]

# ==================== Build crisp 3-row, 8-panel figure (10.6 x 9.2 in) ====================
fig = plt.figure(figsize=(10.6, 9.2))
outer = GridSpec(
    3, 1, figure=fig,
    left=0.068, right=0.980, top=0.946, bottom=0.062,
    hspace=0.45, height_ratios=[0.88, 1.06, 1.06],
)

# Row 1: Panel A (5 saddle plots + colorbar) & Panel B (strength + similarity)
gs_r1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.38, width_ratios=[1.24, 1.0])
gs_r1_a = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_r1[0, 0], wspace=0.10)
gs_r1_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_r1[0, 1], wspace=0.40, width_ratios=[1.10, 0.96])

# Row 2: Panel C (loop expr), Panel D (CID expr), Panel E (top double-anchor proteins)
gs_r2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.38, width_ratios=[1.05, 0.78, 1.17])

# Row 3: Panel F (NapM controls), Panel G (bipartite rank), Panel H (multilayer 229 DBPs)
gs_r3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.36, width_ratios=[1.05, 1.00, 0.95])


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.6)


# ---- Row 1, Panel A: 5 re-evaluated saddle matrices (Extended Data Fig. 6c,d) ----
axA_list = []
for k, (cond_key, short_lbl, str_disp, _) in enumerate(SAD_ORDER):
    ax = fig.add_subplot(gs_r1_a[0, k])
    axA_list.append(ax)
    S_log2 = np.log2(np.clip(sad_mats[cond_key], 0.25, 4.0))
    imA = ax.imshow(S_log2, cmap="RdBu_r", vmin=-0.85, vmax=0.85, interpolation="nearest")
    ax.set_title(f"{short_lbl}\n({str_disp} log$_2$)", fontsize=7.6, fontweight="bold", pad=4, color=C_TEXT)
    ax.set_xticks([])
    ax.set_yticks([])
    if k == 0:
        ax.set_ylabel("E1 rank (20 groups)", fontsize=7.2, labelpad=3)
    if k == 2:
        ax.set_xlabel("E1 rank (ori–ter axis, 2.5th–97.5th pct)", fontsize=7.2, labelpad=4)

pA_last = axA_list[-1].get_position()
caxA = fig.add_axes([pA_last.x1 + 0.008, pA_last.y0, 0.009, pA_last.height])
cbA = fig.colorbar(imA, cax=caxA)
cbA.set_label(r"$\log_2$(O/E)", fontsize=7.0, labelpad=2)
cbA.ax.tick_params(labelsize=6.6)

# ---- Row 1, Panel B1: Axial segregation strength (Extended Data Fig. 6e) ----
axB1 = fig.add_subplot(gs_r1_b[0, 0])
xb1 = np.arange(len(SAD_ORDER))
# Match exact Extended Data Fig. 6e values + control9 jackknife SE
vals_b1 = [float(s_disp) for _, _, s_disp, _ in SAD_ORDER]
ses_b1 = [float(sad_df.loc[k, "jackknife_se"]) for k, _, _, _ in SAD_ORDER]
cols_b1 = [c for _, _, _, c in SAD_ORDER]
axB1.bar(xb1, vals_b1, yerr=ses_b1, width=0.62, color=cols_b1, edgecolor="black", lw=0.45,
         capsize=2.4, error_kw=dict(lw=0.85, color="black"), alpha=0.88)
axB1.axhline(0.03, color="#666666", ls=":", lw=0.8)
p_notes = ["", "5.9e-4", "2.6e-8", "0.048", "n.s."]
for idx, (v, se, pn) in enumerate(zip(vals_b1, ses_b1, p_notes)):
    if pn:
        axB1.text(idx, v + se + 0.06, pn, ha="center", va="bottom", fontsize=6.1,
                  color=C_TEXT if pn != "n.s." else C_SLATE, fontweight="bold" if pn != "n.s." else "normal")
axB1.set_xticks(xb1, [lbl for _, lbl, _, _ in SAD_ORDER], rotation=24, ha="right", fontsize=6.8)
axB1.set_ylim(0, 1.72)
axB1.set_ylabel(r"Segregation ($\log_2$)")
axB1.set_title("Axial segregation\n(Ext. Data Fig. 6e)", loc="left", pad=4, fontsize=8.1, fontweight="bold", color=C_TEXT)
clean_spines(axB1)

# ---- Row 1, Panel B2: Saddle similarity to WT (Extended Data Fig. 6f) ----
axB2 = fig.add_subplot(gs_r1_b[0, 1])
SIM_ORDER = SAD_ORDER[1:]
xb2 = np.arange(len(SIM_ORDER))
vals_b2 = [float(sim_df.loc[k, "pearson_r_vs_WT"]) for k, _, _, _ in SIM_ORDER]
cols_b2 = [c for _, _, _, c in SIM_ORDER]
axB2.bar(xb2, vals_b2, width=0.58, color=cols_b2, edgecolor="black", lw=0.45, alpha=0.90)
axB2.axhline(wt_ceiling, color="#222222", ls="--", lw=0.95, zorder=4)
axB2.text(3.38, wt_ceiling + 0.010, f"WT rep ceiling ({wt_ceiling:.2f})", ha="right", va="bottom",
          fontsize=6.1, color="#222222")
for idx, v in enumerate(vals_b2):
    axB2.text(idx, v - 0.014, f"{v:.2f}", ha="center", va="top", fontsize=6.5, fontweight="bold", color="white")
axB2.set_xticks(xb2, [lbl for _, lbl, _, _ in SIM_ORDER], rotation=24, ha="right", fontsize=6.8)
axB2.set_ylim(0.72, 1.07)
axB2.set_ylabel("Saddle Pearson r vs WT")
axB2.text(3.0, 0.855, "Null max\nr = 0.21\n(P < 0.01)", ha="center", va="bottom",
          fontsize=5.8, color=C_SLATE, linespacing=1.15)
axB2.set_title("Pattern similarity\n(Ext. Data Fig. 6f)", loc="left", pad=4, fontsize=8.1, fontweight="bold", color=C_TEXT)
clean_spines(axB2)

# ---- Row 2, Panel C (kept from previous F): Loop-anchor transcriptional coordination (Fig. 1e) ----
axC = fig.add_subplot(gs_r2[0, 0])
groups_c = [loops_div, r1, r2, r3]
log_groups = [np.log10(g + 1.0) for g in groups_c]
meds_orig = [np.median(g) for g in groups_c]
labels_c = ["Loops\n(n=41)", "Random", "Dist.\nmatched", "Dist.+expr\nmatched"]
cols_c = [C_NAPM, "#4A6FA5", "#7A9CC6", "#B8C5D6"]
bp_c = axC.boxplot(
    log_groups, positions=range(4), widths=0.52, patch_artist=True, showfliers=False,
    medianprops=dict(color="black", lw=1.35),
    whiskerprops=dict(color="#444444", lw=0.8),
    capprops=dict(color="#444444", lw=0.8),
)
for patch, col in zip(bp_c["boxes"], cols_c):
    patch.set_facecolor(col)
    patch.set_alpha(0.85)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)
pvals_c = [None, p_r1, p_r2, p_r3]
for idx in range(4):
    q3 = np.percentile(log_groups[idx], 75)
    iqr = q3 - np.percentile(log_groups[idx], 25)
    whisk_top = np.max(log_groups[idx][log_groups[idx] <= q3 + 1.5 * iqr])
    if idx == 0:
        axC.text(idx, whisk_top + 0.06, f"med.\n{meds_orig[idx]:.2f}", ha="center", va="bottom",
                 fontsize=6.7, fontweight="bold", color=C_NAPM, linespacing=1.1)
    else:
        p = pvals_c[idx]
        star = "**" if p < 0.01 else "*"
        axC.text(idx, whisk_top + 0.05, f"{star}\nmed. {meds_orig[idx]:.2f}\nP={p:.3g}",
                 ha="center", va="bottom", fontsize=6.1, color=C_TEXT, linespacing=1.12)
axC.set_xticks(range(4), labels_c)
axC.set_ylim(-0.04, 2.40)
axC.set_ylabel(r"Expression divergence $\log_{10}(\mathrm{max}/\mathrm{min})$")
axC.set_title("Loop-anchor coordination (Fig. 1e)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axC)

# ---- Row 2, Panel D (NEW — Fig. 1f): CID gene-set multi-sample co-expression ----
axD = fig.add_subplot(gs_r2[0, 1])
bp_d = axD.boxplot(
    [cid_obs, cid_rnd], positions=[0, 1], widths=0.48, patch_artist=True, showfliers=False,
    medianprops=dict(color="black", lw=1.35),
    whiskerprops=dict(color="#444444", lw=0.8),
    capprops=dict(color="#444444", lw=0.8),
)
for patch, col in zip(bp_d["boxes"], [C_TEAL, "#94A3B8"]):
    patch.set_facecolor(col)
    patch.set_alpha(0.82)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.6)
axD.plot([0, 0, 1, 1], [0.390, 0.398, 0.398, 0.390], color="#333333", lw=0.85)
axD.text(0.5, 0.402, "* empirical P = 0.045\nmean r: 0.302 vs 0.289 (+4.6%)",
         ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=C_TEXT, linespacing=1.18)
axD.set_xticks([0, 1], ["CID sets", "Random"])
axD.set_ylim(0.18, 0.452)
axD.set_ylabel("Mean within-set Pearson r")
axD.set_title("CID co-expression (Fig. 1f)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axD)

# ---- Row 2, Panel E (kept from previous B): Top 10 double-anchor proteins across 41 loops ----
axE = fig.add_subplot(gs_r2[0, 2])
top10 = bridging.head(10).iloc[::-1].reset_index(drop=True)
labels_e = ["NapM (Rv0047c)" if p == "Rv0047c" else p for p in top10.protein]
cols_e = [C_NAPM if p == "Rv0047c" else "#64748B" for p in top10.protein]
ye = np.arange(len(top10))
axE.barh(ye, top10.n_loops_bridged, color=cols_e, edgecolor="black", lw=0.45, height=0.70)
for idx, (val, p) in enumerate(zip(top10.n_loops_bridged, top10.protein)):
    axE.text(
        val + 0.35, idx, f"{int(val)}", va="center", ha="left",
        fontsize=7.1, fontweight="bold" if p == "Rv0047c" else "normal",
        color=C_NAPM if p == "Rv0047c" else C_TEXT,
    )
axE.set_yticks(ye, labels_e)
for tick_lbl in axE.get_yticklabels():
    if "NapM" in tick_lbl.get_text():
        tick_lbl.set_fontweight("bold")
        tick_lbl.set_color(C_NAPM)
axE.set_xlim(0, 20.5)
axE.set_xlabel("Loops with both anchors bound (of 41)")
axE.set_title("Top double-anchor proteins (41 loops)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axE)

# ---- Row 3, Panel F (kept from previous C): NapM double-anchor enrichment controls ----
axF = fig.add_subplot(gs_r3[0, 0])
bins_f = np.arange(0.5, 24.5, 1.0)
axF.hist(null_int, bins=bins_f, density=True, color="#94A3B8", edgecolor="white", lw=0.45, alpha=0.88,
         label="Circular-shift null (10,000)")
ymax_f = axF.get_ylim()[1]
axF.plot([17, 17], [0, ymax_f * 1.02], color=C_NAPM, lw=2.0, zorder=4, label="Observed NapM (17/41)")
axF.scatter([17], [ymax_f * 1.02], color=C_NAPM, s=22, zorder=5)
axF.set_xlim(1, 24.0)
axF.set_ylim(0, ymax_f * 1.98)
axF.set_xlabel("NapM doubly bound loops (of 41)")
axF.set_ylabel("Density")
axF.legend(frameon=False, loc="upper left", ncol=1, fontsize=6.5, handlelength=1.4, labelspacing=0.22)
axF.text(
    0.98, 0.77,
    f"Circular shift: 17 vs {uni_both.loc['interval','null_mean']:.1f} ± {uni_both.loc['interval','null_sd']:.1f} (P = {uni_both.loc['interval','p_empirical']:.3f})\n"
    f"Dist.-matched: 17 vs {bind_sum.loc['rv_bridge','null_mean']:.1f} ± {bind_sum.loc['rv_bridge','null_sd']:.1f} (P = {bind_sum.loc['rv_bridge','empirical_p']:.3f})\n"
    f"Curated bins: 17 vs {uni_both.loc['curated','null_mean']:.1f} (P = {uni_both.loc['curated','p_empirical']:.3f})\n"
    f"Network edges: 15 vs {uni_both.loc['network','null_mean']:.1f} (P = {uni_both.loc['network','p_empirical']:.3f})",
    transform=axF.transAxes, ha="right", va="top", fontsize=6.05, linespacing=1.18,
    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CBD5E1", lw=0.55, alpha=0.96),
)
axF.set_title("NapM double-anchor enrichment controls", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axF)

# ---- Row 3, Panel G (kept from previous D — NO Rv0081 label!): Bipartite network rank ----
axG = fig.add_subplot(gs_r3[0, 1])
ranks = np.arange(1, len(prot_deg) + 1)
degs = np.array([d for _, d in prot_deg])
napm_idx = [i for i, (p, _) in enumerate(prot_deg) if p == "Rv0047c"][0]
napm_rank = napm_idx + 1
napm_deg = degs[napm_idx]

axG.plot(ranks, degs, "-", color="#94A3B8", lw=1.1, zorder=2)
axG.scatter(ranks, degs, s=16, color="#64748B", edgecolor="white", lw=0.3, zorder=3)
axG.scatter([napm_rank], [napm_deg], s=58, color=C_NAPM, edgecolor="black", lw=0.7, zorder=5)
axG.annotate(
    f"NapM (Rv0047c)\ndegree = {napm_deg} (rank {napm_rank}/49)",
    xy=(napm_rank, napm_deg), xytext=(15.5, 42.0),
    fontsize=7.0, fontweight="bold", color=C_NAPM, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#F5B7B1", lw=0.6),
    arrowprops=dict(arrowstyle="-|>", color=C_NAPM, lw=1.0),
)
axG.set_xlim(-1, 51)
axG.set_ylim(0, 54)
axG.set_xlabel("Protein rank in loop-anchor network (n = 49)")
axG.set_ylabel("Bipartite degree (DNA anchor bins)")
axG.set_title("Loop-anchor network rank", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axG)

# ---- Row 3, Panel H (kept from previous E): NapM multilayer connectivity (229 DBPs, Fig. 2f) ----
axH = fig.add_subplot(gs_r3[0, 2])
layers = [
    ("loop", "Loops\n(both anchors)", "Top 1%"),
    ("operon", "Operons\n(of 749)", "Top 3%"),
    ("ppi", "PPI\npartners", "Top 6%"),
]
folds_h = [float(napm_row[col] / other_rows[col].mean()) for col, _, _ in layers]
raw_lbls = [
    f"{int(napm_row[col])} vs {other_rows[col].mean():.2f}" if col != "operon"
    else f"{int(napm_row[col])} vs {other_rows[col].mean():.1f}"
    for col, _, _ in layers
]
xh = np.arange(len(layers))
bars_h = axH.bar(xh, folds_h, width=0.52, color=[C_NAPM, "#2B6CB0", "#4A5568"], edgecolor="black", lw=0.55)
axH.axhline(1.0, color="#444444", ls="--", lw=0.85)
for idx, (b, fval, rlbl, (_, _, pct)) in enumerate(zip(bars_h, folds_h, raw_lbls, layers)):
    axH.text(
        b.get_x() + b.get_width() / 2, fval + 0.55,
        f"{fval:.1f}× ({pct})\n[{rlbl}]",
        ha="center", va="bottom", fontsize=6.6, fontweight="bold" if idx == 0 else "normal",
        color=C_NAPM if idx == 0 else C_TEXT, linespacing=1.18,
    )
axH.set_xticks(xh, [lbl for _, lbl, _ in layers])
axH.set_ylim(0, 22.0)
axH.set_ylabel("Fold over mean of 228 DBPs")
axH.set_title("NapM multilayer rank (Fig. 2f, 229 DBPs)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axH)

# Aligned figure-level panel letters A-H
pA0 = axA_list[0].get_position()
pB1 = axB1.get_position()
pC, pD, pE = axC.get_position(), axD.get_position(), axE.get_position()
pF, pG, pH = axF.get_position(), axG.get_position(), axH.get_position()

y_top_r1 = max(pA0.y1, pB1.y1) + 0.044
fig.text(0.012, y_top_r1, "A", fontsize=13.0, fontweight="bold", va="top")
fig.text(pB1.x0 - 0.048, y_top_r1, "B", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pC.y1 + 0.032, "C", fontsize=13.0, fontweight="bold", va="top")
fig.text(pD.x0 - 0.052, pD.y1 + 0.032, "D", fontsize=13.0, fontweight="bold", va="top")
fig.text(pE.x0 - 0.088, pE.y1 + 0.032, "E", fontsize=13.0, fontweight="bold", va="top")
fig.text(0.012, pF.y1 + 0.032, "F", fontsize=13.0, fontweight="bold", va="top")
fig.text(pG.x0 - 0.052, pG.y1 + 0.032, "G", fontsize=13.0, fontweight="bold", va="top")
fig.text(pH.x0 - 0.052, pH.y1 + 0.032, "H", fontsize=13.0, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_17.pdf"
png_p = OUT_FIG / "Response_Fig_17.png"
jpg_p = OUT_FIG / "Response_Fig_17.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print(f"Saved redesigned Response_Fig_17.{{pdf,png,jpg}} | CID mean r: {cid_obs.mean():.3f} vs {cid_rnd.mean():.3f}")
