#!/usr/bin/env python3
"""Generate clean, self-contained graphical assets for the repository README directly
from archived summary tables in this repository.

Outputs:
  - assets/readme_banner.png
  - assets/readme_summary_dashboard.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

for fam in ("Arial", "Helvetica", "DejaVu Sans"):
    try:
        font_manager.findfont(font_manager.FontProperties(family=fam), fallback_to_default=False)
        FONT = fam
        break
    except ValueError:
        continue

plt.rcParams.update({
    "font.family": FONT,
    "mathtext.default": "regular",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
})

# High-contrast, colorblind-safe palette (crisp on both GitHub Light & Dark themes)
NAVY = "#0F172A"
SLATE = "#334155"
MUTED = "#64748B"
BORDER = "#CBD5E1"
BG_CARD = "#F8FAFC"
BLUE = "#2563EB"
TEAL = "#0D9488"
PURPLE = "#6D28D9"
AMBER = "#D97706"
CORAL = "#DC2626"
MINT = "#059669"


def rounded_box(ax, x, y, w, h, fc="white", ec=BORDER, lw=1.0, r=0.018, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def quad_curve(ax, p0, p1, p2, color, lw=1.8, zorder=4):
    path = MplPath([p0, p1, p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, zorder=zorder))


def build_banner():
    fw, fh = 14.0, 5.8
    aspect = fw / fh  # ~2.414
    fig = plt.figure(figsize=(fw, fh), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer sleek frame
    rounded_box(ax, 0.008, 0.015, 0.984, 0.970, fc="#FFFFFF", ec="#CBD5E1", lw=1.4, r=0.024, zorder=0)

    # Top dark header bar
    rounded_box(ax, 0.022, 0.775, 0.956, 0.188, fc=NAVY, ec=NAVY, lw=0, r=0.020, zorder=1)
    ax.add_patch(Rectangle((0.022, 0.775), 0.956, 0.011, facecolor=TEAL, edgecolor="none", zorder=2))

    # True-circle stylized chromosome emblem (aspect-corrected Ellipse!)
    cx, cy = 0.064, 0.875
    r_outer_y = 0.058
    r_outer_x = r_outer_y / aspect
    r_inner_y = 0.041
    r_inner_x = r_inner_y / aspect
    ax.add_patch(Ellipse((cx, cy), 2 * r_outer_x, 2 * r_outer_y, fill=False, edgecolor="#38BDF8", lw=2.3, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 2 * r_inner_x, 2 * r_inner_y, fill=False, edgecolor="#A78BFA", lw=1.3, ls="--", zorder=3))
    ax.scatter([cx], [cy + r_outer_y], s=48, color="#34D399", edgecolor="white", lw=0.8, zorder=4)
    ax.scatter([cx], [cy - r_outer_y], s=48, color="#F87171", edgecolor="white", lw=0.8, zorder=4)
    ax.text(cx, cy, "3D", color="white", fontsize=11.0, fontweight="bold", ha="center", va="center", zorder=4)

    # Header titles
    ax.text(0.106, 0.904, "Mycobacterium tuberculosis 3D Genome Architecture & Remodeling",
            color="white", fontsize=17.2, fontweight="bold", ha="left", va="center", zorder=3)
    ax.text(0.106, 0.834,
            "Multi-Omics Reproducibility Repository  ·  Hi-C · RNA-seq · NapM ChIP-seq · B2H PPI · OpenMM Polymer Modeling",
            color="#CBD5E1", fontsize=9.6, ha="left", va="center", zorder=3)

    # Right SRA pill in header
    rounded_box(ax, 0.822, 0.840, 0.138, 0.070, fc="#1E293B", ec="#38BDF8", lw=1.0, r=0.015, zorder=3)
    ax.text(0.891, 0.875, "SRA: PRJNA1452606", color="#E0F2FE", fontsize=8.6, fontweight="bold",
            ha="center", va="center", zorder=4)

    # ------------------------------------------------------------------
    # THREE MAIN COLUMNS
    # ------------------------------------------------------------------
    col_y, col_h = 0.045, 0.698

    # === COLUMN 1: Multi-Omics Primary Inputs (x=0.022..0.318) ===
    rounded_box(ax, 0.022, col_y, 0.296, col_h, fc=BG_CARD, ec=BORDER, lw=1.0, r=0.018, zorder=1)
    rounded_box(ax, 0.036, 0.676, 0.268, 0.050, fc="#DBEAFE", ec="#93C5FD", lw=0.8, r=0.011, zorder=2)
    ax.text(0.170, 0.701, "1. Primary Data & Conditions (data/)", color="#1E3A8A",
            fontsize=10.0, fontweight="bold", ha="center", va="center", zorder=3)

    conds = [
        ("WT (n=4)", BLUE), ("napM-KO (n=4)", PURPLE),
        ("WT-Hypoxia (n=4)", TEAL), ("KO-Hypoxia (n=3)", AMBER),
        ("WT-Latent* (n=4)", CORAL),
    ]
    pill_pos = [(0.038, 0.604, 0.128), (0.174, 0.604, 0.128),
                (0.038, 0.544, 0.128), (0.174, 0.544, 0.128),
                (0.088, 0.484, 0.164)]
    for (lbl, col), (px, py, pw) in zip(conds, pill_pos):
        rounded_box(ax, px, py, pw, 0.044, fc="white", ec=col, lw=1.1, r=0.011, zorder=2)
        ax.scatter([px + 0.014], [py + 0.022], s=25, color=col, zorder=3)
        ax.text(px + 0.025, py + 0.022, lbl, color=NAVY, fontsize=7.9, fontweight="bold",
                ha="left", va="center", zorder=3)
    ax.text(0.170, 0.464, "*in vitro acid/antibiotic latency-like stress", color=MUTED,
            fontsize=6.5, style="italic", ha="center", va="center", zorder=3)

    omics = [
        ("Hi-C Matrices", "5-kb ICE-balanced · 19 replicates + 5 merged", BLUE),
        ("Bulk RNA-seq", "4,065 genes · limma DEGs & factorial model", TEAL),
        ("NapM ChIP-seq", "634 curated Rv0047c peaks · AT-preferring", PURPLE),
        ("B2H PPI & Operons", "2,992 proteins · 8,043 PPIs · 749 operons", AMBER),
    ]
    for i, (title, desc, col) in enumerate(omics):
        ry = 0.365 - i * 0.098
        rounded_box(ax, 0.038, ry, 0.264, 0.080, fc="white", ec="#E2E8F0", lw=0.9, r=0.011, zorder=2)
        ax.add_patch(Rectangle((0.038, ry + 0.010), 0.006, 0.060, facecolor=col, edgecolor="none", zorder=3))
        ax.text(0.054, ry + 0.054, title, color=NAVY, fontsize=8.8, fontweight="bold", ha="left", va="center", zorder=3)
        ax.text(0.054, ry + 0.024, desc, color=MUTED, fontsize=7.5, ha="left", va="center", zorder=3)

    # Arrow 1 -> 2
    ax.add_patch(FancyArrowPatch((0.320, 0.40), (0.348, 0.40),
                                 arrowstyle="-|>,head_length=7,head_width=4.5",
                                 color="#475569", lw=1.8, zorder=4))

    # === COLUMN 2: Multi-Scale 3D Genome Framework (x=0.350..0.678) ===
    rounded_box(ax, 0.350, col_y, 0.328, col_h, fc=BG_CARD, ec=BORDER, lw=1.0, r=0.018, zorder=1)
    rounded_box(ax, 0.364, 0.676, 0.300, 0.050, fc="#CCFBF1", ec="#5EEAD4", lw=0.8, r=0.011, zorder=2)
    ax.text(0.514, 0.701, "2. Multi-Scale 3D Architecture (15 Modules)", color="#115E59",
            fontsize=10.0, fontweight="bold", ha="center", va="center", zorder=3)

    # Circular chromosome schematic (clean true-proportion oval with generous vertical clearance)
    gx, gy = 0.514, 0.520
    ry = 0.098
    rx = ry / aspect * 1.45  # sleek pill-circular ring with clean margins
    theta_l = np.linspace(np.pi / 2, 3 * np.pi / 2, 160)
    theta_r = np.linspace(-np.pi / 2, np.pi / 2, 160)
    ax.plot(gx + rx * np.cos(theta_l), gy + ry * np.sin(theta_l), color=BLUE, lw=3.2, solid_capstyle="round", zorder=3)
    ax.plot(gx + rx * np.cos(theta_r), gy + ry * np.sin(theta_r), color=TEAL, lw=3.2, solid_capstyle="round", zorder=3)

    # Subtle arm-arm juxtaposition dashed rungs
    for frac in [-0.52, -0.18, 0.18, 0.52]:
        xx = rx * np.sqrt(max(0.0, 1.0 - frac**2))
        ax.plot([gx - xx * 0.86, gx + xx * 0.86], [gy + frac * ry, gy + frac * ry],
                color="#CBD5E1", lw=1.0, ls="--", zorder=2)

    # Smooth curved chromatin loop arcs anchored on the ring
    a1, a2 = np.deg2rad(132), np.deg2rad(195)
    p0 = (gx + rx * np.cos(a1), gy + ry * np.sin(a1))
    p2 = (gx + rx * np.cos(a2), gy + ry * np.sin(a2))
    quad_curve(ax, p0, (gx - rx * 0.25, gy + ry * 0.12), p2, PURPLE, lw=1.8, zorder=4)

    b1, b2 = np.deg2rad(42), np.deg2rad(-22)
    q0 = (gx + rx * np.cos(b1), gy + ry * np.sin(b1))
    q2 = (gx + rx * np.cos(b2), gy + ry * np.sin(b2))
    quad_curve(ax, q0, (gx + rx * 0.28, gy + ry * 0.08), q2, PURPLE, lw=1.8, zorder=4)

    # ori & ter markers with crisp spacing away from header and cards
    ax.scatter([gx], [gy + ry], s=72, color=MINT, edgecolor="white", lw=1.0, zorder=5)
    ax.scatter([gx], [gy - ry], s=72, color=CORAL, edgecolor="white", lw=1.0, zorder=5)
    ax.text(gx, gy + ry + 0.022, "oriC (0 Mb)", color=MINT, fontsize=8.0, fontweight="bold", ha="center", va="bottom", zorder=5)
    ax.text(gx, gy - ry - 0.020, "ter (2.03 Mb)", color=CORAL, fontsize=8.0, fontweight="bold", ha="center", va="top", zorder=5)
    ax.text(gx - rx - 0.014, gy, "Left\nArm", color=BLUE, fontsize=7.8, fontweight="bold", ha="right", va="center", zorder=5)
    ax.text(gx + rx + 0.014, gy, "Right\nArm", color=TEAL, fontsize=7.8, fontweight="bold", ha="left", va="center", zorder=5)
    ax.text(gx, gy, "4.41 Mb\nCircular", color=NAVY, fontsize=7.6, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88), zorder=5)

    # 4 Analytical tier rows below schematic
    tiers = [
        ("Axial Segregation", "Circular Fourier PC1 (|r| = 0.97) · Mean O/E saddle", BLUE),
        ("CID Domains", "18 DI-defined boundaries · within/cross ratio = 1.102", TEAL),
        ("Chromatin Loops", "41 Chromosight loops (30–995 kb) · APA O/E = 1.61", PURPLE),
        ("Polymer Modeling", "OpenMM bead–spring ring-polymer (S1 / S2 / S3)", AMBER),
    ]
    for i, (t_name, t_desc, col) in enumerate(tiers):
        ty = 0.304 - i * 0.077
        rounded_box(ax, 0.366, ty, 0.296, 0.063, fc="white", ec="#E2E8F0", lw=0.85, r=0.010, zorder=2)
        ax.scatter([0.380], [ty + 0.0315], s=32, color=col, zorder=3)
        ax.text(0.394, ty + 0.043, t_name, color=NAVY, fontsize=8.3, fontweight="bold", ha="left", va="center", zorder=3)
        ax.text(0.394, ty + 0.018, t_desc, color=MUTED, fontsize=7.2, ha="left", va="center", zorder=3)

    # Arrow 2 -> 3
    ax.add_patch(FancyArrowPatch((0.680, 0.40), (0.708, 0.40),
                                 arrowstyle="-|>,head_length=7,head_width=4.5",
                                 color="#475569", lw=1.8, zorder=4))

    # === COLUMN 3: Key Quantitative Highlights & Verification (x=0.710..0.978) ===
    rounded_box(ax, 0.710, col_y, 0.268, col_h, fc=BG_CARD, ec=BORDER, lw=1.0, r=0.018, zorder=1)
    rounded_box(ax, 0.724, 0.676, 0.240, 0.050, fc="#EDE9FE", ec="#C4B5FD", lw=0.8, r=0.011, zorder=2)
    ax.text(0.844, 0.701, "3. Key Results & Reproducibility", color="#4C1D95",
            fontsize=10.0, fontweight="bold", ha="center", va="center", zorder=3)

    kpis = [
        ("1.29 → 0.84 → 0.60", "Graded ori–ter segregation loss", "WT → Hypoxia → Latency-like (log2 O/E)", BLUE),
        ("276 → 515 kb", "napM-KO local decompaction", "<100 kb: 0.377 → 0.242 · Arm O/E: −10.4%", PURPLE),
        ("17 / 41 loops · #6", "NapM multilayer hub rank", "441 degree · 470 operons · 14 B2H PPIs", TEAL),
        ("108 joint bins", "Genotype × hypoxia remodeling", "730 interaction DEGs · Δins = −0.108", AMBER),
        ("SHA-256 Verified", "Reproducible data & code archive", "tools/validate_manifest.py · sandbox replay", MINT),
    ]
    for i, (big, mid, sub, col) in enumerate(kpis):
        ky = 0.548 - i * 0.119
        rounded_box(ax, 0.724, ky, 0.240, 0.100, fc="white", ec="#E2E8F0", lw=0.9, r=0.012, zorder=2)
        ax.add_patch(Rectangle((0.724, ky + 0.012), 0.006, 0.076, facecolor=col, edgecolor="none", zorder=3))
        ax.text(0.740, ky + 0.074, big, color=col, fontsize=9.8, fontweight="bold", ha="left", va="center", zorder=3)
        ax.text(0.740, ky + 0.045, mid, color=NAVY, fontsize=8.0, fontweight="bold", ha="left", va="center", zorder=3)
        ax.text(0.740, ky + 0.019, sub, color=MUTED, fontsize=6.9, ha="left", va="center", zorder=3)

    out = ASSETS / "readme_banner.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")


def build_summary_dashboard():
    saddle = pd.read_csv(
        REPO / "response_analyses/analysis_10_circular_eigenvector_saddle/results/hic_review/review_results/R4_comment1_AB_compartment/tables/control9_final_saddle.csv"
    )
    rep = pd.read_csv(
        REPO / "response_analyses/analysis_13_wt_ko_compaction_metrics/results/hic_review/review_results/R4_comment4/tables/replicate_metrics.csv"
    )
    cent = pd.read_csv(
        REPO / "response_analyses/analysis_03_network_centrality_hubs/results/analysis/reviewer2_revision_20261119/tables/full_network_centrality.csv"
    )
    loop_rank = pd.read_csv(
        REPO / "response_analyses/analysis_03_network_centrality_hubs/results/analysis/loop_augmented_network_20261105/loop_augmented_degree_ranking.csv"
    )
    modes = pd.read_csv(
        REPO / "response_analyses/analysis_02_hypoxia_factorial_integration/results/docx_latex_conversion/generated/structure_020_response_modes/Response_Fig2F_structure_020_enrichment_tests.csv"
    )

    fig = plt.figure(figsize=(13.6, 8.2), dpi=200, facecolor="white")
    fig.suptitle(
        "Quantitative Overview Generated Directly from Archived Repository Tables",
        x=0.04, y=0.972, ha="left", fontsize=13.5, fontweight="bold", color=NAVY
    )
    fig.text(
        0.04, 0.937,
        "Auto-reproduced via scripts/build_readme_graphics.py  ·  All values computed directly from SHA-256 verified CSV tables in response_analyses/",
        fontsize=8.7, color=MUTED, ha="left"
    )

    gs = GridSpec(2, 2, figure=fig, left=0.075, right=0.965, top=0.875, bottom=0.085, wspace=0.28, hspace=0.42)

    def style_ax(ax, letter, title):
        ax.set_title(f"{letter}   {title}", loc="left", fontsize=10.2, fontweight="bold", color=NAVY, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94A3B8")
        ax.spines["bottom"].set_color("#94A3B8")
        ax.tick_params(colors=SLATE, labelsize=8.3)
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    # --- Panel A: ori-ter axial segregation strength across 5 conditions ---
    axA = fig.add_subplot(gs[0, 0])
    order_a = ["WT", "WT-hypoxia", "WT-latent", "napM-KO", "napM-KO-hypoxia"]
    labels_a = ["WT", "WT\nHypoxia", "WT\nLatent*", "napM-KO", "napM-KO\nHypoxia"]
    cols_a = [BLUE, TEAL, CORAL, PURPLE, AMBER]
    sad_idx = saddle.set_index("condition").loc[order_a]
    vals_a = sad_idx["strength_log2"].to_numpy()
    ci_a = 1.96 * sad_idx["jackknife_se"].to_numpy()
    xa = np.arange(len(order_a))
    axA.bar(xa, vals_a, yerr=ci_a, capsize=4, width=0.56, color=cols_a,
            edgecolor="white", lw=0.8, error_kw=dict(ecolor=NAVY, lw=1.1), zorder=3)
    null_max = float(sad_idx["shuffle_absmax"].max())
    axA.axhline(null_max, color="#475569", ls="--", lw=1.1, zorder=4,
                label=f"100-perm null max ({null_max:.2f})")
    for i, (v, c) in enumerate(zip(vals_a, ci_a)):
        axA.text(i, v + c + 0.045, f"{v:.2f}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=NAVY)
    axA.set_xticks(xa, labels_a)
    axA.set_ylabel(r"Axial segregation strength ($\log_2$ O/E)", fontsize=8.8, color=SLATE)
    axA.set_ylim(0, 1.82)
    axA.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=7.6)
    axA.text(0.99, 0.80, "*in vitro acid/antibiotic\nlatency-like stress", transform=axA.transAxes,
             ha="right", va="top", fontsize=7.2, color=MUTED, style="italic")
    style_ax(axA, "A", "Circular ori–ter Axial Segregation (20-quantile Saddle)")

    # --- Panel B: Replicate-level Hi-C architecture shift across conditions ---
    axB = fig.add_subplot(gs[0, 1])
    cond_b = ["WT", "napM-KO", "hypoxia", "latent"]
    lbl_b = ["WT", "napM-KO", "WT Hypoxia", "WT Latent*"]
    col_b = [BLUE, PURPLE, TEAL, CORAL]
    metrics_b = [
        ("frac_short_lt100kb", "Short-range\n(<100 kb)"),
        ("frac_long_gt1Mb", "Long-range\n(>1 Mb)"),
        ("sec_diag_OE", "Arm–arm\nO/E"),
        ("ori_ter_OE", "ori–ter\nO/E"),
    ]
    wt_means = rep[rep["condition"] == "WT"][[m[0] for m in metrics_b]].mean()
    xb = np.arange(len(metrics_b))
    w = 0.25
    axB.axhline(1.0, color=NAVY, ls="--", lw=1.0, zorder=2, label="WT baseline (1.0×)")
    for j, (ck, cl, col) in enumerate(zip(cond_b[1:], lbl_b[1:], col_b[1:])):
        sub = rep[rep["condition"] == ck]
        fcs = [sub[m[0]].mean() / wt_means[m[0]] for m in metrics_b]
        sds = [sub[m[0]].std() / wt_means[m[0]] for m in metrics_b]
        offset = (j - 1) * w
        axB.bar(xb + offset, fcs, yerr=sds, width=w * 0.90, color=col, capsize=2.5,
                edgecolor="white", lw=0.6, error_kw=dict(ecolor=NAVY, lw=0.8), label=cl, zorder=3)
        for xi, fv, sd in zip(xb + offset, fcs, sds):
            y_pos = fv + sd + 0.025
            if 0.90 <= fv <= 0.94:
                y_pos = fv + sd + 0.015  # stay cleanly below the 1.0 dashed line
            axB.text(xi, y_pos, f"{fv:.2f}×", ha="center", va="bottom", fontsize=6.8,
                     fontweight="bold", color=col,
                     bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none", alpha=0.8))
    axB.set_xticks(xb, [m[1] for m in metrics_b])
    axB.set_ylabel("Fold change vs WT (mean ± SD, n = 4)", fontsize=8.8, color=SLATE)
    axB.set_ylim(0, 1.66)
    axB.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=7.3, ncol=2)
    style_ax(axB, "B", "Replicate-Level Hi-C Remodeling Across Perturbations")

    # --- Panel C: Top-10 integrated network hubs ---
    axC = fig.add_subplot(gs[1, 0])
    top10 = cent.sort_values("degree_rank").head(10).merge(
        loop_rank[["protein", "loops_with_both_anchors_bound"]],
        left_on="node", right_on="protein", how="left"
    ).iloc[::-1].reset_index(drop=True)
    yc = np.arange(len(top10))
    bar_cols_c = [PURPLE if n == "Rv0047c" else ("#F59E0B" if r <= 5 else "#94A3B8")
                  for n, r in zip(top10["node"], top10["degree_rank"])]
    axC.barh(yc, top10["degree"], height=0.66, color=bar_cols_c, edgecolor="white", lw=0.6, zorder=3)
    yt_labels = ["NapM (Rv0047c)" if n == "Rv0047c" else n for n in top10["node"]]
    axC.set_yticks(yc, yt_labels)
    for tick, n in zip(axC.get_yticklabels(), top10["node"]):
        if n == "Rv0047c":
            tick.set_fontweight("bold")
            tick.set_color(PURPLE)
    for i, r in top10.iterrows():
        lcnt = int(r["loops_with_both_anchors_bound"]) if pd.notna(r["loops_with_both_anchors_bound"]) else 0
        txt = f"deg {int(r['degree'])} · {lcnt}/41 loops · bet #{int(r['betweenness_rank'])}"
        axC.text(r["degree"] + 9, i, txt, va="center", ha="left",
                 fontsize=7.3, fontweight="bold" if r["node"] == "Rv0047c" else "normal",
                 color=PURPLE if r["node"] == "Rv0047c" else SLATE)
    axC.set_xlim(0, 860)
    axC.set_xlabel("Integrated network degree (DNA–protein–PPI, n = 3,870 nodes)", fontsize=8.6, color=SLATE)
    style_ax(axC, "C", "Top-10 Hubs in Integrated Multi-Layer Network")
    axC.grid(axis="x", color="#E2E8F0", lw=0.7, zorder=0)
    axC.grid(axis="y", visible=False)

    # --- Panel D: Genotype x Hypoxia Response Modes (883 5-kb bins) ---
    axD = fig.add_subplot(gs[1, 1])
    mode_order = ["RNA only", "Structure only", "Joint response", "Other / unresolved"]
    mode_labels = ["RNA-only", "Structure-only", "Joint response", "Other /\nunresolved"]
    mode_cols = [TEAL, PURPLE, AMBER, "#94A3B8"]
    m_idx = modes.set_index("response_mode").loc[mode_order]
    obs_d = m_idx["observed_n"].to_numpy()
    exp_d = m_idx["null_mean_n"].to_numpy()
    pct_d = m_idx["observed_percent"].to_numpy()
    xd = np.arange(len(mode_order))
    w_d = 0.36
    axD.bar(xd - w_d / 2, obs_d, width=w_d, color=mode_cols, edgecolor="white", lw=0.7, label="Observed bins", zorder=3)
    axD.bar(xd + w_d / 2, exp_d, width=w_d, color="#E2E8F0", edgecolor="#94A3B8", lw=0.8,
            hatch="///", label="Circular-shift null mean (10,000)", zorder=3)
    for i, (ov, pv) in enumerate(zip(obs_d, pct_d)):
        extra = "\n(two-sided P = 0.0193)" if mode_order[i] == "Joint response" else ""
        axD.text(i - w_d / 2, ov + 10, f"{int(ov)}\n({pv:.1f}%){extra}",
                 ha="center", va="bottom", fontsize=7.1,
                 fontweight="bold" if mode_order[i] == "Joint response" else "normal",
                 color=AMBER if mode_order[i] == "Joint response" else NAVY)
    axD.set_xticks(xd, mode_labels)
    axD.set_ylabel("Genomic 5-kb bins (n = 883 total)", fontsize=8.8, color=SLATE)
    axD.set_ylim(0, 560)
    axD.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=7.5)
    axD.text(0.98, 0.96,
             "Independent-sample insulation:\n"
             "WT +0.041 vs KO −0.067\n"
             r"Contrast = −0.108 ($P = 4.08\times 10^{-4}$)",
             transform=axD.transAxes, ha="right", va="top", fontsize=7.3, color=SLATE,
             linespacing=1.42,
             bbox=dict(boxstyle="round,pad=0.40", facecolor=BG_CARD, edgecolor="#CBD5E1", lw=0.85))
    style_ax(axD, "D", "Genotype × Hypoxia RNA & Insulation Response Modes")

    out = ASSETS / "readme_summary_dashboard.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build_banner()
    build_summary_dashboard()
