#!/usr/bin/env python3
"""fc.py — shared style + data loading for R1_mechanism_reanalysis/07_figures.

Reads only pre-computed tables in RA/01..06 (no statistics re-run).
Seed 42. Palette per specification.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review"
RA = f"{ROOT}/hic_review/review_results/R1_mechanism_reanalysis"
OUT = f"{RA}/07_figures"
FIG = f"{OUT}/figures"
SD = f"{OUT}/source_data"
COOL = f"{ROOT}/hic_review/review_data/cool_file"
GTF = f"{ROOT}/hic_review/annotation.gtf"
N_BINS, BIN, GENOME = 883, 5000, 4411532

# palette (fixed by specification)
C = dict(WT="#355F7A", KO="#A5574D", chip="#665A7A", hypoxia="#4F7C73",
         latent="#8A7567", ns="#C6C8CA", bg="#E5E5E3", text="#2A2A2A",
         matched="#9A9D9F", light="#D9DBDC", pos="#B2182B", neg="#2166AC")
CMAP_DIV = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#2166AC", "#F7F7F7", "#B2182B"])
CMAP_DIV.set_bad("#BFBFBF")  # masked/NaN grey, distinct from true zero (white)
CMAP_HIC = mcolors.LinearSegmentedColormap.from_list(
    "hic", ["#FFF5F0", "#FB6A4A", "#99000D"])
CMAP_HIC.set_bad("#BFBFBF")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "text.color": C["text"], "axes.labelcolor": C["text"],
    "xtick.color": C["text"], "ytick.color": C["text"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False})
MM = 1 / 25.4
FIGW = 180 * MM       # double column
FIGW1 = 85 * MM       # single column


def panel_label(ax, s, x=-0.14, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)


class SourceData:
    """Collect one DataFrame per panel; write a multi-sheet xlsx."""
    def __init__(self, figname):
        self.path = f"{SD}/{figname}_source_data.xlsx"
        self.sheets = {}

    def add(self, panel, df):
        self.sheets[panel] = df

    def write(self):
        os.makedirs(SD, exist_ok=True)
        with pd.ExcelWriter(self.path, engine="openpyxl") as xw:
            for panel, df in self.sheets.items():
                df.to_excel(xw, sheet_name=panel[:31], index=False)
        print("wrote", self.path)


def load_tables():
    d = {}
    d["b"] = pd.read_csv(f"{RA}/03_structure_metrics/bin_level_v2.csv")
    d["rna_bin"] = pd.read_csv(f"{RA}/04_rna_integration/bin_level_rna_v2.csv")
    d["genes"] = pd.read_csv(f"{RA}/04_rna_integration/gene_level_rna_v2.csv")
    d["peaks"] = pd.read_csv(f"{RA}/01_chip_curated/NapM_binding_curated.tsv", sep="\t")
    d["loops"] = pd.read_csv(f"{RA}/02_loop_curated/loops_41_curated.tsv", sep="\t")
    d["cid"] = pd.read_csv(f"{RA}/02_loop_curated/wt_cid_boundaries_curated.tsv", sep="\t")
    d["pairs"] = pd.read_csv(f"{RA}/05_matched_controls/matched_pairs.csv")
    d["gpairs"] = pd.read_csv(f"{RA}/05_matched_controls/matched_pairs_genes.csv")
    d["meta"] = pd.read_csv(f"{RA}/06_permutation_tests/fig1b_metaprofile.csv")
    d["stats13"] = pd.read_csv(f"{RA}/06_permutation_tests/stats_fig1_fig3_summary.csv")
    d["stats2"] = pd.read_csv(f"{RA}/06_permutation_tests/stats_fig2_summary.csv")
    d["classes"] = pd.read_csv(f"{RA}/06_permutation_tests/stats_fig2_classes.csv")
    return d


def stat(stats_df, test):
    """Return the summary row for a test name."""
    r = stats_df[stats_df["test"] == test]
    assert len(r) == 1, test
    return r.iloc[0]


def load_genes_gtf():
    import re
    rows = []
    for line in open(GTF):
        if line.startswith("#"):
            continue
        f = line.rstrip().split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        gid = re.search(r'gene_id "([^"]+)"', f[8]).group(1)
        rows.append(dict(gene_id=gid, start=int(f[3]), end=int(f[4]),
                         strand=f[6]))
    return pd.DataFrame(rows)


def oe_triangle(clr, i0, i1, maxdist=None):
    """45-degree rotated upper-triangle O/E for bins [i0,i1) of a cooler.
    Expected = mean of each diagonal computed genome-wide (plotting only).
    Returns masked image and extent in Mb."""
    import cooler
    bal = clr.bins()[:]["weight"].values
    mat = clr.matrix(balance=True, sparse=False)[:, :]
    mat[~np.isfinite(mat)] = np.nan
    n = mat.shape[0]
    idx = np.arange(n)
    exp = np.full(n, np.nan)
    for off in range(n):
        v = np.diag(mat, off)
        v = v[np.isfinite(v) & (v > 0)]
        exp[off] = v.mean() if len(v) else np.nan
    i, j = np.meshgrid(idx, idx, indexing="ij")
    dist = np.abs(j - i)
    with np.errstate(invalid="ignore", divide="ignore"):
        oe = np.where((mat > 0) & np.isfinite(mat) & (exp[dist] > 0),
                      mat / exp[dist], np.nan)
    sub = oe[i0:i1, i0:i1]
    m = sub.shape[0]
    img = np.full((m, 2 * m - 1), np.nan)
    for a in range(m):
        for bb in range(a, m):
            if maxdist is not None and (bb - a) > maxdist:
                continue
            img[bb - a, a + bb] = sub[a, bb]
    filled = img.copy()
    for yy in range(m):
        row = img[yy]
        for xx in range(2 * m - 1):
            if not np.isfinite(row[xx]):
                nb = [row[xx + d] for d in (-1, 1)
                      if 0 <= xx + d < 2 * m - 1 and np.isfinite(row[xx + d])]
                if nb:
                    filled[yy, xx] = np.mean(nb)
    ext = [i0 * BIN / 1e6, (i1 - 1) * BIN / 1e6, 0, (m - 1) * BIN / 1e6]
    return np.ma.masked_invalid(filled), ext
