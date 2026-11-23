#!/usr/bin/env python3
"""
s01_build_tables.py — Build the two integrated analysis tables for the NapM
reviewer-response analysis (R1_mechanism_zhenghe).

Units
-----
* Gene-level table   : one row per annotated gene (GTF, H37Rv).
* Bin/region table   : one row per 5 kb bin (883 bins, chromosome "Chromosome").

Gene -> bin mapping (pre-registered): a gene is assigned to the bin containing
its transcription start site (TSS; strand-aware). Structural values of a bin
shared by several genes are annotated on each gene but are NEVER treated as
independent observations: all structure-level statistics use bins as units.

Pre-registered parameters
-------------------------
BIN                = 5000 bp
INSULATION_WINDOW  = 5 bins (25 kb) primary; sensitivity 2 (10 kb) and 10 (50 kb)
LONG_RANGE         = >100 kb circular distance
EQ_INS             = 0.25  (log2 insulation equivalence bound)
RNA detected       = BH-FDR < 0.05 and |log2FC| >= 1 (limma tables)
RNA supports-small = BH-FDR >= 0.05 and |log2FC| < 0.5 (heuristic equivalence,
                     because the provided limma tables carry no SE/padj)
ChIP occupancy     = 10-bp signal >= Q3 + 1.5*IQR of log10(signal+1),
                     runs merged across <=100 bp gaps, regions >= 50 bp;
                     a bin is "bound" if >=50% of it is covered by regions.
                     Sensitivity: 99th-percentile threshold.
Random seed        = 42

Environment: hic_review/.venv_cooltools (cooler 0.10.4, cooltools 0.7.1,
numpy 1.26.4, pandas 2.2.2, scipy 1.13.1).
"""
import json
import re
import numpy as np
import pandas as pd
import cooler
from scipy import stats

RNG = np.random.default_rng(42)

ROOT = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review"
DATA = f"{ROOT}/review_data"
OUT = f"{ROOT}/review_results/R1_mechanism_zhenghe"
TAB = f"{OUT}/tables"

BIN = 5000
N_BINS = 883
GENOME_LEN = N_BINS * BIN
W_PRIMARY = 5
W_SENS = [2, 10]
LR_DIST = 20            # bins, 100 kb
EQ_INS = 0.25
COND_FILES = {
    "WT": "cool_file/wt_5k.cool",
    "KO": "cool_file/napm_ko_5k.cool",
    "hypoxia": "cool_file/hypoxia_5k.mcool",
    "latent": "cool_file/latent_5k.mcool",
    "KO_hypoxia": "cool_file/napmhypoxia_5k.mcool",
}
REP_FILES = {
    "WT": [f"wt_rep_icedmatrix/wt_{i}_5000_iced.matrix" for i in range(1, 5)],
    "KO": [f"wt_rep_icedmatrix/napm_ko_{i}_5000_iced.matrix" for i in range(1, 5)],
    "hypoxia": [f"wt_rep_icedmatrix/hypoxia_{i}_5000_iced.matrix" for i in range(1, 5)],
    "latent": [f"wt_rep_icedmatrix/latent_{i}_5000_iced.matrix" for i in range(1, 5)],
}


# ---------------------------------------------------------------- utilities
def bh_fdr(p):
    p = np.asarray(p, float)
    n = np.isfinite(p).sum()
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def load_dense_cool(uri):
    c = cooler.Cooler(uri)
    b = c.bins()[:]
    w = b["weight"].values.astype(float)
    mat = c.matrix(balance=False, sparse=False)[:].astype(float)
    bal = mat * w[:, None] * w[None, :]
    bal[~np.isfinite(bal)] = np.nan
    return bal, w


def load_dense_iced(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["b1", "b2", "v"])
    mat = np.zeros((N_BINS, N_BINS))
    i = df.b1.values - 1
    j = df.b2.values - 1
    mat[i, j] = df.v.values
    mat[j, i] = df.v.values
    return mat


def circ_diamond_insulation(mat, w):
    """Circular diamond insulation score, log2 vs genome-wide median.

    score_i = mean contacts between the w bins up- and downstream of bin i
    (wrapping around the circular chromosome); NaN-robust.
    """
    n = mat.shape[0]
    score = np.full(n, np.nan)
    for i in range(n):
        up = [(i - w + 1 + k) % n for k in range(w)]
        dn = [(i + 1 + k) % n for k in range(w)]
        sub = mat[np.ix_(up, dn)]
        if np.isfinite(sub).sum() >= 0.5 * sub.size:
            score[i] = np.nanmean(sub)
    med = np.nanmedian(score)
    return np.log2(score / med)


def long_range_fraction(mat, dist_bins=LR_DIST):
    """Per-bin fraction of contacts beyond dist_bins (circular distance)."""
    n = mat.shape[0]
    idx = np.arange(n)
    D = np.abs(idx[:, None] - idx[None, :])
    D = np.minimum(D, n - D)
    lr_mask = D > dist_bins
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.nansum(np.where(lr_mask, mat, 0.0), axis=1)
        tot = np.nansum(np.where(np.isfinite(mat), mat, 0.0), axis=1)
    out = np.where(tot > 0, lr / tot, np.nan)
    return out


def classify_3way(est, lo, hi, bound):
    """detected increase / detected decrease / supports small / cannot determine."""
    if not np.isfinite(est) or not np.isfinite(lo) or not np.isfinite(hi):
        return "cannot_determine"
    if lo > bound:
        return "detected_increase"
    if hi < -bound:
        return "detected_decrease"
    if lo >= -bound and hi <= bound:
        return "supports_small_change"
    return "cannot_determine"


# ------------------------------------------------------------------- genes
def parse_gtf():
    rows = []
    for line in open(f"{ROOT}/annotation.gtf"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        attr = f[8]
        gid = re.search(r'gene_id "([^"]+)"', attr).group(1)
        gn = re.search(r'gene_name "([^"]+)"', attr)
        rows.append(dict(gene_id=gid,
                         gene_name=gn.group(1) if gn else "",
                         start=int(f[3]), end=int(f[4]), strand=f[6]))
    g = pd.DataFrame(rows)
    g["tss"] = np.where(g.strand == "+", g.start, g.end)
    g["bin_tss"] = np.clip((g.tss - 1) // BIN, 0, N_BINS - 1).astype(int)
    return g


# -------------------------------------------------------------------- ChIP
def load_chip():
    bg = pd.read_csv(f"{DATA}/chip_seq_data/Rv0047c.bedgraph", sep="\t",
                     header=None, names=["chrom", "start", "end", "val"])
    L = int(bg.end.max())
    sig = np.zeros(L)
    # bedgraph is contiguous 10-bp; fill via searchsorted-safe assignment
    st = bg.start.values
    sig[st] = bg.val.values  # 10-bp resolution grid aligned at multiples of 10
    return sig, L


def call_bound_regions(sm, thresh, min_len=200):
    above = sm >= thresh
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(sm)]
    regions = list(zip(starts, ends))
    # merge gaps <= 100 bp
    merged = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= 100:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged = [(s, e) for s, e in merged if e - s >= min_len]
    return merged


def bin_chip(sm, sig, regions):
    """Per-bin mean smoothed signal; bound = bin overlaps >=1 called region.

    Rationale: the ChIP-seq track is dominated by narrow (~300 bp) peaks, so a
    fractional-coverage rule at 5-kb resolution would discard nearly all real
    occupancy. Peak-overlap is the pre-registered bin-level rule.
    """
    bin_mean = np.array([sm[i * BIN:(i + 1) * BIN].mean() if (i + 1) * BIN <= len(sm)
                         else sm[i * BIN:].mean() for i in range(N_BINS)])
    bound = np.zeros(N_BINS, bool)
    cov = np.zeros(N_BINS)
    for s, e in regions:
        b0, b1 = s // BIN, min((e - 1) // BIN, N_BINS - 1)
        for b in range(b0, b1 + 1):
            ov = min(e, (b + 1) * BIN) - max(s, b * BIN)
            cov[b] += max(ov, 0)
            bound[b] = True
    return bin_mean, bound, cov / BIN


# --------------------------------------------------------------------- RNA
def load_degs():
    out = {}
    for name, fn in [("KOvsWT", "KOvsWT_supo_DEG_limma.csv"),
                     ("hypoxiavsWT", "hypoxiaWTvsWT_supo_DEG_limma.csv"),
                     ("latentvsWT", "LatentvsWT_supo_DEG_limma.csv"),
                     ("KOhypoxiavsWT", "hypoxia_KOvsWT_supo_DEG_limma.csv")]:
        d = pd.read_csv(f"{DATA}/rnaseq/{fn}", index_col=0)
        d = d[["avg_log2FC", "p_val"]].rename(
            columns={"avg_log2FC": f"log2FC_{name}", "p_val": f"p_{name}"})
        d[f"padj_{name}"] = bh_fdr(d[f"p_{name}"].values)
        out[name] = d
    deg = pd.concat(out.values(), axis=1)
    deg.index.name = "gene_id"
    return deg


def rna_class(row, name):
    fc, q = row[f"log2FC_{name}"], row[f"padj_{name}"]
    if not np.isfinite(fc) or not np.isfinite(q):
        return "cannot_determine"
    if q < 0.05 and abs(fc) >= 1:
        return "detected_increase" if fc > 0 else "detected_decrease"
    if q >= 0.05 and abs(fc) < 0.5:
        return "supports_small_change"
    return "cannot_determine"


def load_counts_and_interaction():
    """log2CPM counts + genotype x condition interaction model (OLS, numpy).

    Design (8 samples): intercept, genotype (KO=1), condition (hypoxia=1),
    interaction. Residual df = 4 (2 reps x 4 groups; no variance moderation,
    which is conservative and documented as a limitation).
    """
    m = pd.read_csv(f"{DATA}/rnaseq/AKR_RNA_seq_merged.csv")
    mapping = json.load(open(f"{TAB}/ebg_to_rv_mapping_raw.json"))
    midx = np.array(mapping["merged_idx"])
    rv = np.array(mapping["rv"])

    cnt = m[["WT-rep1", "WT-rep2", "NapM-KO-rep1", "NapM-KO-rep2",
             "Hypoxia-rep1", "Hypoxia-rep2",
             "NapM_Ko-Hypoxia-rep1", "NapM_Ko-Hypoxia-rep2"]].values.astype(float)
    lib = cnt.sum(axis=0)
    logcpm = np.log2(cnt / lib * 1e6 + 0.5)
    # order samples: WT base, KO base, WT hyp, KO hyp
    cols = {"WT": [0, 1], "KO": [2, 3], "hypoxia": [4, 5], "KO_hypoxia": [6, 7]}

    X = np.array([[1, 0, 0, 0], [1, 0, 0, 0],
                  [1, 1, 0, 0], [1, 1, 0, 0],
                  [1, 0, 1, 0], [1, 0, 1, 0],
                  [1, 1, 1, 1], [1, 1, 1, 1]], float)
    XtXinv = np.linalg.inv(X.T @ X)
    # validated genes only (KO-count validated mapping)
    val = json.load(open(f"{TAB}/ebg_map_validation.json"))
    valid = np.array(val["valid_mask"])
    rows = []
    df_resid = 4
    for gi in np.where(valid)[0]:
        y = logcpm[midx[gi]]
        beta = XtXinv @ X.T @ y
        resid = y - X @ beta
        s2 = (resid @ resid) / df_resid
        se = np.sqrt(np.diag(XtXinv) * s2)
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), df_resid)
        # KO hypoxia response = b_cond + b_int
        c_vec = np.array([0, 0, 1, 1])
        b_kr = c_vec @ beta
        se_kr = np.sqrt(c_vec @ XtXinv @ c_vec * s2)
        rows.append(dict(gene_id=rv[gi],
                         base_mean_logcpm=y[[0, 1]].mean(),
                         ko_effect_base=beta[1], se_ko=se[1], p_ko=p[1],
                         hyp_response_wt=beta[2], se_hyp=se[2], p_hyp=p[2],
                         interaction=beta[3], se_int=se[3], p_int=p[3],
                         hyp_response_ko=b_kr, se_hyp_ko=se_kr,
                         p_hyp_ko=2 * stats.t.sf(abs(b_kr / se_kr), df_resid)))
    inter = pd.DataFrame(rows)
    ren = {"p_ko": "padj_ko", "p_hyp": "padj_hyp",
           "p_int": "padj_int", "p_hyp_ko": "padj_hyp_ko"}
    for c, cn in ren.items():
        inter[cn] = bh_fdr(inter[c].values)
    return logcpm, midx, rv, valid, cols, inter


# --------------------------------------------------------------------- main
def main():
    unanalyzable = []

    # ---- genes
    genes = parse_gtf()
    print(f"genes: {len(genes)}")

    # ---- ChIP occupancy
    sig, L = load_chip()
    k1k = np.ones(100) / 100          # 1-kb rolling mean (100 x 10-bp)
    sm = np.convolve(sig, k1k, mode="same")
    thr = np.quantile(sm, 0.99)       # primary: top 1% of smoothed signal
    thr_alt = np.quantile(sm, 0.975)  # sensitivity threshold
    regions = call_bound_regions(sm, thr)
    regions_alt = call_bound_regions(sm, thr_alt)
    bin_chip_mean, bound, bound_frac = bin_chip(sm, sig, regions)
    _, bound_alt, _ = bin_chip(sm, sig, regions_alt)
    print(f"ChIP smoothed 1kb; threshold {thr:.0f} (alt {thr_alt:.0f}); "
          f"{len(regions)} regions; {bound.sum()} bound bins "
          f"(alt {bound_alt.sum()})")

    reg_tbl = pd.DataFrame(regions, columns=["start", "end"])
    reg_tbl["length_bp"] = reg_tbl.end - reg_tbl.start
    reg_tbl["bin_start"] = reg_tbl.start // BIN
    reg_tbl["bin_end"] = (reg_tbl.end - 1) // BIN
    reg_tbl["center"] = (reg_tbl.start + reg_tbl.end) // 2
    reg_tbl["region_id"] = [f"NR{i:03d}" for i in range(len(reg_tbl))]
    reg_tbl.to_csv(f"{TAB}/napm_bound_regions.csv", index=False)

    # ---- Hi-C merged matrices
    mats, weights = {}, {}
    for cond, fn in COND_FILES.items():
        uri = f"{DATA}/{fn}::/resolutions/5000"
        mats[cond], weights[cond] = load_dense_cool(uri)
        bad = (~np.isfinite(weights[cond])).sum()
        if bad:
            unanalyzable.append(dict(item=f"{cond} bins with NaN balance weight",
                                     n=int(bad), reason="failed ICE convergence"))
    # ---- Hi-C replicates (iced)
    rep_mats = {c: [load_dense_iced(f"{DATA}/{f}") for f in fs]
                for c, fs in REP_FILES.items()}
    unanalyzable.append(dict(
        item="KO-hypoxia Hi-C replicates", n=4,
        reason="no independent biological replicates exist; only the merged "
               "mcool is available, so genotype x condition interaction for "
               "Hi-C can only be reported as effect size + description"))

    # ---- insulation & long-range, merged + replicates
    ins = {c: circ_diamond_insulation(m, W_PRIMARY) for c, m in mats.items()}
    ins_sens = {w: {c: circ_diamond_insulation(m, w) for c, m in mats.items()}
                for w in W_SENS}
    lr = {c: long_range_fraction(m) for c, m in mats.items()}
    ins_rep = {c: [circ_diamond_insulation(m, W_PRIMARY) for m in ms]
               for c, ms in rep_mats.items()}
    lr_rep = {c: [long_range_fraction(m) for m in ms]
              for c, ms in rep_mats.items()}

    def pair_ci(a_reps, b_reps):
        """all pairwise differences; 5-95 percentile as 90% CI"""
        ds = np.array([[b - a for b in b_reps] for a in a_reps])  # (4,4,n)
        lo = np.nanpercentile(ds, 5, axis=(0, 1))
        hi = np.nanpercentile(ds, 95, axis=(0, 1))
        pt = np.nanmedian(ds, axis=(0, 1))
        return pt, lo, hi

    d_ins_pt, d_ins_lo, d_ins_hi = pair_ci(ins_rep["WT"], ins_rep["KO"])
    ins_class = [classify_3way(e, l, h, EQ_INS)
                 for e, l, h in zip(d_ins_pt, d_ins_lo, d_ins_hi)]
    d_lr_pt, d_lr_lo, d_lr_hi = pair_ci(lr_rep["WT"], lr_rep["KO"])

    # condition effects (hypoxia direction)
    d_hyp_wt_pt, d_hyp_wt_lo, d_hyp_wt_hi = pair_ci(ins_rep["WT"], ins_rep["hypoxia"])
    # KO hypoxia response: merged KO_hypoxia minus each KO replicate (no KO-hyp reps)
    d_hyp_ko_pt = ins["KO_hypoxia"] - np.nanmedian(np.array(ins_rep["KO"]), axis=0)
    hrep = np.array(ins_rep["hypoxia"]); wrep = np.array(ins_rep["WT"])
    inter_pt = d_hyp_ko_pt - d_hyp_wt_pt
    # CI for the WT-side hypoxia response only (KO-hyp has no replicates)
    inter_lo = d_hyp_ko_pt - d_hyp_wt_hi
    inter_hi = d_hyp_ko_pt - d_hyp_wt_lo

    # ---- RNA
    deg = load_degs()
    for nm in ["KOvsWT", "hypoxiavsWT", "latentvsWT", "KOhypoxiavsWT"]:
        deg[f"class_{nm}"] = [rna_class(r, nm) for _, r in deg.iterrows()]
    logcpm, midx, rv_ids, valid, cols, inter = load_counts_and_interaction()
    inter.to_csv(f"{TAB}/rna_interaction_model.csv", index=False)

    # expression baseline (WT mean log2cpm) for mapped genes
    expr_wt = pd.Series(logcpm[midx][:, cols["WT"]].mean(axis=1), index=rv_ids)
    expr_wt_valid = expr_wt[valid]

    # ---- bin-level table
    btab = pd.DataFrame(dict(
        bin=np.arange(N_BINS),
        start=np.arange(N_BINS) * BIN,
        end=(np.arange(N_BINS) + 1) * BIN))
    btab["weight_WT"] = weights["WT"]
    btab["weight_KO"] = weights["KO"]
    btab["chip_mean_signal"] = bin_chip_mean
    btab["chip_bound"] = bound
    btab["chip_bound_alt"] = bound_alt
    btab["bound_region_frac"] = bound_frac
    btab["ins_WT"] = ins["WT"]
    btab["ins_KO"] = ins["KO"]
    btab["ins_hypoxia"] = ins["hypoxia"]
    btab["ins_latent"] = ins["latent"]
    btab["ins_KO_hypoxia"] = ins["KO_hypoxia"]
    for w in W_SENS:
        btab[f"ins_WT_w{w}"] = ins_sens[w]["WT"]
        btab[f"ins_KO_w{w}"] = ins_sens[w]["KO"]
        btab[f"d_ins_KO_WT_w{w}"] = ins_sens[w]["KO"] - ins_sens[w]["WT"]
    btab["d_ins_KO_WT"] = d_ins_pt
    btab["d_ins_KO_WT_lo"] = d_ins_lo
    btab["d_ins_KO_WT_hi"] = d_ins_hi
    btab["ins_change_class"] = ins_class
    btab["lrfrac_WT"] = lr["WT"]
    btab["lrfrac_KO"] = lr["KO"]
    btab["d_lrfrac_KO_WT"] = d_lr_pt
    btab["d_lrfrac_KO_WT_lo"] = d_lr_lo
    btab["d_lrfrac_KO_WT_hi"] = d_lr_hi
    btab["d_ins_hyp_WT"] = d_hyp_wt_pt
    btab["d_ins_hyp_WT_lo"] = d_hyp_wt_lo
    btab["d_ins_hyp_WT_hi"] = d_hyp_wt_hi
    btab["d_ins_hyp_KO"] = d_hyp_ko_pt
    btab["d_ins_interaction"] = inter_pt
    btab["d_ins_interaction_lo"] = inter_lo
    btab["d_ins_interaction_hi"] = inter_hi

    # replicate-level sample points (for plotting)
    for c in ["WT", "KO", "hypoxia", "latent"]:
        for r in range(4):
            btab[f"ins_{c}_rep{r+1}"] = ins_rep[c][r]
            btab[f"lrfrac_{c}_rep{r+1}"] = lr_rep[c][r]

    # ---- RNA summaries per bin (genes assigned by TSS bin)
    g2 = genes.merge(deg, left_on="gene_id", right_index=True, how="left")
    bin_rna = []
    for b in range(N_BINS):
        sub = g2[g2.bin_tss == b]
        fc = sub["log2FC_KOvsWT"].dropna()
        det = sub["class_KOvsWT"]
        bin_rna.append(dict(
            bin=b, n_genes=len(sub),
            rna_mean_log2FC_KOvsWT=fc.mean() if len(fc) else np.nan,
            rna_mean_abs_log2FC_KOvsWT=fc.abs().mean() if len(fc) else np.nan,
            n_rna_up=int((det == "detected_increase").sum()),
            n_rna_down=int((det == "detected_decrease").sum()),
            n_rna_small=int((det == "supports_small_change").sum()),
            n_rna_undet=int((det == "cannot_determine").sum())))
    brna = pd.DataFrame(bin_rna)
    btab = btab.merge(brna, on="bin")

    # baseline expression per bin (mean WT log2cpm of genes with TSS in bin)
    gexp = genes.copy()
    gexp["expr_wt"] = gexp.gene_id.map(expr_wt_valid)
    bexp = gexp.groupby("bin_tss")["expr_wt"].mean().rename("expr_wt_mean")
    btab = btab.merge(bexp, left_on="bin", right_index=True, how="left")

    # ---- matched background (greedy NN without replacement)
    covars = ["weight_WT", "ins_WT", "expr_wt_mean"]
    Z = btab[covars].copy()
    for c in covars:
        Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
    Z = Z.fillna(Z.median())
    bound_idx = np.where(btab.chip_bound.values)[0]
    pool = [i for i in range(N_BINS)
            if not btab.chip_bound.values[i] and i not in bound_idx]
    Zv = Z.values
    matched = {}
    pool_set = set(pool)
    for b in bound_idx:
        best, bd = None, np.inf
        for i in pool_set:
            d = np.sum((Zv[b] - Zv[i]) ** 2)
            if d < bd:
                bd, best = d, i
        if best is not None:
            matched[b] = best
            pool_set.discard(best)
    btab["matched_background"] = False
    btab["matched_to_bin"] = np.nan
    for b, i in matched.items():
        btab.loc[btab.bin == i, "matched_background"] = True
        btab.loc[btab.bin == b, "matched_to_bin"] = i
    print(f"matched background: {len(matched)} pairs of {len(bound_idx)} bound bins")

    btab.to_csv(f"{TAB}/bin_level_integrated.csv", index=False)

    # ---- gene-level table
    gtab = g2.copy()
    gtab["n_genes_share_bin"] = gtab.groupby("bin_tss")["gene_id"].transform("count")
    def _safe_mean(a):
        return a.mean() if len(a) else np.nan
    gtab["chip_mean_gene"] = [
        _safe_mean(sm[max(s - 1, 0):e]) for s, e in zip(gtab.start, gtab.end)]
    gtab["chip_mean_upstream500"] = [
        _safe_mean(sm[max(t - 501, 0):t - 1]) if st == "+"
        else _safe_mean(sm[t:min(t + 500, len(sm))])
        for t, st in zip(gtab.tss, gtab.strand)]
    bcols = btab.set_index("bin")
    gtab["bin_chip_bound"] = gtab.bin_tss.map(bcols.chip_bound)
    gtab["bin_d_ins_KO_WT"] = gtab.bin_tss.map(bcols.d_ins_KO_WT)
    gtab["bin_ins_change_class"] = gtab.bin_tss.map(bcols.ins_change_class)
    gtab = gtab.merge(inter[["gene_id", "ko_effect_base", "p_ko", "padj_ko",
                             "hyp_response_wt", "p_hyp", "padj_hyp",
                             "interaction", "p_int", "padj_int",
                             "hyp_response_ko", "p_hyp_ko", "padj_hyp_ko"]],
                      on="gene_id", how="left")
    gtab.to_csv(f"{TAB}/gene_level_integrated.csv", index=False)

    # ---- unanalyzable
    n_unmapped = int((~valid).sum())
    unanalyzable.append(dict(
        item="genes in interaction model", n=n_unmapped,
        reason="EBG<->Rv identifier mapping could not be validated with "
               "independent KO counts (low/duplicated count vectors)"))
    n_nogenes = int((btab.n_genes == 0).sum())
    unanalyzable.append(dict(
        item="bins without RNA summary", n=n_nogenes,
        reason="no annotated TSS inside bin; bin-level RNA summary undefined"))
    chip_miss = ["KO", "hypoxia", "latent", "KO-hypoxia"]
    unanalyzable.append(dict(
        item=f"ChIP-seq in conditions {chip_miss}", n=4,
        reason="only baseline WT ChIP-seq exists (single sample, no input); "
               "occupancy is a baseline covariate, no condition comparison possible"))
    pd.DataFrame(unanalyzable).to_csv(f"{TAB}/unanalyzable_items.csv", index=False)

    # summary
    print("\n=== summary ===")
    print("ins change classes:", pd.Series(ins_class).value_counts().to_dict())
    print("RNA KOvsWT classes:", deg.class_KOvsWT.value_counts().to_dict())
    print("interaction genes:", len(inter),
          "FDR<0.05:", int((inter.padj_int < 0.05).sum()))
    print("tables written to", TAB)


if __name__ == "__main__":
    main()
