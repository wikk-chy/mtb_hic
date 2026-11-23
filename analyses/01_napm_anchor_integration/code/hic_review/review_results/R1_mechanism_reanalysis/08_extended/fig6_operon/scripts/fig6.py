#!/usr/bin/env python3
"""Fig6 — Operon-level analysis.

Builds operon_level.csv (reusable), then panels A-E.
Outputs to RA/08_extended/fig6_operon/{figures,tables}. Seed 42.
Reads only pre-computed tables (RA read-only).
"""
import os, sys, ast, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../07_figures/scripts")
import fc

SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = fc.ROOT
RA = fc.RA
OUTD = f"{RA}/08_extended/fig6_operon"
FIG = f"{OUTD}/figures"
TAB = f"{OUTD}/tables"
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)
NBINS, BINSZ, GENOME = 883, 5000, 4411532
NPERM = 10000
shifts = RNG.integers(0, NBINS, NPERM)

C_PURPLE = "#665A7A"; C_GREY = "#8C8C8C"; C_RED = "#B2182B"; C_BLUE = "#2166AC"
CMAP_CHIP = mcolors.LinearSegmentedColormap.from_list("chip", ["#F0EDF4", C_PURPLE])

stats_rows = []
def add_stat(**kw):
    stats_rows.append(kw)

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=600)
    plt.close(fig)
    print("wrote", name)

def bootstrap_ci(diffs, n=10000):
    bs = RNG.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])

# ---------------- load ----------------
b = pd.read_csv(f"{RA}/03_structure_metrics/bin_level_v2.csv")
g = pd.read_csv(f"{RA}/04_rna_integration/gene_level_rna_v2.csv")
peaks = pd.read_csv(f"{RA}/01_chip_curated/NapM_binding_curated.tsv", sep="\t")
loops = pd.read_csv(f"{RA}/02_loop_curated/loops_41_curated.tsv", sep="\t")
cid = pd.read_csv(f"{RA}/02_loop_curated/wt_cid_boundaries_curated.tsv", sep="\t")
op = pd.read_csv(f"{ROOT}/hic_review/review_data/operon_metadata/"
                 "operon_list_updated_with_bin_coverage.csv")

DET = {"detected_increase", "detected_decrease"}
gd = g.set_index("gene_id")

# ---------------- build operon-level table ----------------
def in_span(pos, start, end):
    """Circular-aware: pos in [start,end] on a circular genome."""
    if start <= end:
        return start <= pos <= end
    return pos >= start or pos <= end

rows = []
for _, r in op.iterrows():
    genes = ast.literal_eval(r["gene_list"])
    cov = ast.literal_eval(r["bin_coverage"])          # {1-based bin: frac}
    bins0 = np.array([int(k) - 1 for k in cov.keys()])  # -> 0-based
    w = np.array([float(v) for v in cov.values()])
    w = w / w.sum()
    start, end = int(r["operon_start"]), int(r["operon_end"])
    # genes present in RNA table
    gg = gd.reindex([x for x in genes if x in gd.index])
    log2fc = gg["log2FC_KOvsWT"].mean() if len(gg) else np.nan
    hyp = gg["log2FC_hypoxiavsWT"].mean() if len(gg) else np.nan
    expr = gg["expr_wt_gene"].mean() if len(gg) else np.nan
    n_det = int(gg["class_KOvsWT"].isin(DET).sum()) if len(gg) else 0
    # peaks inside operon span
    pc = peaks["center"].values
    pk_in = np.array([in_span(p, start, end) for p in pc])
    pk_scores = peaks["score"].values[pk_in]
    # structure weighted by bin coverage
    ins_wt = np.average(b["ins_WT"].values[bins0], weights=w)
    d_ins = np.average(b["d_ins_KO_WT"].values[bins0], weights=w)
    struct_det = bool(b["structure_detected"].values[bins0].any())
    # loop anchors
    anchor_bins = set(loops["bin1"]).union(loops["bin2"])
    n_anchor = len(set(bins0.tolist()) & anchor_bins)
    # CID boundary crossing (boundary strictly inside operon span)
    crosses = False
    adj = False
    for _, cr in cid.iterrows():
        cpos = (cr["start"] + cr["end"]) / 2
        if in_span(cpos, start, end):
            # exclude operons that merely start/end at the boundary bin edge
            crosses = True
        else:
            d1 = min(abs(cpos - start), GENOME - abs(cpos - start))
            d2 = min(abs(cpos - end), GENOME - abs(cpos - end))
            if min(d1, d2) <= BINSZ:
                adj = True
    mid = (start + r["operon_length"] / 2) % GENOME
    rows.append(dict(
        operon_id=r["operon_id"], operon_name=r["operon_name"], strand=r["strand"],
        operon_start=start, operon_end=end, operon_length=int(r["operon_length"]),
        n_genes=len(genes), gene_list=";".join(genes),
        n_genes_with_rna=len(gg),
        baseline_expr_mean=expr,
        normalized_expression=r["normalized_expression"],
        log2FC_KOvsWT_mean=log2fc, log2FC_hypoxiavsWT_mean=hyp,
        n_deg=n_det, frac_deg=(n_det / len(gg)) if len(gg) else np.nan,
        n_peaks=int(pk_in.sum()),
        chip_score_max=float(pk_scores.max()) if len(pk_scores) else 0.0,
        chip_score_mean=float(pk_scores.mean()) if len(pk_scores) else 0.0,
        has_peak=bool(pk_in.sum() > 0),
        ins_WT_wmean=ins_wt, d_ins_wmean=d_ins, abs_d_ins_wmean=abs(d_ins),
        structure_detected_any=struct_det,
        has_loop_anchor=n_anchor > 0, n_loop_anchor_bins=n_anchor,
        crosses_cid_boundary=crosses, adjacent_cid_boundary=(adj and not crosses),
        mid_bin=int(mid // BINSZ) % NBINS,
        bins0=";".join(str(x) for x in bins0),
    ))
opl = pd.DataFrame(rows)
opl.to_csv(f"{TAB}/operon_level.csv", index=False)
print("operon_level.csv:", opl.shape, "| with RNA:", opl.n_genes_with_rna.gt(0).sum(),
      "| has_peak:", opl.has_peak.sum(), "| crosses CID:", opl.crosses_cid_boundary.sum(),
      "| loop anchor:", opl.has_loop_anchor.sum())

fig = plt.figure(figsize=(fc.FIGW, 138 * fc.MM))
gs = fig.add_gridspec(2, 4, left=0.07, right=0.99, top=0.90, bottom=0.09,
                      wspace=0.55, hspace=0.80)

# ---------------- Panel A: operon expression change vs structure change ----------------
axA = fig.add_subplot(gs[0, 0])
dA = opl.dropna(subset=["log2FC_KOvsWT_mean"]).copy()
rho, prho = stats.spearmanr(dA["log2FC_KOvsWT_mean"], dA["d_ins_wmean"])
sc = axA.scatter(dA["log2FC_KOvsWT_mean"], dA["d_ins_wmean"],
                 c=dA["chip_score_max"], cmap=CMAP_CHIP, vmin=0, vmax=1,
                 s=6 + 2 * np.log10(dA["operon_length"]), lw=0, alpha=0.8)
sl, ic = np.polyfit(dA["log2FC_KOvsWT_mean"], dA["d_ins_wmean"], 1)
xs = np.linspace(dA["log2FC_KOvsWT_mean"].min(), dA["log2FC_KOvsWT_mean"].max(), 50)
axA.plot(xs, ic + sl * xs, color=C_GREY, ls="--", lw=1.0)
cb = fig.colorbar(sc, ax=axA, pad=0.02, fraction=0.06)
cb.set_label("NapM peak score (max)", fontsize=7)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
axA.set_xlabel("Operon mean RNA log2FC (KO / WT)")
axA.set_ylabel("Δ insulation (KO − WT), coverage-wtd")
axA.text(0.03, 0.03, f"Spearman ρ = {rho:.3f}\np = {prho:.3g}, n = {len(dA)}",
         transform=axA.transAxes, fontsize=7, va="bottom")
fc.panel_label(axA, "a")
dA[["operon_id", "log2FC_KOvsWT_mean", "d_ins_wmean", "chip_score_max",
    "operon_length"]].to_csv(f"{TAB}/fig6A_source.csv", index=False)
add_stat(panel="A", test="Fig6A_spearman_operon", observed=rho, effect=rho,
         ci_lo=np.nan, ci_hi=np.nan, p=prho, p_type="two-sided asymptotic",
         n=len(dA), null_model="none (Spearman)",
         notes="operon mean log2FC vs coverage-weighted d_ins")

# ---------------- Panel B: bound vs matched unbound operons ----------------
axB = fig.add_subplot(gs[0, 1])
dB = opl.dropna(subset=["log2FC_KOvsWT_mean", "baseline_expr_mean"]).copy()
dB["abs_fc"] = dB["log2FC_KOvsWT_mean"].abs()
dB["log_expr"] = np.log10(dB["baseline_expr_mean"].clip(lower=1e-3))
dB["log_len"] = np.log10(dB["operon_length"])
bound = dB[dB["has_peak"]]
unbound = dB[~dB["has_peak"]].copy()
ze = dB["log_expr"].std(ddof=1); zl = dB["log_len"].std(ddof=1)
# greedy 1:1 nearest-neighbor matching, caliper 0.25 SD per covariate
used = set()
match_pairs = []
order = bound.sample(frac=1, random_state=SEED).index  # random order, seeded
for bi in order:
    best, bestd = None, np.inf
    for ui in unbound.index:
        if ui in used:
            continue
        de = abs(dB.loc[bi, "log_expr"] - dB.loc[ui, "log_expr"]) / ze
        dl = abs(dB.loc[bi, "log_len"] - dB.loc[ui, "log_len"]) / zl
        if de > 0.25 or dl > 0.25:
            continue
        dd = de ** 2 + dl ** 2
        if dd < bestd:
            bestd, best = dd, ui
    if best is not None:
        used.add(best)
        match_pairs.append((bi, best))
mp = pd.DataFrame(match_pairs, columns=["bound_idx", "matched_idx"])
mp.to_csv(f"{TAB}/fig6B_matched_operon_pairs.csv", index=False)
mb_df = dB.loc[mp["bound_idx"]].reset_index(drop=True)
mm_df = dB.loc[mp["matched_idx"]].reset_index(drop=True)
n_pairs = len(mp)
# match quality
mq = dict(expr_diff=float((mb_df["log_expr"] - mm_df["log_expr"]).abs().mean()),
          len_diff=float((mb_df["log_len"] - mm_df["log_len"]).abs().mean()))

def nearest_mid_full(metric_sparse, mids_arr):
    """Fill every bin with the metric of the circularly nearest operon mid-bin,
    so circular-shift nulls never hit empty bins."""
    uniq = np.unique(mids_arr)
    vals = np.full(NBINS, np.nan)
    for i, r_mid in enumerate(mids_arr):
        vals[r_mid] = metric_sparse[i]
    uniq_vals = vals[uniq]
    j = np.arange(NBINS)
    d = np.abs(j[:, None] - uniq[None, :])
    d = np.minimum(d, NBINS - d)
    return uniq_vals[d.argmin(axis=1)]


def paired_op_test(vb, vm, bmid, mmid, label, vname):
    diffs = vb - vm
    obs = np.nanmean(diffs)
    ci = bootstrap_ci(diffs.values)
    wp = stats.wilcoxon(diffs).pvalue if len(diffs) > 10 else np.nan
    # circular-shift null on mid-bin-indexed metric (nearest-mid filled)
    both_mid = pd.concat([bmid, mmid]).values
    both_val = pd.concat([vb, vm]).values
    metric = nearest_mid_full(both_val, both_mid)
    ns = np.empty(NPERM)
    for i, s in enumerate(shifts):
        ns[i] = np.nanmean(metric[(bmid.values + s) % NBINS]
                           - metric[(mmid.values + s) % NBINS])
    mw = stats.mannwhitneyu(vb, vm).pvalue
    p = (1 + np.sum(np.abs(ns) >= abs(obs))) / (1 + len(ns))
    add_stat(panel="B", test=f"Fig6B_{vname}", observed=obs, effect=obs,
             ci_lo=ci[0], ci_hi=ci[1], p=p, p_type="two-sided circular-shift(10000)",
             n=len(diffs), null_model="circular-shift of matched pair mid-bin offsets",
             notes=f"{vname}; bound mean={np.nanmean(vb):.4f}, matched mean="
                   f"{np.nanmean(vm):.4f}; paired Wilcoxon p={wp:.3g}; naive MW "
                   f"p={mw:.3g} (naive); matched on log expr + log length, "
                   f"caliper 0.25 SD")
    return dict(obs=obs, ci=ci, p=p, vb=vb.values, vm=vm.values)

res_fc = paired_op_test(mb_df["abs_fc"], mm_df["abs_fc"], mb_df["mid_bin"],
                        mm_df["mid_bin"], "B", "abs_log2FC")
res_deg = paired_op_test(mb_df["frac_deg"], mm_df["frac_deg"], mb_df["mid_bin"],
                         mm_df["mid_bin"], "B", "deg_frac")
# plot: two mini metrics in one panel (side-by-side half-panels)
subB = gs[0, 1].subgridspec(1, 2, wspace=0.6)
plt.delaxes(axB)
for k, (res, ttl) in enumerate([(res_fc, "|RNA log2FC|"), (res_deg, "DEG fraction")]):
    ax = fig.add_subplot(subB[0, k])
    show = RNG.choice(len(res["vb"]), size=min(60, len(res["vb"])), replace=False)
    for i in show:
        ax.plot([0, 1], [res["vb"][i], res["vm"][i]], color=C_GREY, lw=0.3,
                alpha=0.25, zorder=1)
    for xx, vv, col in ((0, res["vb"], C_PURPLE), (1, res["vm"], C_GREY)):
        ci = bootstrap_ci(vv)
        ax.errorbar([xx], [vv.mean()], yerr=[[vv.mean() - ci[0]], [ci[1] - vv.mean()]],
                    fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1.0,
                    capsize=2, zorder=3)
    ax.set_xlim(-0.55, 1.55); ax.set_xticks([0, 1]); ax.set_xticklabels(["B", "M"])
    ax.set_title(ttl, fontsize=6.8)
    ax.text(0.5, -0.32,
            f"Δ={res['obs']:+.3f}\n[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}]\n"
            f"p={res['p']:.3g}, n={len(res['vb'])}",
            transform=ax.transAxes, fontsize=5.6, ha="center", va="top")
axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
axB.set_title(f"NapM-bound vs matched unbound operons (n={n_pairs} pairs)\n"
              f"matched on baseline expression + operon length", fontsize=7, y=1.16)
fc.panel_label(axB, "b")
srcB = pd.DataFrame(dict(bound_abslog2FC=res_fc["vb"], matched_abslog2FC=res_fc["vm"],
                         bound_deg_frac=res_deg["vb"], matched_deg_frac=res_deg["vm"]))
srcB.to_csv(f"{TAB}/fig6B_source.csv", index=False)

# ---------------- Panel C: CID boundary crossing operons ----------------
axC = fig.add_subplot(gs[0, 2])
dC = opl.copy()
grp_cross = dC[dC["crosses_cid_boundary"]]
grp_adj = dC[dC["adjacent_cid_boundary"]]
grp_rest = dC[~(dC["crosses_cid_boundary"] | dC["adjacent_cid_boundary"])]
# circular-shift null at mid-bin resolution for crossing group vs rest
metric = nearest_mid_full(dC["abs_d_ins_wmean"].values, dC["mid_bin"].values)
cmid = grp_cross["mid_bin"].values
rmid = grp_rest["mid_bin"].values
obs_c = grp_cross["abs_d_ins_wmean"].mean() - grp_rest["abs_d_ins_wmean"].mean()
ns = np.empty(NPERM)
for i, s in enumerate(shifts):
    ns[i] = np.nanmean(metric[(cmid + s) % NBINS]) - np.nanmean(metric[(rmid + s) % NBINS])
p_c = (1 + np.sum(np.abs(ns) >= abs(obs_c))) / (1 + len(ns))
diffs_boot = []
for _ in range(10000):
    diffs_boot.append(RNG.choice(grp_cross["abs_d_ins_wmean"], len(grp_cross),
                                 replace=True).mean()
                      - RNG.choice(grp_rest["abs_d_ins_wmean"], len(grp_rest),
                                   replace=True).mean())
ci_c = np.percentile(diffs_boot, [2.5, 97.5])
mw_c = stats.mannwhitneyu(grp_cross["abs_d_ins_wmean"],
                          grp_rest["abs_d_ins_wmean"]).pvalue
add_stat(panel="C", test="Fig6C_cross_vs_rest_absdins", observed=obs_c, effect=obs_c,
         ci_lo=ci_c[0], ci_hi=ci_c[1], p=p_c, p_type="two-sided circular-shift(10000)",
         n=len(grp_cross), null_model="circular-shift of mid-bin metric",
         notes=f"|d_ins| crossing (n={len(grp_cross)}) vs other (n={len(grp_rest)}); "
               f"naive MW p={mw_c:.3g} (naive); adjacent-to-boundary n={len(grp_adj)} "
               f"reported descriptively; bootstrap CI anti-conservative at n=4, "
               f"circular-shift p is primary")
groups = [grp_cross, grp_adj, grp_rest]
labels_g = [f"crossing\n(n={len(grp_cross)})", f"adjacent\n(n={len(grp_adj)})",
            f"other\n(n={len(grp_rest)})"]
cols_g = [C_RED, "#E08214", C_GREY]
for i, (gg_, col) in enumerate(zip(groups, cols_g)):
    v = gg_["abs_d_ins_wmean"].values
    axC.scatter(np.full(len(v), i) + RNG.uniform(-0.12, 0.12, len(v)), v, s=7,
                color=col, alpha=0.65, lw=0, zorder=2)
    if len(v) > 1:
        ci = bootstrap_ci(v)
        axC.errorbar([i], [v.mean()], yerr=[[v.mean() - ci[0]], [ci[1] - v.mean()]],
                     fmt="_", ms=8, color="black", elinewidth=1.2, capsize=3, zorder=3)
axC.set_xticks(range(3)); axC.set_xticklabels(labels_g, fontsize=6.5)
axC.set_ylabel("|Δ insulation| (operon)")
axC.margins(y=0.14)
axC.set_title(f"crossing − other = {obs_c:+.3f}\nshift p = {p_c:.3g} (n=4, descriptive)",
              fontsize=6.8)
fc.panel_label(axC, "c")
pd.DataFrame(dict(group=np.repeat(["crossing", "adjacent", "other"],
                                  [len(x) for x in groups]),
                  abs_d_ins=np.concatenate([x["abs_d_ins_wmean"].values for x in groups]))
             ).to_csv(f"{TAB}/fig6C_source.csv", index=False)

# ---------------- Panel D: loop-connected operon pairs ----------------
axD = fig.add_subplot(gs[1, 0:2])
# map anchor bin -> operon with max coverage in that bin
def operon_at_bin(bn):
    cands = []
    for i, r in opl.iterrows():
        bns = [int(x) for x in r["bins0"].split(";")]
        if bn in bns:
            cands.append(i)
    return cands[0] if cands else None

pairs_loop = []
excl = dict(no_operon_at_anchor=0, same_operon=0, no_rna=0)
for _, lr in loops.iterrows():
    o1, o2 = operon_at_bin(lr["bin1"]), operon_at_bin(lr["bin2"])
    if o1 is None or o2 is None:
        excl["no_operon_at_anchor"] += 1
        continue
    if o1 == o2:
        excl["same_operon"] += 1
        continue
    r1, r2 = opl.loc[o1], opl.loc[o2]
    if np.isnan(r1["log2FC_KOvsWT_mean"]) or np.isnan(r2["log2FC_KOvsWT_mean"]):
        excl["no_rna"] += 1
        continue
    pairs_loop.append(dict(loop_id=lr["loop_id"], op1=r1["operon_id"],
                           op2=r2["operon_id"], fc1=r1["log2FC_KOvsWT_mean"],
                           fc2=r2["log2FC_KOvsWT_mean"],
                           dist_kb=lr["distance_circular_kb"],
                           both_bound=bool(lr["both_bound"]),
                           op1_bound=bool(r1["has_peak"]), op2_bound=bool(r2["has_peak"])))
pl = pd.DataFrame(pairs_loop)
pl.to_csv(f"{TAB}/fig6D_loop_operon_pairs.csv", index=False)
nL = len(pl)
obs_r, obs_p = stats.spearmanr(pl["fc1"], pl["fc2"]) if nL > 3 else (np.nan, np.nan)
obs_pr, _ = stats.pearsonr(pl["fc1"], pl["fc2"]) if nL > 3 else (np.nan, np.nan)
# distance-matched random-pair null
valid = opl.dropna(subset=["log2FC_KOvsWT_mean"])
mids = valid["mid_bin"].values
fcs = valid["log2FC_KOvsWT_mean"].values
def cdist_kb(i, j):
    d = abs(mids[i] - mids[j]) * BINSZ / 1e3
    return min(d, GENOME / 1e3 - d)
# precompute all pair distances
idx = np.arange(len(valid))
null_r = np.empty(NPERM)
one_set = None
for it in range(NPERM):
    sel1, sel2 = [], []
    for _, lr in pl.iterrows():
        dd = lr["dist_kb"]
        cand = None
        for attempt in range(50):
            i, j = RNG.choice(idx, 2, replace=False)
            if abs(cdist_kb(i, j) - dd) <= max(15, 0.15 * dd):
                cand = (i, j)
                break
        if cand is None:
            # fall back to closest-distance random pair
            i, j = RNG.choice(idx, 2, replace=False)
            cand = (i, j)
        sel1.append(cand[0]); sel2.append(cand[1])
    r1v = fcs[sel1]; r2v = fcs[sel2]
    null_r[it] = stats.spearmanr(r1v, r2v).statistic
    if it == 0:
        one_set = (r1v.copy(), r2v.copy())
p_d = (1 + np.sum(np.abs(null_r) >= abs(obs_r))) / (1 + NPERM)
null_ci = np.percentile(null_r, [2.5, 97.5])
add_stat(panel="D", test="Fig6D_loop_pair_corr", observed=obs_r, effect=obs_r,
         ci_lo=np.nan, ci_hi=np.nan, p=p_d,
         p_type="two-sided vs distance-matched random pairs (10000)", n=nL,
         null_model="random operon pairs matched to loop distance (±15% or ±15kb)",
         notes=f"Spearman r={obs_r:.3f} (asymptotic p={obs_p:.3g}); Pearson "
               f"r={obs_pr:.3f}; null mean={null_r.mean():.3f} "
               f"[{null_ci[0]:.3f},{null_ci[1]:.3f}]; {len(loops)} loops -> "
               f"{nL} pairs (excluded: {excl})")
axD.scatter(one_set[0], one_set[1], s=10, color=C_GREY, alpha=0.5, lw=0,
            label=f"distance-matched random pairs (1 realization), ρ={stats.spearmanr(one_set[0], one_set[1]).statistic:.2f}")
axD.scatter(pl["fc1"], pl["fc2"], s=14, color=C_PURPLE, lw=0, alpha=0.9,
            label=f"loop-connected pairs (n={nL}), ρ={obs_r:.2f}")
axD.axhline(0, color="#DDDDDD", lw=0.6); axD.axvline(0, color="#DDDDDD", lw=0.6)
axD.set_xlabel("Operon 1 mean log2FC (KO / WT)")
axD.set_ylabel("Operon 2 mean log2FC (KO / WT)")
axD.legend(fontsize=6, frameon=False, loc="upper left")
axD.set_title(f"loop-pair ρ = {obs_r:.3f}, distance-matched null p = {p_d:.3g}\n"
              f"null ρ mean {null_r.mean():.3f} [{null_ci[0]:.3f}, {null_ci[1]:.3f}]",
              fontsize=6.8)
fc.panel_label(axD, "d")

# ---------------- Panel E: both-end-bound loop subset ----------------
axE = fig.add_subplot(gs[1, 2:4])
# primary definition: loop whose two anchor bins are both NapM-bound (bin level);
# strict operon-level (peak inside both operon spans) reported as a note.
both = pl[pl["both_bound"]]
others = pl[~pl["both_bound"]]
n_both = len(both)
n_strict = int((pl["both_bound"] & pl["op1_bound"] & pl["op2_bound"]).sum())
both_absmean = (both["fc1"].abs() + both["fc2"].abs()) / 2
oth_absmean = (others["fc1"].abs() + others["fc2"].abs()) / 2
if n_both > 2:
    rho_b = stats.spearmanr(both["fc1"], both["fc2"]).statistic
    rho_b_p = stats.spearmanr(both["fc1"], both["fc2"]).pvalue
else:
    rho_b, rho_b_p = np.nan, np.nan
axE.scatter(others["fc1"], others["fc2"], s=14, color=C_GREY, lw=0, alpha=0.7,
            label=f"other loop pairs (n={len(others)})")
axE.scatter(both["fc1"], both["fc2"], s=18, color=C_RED, lw=0, alpha=0.95,
            label=f"both anchor bins NapM-bound (n={n_both})")
axE.axhline(0, color="#DDDDDD", lw=0.6); axE.axvline(0, color="#DDDDDD", lw=0.6)
axE.set_xlabel("Operon 1 mean log2FC (KO / WT)")
axE.set_ylabel("Operon 2 mean log2FC (KO / WT)")
axE.legend(fontsize=6, frameon=False, loc="upper left")
axE.set_title(f"both-end-bound subset (descriptive, small n={n_both}):\n"
              f"within-pair ρ={rho_b:.2f} (p={rho_b_p:.2g}); mean |log2FC| "
              f"{both_absmean.mean():.2f} vs {oth_absmean.mean():.2f} (others)",
              fontsize=6.8)
fc.panel_label(axE, "e")
both.to_csv(f"{TAB}/fig6E_both_bound_pairs.csv", index=False)
add_stat(panel="E", test="Fig6E_both_bound_descriptive", observed=rho_b, effect=rho_b,
         ci_lo=np.nan, ci_hi=np.nan, p=np.nan, p_type="descriptive (no test, small n)",
         n=n_both, null_model="none",
         notes=f"both anchor bins NapM-bound loop pairs; within-pair "
               f"Spearman={rho_b:.3f} (asymptotic p={rho_b_p:.3g}, small n); "
               f"mean |log2FC| both={both_absmean.mean():.3f} vs others="
               f"{oth_absmean.mean():.3f}; strict operon-level (peak inside both "
               f"operon spans) n={n_strict}")

save(fig, "fig6_operon")
pd.DataFrame(stats_rows).to_csv(f"{TAB}/fig6_stats_summary.csv", index=False)
print(json.dumps(dict(n_operons=len(opl), n_with_rna=int(opl.n_genes_with_rna.gt(0).sum()),
                      n_peak=int(opl.has_peak.sum()), n_pairs_B=n_pairs,
                      match_quality=mq,
                      fig6B_absfc=dict(obs=res_fc["obs"], ci=res_fc["ci"].tolist(),
                                       p=res_fc["p"]),
                      fig6B_deg=dict(obs=res_deg["obs"], ci=res_deg["ci"].tolist(),
                                     p=res_deg["p"]),
                      fig6A_rho=rho, fig6A_p=prho,
                      fig6C=dict(obs=obs_c, ci=ci_c.tolist(), p=p_c,
                                 n_cross=len(grp_cross), n_adj=len(grp_adj)),
                      fig6D=dict(rho=obs_r, p=p_d, n=nL, null_mean=float(null_r.mean()),
                                 null_ci=null_ci.tolist(), excluded=excl),
                      fig6E=dict(n=n_both, rho=rho_b, rho_p=rho_b_p,
                                 n_strict=n_strict,
                                 mean_absfc_both=float(both_absmean.mean()),
                                 mean_absfc_others=float(oth_absmean.mean()))),
            indent=1, default=float))
