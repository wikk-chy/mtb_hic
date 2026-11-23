#!/usr/bin/env python3
"""Fig5 — Local transcription does not quantitatively explain local structural remodeling.

Outputs to RA/08_extended/fig5_rna/{figures,tables}. Seed 42.
Reads only pre-computed tables in RA/01..06 (read-only).
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../07_figures/scripts")
import fc  # shared style

SEED = 42
RNG = np.random.default_rng(SEED)
RA = fc.RA
OUTD = f"{RA}/08_extended/fig5_rna"
FIG = f"{OUTD}/figures"
TAB = f"{OUTD}/tables"
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)
NBINS = 883
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

def paired_shift_test(bound_idx, matched_idx, values, label, value_name):
    """Paired test with circular-shift null (keeps pair offsets), as in s06_perm."""
    vb = values[bound_idx]; vm = values[matched_idx]
    ok = ~(np.isnan(vb) | np.isnan(vm))
    B, M = bound_idx[ok], matched_idx[ok]
    diffs = values[B] - values[M]
    obs = diffs.mean()
    ci = bootstrap_ci(diffs)
    wp = stats.wilcoxon(diffs).pvalue if len(diffs) > 10 else np.nan
    ns = np.empty(NPERM)
    for i, s in enumerate(shifts):
        ns[i] = np.nanmean(values[(B + s) % NBINS] - values[(M + s) % NBINS])
    mw = stats.mannwhitneyu(values[B], values[M]).pvalue
    p = (1 + np.sum(np.abs(ns) >= abs(obs))) / (1 + len(ns))
    add_stat(panel=label, test=f"{label}_{value_name}", observed=obs, effect=obs,
             ci_lo=ci[0], ci_hi=ci[1], p=p, p_type="two-sided circular-shift(10000)",
             n=int(ok.sum()), null_model="circular-shift of matched pair offsets",
             notes=f"{value_name}; paired Wilcoxon p={wp:.3g}; naive MW p={mw:.3g} (naive)")
    return dict(obs=obs, ci=ci, p=p, n=int(ok.sum()), wp=wp, mw=mw,
                mb=np.nanmean(values[B]), mm=np.nanmean(values[M]))

# ---------------- load ----------------
b = pd.read_csv(f"{RA}/03_structure_metrics/bin_level_v2.csv")
g = pd.read_csv(f"{RA}/04_rna_integration/gene_level_rna_v2.csv")
bp = pd.read_csv(f"{RA}/05_matched_controls/matched_pairs.csv")
gp = pd.read_csv(f"{RA}/05_matched_controls/matched_pairs_genes.csv")
nested = pd.read_csv(f"{RA}/../R1_mechanism/R1_mechanism_zhenghe/tables/stats_fig2_nested_models.csv")
cmp_df = pd.read_csv(f"{RA}/../R1_mechanism/R1_mechanism_zhenghe/tables/stats_fig2_model_comparison.csv")
sp_row = pd.read_csv(f"{RA}/06_permutation_tests/stats_fig2_summary.csv")
sp_all = sp_row[sp_row.test == "Fig2d_spearman_rna_vs_dins_all"].iloc[0]
RHO, RHO_P, RHO_N = sp_all["observed"], sp_all["p"], int(sp_all["n"])

x_all = b["rna_mean_log2FC_KOvsWT"].values
y_all = b["d_ins_KO_WT"].values
chip = b["chip_bin_score_max"].values
m = ~np.isnan(x_all)
x, y, ch = x_all[m], y_all[m], chip[m]
n_af = m.sum()

fig = plt.figure(figsize=(fc.FIGW, 120 * fc.MM))
gs = fig.add_gridspec(2, 4, left=0.098, right=0.99, top=0.93, bottom=0.09,
                      wspace=0.5, hspace=0.55)

# ---------------- Panel A: continuous scatter ----------------
axA = fig.add_subplot(gs[0, 0])
hb = axA.hexbin(x, y, gridsize=45, cmap="Greys", mincnt=1, alpha=0.45,
                linewidths=0, zorder=1)
sc = axA.scatter(x, y, c=ch, cmap=CMAP_CHIP, s=4, lw=0, alpha=0.85, zorder=2,
                 vmin=0, vmax=1)
sl, ic = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
axA.plot(xs, ic + sl * xs, color=C_GREY, ls="--", lw=1.0, zorder=3)
cb = fig.colorbar(sc, ax=axA, pad=0.02, fraction=0.06)
cb.set_label("NapM ChIP score (max)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_linewidth(0.5)
axA.set_xlabel("Bin mean RNA log2FC (KO / WT)")
axA.set_ylabel("Δ insulation (KO − WT)")
axA.text(0.03, 0.03, f"Spearman ρ = {RHO:.3f}\np = {RHO_P:.2f}, n = {RHO_N}",
         transform=axA.transAxes, fontsize=7, va="bottom")
fc.panel_label(axA, "a")
srcA = pd.DataFrame(dict(bin=b["bin"].values[m], rna_mean_log2FC=x, d_ins_KO_WT=y,
                         chip_bin_score_max=ch))
srcA.to_csv(f"{TAB}/fig5A_source.csv", index=False)
add_stat(panel="A", test="Fig5A_spearman_rna_vs_dins", observed=RHO, effect=RHO,
         ci_lo=np.nan, ci_hi=np.nan, p=RHO_P, p_type="two-sided asymptotic",
         n=RHO_N, null_model="none (Spearman)",
         notes="reused from 06_permutation_tests/stats_fig2_summary.csv")

# ---------------- Panel B: matched estimation plots ----------------
DET = {"detected_increase", "detected_decrease"}
gd = g.set_index("gene_id")
gidx = {gid: i for i, gid in enumerate(g["gene_id"])}
g_bin_tss = g["bin_tss"].values
g_absfc = g["log2FC_KOvsWT"].abs().values
g_absdins = g["bin_d_ins_KO_WT"].abs().values
g_deg = g["class_KOvsWT"].isin(DET).astype(float).values

b_absrna = b["rna_mean_log2FC_KOvsWT"].abs().values
b_absdins = b["d_ins_KO_WT"].abs().values
b_deg = (b["rna_bin_class"] == "detected_change").astype(float).values

gb_i = np.array([gidx[v] for v in gp["bound_gene"]])
gm_i = np.array([gidx[v] for v in gp["matched_gene"]])
gb_bin = g_bin_tss[gb_i]; gm_bin = g_bin_tss[gm_i]
bb_bin = bp["bound_bin"].values; mb_bin = bp["matched_bin"].values

metrics = [
    ("gene", "|RNA log2FC| (gene)", g_absfc, gb_i, gm_i, gb_bin, gm_bin, "abslog2FC"),
    ("gene", "|Δins| (gene bin)", g_absdins, gb_i, gm_i, gb_bin, gm_bin, "absdins_genebin"),
    ("gene", "P(DEG) (gene)", g_deg, gb_i, gm_i, gb_bin, gm_bin, "DEG_gene"),
    ("bin", "|RNA log2FC| (bin)", b_absrna, bb_bin, mb_bin, bb_bin, mb_bin, "abslog2FC_bin"),
    ("bin", "|Δins| (bin)", b_absdins, bb_bin, mb_bin, bb_bin, mb_bin, "absdins_bin"),
    ("bin", "P(RNA detected) (bin)", b_deg, bb_bin, mb_bin, bb_bin, mb_bin, "DEG_bin"),
]
# nested gridspec inside gs[0,1:3]: 2 rows x 3 cols mini-panels
subB = gs[0, 1:3].subgridspec(2, 3, wspace=0.45, hspace=2.1)
srcB = {}
for k, (lvl, ttl, vals, bi_arr, mi_arr, bbin, mbin, vname) in enumerate(metrics):
    r, c = divmod(k, 3)
    ax = fig.add_subplot(subB[r, c])
    vb, vm = vals[bi_arr], vals[mi_arr]
    res = paired_shift_test(bbin, mbin, vals, f"Fig5B_{lvl}", vname)
    show = RNG.choice(len(vb), size=min(60, len(vb)), replace=False)
    for i in show:
        if np.isfinite(vb[i]) and np.isfinite(vm[i]):
            ax.plot([0, 1], [vb[i], vm[i]], color=C_GREY, lw=0.3, alpha=0.25,
                    zorder=1)
    for xx, vv, col, lab in ((0, vb, C_PURPLE, "bound"), (1, vm, C_GREY, "matched")):
        vv = vv[np.isfinite(vv)]
        ci = bootstrap_ci(vv)
        ax.errorbar([xx], [vv.mean()], yerr=[[vv.mean() - ci[0]], [ci[1] - vv.mean()]],
                    fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=1.0,
                    capsize=2, zorder=3, label=lab if k == 0 else None)
    ax.set_xlim(-0.55, 1.55); ax.set_xticks([0, 1]); ax.set_xticklabels(["B", "M"])
    ax.set_title(ttl.replace(" (", "\n("), fontsize=6.8)
    ax.text(0.5, -0.78,
            f"Δ={res['obs']:+.3f} [{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}]\n"
            f"p={res['p']:.3g}, n={res['n']}",
            transform=ax.transAxes, fontsize=6.2, ha="center", va="top")
    srcB[ttl] = pd.DataFrame(dict(bound=vb, matched=vm))
    if k == 0:
        ax.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.2,
                  borderaxespad=0.1)
axB = fig.add_subplot(gs[0, 1:3]); axB.axis("off")
fc.panel_label(axB, "b")
srcB_out = pd.concat({k: v for k, v in srcB.items()}, axis=1)
srcB_out.to_csv(f"{TAB}/fig5B_source.csv", index=False)

# ---------------- Panel C: 2x2 joint classification ----------------
axC = fig.add_subplot(gs[0, 3])
sd = b["structure_detected"].values
rc = b["rna_bin_class"].values
rna_det = rc == "detected_change"
rna_undet = np.isin(rc, ["cannot_determine", "no_data"])
rna_notdet = rc == "supports_small_change"
det_ok = ~rna_undet
c11 = int(np.sum(sd & rna_det & det_ok)); c10 = int(np.sum(sd & rna_notdet))
c01 = int(np.sum(~sd & rna_det & det_ok)); c00 = int(np.sum(~sd & rna_notdet))
u1 = int(np.sum(sd & rna_undet)); u0 = int(np.sum(~sd & rna_undet))
odds, fisher_p = stats.fisher_exact([[c11, c10], [c01, c00]])
width = 0.38
xs2 = np.arange(3)
labels3 = ["struct+\nRNA+", "struct+\nRNA−", "struct−\nRNA+"]
counts = [c11, c10, c01]
# simpler grouped view: x = structure status, stacked RNA status
bot = np.zeros(2)
for vals2, col, lab in (([c11, c01], C_PURPLE, "RNA detected"),
                        ([c10, c00], "#D9DBDC", "RNA not detected"),
                        ([u1, u0], "#F0F0F0", "undetermined")):
    axC.bar([0, 1], vals2, width, bottom=bot, color=col, edgecolor="white",
            lw=0.5, label=lab)
    for xi, (vv, bb_) in enumerate(zip(vals2, bot)):
        if vv > 0:
            axC.text(xi, bb_ + vv / 2, str(int(vv)), ha="center", va="center",
                     fontsize=6.5, color="white" if lab == "RNA detected" else "#555555")
    bot += np.array(vals2, dtype=float)
axC.set_xticks([0, 1]); axC.set_xticklabels(["structure\ndetected", "structure\nnot detected"])
axC.set_ylabel("Bins")
axC.legend(fontsize=5.8, frameon=False, loc="upper right", borderaxespad=0.1,
           handlelength=1.2, labelspacing=0.3)
axC.text(0.5, 0.52, f"OR={odds:.2f}, Fisher p={fisher_p:.3g}\nundet. not merged",
         transform=axC.transAxes, fontsize=6, va="center", ha="center",
         bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2))
fc.panel_label(axC, "c")
srcC = pd.DataFrame(dict(category=["struct+RNA+", "struct+RNA-", "struct-RNA+",
                                   "struct-RNA-", "struct+undet", "struct-undet"],
                         n=[c11, c10, c01, c00, u1, u0]))
srcC.to_csv(f"{TAB}/fig5C_source.csv", index=False)
add_stat(panel="C", test="Fig5C_fisher_2x2", observed=odds, effect=odds,
         ci_lo=np.nan, ci_hi=np.nan, p=fisher_p, p_type="two-sided Fisher exact",
         n=int(c11 + c10 + c01 + c00), null_model="none",
         notes=f"2x2 [[{c11},{c10}],[{c01},{c00}]]; undetermined struct+={u1}, struct-={u0}")

# ---------------- Panel D: multiple regression ----------------
def ols(X, yv):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
    resid = yv - X1 @ beta
    sse = resid @ resid
    dof = len(yv) - X1.shape[1]
    sigma2 = sse / dof
    XtXinv = np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(XtXinv) * sigma2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ci = np.column_stack([beta - 1.96 * se, beta + 1.96 * se])
    sst = ((yv - yv.mean()) ** 2).sum()
    return dict(beta=beta, se=se, ci=ci, t=t, p=p, sse=sse, r2=1 - sse / sst,
                dof=dof)

pred_names = ["rna_mean_log2FC_KOvsWT", "expr_wt_mean", "ins_WT", "weight_WT", "n_genes"]
Xf = b.loc[m, pred_names].values.astype(float)
yv = y.copy()
ok = ~np.isnan(Xf).any(axis=1)
Xf, yv = Xf[ok], yv[ok]
# standardize predictors for comparable coefficients
Xmean, Xsd = Xf.mean(0), Xf.std(0, ddof=1)
Xz = (Xf - Xmean) / Xsd
full = ols(Xz, yv)
base = ols(Xz[:, 1:], yv)
partial_r2 = (base["sse"] - full["sse"]) / base["sse"]
Fstat = ((base["sse"] - full["sse"]) / 1) / (full["sse"] / full["dof"])
Fp = stats.f.sf(Fstat, 1, full["dof"])
# 10-block circular CV
fold = (np.arange(len(yv)) * 10) // len(yv)  # bins already sorted by position
def cv_r2(Xm):
    preds = np.empty(len(yv))
    for f in range(10):
        tr, te = fold != f, fold == f
        Xtr = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        beta, *_ = np.linalg.lstsq(Xtr, yv[tr], rcond=None)
        Xte = np.column_stack([np.ones(te.sum()), Xm[te]])
        preds[te] = Xte @ beta
    return 1 - ((yv - preds) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
cv_full, cv_base = cv_r2(Xz), cv_r2(Xz[:, 1:])
add_stat(panel="D", test="Fig5D_rna_coef_std", observed=full["beta"][1],
         effect=full["beta"][1], ci_lo=full["ci"][1, 0], ci_hi=full["ci"][1, 1],
         p=full["p"][1], p_type="OLS t (two-sided)", n=len(yv),
         null_model="none", notes="standardized predictors; outcome d_ins_KO_WT")
add_stat(panel="D", test="Fig5D_partialR2_rna", observed=partial_r2, effect=partial_r2,
         ci_lo=np.nan, ci_hi=np.nan, p=Fp, p_type="F-test nested (1 df)", n=len(yv),
         null_model="baseline model without RNA",
         notes=f"R2_full={full['r2']:.4f}, R2_base={base['r2']:.4f}, "
               f"CV_R2_full={cv_full:.4f}, CV_R2_base={cv_base:.4f} (10-block)")
srcD = pd.DataFrame(dict(
    predictor=["intercept"] + pred_names,
    beta=full["beta"], ci_lo=full["ci"][:, 0], ci_hi=full["ci"][:, 1],
    t=full["t"], p=full["p"]))
srcD.to_csv(f"{TAB}/fig5D_source.csv", index=False)

axD = fig.add_subplot(gs[1, 0])
ypos = np.arange(len(pred_names))[::-1]
for i, pn in enumerate(pred_names):
    col = C_PURPLE if pn == "rna_mean_log2FC_KOvsWT" else C_GREY
    axD.errorbar(full["beta"][i + 1], ypos[i],
                 xerr=[[full["beta"][i + 1] - full["ci"][i + 1, 0]],
                       [full["ci"][i + 1, 1] - full["beta"][i + 1]]],
                 fmt="o", ms=4, color=col, ecolor=col, elinewidth=1.0, capsize=2)
axD.axvline(0, color="#BBBBBB", lw=0.7, ls=":")
axD.set_yticks(ypos)
axD.set_yticklabels(["RNA log2FC", "expr WT", "ins WT", "weight WT", "n genes"],
                    fontsize=7)
axD.set_xlabel("Standardized coefficient (95% CI)")
axD.set_title(f"partial R²(RNA) = {partial_r2:.4f}, F p = {Fp:.3g}\n"
              f"CV R² full {cv_full:.3f} vs baseline {cv_base:.3f}", fontsize=6.8)
fc.panel_label(axD, "d")

# ---------------- Panel E: nested models incremental ----------------
axE = fig.add_subplot(gs[1, 1])
models = nested["model"].values
r2s = nested["cv_R2"].values
labels_m = ["M0\nbaseline", "M1\n+RNA", "M2\n+occupancy"]
cols_m = [C_GREY, C_BLUE, C_PURPLE]
axE.plot(range(3), r2s, "-", color="#CCCCCC", lw=0.8, zorder=1)
for i, (r2v, col) in enumerate(zip(r2s, cols_m)):
    axE.plot(i, r2v, "o", ms=6, color=col, zorder=2)
    axE.annotate(f"{r2v:.4f}", (i, r2v), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=6.5)
axE.set_xticks(range(3)); axE.set_xticklabels(labels_m, fontsize=7)
pad = 0.0012
axE.set_ylim(r2s.min() - pad, r2s.max() + pad * 2.6)
axE.set_ylabel("10-block CV R²")
dR21 = cmp_df["dR2_M1_vs_M0"].iloc[0]
dR32 = cmp_df["dR2_M2_vs_M1"].iloc[0]
lo32, hi32 = cmp_df["ci_lo"].iloc[0], cmp_df["ci_hi"].iloc[0]
axE.set_title(f"ΔR²(M1−M0) = {dR21:+.5f}\nΔR²(M2−M1) = {dR32:+.5f} [{lo32:+.5f}, {hi32:+.5f}]",
              fontsize=6.8)
fc.panel_label(axE, "e")
srcE = nested.copy()
srcE["dR2_M2_vs_M1"] = dR32; srcE["ci_lo"] = lo32; srcE["ci_hi"] = hi32
srcE["dR2_M1_vs_M0"] = dR21
srcE.to_csv(f"{TAB}/fig5E_source.csv", index=False)
add_stat(panel="E", test="Fig5E_nested_cv", observed=dR32, effect=dR32, ci_lo=lo32,
         ci_hi=hi32, p=np.nan, p_type="block bootstrap CI (reused)",
         n=int(nested["n_bins"].iloc[0]), null_model="none",
         notes="reused from R1_mechanism_zhenghe; M0/M1/M2 CV R2="
               + ",".join(f"{v:.4f}" for v in r2s))

# ---------------- Panel F: nonlinear check ----------------
def lowess(xv, yv2, frac=0.35, it=2, npts=120):
    n = len(xv)
    k = max(int(frac * n), 3)
    grid = np.linspace(xv.min(), xv.max(), npts)
    order = np.argsort(xv)
    xs_s, ys_s = xv[order], yv2[order]
    robust = np.ones(n)
    out = np.empty(npts)
    for itn in range(it + 1):
        for gi, gx in enumerate(grid):
            d = np.abs(xs_s - gx)
            h = np.partition(d, k - 1)[k - 1]
            w = (1 - (d / h) ** 3) ** 3
            w[d >= h] = 0
            w *= robust
            W = w.sum()
            xm = (w * xs_s).sum() / W
            ym = (w * ys_s).sum() / W
            cov = (w * (xs_s - xm) * (ys_s - ym)).sum() / W
            var = (w * (xs_s - xm) ** 2).sum() / W
            b1 = cov / var if var > 0 else 0.0
            out[gi] = ym + b1 * (gx - xm) if itn == it else 0
        if itn < it:
            # fit once on full data to get residuals for robustness weights
            preds = np.empty(n)
            for i2 in range(n):
                d = np.abs(xs_s - xs_s[i2])
                h = np.partition(d, k - 1)[k - 1]
                w = (1 - (d / h) ** 3) ** 3
                w[d >= h] = 0
                w *= robust
                W = w.sum()
                xm = (w * xs_s).sum() / W
                ym = (w * ys_s).sum() / W
                cov = (w * (xs_s - xm) * (ys_s - ym)).sum() / W
                var = (w * (xs_s - xm) ** 2).sum() / W
                b1 = cov / var if var > 0 else 0.0
                preds[i2] = ym + b1 * (xs_s[i2] - xm)
            r = np.abs(ys_s - preds)
            med = np.median(r)
            if med == 0:
                break
            u = r / (6 * med)
            robust = (1 - u ** 2) ** 2
            robust[u >= 1] = 0
    return grid, out

def quantile_reg(xv, yv2, tau):
    n = len(xv)
    # min tau*sum(u) + (1-tau)*sum(v), y - (b0+b1 x) = u - v
    c = np.concatenate([[0, 0], tau * np.ones(n), (1 - tau) * np.ones(n)])
    Aeq = np.zeros((n, 2 + 2 * n))
    Aeq[:, 0] = 1
    Aeq[:, 1] = xv
    for i in range(n):
        Aeq[i, 2 + i] = -1
        Aeq[i, 2 + n + i] = 1
    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=Aeq, b_eq=yv2, bounds=bounds, method="highs")
    return res.x[0], res.x[1]

axF = fig.add_subplot(gs[1, 2:4])
axF.scatter(x, y, s=3, lw=0, color="#C9C9C9", alpha=0.7, zorder=1)
gx, gy = lowess(x, y, frac=0.35, it=2)
axF.plot(gx, gy, color=C_PURPLE, lw=1.6, zorder=4, label="LOWESS (f=0.35)")
qr_styles = {0.1: (C_BLUE, ":"), 0.5: (C_RED, "-"), 0.9: (C_BLUE, ":")}
qr_slopes = {}
for tau, (col, ls) in qr_styles.items():
    b0, b1 = quantile_reg(x, y, tau)
    qr_slopes[tau] = (b0, b1)
    axF.plot(xs, b0 + b1 * xs, color=col, ls=ls, lw=1.0, zorder=3,
             label=f"QR τ={tau}" if tau != 0.1 else "QR τ=0.1/0.9")
# bootstrap CI for median slope
nb = 500
bs_slopes = np.empty(nb)
for i in range(nb):
    idx = RNG.integers(0, len(x), len(x))
    bs_slopes[i] = quantile_reg(x[idx], y[idx], 0.5)[1]
med_ci = np.percentile(bs_slopes, [2.5, 97.5])
axF.set_xlabel("Bin mean RNA log2FC (KO / WT)")
axF.set_ylabel("Δ insulation (KO − WT)")
axF.legend(fontsize=6, frameon=False, loc="upper left", borderaxespad=0.1)
axF.set_title(f"QR median slope = {qr_slopes[0.5][1]:+.4f} "
              f"[{med_ci[0]:+.4f}, {med_ci[1]:+.4f}]\n"
              f"LOWESS range = {gy.max() - gy.min():.4f} (Δins units)", fontsize=6.8)
fc.panel_label(axF, "f")
srcF = pd.DataFrame(dict(rna_mean_log2FC=x, d_ins_KO_WT=y))
srcF.to_csv(f"{TAB}/fig5F_source.csv", index=False)
pd.DataFrame(dict(quantile=[0.1, 0.5, 0.9],
                  intercept=[qr_slopes[t][0] for t in (0.1, 0.5, 0.9)],
                  slope=[qr_slopes[t][1] for t in (0.1, 0.5, 0.9)])).to_csv(
    f"{TAB}/fig5F_quantile_fits.csv", index=False)
pd.DataFrame(dict(grid_x=gx, lowess_y=gy)).to_csv(f"{TAB}/fig5F_lowess.csv", index=False)
add_stat(panel="F", test="Fig5F_qr_median_slope", observed=qr_slopes[0.5][1],
         effect=qr_slopes[0.5][1], ci_lo=med_ci[0], ci_hi=med_ci[1], p=np.nan,
         p_type="bootstrap 95% CI (500)", n=len(x), null_model="none",
         notes=f"QR slopes: tau0.1={qr_slopes[0.1][1]:+.4f}, tau0.5={qr_slopes[0.5][1]:+.4f}, "
               f"tau0.9={qr_slopes[0.9][1]:+.4f}; LOWESS y-range={gy.max()-gy.min():.4f}")

save(fig, "fig5_rna_structure")
pd.DataFrame(stats_rows).to_csv(f"{TAB}/fig5_stats_summary.csv", index=False)
print(json.dumps(dict(partial_r2=partial_r2, Fp=Fp,
                      rna_beta=full["beta"][1].tolist(),
                      rna_ci=full["ci"][1].tolist(), rna_p=full["p"][1].tolist(),
                      cv_full=cv_full, cv_base=cv_base,
                      qr_med_slope=qr_slopes[0.5][1], med_ci=med_ci.tolist(),
                      lowess_range=float(gy.max() - gy.min()),
                      fisher_or=odds, fisher_p=fisher_p), indent=1))
