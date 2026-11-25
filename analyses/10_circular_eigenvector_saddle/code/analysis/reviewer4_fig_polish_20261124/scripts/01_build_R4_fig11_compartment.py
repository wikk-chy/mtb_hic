#!/usr/bin/env python3
"""
Rebuild Response Fig. 11 (Reviewer 4, Comment 1 — A/B compartment & ori-ter axis controls):
100% vector-editable TrueType Arial (pdf.fonttype=42), standard print scale (9.2 x 7.9 in),
consistent readable fonts, and ZERO text/axis/legend overlaps.
Exact same underlying WT Hi-C data and seed=42 computations as 06_combined_figure.py.
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from scipy import stats
from scipy.signal import savgol_filter
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
BASE = ROOT / "hic_review"
OUT_FIG = ROOT / "analysis/reviewer4_fig_polish_20261124/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

BIN = 5000
GENOME = 4411532
GROUPS = 20
CORNER_FRAC = 0.20
NSHUF = 100
TER_BP = 2030000
rng = np.random.default_rng(42)

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
    "font.size": 7.6,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.9,
    "axes.linewidth": 0.7,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})


def saddle(pc, X, groups=GROUPS, qlo=0.02, qhi=0.98):
    lo, hi = np.quantile(pc, [qlo, qhi])
    sel = np.where((pc >= lo) & (pc <= hi))[0]
    order = sel[np.argsort(pc[sel])]
    g = np.full(len(pc), -1)
    for gi, grp in enumerate(np.array_split(order, groups)):
        g[grp] = gi
    S = np.full((groups, groups), np.nan)
    for a in range(groups):
        for b in range(groups):
            m = np.isfinite(X[np.ix_(g == a, g == b)])
            if m.any():
                S[a, b] = np.nanmean(X[np.ix_(g == a, g == b)])
    return S


def saddle_strength(S, mode="corner"):
    c = max(1, int(round(S.shape[0] * CORNER_FRAC)))
    aa_c = np.concatenate([S[:c, :c].ravel(), S[-c:, -c:].ravel()])
    ab_c = np.concatenate([S[:c, -c:].ravel(), S[-c:, :c].ravel()])
    if mode == "diff":
        return np.nanmean(aa_c) - np.nanmean(ab_c)
    return np.log2(np.nanmean(aa_c) / np.nanmean(ab_c))


def oe_pipeline(Mv, pos_v):
    n = Mv.shape[0]
    dp = np.abs(pos_v[:, None] - pos_v[None, :])
    dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
    cnt = np.zeros(dbin.max() + 1)
    summ = np.zeros(dbin.max() + 1)
    nz = Mv > 0
    np.add.at(cnt, dbin[nz], 1)
    np.add.at(summ, dbin[nz], Mv[nz])
    exp = np.where(cnt > 0, summ / np.maximum(cnt, 1), np.nan)
    OE = np.full((n, n), np.nan)
    ok = nz & np.isfinite(exp[dbin]) & (exp[dbin] > 0)
    OE[ok] = Mv[ok] / exp[dbin[ok]]
    fin = np.isfinite(OE).all(axis=0)
    OE, pos = OE[np.ix_(fin, fin)], pos_v[fin]
    C = np.corrcoef(OE)
    okc = np.isfinite(C).all(axis=0)
    OE, pos, C = OE[np.ix_(okc, okc)], pos[okc], C[np.ix_(okc, okc)]
    ev, vec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return OE, pos, ev[o] / ev.sum(), vec[:, o[0]]


# Real WT data
clr = cooler.Cooler(str(BASE / "review_data/cool_file/wt_5k.cool") + "::/resolutions/5000")
Mbal = clr.matrix(balance=True)[:]
valid = ~np.isnan(Mbal).all(axis=0)
pos_v = (np.where(valid)[0] * BIN + BIN / 2).astype(float)
OE_real, pos, frac_real, pc1 = oe_pipeline(Mbal[np.ix_(valid, valid)], pos_v)
n2 = OE_real.shape[0]
dp_r = np.abs(pos[:, None] - pos[None, :])
dbin_r = np.round(np.minimum(dp_r, GENOME - dp_r) / BIN).astype(int)
OE_mask = OE_real.copy()
OE_mask[dbin_r < 2] = np.nan
theta = 2 * np.pi * pos / GENOME
cos_ax, sin_ax = np.cos(theta), np.sin(theta)
if stats.pearsonr(pc1, cos_ax)[0] < 0:
    pc1 = -pc1
r_pc1_cos = stats.pearsonr(pc1, cos_ax)[0]

S_orig = saddle(pc1, OE_mask)
str_orig = saddle_strength(S_orig)
shuf = np.array([saddle_strength(saddle(rng.permutation(pc1), OE_mask)) for _ in range(NSHUF)])
_ = np.nanmean([saddle(rng.permutation(pc1), OE_mask) for _ in range(NSHUF)], axis=0)
p_shuf = (np.sum(np.abs(shuf) >= abs(str_orig)) + 1) / (NSHUF + 1)

# Synthetic P(s) only
Mraw = np.nan_to_num(clr.matrix(balance=False)[:], nan=0.0)[np.ix_(valid, valid)]
dp = np.abs(pos_v[:, None] - pos_v[None, :])
dbin = np.round(np.minimum(dp, GENOME - dp) / BIN).astype(int)
cnt = np.zeros(dbin.max() + 1)
summ = np.zeros(dbin.max() + 1)
nz = Mraw > 0
np.add.at(cnt, dbin[nz], 1)
np.add.at(summ, dbin[nz], Mraw[nz])
Ps = np.where(cnt > 0, summ / np.maximum(cnt, 1), 0.0)
lam = Ps[dbin]
iu = np.triu_indices(len(pos_v), k=1)
synth = np.zeros_like(Mraw)
synth[iu] = rng.poisson(lam[iu])
synth[(iu[1], iu[0])] = synth[iu]
np.fill_diagonal(synth, Ps[0])
OE_syn, pos_syn, frac_syn, pc1_syn = oe_pipeline(synth, pos_v)
theta_syn = 2 * np.pi * pos_syn / GENOME
r_syn_cos = abs(stats.pearsonr(pc1_syn, np.cos(theta_syn))[0])
dp_s = np.abs(pos_syn[:, None] - pos_syn[None, :])
dbin_s = np.round(np.minimum(dp_s, GENOME - dp_s) / BIN).astype(int)
OE_syn_mask = OE_syn.copy()
OE_syn_mask[dbin_s < 2] = np.nan
str_syn = saddle_strength(saddle(pc1_syn, OE_syn_mask))

# Residual (ori-ter axis removed)
dbin2 = np.round(np.minimum(np.abs(pos[:, None] - pos[None, :]),
                            GENOME - np.abs(pos[:, None] - pos[None, :])) / BIN).astype(int)
ii, jj = np.triu_indices(n2, k=1)
dc = dbin2[ii, jj]
dax = np.abs(cos_ax[ii] - cos_ax[jj])
y = OE_real[ii, jj]
resid = np.empty_like(y)
for d in np.unique(dc):
    m = dc == d
    x, ym = dax[m], y[m]
    if m.sum() < 10 or np.var(x) == 0:
        p = np.full(m.sum(), ym.mean())
    else:
        b = np.cov(x, ym, ddof=0)[0, 1] / np.var(x)
        p = ym.mean() - b * x.mean() + b * x
    resid[m] = ym - p
R = np.zeros((n2, n2))
R[ii, jj] = resid
R[jj, ii] = resid
Cr = np.corrcoef(R)
evr, vecr = np.linalg.eigh(Cr)
o = np.argsort(evr)[::-1]
evr, E1 = evr[o] / evr.sum(), vecr[:, o[0]]
h2 = np.column_stack([np.cos(2 * theta), np.sin(2 * theta), np.ones(n2)])
coef, *_ = np.linalg.lstsq(h2, E1, rcond=None)
h2_fit = h2[:, :2] @ coef[:2]
r_E1_h2 = stats.pearsonr(E1, h2_fit)[0]
if r_E1_h2 < 0:
    E1, r_E1_h2 = -E1, -r_E1_h2
R_mask = R.copy()
R_mask[dbin_r < 2] = np.nan
S_res = saddle(E1, R_mask)
str_res_diff = saddle_strength(S_res, mode="diff")
str_orig_diff = saddle_strength(S_orig, mode="diff")
shuf_res = np.array([saddle_strength(saddle(rng.permutation(E1), R_mask), mode="diff") for _ in range(NSHUF)])
p_res = (np.sum(shuf_res >= str_res_diff) + 1) / (NSHUF + 1)

# GC / gene density
seq = []
with open(BASE / "genome.fa") as fh:
    for line in fh:
        if not line.startswith(">"):
            seq.append(line.strip())
seq = "".join(seq).upper()
gtf = pd.read_csv(BASE / "annotation.gtf", sep="\t", comment="#", header=None,
                  usecols=[2, 3, 4], names=["feature", "start", "end"])
genes = gtf[gtf.feature == "gene"]
gc = np.array([(seq[int(p - BIN / 2):int(p + BIN / 2)].count("G") +
                seq[int(p - BIN / 2):int(p + BIN / 2)].count("C")) / BIN for p in pos])
gd = np.array([((genes["start"] <= p + BIN / 2) & (genes["end"] >= p - BIN / 2)).sum() for p in pos])
r_gc, p_gc = stats.pearsonr(pc1, gc)
r_gd, p_gd = stats.pearsonr(pc1, gd)

# Residual E1 zero-crossings / four zones
w = 31
E1pad = np.concatenate([E1[-3 * w:], E1, E1[:3 * w]])
E1s = savgol_filter(E1pad, w, 3)[3 * w:3 * w + n2]
sign_c = np.sign(E1s)
cross_idx = np.where(np.diff(np.append(sign_c, sign_c[0])) != 0)[0]
cross_bp = []
for k in cross_idx:
    k2 = (k + 1) % n2
    p0 = pos[k]
    p1 = pos[k2] if k2 > k else pos[k2] + GENOME
    v0, v1 = E1s[k], E1s[k2]
    cross_bp.append((p0 + (p1 - p0) * (-v0) / (v1 - v0)) % GENOME)
cross_bp = np.sort(cross_bp)

segments = []
ncross = len(cross_bp)
for s in range(ncross):
    s0 = cross_bp[s]
    s1 = cross_bp[(s + 1) % ncross]
    wrap = s1 <= s0
    end_lin = s1 + GENOME if wrap else s1
    mid = (s0 + end_lin) / 2 % GENOME
    sign = "+" if np.interp(mid, pos, E1s, period=GENOME) > 0 else "-"
    segments.append(dict(segment=s + 1, start_bp=int(round(s0)), end_bp=int(round(s1)),
                         wraps_origin=wrap, E1_sign=sign))
segdf = pd.DataFrame(segments)

# ==================== Build clean, zero-overlap figure ====================
fig = plt.figure(figsize=(9.2, 8.0))
outer = gridspec.GridSpec(
    3, 1, figure=fig,
    left=0.075, right=0.975, top=0.945, bottom=0.065,
    hspace=0.45, height_ratios=[0.95, 0.98, 1.12],
)

gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.28, width_ratios=[2.15, 1.0])
gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.24, width_ratios=[1.0, 2.55])
gs_r2_c = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_r2[0, 1], wspace=0.14, width_ratios=[1.65, 0.90, 0.90])
gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.25, width_ratios=[1.02, 1.32])
gs_r3_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_r3[0, 0], wspace=0.22)

C_RED = "#C0392B"
C_TEXT = "#1F2933"


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


# ---- Row 1: Panel A (track + saddle strength bars) ----
axA1 = fig.add_subplot(gs_r1[0, 0])
axA1.plot(pos / 1e6, pc1 / np.std(pc1), ".", ms=2.4, color=C_RED,
          label=f"Real WT PC1 (|r|cos = {abs(r_pc1_cos):.2f})")
axA1.plot(pos_syn / 1e6, pc1_syn / np.std(pc1_syn), ".", ms=2.4, color="#8A94A0",
          alpha=0.55, label=f"Synthetic P(s)-only PC1 (|r|cos = {r_syn_cos:.2f})")
axA1.axhline(0, color="#444444", lw=0.6)
axA1.set_xlim(-0.1, 4.5)
axA1.set_ylim(-3.1, 4.4)
axA1.set_xlabel("Genomic position (Mb)")
axA1.set_ylabel("PC1 (z-scored)")
axA1.set_title("PC1 along genome: real vs synthetic", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axA1.legend(markerscale=2.8, frameon=False, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=1.0)
clean_spines(axA1)

axA2 = fig.add_subplot(gs_r1[0, 1])
valsA = [str_orig, str_syn, shuf.mean()]
axA2.bar([0, 1, 2], valsA, color=[C_RED, "#8A94A0", "#CBD5E1"], edgecolor="black", lw=0.5, width=0.58)
axA2.errorbar(2, shuf.mean(), yerr=shuf.std(), color="black", capsize=3, lw=1.0)
axA2.set_xticks([0, 1, 2], ["Real\nWT", "Synthetic\nP(s) only", "Shuffled\n(mean)"])
axA2.set_ylabel("Saddle strength\n" + r"$\log_2$((AA+BB)/AB)")
axA2.axhline(0, color="#444444", lw=0.6)
axA2.set_ylim(-0.05, str_orig * 1.24)
for x, v in zip([0, 1, 2], valsA):
    axA2.text(x, v + 0.045, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold" if x == 0 else "normal")
axA2.set_title("No compartment signal from\ndistance decay alone", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axA2)

# ---- Row 2: Panel B (shuffle) + Panel C (PC1 identity: 3 panels sharing y-axis cleanly!) ----
axB = fig.add_subplot(gs_r2[0, 0])
axB.hist(shuf, bins=12, color="#64748B", edgecolor="#334155", lw=0.6, alpha=0.95)
axB.axvline(str_orig, color=C_RED, lw=1.6)
axB.set_xlim(-0.08, str_orig * 1.12)
axB.text(str_orig - 0.06, axB.get_ylim()[1] * 0.80,
         f"Observed = {str_orig:.2f}\nEmpirical P < {max(p_shuf, 1/(NSHUF+1)):.2f}",
         color=C_RED, fontsize=6.9, fontweight="bold", va="top", ha="right")
axB.set_xlabel("Saddle strength " + r"$\log_2$((AA+BB)/AB)")
axB.set_ylabel("Count")
axB.set_title(f"Shuffle control (n = {NSHUF})", loc="left", pad=6, fontweight="bold", color=C_TEXT)
clean_spines(axB)

axC1 = fig.add_subplot(gs_r2_c[0, 0])
h1 = np.column_stack([cos_ax, sin_ax, np.ones(n2)])
c1, *_ = np.linalg.lstsq(h1, pc1, rcond=None)
xx = np.linspace(0, GENOME, 500)
fit1 = c1[0] * np.cos(2 * np.pi * xx / GENOME) + c1[1] * np.sin(2 * np.pi * xx / GENOME) + c1[2]
axC1.plot(pos / 1e6, pc1, ".", ms=2.4, color=C_RED)
axC1.plot(xx / 1e6, fit1, color="black", lw=1.2, label=f"1st-harmonic fit, |r| = {abs(r_pc1_cos):.2f}")
axC1.axhline(0, color="#888888", lw=0.6)
axC1.set_ylim(-0.072, 0.068)
axC1.set_xlabel("Genomic position (Mb)")
axC1.set_ylabel("PC1")
axC1.set_title("PC1 = ori–ter axis (Fourier mode 1)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axC1.legend(frameon=False, loc="lower left", fontsize=6.6, handlelength=1.4)
clean_spines(axC1)

axC2 = fig.add_subplot(gs_r2_c[0, 1], sharey=axC1)
axC2.plot(gc, pc1, ".", ms=2.2, color="#3B7EA1", alpha=0.75)
axC2.axhline(0, color="#888888", lw=0.5)
axC2.set_xlabel("GC content / 5 kb")
axC2.set_title(f"r = {r_gc:.2f} (n.s.)", loc="center", pad=6, fontweight="bold", color=C_TEXT)
axC2.tick_params(axis="y", labelleft=False)
clean_spines(axC2)

axC3 = fig.add_subplot(gs_r2_c[0, 2], sharey=axC1)
axC3.plot(gd, pc1, ".", ms=2.2, color="#2E8B57", alpha=0.75)
axC3.axhline(0, color="#888888", lw=0.5)
axC3.set_xlabel("Genes / 5 kb")
axC3.set_title(f"r = {r_gd:.2f} (n.s.)", loc="center", pad=6, fontweight="bold", color=C_TEXT)
axC3.tick_params(axis="y", labelleft=False)
clean_spines(axC3)

# ---- Row 3: Panel D (saddle heatmaps) + Panel E (residual E1 four-zone track) ----
axD1 = fig.add_subplot(gs_r3_d[0, 0])
vmax = np.nanpercentile(np.abs(S_orig - 1), 98)
axD1.imshow(S_orig, cmap="coolwarm", vmin=1 - vmax, vmax=1 + vmax, interpolation="nearest")
axD1.set_title(f"Original\nstrength {str_orig:.2f} " + r"$\log_2$", fontsize=7.6, pad=5, fontweight="bold", color=C_TEXT)

axD2 = fig.add_subplot(gs_r3_d[0, 1])
vr = np.nanpercentile(np.abs(S_res - np.nanmean(S_res)), 98)
res_mean = np.nanmean(S_res)
axD2.imshow(S_res, cmap="coolwarm", vmin=res_mean - vr, vmax=res_mean + vr, interpolation="nearest")
axD2.set_title(f"ori–ter axis removed\ncorner diff. = {str_res_diff:.2f} ({100*str_res_diff/str_orig_diff:.0f}%; P = {max(p_res, 1/(NSHUF+1)):.2f})",
               fontsize=7.3, pad=5, fontweight="bold", color=C_TEXT)

for k_d, ax in enumerate((axD1, axD2)):
    ax.set_xlabel("Sorted bins")
    if k_d == 0:
        ax.set_ylabel("Sorted bins")
    ax.set_xticks([])
    ax.set_yticks([])

axE = fig.add_subplot(gs_r3[0, 1])
xx = np.linspace(0, GENOME, 1000)
fit2 = coef[0] * np.cos(4 * np.pi * xx / GENOME) + coef[1] * np.sin(4 * np.pi * xx / GENOME)
seg_colors = {"+": "#F8D7DA", "-": "#D6E4FA"}
for _, s in segdf.iterrows():
    if s.wraps_origin:
        axE.axvspan(s.start_bp / 1e6, GENOME / 1e6, color=seg_colors[s.E1_sign], alpha=0.55, lw=0)
        axE.axvspan(0, s.end_bp / 1e6, color=seg_colors[s.E1_sign], alpha=0.55, lw=0)
    else:
        axE.axvspan(s.start_bp / 1e6, s.end_bp / 1e6, color=seg_colors[s.E1_sign], alpha=0.55, lw=0)
axE.plot(pos / 1e6, E1, ".", ms=2.6, color="#1A202C", zorder=3)
axE.plot(xx / 1e6, fit2, color=C_RED, lw=1.4, label=f"2nd-harmonic fit, |r| = {abs(r_E1_h2):.2f}", zorder=4)
for c in cross_bp:
    axE.axvline(c / 1e6, color="#64748B", ls=":", lw=0.85)
axE.axvline(TER_BP / 1e6, color="#7C3AED", ls="--", lw=1.1, zorder=2)
axE.axvline(0, color="#15803D", ls="--", lw=1.1, zorder=2)
axE.axhline(0, color="#64748B", lw=0.6)

axE.set_xlim(-0.08, 4.50)
axE.set_ylim(-0.118, 0.132)
axE.text(0.06, -0.112, "ori", color="#15803D", fontsize=6.9, fontweight="bold", va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"), zorder=5)
axE.text(TER_BP / 1e6 + 0.06, -0.112, "ter", color="#7C3AED", fontsize=6.9, fontweight="bold", va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.85, edgecolor="none"), zorder=5)

for _, s in segdf.iterrows():
    if s.wraps_origin:
        # Place cleanly inside the right segment (3.73-4.41 Mb) so it never clips!
        mid_x = (s.start_bp / 1e6 + 4.41) / 2.0
        lbl = f"{s.start_bp/1e6:.2f}–{s.end_bp/1e6:.2f} Mb\n({s.E1_sign}; wraps ori)"
    else:
        mid_x = (s.start_bp + s.end_bp) / 2e6
        lbl = f"{s.start_bp/1e6:.2f}–{s.end_bp/1e6:.2f} Mb\n({s.E1_sign})"
    axE.text(mid_x, 0.124, lbl, ha="center", va="top", fontsize=5.85, color=C_TEXT, linespacing=1.18)

axE.set_xlabel("Genomic position (Mb)")
axE.set_ylabel("Residual E1")
axE.set_title("Residual E1: four-zone pattern (Fourier mode 2)", loc="left", pad=6, fontweight="bold", color=C_TEXT)
axE.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0", fontsize=6.5)
clean_spines(axE)

# Aligned figure-level panel letters A-E
pA, pB, pC = axA1.get_position(), axB.get_position(), axC1.get_position()
pD, pE = axD1.get_position(), axE.get_position()
fig.text(0.018, pA.y1 + 0.032, "A", fontsize=12, fontweight="bold", va="top")
fig.text(0.018, pB.y1 + 0.032, "B", fontsize=12, fontweight="bold", va="top")
fig.text(pC.x0 - 0.056, pC.y1 + 0.032, "C", fontsize=12, fontweight="bold", va="top")
fig.text(0.018, pD.y1 + 0.042, "D", fontsize=12, fontweight="bold", va="top")
fig.text(pE.x0 - 0.060, pE.y1 + 0.032, "E", fontsize=12, fontweight="bold", va="top")

pdf_p = OUT_FIG / "Response_Fig_11.pdf"
png_p = OUT_FIG / "Response_Fig_11.png"
jpg_p = OUT_FIG / "Response_Fig_11.jpg"
fig.savefig(pdf_p, dpi=300)
fig.savefig(png_p, dpi=300)
plt.close(fig)
Image.open(png_p).convert("RGB").save(jpg_p, quality=95, dpi=(300, 300))
print("Saved Response_Fig_11.{pdf,png,jpg}")
