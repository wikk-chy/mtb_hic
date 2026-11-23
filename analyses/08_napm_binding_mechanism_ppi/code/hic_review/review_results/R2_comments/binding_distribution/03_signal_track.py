#!/usr/bin/env python3
"""R2 Comment 6 - binding_distribution, part 3.

Continuous-signal view (not only called peaks): genome-wide statistics of the
Rv0047c bedgraph and a whole-genome signal track figure with curated peak
positions marked.

Outputs: tables/signal_genomewide_stats.csv,
         figures/fig3_signal_track.{png,pdf}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
BEDGRAPH = BASE + "hic_review/review_data/chip_seq_data/Rv0047c.bedgraph"
PEAKS = BASE + "hic_review/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
OUT_T = BASE + "hic_review/review_results/R2_comments/binding_distribution/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/binding_distribution/figures/"
GENOME_LEN = 4_411_532
BIN = 5000

bg = pd.read_csv(BEDGRAPH, sep="\t", header=None,
                 names=["chrom", "start", "end", "signal"]).sort_values("start")
w = (bg.end - bg.start).values          # interval widths (bp) -> bp-weighted stats
sig = bg.signal.values
assert w.sum() == GENOME_LEN, f"bedgraph covers {w.sum()} bp != {GENOME_LEN}"

def frac_above(q):
    thr = np.quantile(sig, q)           # unweighted quantile of interval values
    return thr, w[sig >= thr].sum() / GENOME_LEN

rows = [("signal_mean", round(sig.mean(), 1)),
        ("signal_sd", round(sig.std(), 1)),
        ("signal_median", round(np.median(sig), 1)),
        ("signal_max", round(sig.max(), 1)),
        ("n_intervals", len(bg))]
for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    thr, frac = frac_above(q)
    rows.append((f"frac_bp_above_q{int(q*100)}", round(frac, 4)))
    rows.append((f"signal_at_q{int(q*100)}", round(thr, 1)))
pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
    OUT_T + "signal_genomewide_stats.csv", index=False)

# ---- 5-kb binned mean signal for the track --------------------------------
nbins = int(np.ceil(GENOME_LEN / BIN))
bin_sig = np.zeros(nbins)
bin_w = np.zeros(nbins)
for s, e, v in zip(bg.start.values, bg.end.values, sig):
    b0, b1 = s // BIN, min((e - 1) // BIN, nbins - 1)
    for b in range(b0, b1 + 1):
        ov = min(e, (b + 1) * BIN) - max(s, b * BIN)
        if ov > 0:
            bin_sig[b] += v * ov
            bin_w[b] += ov
bin_sig = bin_sig / np.maximum(bin_w, 1)
x = (np.arange(nbins) + 0.5) * BIN / 1e6

peaks = pd.read_csv(PEAKS, sep="\t")
peak_gb = peaks.assign(gb=peaks.center // BIN).groupby("gb").score.max()

fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.2), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 1]))
ax = axes[0]
ax.fill_between(x, 0, bin_sig, color="#9ecae1", lw=0)
ax.plot(x, bin_sig, color="#3182bd", lw=0.4)
ax.set_ylabel("ChIP signal\n(5-kb mean, VPM-like)")
for q, ls, c in [(0.95, "--", "grey"), (0.99, ":", "black")]:
    thr = np.quantile(sig, q)
    ax.axhline(thr, ls=ls, lw=0.6, color=c)
    ax.text(4.42, thr, f"q{int(q*100)}", fontsize=6, va="center")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("A  Rv0047c continuous ChIP-seq signal, whole genome", loc="left",
             fontweight="bold")

ax = axes[1]
ax.stem(peak_gb.index * BIN / 1e6, peak_gb.values, linefmt="#6a51a3",
        markerfmt=" ", basefmt=" ")
ax.set_ylabel("peak score")
ax.set_xlabel("genomic position (Mb)")
ax.set_title("B  634 curated peaks (score)", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig3_signal_track.png", dpi=300)
fig.savefig(OUT_F + "fig3_signal_track.pdf")
print(pd.DataFrame(rows, columns=["metric", "value"]).to_string(index=False))
print("OK")
