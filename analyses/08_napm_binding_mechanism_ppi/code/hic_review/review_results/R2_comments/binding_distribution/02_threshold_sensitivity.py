#!/usr/bin/env python3
"""R2 Comment 6 - binding_distribution, part 2.

Threshold sensitivity of NapM binding coverage, three definitions:
  (a) curated 634-peak table: coverage as a function of the score threshold
      and of the VPM threshold (merged union coverage).
  (b) legacy signal-threshold definition from Rv0047c.bedgraph: contiguous
      regions with signal >= genome-wide mean + k*SD, k = 0.5..3.0.
      k = 2.0 reproduces the "mean+2SD" 233-peak definition cited in the
      response draft (review_results/omics/chipseq_peaks_mean2sd.bed is
      absent; recomputed here as instructed).

Outputs: tables/coverage_threshold_sensitivity.csv,
         tables/mean2sd_peaks.bed,
         figures/fig2_coverage_thresholds.{png,pdf}
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
PEAKS = BASE + "hic_review/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
BEDGRAPH = BASE + "hic_review/review_data/chip_seq_data/Rv0047c.bedgraph"
OUT_T = BASE + "hic_review/review_results/R2_comments/binding_distribution/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/binding_distribution/figures/"
GENOME_LEN = 4_411_532

def merged_len(iv):
    iv = sorted(iv)
    if not iv:
        return 0, 0
    m = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return sum(e - s for s, e in m), len(m)

peaks = pd.read_csv(PEAKS, sep="\t")
peaks["width"] = peaks["end"] - peaks["start"]

rows = []
# (a) score / vpm threshold sweeps on curated peaks
for thr in np.arange(0.70, 1.001, 0.02):
    sub = peaks[peaks.score >= thr]
    ml, nm = merged_len(sub[["start", "end"]].values.tolist())
    rows.append(dict(definition="curated634_score>=thr", threshold=round(thr, 3),
                     n_peaks=len(sub), n_merged=nm, merged_bp=ml,
                     coverage=ml / GENOME_LEN))
for thr in [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]:
    sub = peaks[peaks.vpm >= thr]
    ml, nm = merged_len(sub[["start", "end"]].values.tolist())
    rows.append(dict(definition="curated634_vpm>=thr", threshold=thr,
                     n_peaks=len(sub), n_merged=nm, merged_bp=ml,
                     coverage=ml / GENOME_LEN))

# (b) signal-threshold (legacy mean+k*SD) from bedgraph
bg = pd.read_csv(BEDGRAPH, sep="\t", header=None,
                 names=["chrom", "start", "end", "signal"])
bg = bg.sort_values("start").reset_index(drop=True)
mu, sd = bg.signal.mean(), bg.signal.std()
mean2sd_peaks = None
for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    thr = mu + k * sd
    above = bg.signal >= thr
    # contiguous runs of above-threshold intervals
    runs, in_run = [], False
    for i, ok in enumerate(above.values):
        if ok and not in_run:
            rs = bg.start.iat[i]; in_run = True
        elif not ok and in_run:
            runs.append([rs, bg.end.iat[i - 1]]); in_run = False
    if in_run:
        runs.append([rs, bg.end.iat[len(above) - 1]])
    ml, nm = merged_len(runs)
    widths = [e - s for s, e in runs] if runs else [0]
    rows.append(dict(definition="bedgraph_mean+kSD", threshold=round(k, 2),
                     n_peaks=len(runs), n_merged=nm, merged_bp=ml,
                     coverage=ml / GENOME_LEN,
                     signal_cutoff=round(thr, 1),
                     median_width_bp=np.median(widths)))
    if k == 2.0:
        mean2sd_peaks = pd.DataFrame(runs, columns=["start", "end"])
        mean2sd_peaks.insert(0, "chrom", "Chromosome")

mean2sd_peaks.to_csv(OUT_T + "mean2sd_peaks.bed", sep="\t", index=False, header=False)
out = pd.DataFrame(rows)
out.to_csv(OUT_T + "coverage_threshold_sensitivity.csv", index=False)

# ---- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

d = out[out.definition == "curated634_score>=thr"]
axes[0].plot(d.threshold, 100 * d.coverage, "o-", color="#6a51a3", ms=3, lw=1)
axes[0].set_xlabel("score threshold (curated 634-peak table)")
axes[0].set_ylabel("merged coverage (% genome)")
axes[0].set_title("A  score threshold", loc="left", fontweight="bold")

d = out[out.definition == "curated634_vpm>=thr"]
axes[1].plot(d.threshold, 100 * d.coverage, "o-", color="#3182bd", ms=3, lw=1)
axes[1].set_xlabel("VPM threshold (curated 634-peak table)")
axes[1].set_title("B  VPM threshold", loc="left", fontweight="bold")

d = out[out.definition == "bedgraph_mean+kSD"]
ax = axes[2]
ax.plot(d.threshold, 100 * d.coverage, "o-", color="#de2d26", ms=3, lw=1)
ax2 = ax.twinx()
ax2.plot(d.threshold, d.n_peaks, "s--", color="grey", ms=3, lw=0.8)
ax2.set_ylabel("n regions", color="grey")
ax.set_xlabel("signal threshold (mean + k SD)")
ax.set_ylabel("merged coverage (% genome)", color="#de2d26")
ax.set_title("C  signal threshold (bedgraph)", loc="left", fontweight="bold")

for a in axes:
    a.spines[["top"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig2_coverage_thresholds.png", dpi=300)
fig.savefig(OUT_F + "fig2_coverage_thresholds.pdf")

print("bedgraph mean =", round(mu, 2), " SD =", round(sd, 2))
print(out[out.definition == "bedgraph_mean+kSD"].to_string(index=False))
print("baseline score>=0:", out[(out.definition == "curated634_score>=thr") & (out.threshold == 0.7)].to_string(index=False))
print("OK")
