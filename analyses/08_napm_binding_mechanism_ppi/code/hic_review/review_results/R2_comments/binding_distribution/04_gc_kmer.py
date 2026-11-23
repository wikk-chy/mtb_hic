#!/usr/bin/env python3
"""R2 Comment 6 - binding_distribution, part 4.

Sequence composition of NapM peak regions:
  - GC% of merged peak intervals vs (i) genome-wide GC and (ii) length-matched
    random regions (empirical p, seed=42). M. tuberculosis is GC-rich
    (~65.6%), so the background must be stated explicitly.
  - k-mer enrichment (k = 8, canonical reverse-complement collapsed):
    peak sequences vs GC- AND length-matched random background
    (rejection sampling per interval, 5x background, seed=42).
    Simple enumeration; no meme/homer available.

Outputs: tables/gc_composition.csv, tables/kmer8_enrichment.csv,
         figures/fig4_gc_kmer.{png,pdf}
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
from collections import Counter

BASE = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/"
PEAKS = BASE + "hic_review/review_results/R1_mechanism_reanalysis/01_chip_curated/NapM_binding_curated.tsv"
GENOME_FA = BASE + "hic_review/genome.fa"
OUT_T = BASE + "hic_review/review_results/R2_comments/binding_distribution/tables/"
OUT_F = BASE + "hic_review/review_results/R2_comments/binding_distribution/figures/"
GENOME_LEN = 4_411_532
SEED = 42
K = 8
BG_MULT = 5

genome = open(GENOME_FA).read().split("\n", 1)[1].replace("\n", "").upper()
assert len(genome) == GENOME_LEN
comp = str.maketrans("ACGT", "TGCA")
def rc(s):
    return s.translate(comp)[::-1]
def canon(k):
    r = rc(k)
    return k if k <= r else r

def gc_frac(s):
    return (s.count("G") + s.count("C")) / len(s)

def merged_intervals(df):
    iv = df[["start", "end"]].sort_values("start").values
    m = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m

peaks = pd.read_csv(PEAKS, sep="\t")
miv = merged_intervals(peaks)
peak_seqs = [genome[s:e] for s, e in miv]
peak_bp = sum(len(s) for s in peak_seqs)
obs_gc = np.mean([gc_frac(s) for s in peak_seqs])          # interval-mean
obs_gc_bp = sum(s.count("G") + s.count("C") for s in peak_seqs) / peak_bp
genome_gc = gc_frac(genome)

rng = np.random.default_rng(SEED)
# --- length-matched random GC null (1000 sets) ------------------------------
lengths = np.array([e - s for s, e in miv])
n_sets = 1000
null_gc = np.empty(n_sets)
for i in range(n_sets):
    starts = rng.integers(0, GENOME_LEN - lengths.max(), len(lengths))
    # clip lengths that would overrun; sample uniform anyway (genome is circular-ish; ignore edge)
    vals = []
    for st, ln in zip(starts, lengths):
        if st + ln > GENOME_LEN:
            st = GENOME_LEN - ln
        seg = genome[st:st + ln]
        vals.append(seg.count("G") + seg.count("C"))
    null_gc[i] = sum(vals) / lengths.sum()
p_gc = 2 * min((null_gc >= obs_gc_bp).mean(), (null_gc <= obs_gc_bp).mean())

pd.DataFrame([
    ("genome_gc", round(genome_gc, 4)),
    ("peak_gc_bp_weighted", round(obs_gc_bp, 4)),
    ("peak_gc_interval_mean", round(obs_gc, 4)),
    ("null_gc_lenmatched_mean", round(null_gc.mean(), 4)),
    ("null_gc_lenmatched_q025", round(np.quantile(null_gc, 0.025), 4)),
    ("null_gc_lenmatched_q975", round(np.quantile(null_gc, 0.975), 4)),
    ("p_gc_two_sided", round(p_gc, 4)),
    ("peak_bp_total", peak_bp),
    ("n_merged_intervals", len(miv)),
], columns=["metric", "value"]).to_csv(OUT_T + "gc_composition.csv", index=False)

# --- k-mer counting ----------------------------------------------------------
def count_kmers(seqs, k):
    c = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            if "N" not in km:
                c[canon(km)] += 1
    return c

peak_counts = count_kmers(peak_seqs, K)
peak_total = sum(peak_counts.values())

# GC- and length-matched background, 5x
bg_seqs = []
for (s, e), ln in zip(miv, lengths):
    target_gc = gc_frac(genome[s:e])
    got = 0
    best, best_d = None, 1.0
    tries = 0
    while got < BG_MULT and tries < 400:
        st = rng.integers(0, GENOME_LEN - ln)
        seg = genome[st:st + ln]
        d = abs(gc_frac(seg) - target_gc)
        if d < best_d:
            best, best_d = seg, d
        if d <= 0.02:
            bg_seqs.append(seg); got += 1
        tries += 1
    while got < BG_MULT:  # fallback: closest-GC segment found
        bg_seqs.append(best); got += 1
bg_counts = count_kmers(bg_seqs, K)
bg_total = sum(bg_counts.values())

MIN_COUNT = 20  # kmers below this peak-count are too sparse to interpret
n_canon = len(set(list(peak_counts) + list(bg_counts)))
rows = []
for km in set(list(peak_counts) + list(bg_counts)):
    c = peak_counts.get(km, 0)
    bc = bg_counts.get(km, 0)
    # +1 pseudocount against zero-frequency artifacts
    f_p = (c + 1) / (peak_total + n_canon)
    f_b = (bc + 1) / (bg_total + n_canon)
    rows.append((km, c, bc, f_p, f_b, np.log2(f_p / f_b),
                 (km.count("A") + km.count("T")) / K))
kdf = pd.DataFrame(rows, columns=["kmer", "count_peaks", "count_bg",
                                  "freq_peaks", "freq_bg", "log2_enrichment",
                                  "at_frac"])
kdf = kdf.sort_values("log2_enrichment", ascending=False)
kdf.to_csv(OUT_T + "kmer8_enrichment.csv", index=False)
# reliable subset used for reporting/figures
kdf_f = kdf[kdf.count_peaks >= MIN_COUNT]
kdf_f.to_csv(OUT_T + f"kmer8_enrichment_min{MIN_COUNT}.csv", index=False)

# ---- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
ax = axes[0]
ax.hist(null_gc, bins=40, color="lightgrey", edgecolor="white", lw=0.3)
ax.axvline(obs_gc_bp, color="#6a51a3", lw=1.2, label="peaks")
ax.axvline(genome_gc, color="black", ls="--", lw=0.8, label="genome")
ax.set_xlabel("GC fraction"); ax.set_ylabel("count (1000 sets)")
ax.set_title("A  GC vs length-matched null", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7)

top = kdf_f.head(12).iloc[::-1]
ax = axes[1]
ax.barh(top.kmer, top.log2_enrichment, color="#6a51a3", height=0.7)
ax.set_xlabel("log2 enrichment vs GC/len-matched bg")
ax.set_title("B  top enriched 8-mers", loc="left", fontweight="bold")
for lbl in ax.get_yticklabels():
    lbl.set_family("DejaVu Sans Mono"); lbl.set_fontsize(6)

bot = kdf_f.dropna().tail(12).iloc[::-1]
ax = axes[2]
ax.barh(bot.kmer, bot.log2_enrichment, color="#3182bd", height=0.7)
ax.set_xlabel("log2 enrichment (depleted)")
ax.set_title("C  top depleted 8-mers", loc="left", fontweight="bold")
for lbl in ax.get_yticklabels():
    lbl.set_family("DejaVu Sans Mono"); lbl.set_fontsize(6)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_F + "fig4_gc_kmer.png", dpi=300)
fig.savefig(OUT_F + "fig4_gc_kmer.pdf")
print("genome GC", round(genome_gc, 4), "| peak GC(bp)", round(obs_gc_bp, 4),
      "| null", round(null_gc.mean(), 4), "| p", round(p_gc, 4))
print(kdf_f.head(8).to_string(index=False))
print(kdf_f.dropna().tail(8).to_string(index=False))
top_at = kdf_f.head(50).at_frac.mean()
print("mean AT fraction of top-50 enriched 8-mers:", round(top_at, 3),
      "| genome-wide expectation ~0.344")
print("OK")
