import csv
from collections import Counter, defaultdict
from math import comb

path = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Mechanism_bin_evidence_classes.csv"
counts = defaultdict(Counter)
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        counts[row["hypoxia_response_evidence_class"]][row["NapM_bound"] == "True"] += 1

for evidence_class, values in sorted(counts.items()):
    bound = values[True]
    unbound = values[False]
    total = bound + unbound
    print(f"{evidence_class}\tbound={bound}\tunbound={unbound}\ttotal={total}\tbound_pct={100*bound/total:.1f}%")

structural = "NapM-associated_candidate_no_detected_local_RNA_response"
transcription = "Transcription-associated_consistent_pattern"
a = counts[structural][True]
b = counts[structural][False]
c = counts[transcription][True]
d = counts[transcription][False]
N = a + b + c + d
K = a + c
n = a + b
p = sum(comb(n, x) * comb(N - n, K - x) for x in range(a, min(n, K) + 1)) / comb(N, K)
print(f"Structural vs transcription: [{a},{b};{c},{d}], one-sided Fisher enrichment P={p:.6g}")
