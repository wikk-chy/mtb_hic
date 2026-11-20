import csv
import math
from pathlib import Path

SOURCE = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Hypoxia_bin_level.csv")
OUT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/analysis/overlap_analysis_20261103")
OUT.mkdir(parents=True, exist_ok=True)
DETECTED = {"detected_increase", "detected_decrease"}

def as_bool(value):
    return value == "True"

def tested_count(row, tag):
    return int(float(row[f"RNA_{tag}_n_tested_genes"] or 0))

eligible = []
with SOURCE.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if not as_bool(row["NapM_bound"]):
            continue
        if min(tested_count(row, tag) for tag in ("WT_hypoxia", "KO_hypoxia", "interaction")) <= 0:
            continue
        eligible.append({
            "bin": int(row["bin"]),
            "start": int(row["start"]),
            "end": int(row["end"]),
            "structure": row["interaction_ins_class"] in DETECTED,
            "rna": any(as_bool(row[f"RNA_{tag}_any_DEG"]) for tag in ("WT_hypoxia", "KO_hypoxia", "interaction")),
            "evidence_class": row["hypoxia_response_evidence_class"],
        })
eligible.sort(key=lambda row: row["bin"])

a = sum(row["structure"] and row["rna"] for row in eligible)
b = sum(row["structure"] and not row["rna"] for row in eligible)
c = sum(not row["structure"] and row["rna"] for row in eligible)
d = sum(not row["structure"] and not row["rna"] for row in eligible)
n = len(eligible)
expected = (a + b) * (a + c) / n
odds_ratio = (a * d) / (b * c) if b and c else float("inf")

# Fisher exact test on the fixed-margin hypergeometric distribution.
row1 = a + b
col1 = a + c
lo = max(0, row1 + col1 - n)
hi = min(row1, col1)
denominator = math.comb(n, row1)
def probability(x):
    return math.comb(col1, x) * math.comb(n - col1, row1 - x) / denominator
observed_probability = probability(a)
fisher_greater = sum(probability(x) for x in range(a, hi + 1))
fisher_two_sided = sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed_probability * (1 + 1e-12))

# Exhaustive nonzero circular shifts of the RNA-response sequence among eligible bins.
rna = [row["rna"] for row in eligible]
structure = [row["structure"] for row in eligible]
shift_overlaps = []
for shift in range(1, n):
    shifted = rna[-shift:] + rna[:-shift]
    overlap = sum(s and r for s, r in zip(structure, shifted))
    shift_overlaps.append((shift, overlap))
permutation_p = (1 + sum(overlap >= a for _, overlap in shift_overlaps)) / n
shift_values = [overlap for _, overlap in shift_overlaps]
shift_mean = sum(shift_values) / len(shift_values)

with (OUT / "overlap_contingency.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["structure_interaction", "local_RNA_DEG", "n_bins"])
    writer.writerow(["detected", "yes", a])
    writer.writerow(["detected", "no", b])
    writer.writerow(["not_detected", "yes", c])
    writer.writerow(["not_detected", "no", d])

with (OUT / "circular_shift_results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["shift", "overlap_bins"])
    writer.writerows(shift_overlaps)

with (OUT / "overlap_bins.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["bin", "start", "end", "evidence_class"])
    writer.writeheader()
    for row in eligible:
        if row["structure"] and row["rna"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})

summary = [
    ("eligible_NapM_bound_RNA_tested_bins", n),
    ("structure_detected_bins", a + b),
    ("RNA_DEG_bins", a + c),
    ("observed_overlap", a),
    ("expected_overlap_under_independence", expected),
    ("observed_to_expected", a / expected),
    ("odds_ratio", odds_ratio),
    ("fisher_one_sided_enrichment_P", fisher_greater),
    ("fisher_two_sided_P", fisher_two_sided),
    ("circular_shifts_tested", n - 1),
    ("circular_shift_mean_overlap", shift_mean),
    ("circular_shift_one_sided_enrichment_P", permutation_p),
]
with (OUT / "overlap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["metric", "value"])
    writer.writerows(summary)

for key, value in summary:
    print(f"{key}: {value}")
print(f"Contingency: a={a}, b={b}, c={c}, d={d}")
print(f"Outputs: {OUT}")
