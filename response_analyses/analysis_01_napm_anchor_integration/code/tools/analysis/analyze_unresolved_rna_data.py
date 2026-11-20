import csv

path = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Hypoxia_bin_level.csv"
detected = {"detected_increase", "detected_decrease"}
rows = []
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["hypoxia_response_evidence_class"] != "Mixed_or_unresolved":
            continue
        tested_counts = [int(float(row[f"RNA_{tag}_n_tested_genes"] or 0)) for tag in ("WT_hypoxia", "KO_hypoxia", "interaction")]
        row["local_rna_tested"] = min(tested_counts) > 0
        row["tested_counts"] = tested_counts
        row["napm_int"] = row["NapM_bound"] == "True" and row["interaction_ins_class"] in detected
        rows.append(row)

insufficient = [row for row in rows if not row["local_rna_tested"]]
direct = [row for row in insufficient if row["napm_int"]]
print(f"Mixed_or_unresolved total: {len(rows)}")
print(f"Local RNA insufficient: {len(insufficient)} ({100*len(insufficient)/len(rows):.1f}%)")
print(f"Local RNA tested: {len(rows)-len(insufficient)} ({100*(len(rows)-len(insufficient))/len(rows):.1f}%)")
print(f"RNA insufficient AND NapM-bound with significant structural interaction: {len(direct)}")
print(f"RNA insufficient but also unbound or without significant structural interaction: {len(insufficient)-len(direct)}")
print("Tested-gene-count patterns among RNA-insufficient bins:")
patterns = {}
for row in insufficient:
    pattern = tuple(row["tested_counts"])
    patterns[pattern] = patterns.get(pattern, 0) + 1
for pattern, count in sorted(patterns.items(), key=lambda item: (-item[1], item[0])):
    print(f"  WT/KO/interaction={pattern}: {count}")
