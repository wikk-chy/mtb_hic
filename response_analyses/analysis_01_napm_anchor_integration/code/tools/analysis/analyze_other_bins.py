import csv
from collections import Counter

path = "/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/hypoxia_latency_integration_20261030/source_data/Hypoxia_bin_level.csv"
detected = {"detected_increase", "detected_decrease"}
rows = []
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        evidence = row["hypoxia_response_evidence_class"]
        if evidence not in {"Mixed_or_unresolved", "No_detected_structural_response"}:
            continue
        def boolean(name):
            return row[name] == "True"
        def count(name):
            return int(float(row[name] or 0))
        row["any_rna_deg"] = any(boolean(f"RNA_{tag}_any_DEG") for tag in ("WT_hypoxia", "KO_hypoxia", "interaction"))
        row["local_rna_tested"] = min(count(f"RNA_{tag}_n_tested_genes") for tag in ("WT_hypoxia", "KO_hypoxia", "interaction")) > 0
        row["napm_bound"] = boolean("NapM_bound")
        row["int_sd"] = row["interaction_ins_class"] in detected
        row["wt_sd"] = row["WT_hypoxia_ins_class"] in detected
        row["ko_sd"] = row["KO_hypoxia_ins_class"] in detected
        rows.append(row)

for evidence in ("Mixed_or_unresolved", "No_detected_structural_response"):
    group = [r for r in rows if r["hypoxia_response_evidence_class"] == evidence]
    print(f"\n{evidence}: {len(group)}")
    print(f"  any RNA DEG: {sum(r['any_rna_deg'] for r in group)}")
    print(f"  NapM-bound: {sum(r['napm_bound'] for r in group)}")
    print(f"  significant structural interaction: {sum(r['int_sd'] for r in group)}")
    print(f"  structural response in WT or KO: {sum(r['wt_sd'] or r['ko_sd'] for r in group)}")

mixed = [r for r in rows if r["hypoxia_response_evidence_class"] == "Mixed_or_unresolved"]
categories = Counter()
for row in mixed:
    if not row["local_rna_tested"]:
        key = "RNA untested"
    elif not row["napm_bound"]:
        key = "RNA tested, NapM unbound"
    elif not row["int_sd"]:
        key = "RNA tested, NapM-bound, no significant interaction structure"
    else:
        key = "RNA tested, NapM-bound, significant interaction structure"
    categories[key] += 1
print("\nMixed_or_unresolved mutually exclusive breakdown:")
for key, value in categories.items():
    print(f"  {key}: {value}")

no_structure = [r for r in rows if r["hypoxia_response_evidence_class"] == "No_detected_structural_response"]
print("\nNo_detected_structural_response RNA breakdown:")
print(f"  with any RNA DEG: {sum(r['any_rna_deg'] for r in no_structure)}")
print(f"  without any RNA DEG: {sum(not r['any_rna_deg'] for r in no_structure)}")
