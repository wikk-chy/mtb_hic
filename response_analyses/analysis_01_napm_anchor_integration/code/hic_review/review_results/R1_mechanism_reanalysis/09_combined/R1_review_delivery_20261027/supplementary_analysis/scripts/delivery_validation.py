#!/usr/bin/env python3
"""Validate the Reviewer 1 delivery against final figures, tables and documents.

Requires only the Python standard library and pypdf. Run from any directory.
The script prints one line per check and exits non-zero on any failure.
"""
from pathlib import Path
from collections import Counter
import ast
import csv
import math
import re
import sys

from pypdf import PdfReader

DEL = Path(__file__).resolve().parents[2]
OUT = DEL / "supplementary_analysis"
FIG = OUT / "figures"
TAB = OUT / "tables"
SRC = TAB / "corrected_figure_sources"
SCRIPTS = OUT / "scripts"

DOCS = [
    "R1_major_concerns.md",
    "R1_response_draft.md",
    "figure_legends.md",
    "figure_panel_guide_CN.md",
    "analysis_audit.md",
]
EXPECTED_PANELS = {
    1: set("ABCDE"),
    2: set("ABCDEF"),
    3: set("ABCD"),
    4: set("ABCDEFGH"),
}

checks = 0
failures = []


def check(condition, message):
    global checks
    checks += 1
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        failures.append(message)


def rows(relative):
    path = Path(relative)
    if not path.is_absolute():
        path = TAB / path
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def src_rows(name):
    return rows(SRC / name)


def close(actual, expected, tolerance=1e-9):
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def unique_row(data, **criteria):
    hit = [r for r in data if all(r.get(k) == str(v) for k, v in criteria.items())]
    if len(hit) != 1:
        raise AssertionError(f"Expected one row for {criteria}, found {len(hit)}")
    return hit[0]


def contains(text, tokens, label):
    for token in tokens:
        check(token in text, f"{label} contains {token!r}")


print(f"Delivery directory: {DEL}")
print("=" * 72)

# 1. Required documents and their editorial status.
texts = {}
for name in DOCS:
    path = DEL / name
    check(path.is_file(), f"required document exists: {name}")
    if path.is_file():
        texts[name] = path.read_text(encoding="utf-8")
        check(path.stat().st_size > 1000, f"document is non-trivial: {name}")

response = texts.get("R1_response_draft.md", "")
legends = texts.get("figure_legends.md", "")
guide = texts.get("figure_panel_guide_CN.md", "")
audit = texts.get("analysis_audit.md", "")
contains(response, ["[CURRENT SOURCE]", "[TO BE INSERTED", "[TO BE REVISED", "[TO BE DELETED"],
         "response draft")

# 2. PDFs are single-page and contain all panel labels.
pdf_text = {}
for number, expected in EXPECTED_PANELS.items():
    path = FIG / f"Fig{number}_corrected.pdf"
    check(path.is_file(), f"corrected PDF exists: Fig{number}")
    if not path.is_file():
        continue
    reader = PdfReader(str(path))
    check(len(reader.pages) == 1, f"Fig{number} is a single-page PDF")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    pdf_text[number] = text
    labels = set(re.findall(r"(?m)^\s*([A-H])\s*$", text))
    # In Fig4, pypdf merges the locus labels with the adjacent WT track
    # ("WTE"–"WTH") and merges C with the following legend entry.
    labels.update(re.findall(r"(?m)^WT([E-H])$", text))
    for label in expected - labels:
        if re.search(rf"(?m)^{label}\s", text):
            labels.add(label)
    check(expected.issubset(labels),
          f"Fig{number} panel labels include {''.join(sorted(expected))}; found {''.join(sorted(labels))}")

contains(pdf_text.get(3, ""), ["825 matched pairs", "+0.023", "no spatial P reported",
                              "unresolved ≠ unchanged"], "Fig3 PDF text")
for stale in ("817", "122/269", "119/269", "circular shift"):
    check(stale not in pdf_text.get(3, ""), f"Fig3 PDF excludes stale text {stale!r}")
contains(pdf_text.get(4, ""), ["pairwise interval", "315/883", "90% CI rule",
                              "descriptive only."], "Fig4 PDF text")

# 3. Every delivery script is syntactically valid; required scripts are present.
required_scripts = [
    "01_independent_sample_and_response_audit.py",
    "02_spatial_correlation_audit.py",
    "03_interval_binding_sensitivity.py",
    "04_nested_cv_complete_case.py",
    "05_fig3c_matched_effects.py",
    "fig1_corrected.py",
    "fig2_corrected.py",
    "fig3_corrected.py",
    "fig4_corrected.py",
    "delivery_validation.py",
]
for name in required_scripts:
    check((SCRIPTS / name).is_file(), f"required script exists: {name}")
for path in sorted(SCRIPTS.glob("*.py")):
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        ok = True
    except SyntaxError as exc:
        print(f"       {path.name}: {exc}")
        ok = False
    check(ok, f"Python syntax compiles: {path.name}")

# 4. All source CSVs explicitly named in the Chinese guide exist.
for name in sorted(set(re.findall(r"`([^`/]+\.csv)`", guide))):
    check((SRC / name).is_file(), f"guide source table exists: {name}")

# 5. Fig. 1 critical values and binding definitions.
brid = unique_row(src_rows("Fig1_final_D_loop_bridging_primary.csv"), definition="interval", metric="both")
check(close(brid["observed"], 17), "Fig1D observed both-anchor loops = 17")
check(close(brid["null_mean"], 10.897), "Fig1D null mean = 10.897")
check(close(brid["fold_enrichment"], 1.56), "Fig1D fold = 1.56")
check(close(brid["p_empirical"], 0.0328), "Fig1D circular-shift P = 0.0328")
check(len(src_rows("Fig1_final_D_null_both_anchor_counts.csv")) == 10000,
      "Fig1D exports 10,000 null counts")

loop_e = src_rows("Fig1_final_E_loop_strength_stats.csv")[0]
check((int(loop_e["n_bound"]), int(loop_e["n_other"])) == (16, 23),
      "Fig1E groups = 16 bound and 23 other loops")
check(close(loop_e["mean_diff"], 0.0589739086974784), "Fig1E effect matches source")
check(close(loop_e["boot_ci_lo"], -0.2029204933578728), "Fig1E CI lower matches source")
check(close(loop_e["boot_ci_hi"], 0.3345179821930129), "Fig1E CI upper matches source")
check(close(loop_e["permutation_p"], 0.6853314668533147), "Fig1E permutation P matches source")

rob = src_rows("Fig1_final_C_robustness_18variants.csv")
cross = [r for r in rob if float(r["ci_lo"]) <= 0 <= float(r["ci_hi"])]
exceptions = [r for r in rob if r not in cross]
check(len(rob) == 18 and len(cross) == 16, "Fig1C robustness has 16/18 intervals crossing zero")
check(len(exceptions) == 2 and all(float(r["effect"]) < 0 for r in exceptions),
      "Fig1C both robustness exceptions are negative")

qc = rows("binding_definition_matching_qc.csv")
interval_qc = unique_row(qc, binding_definition="interval_454")
check(int(interval_qc["n_bound_feature_complete"]) == 442, "interval binding has 442 feature-complete bins")
check(int(interval_qc["n_pairs"]) == 406, "interval binding has 406 matched pairs")
check(int(interval_qc["n_unmatched_bound"]) == 36, "interval binding has 36 unmatched bound bins")
check(int(interval_qc["n_caliper_exceeded"]) == 107, "interval binding retains 107 caliper-exceeded pairs")

# 6. Fig. 2 unified response, exact independent-sample test and fingerprint.
quad = {r["quadrant"]: int(r["n_bins"]) for r in src_rows("Fig2_final_C_quadrant_counts.csv")}
check(quad == {"concordant": 154, "attenuated": 78, "enhanced": 187, "sign_reversed": 462},
      "Fig2C quadrant counts = 154/78/187/462")
check(sum(quad.values()) == 881, "Fig2C quadrant counts sum to 881")

factorial = src_rows("Fig2_final_D_independent_sample_factorial.csv")
wt = unique_row(factorial, test="WT genome-mean hypoxia response")
ko = unique_row(factorial, test="KO genome-mean hypoxia response")
did = unique_row(factorial, test="genotype contrast of genome-mean hypoxia responses")
check(close(wt["effect"], 0.040975775304809836), "Fig2D WT hypoxia response matches source")
check(close(ko["effect"], -0.06701369051872055), "Fig2D KO hypoxia response matches source")
check(close(did["effect"], -0.10798946582353039), "Fig2D DID matches source")
check(close(did["ci_lo"], -0.11402510176988985), "Fig2D DID CI lower matches source")
check(close(did["ci_hi"], -0.10150013086226893), "Fig2D DID CI upper matches source")
check(close(did["p"], 0.00040816326530612246), "Fig2D exact permutation P = 1/2450")
check(int(did["n_samples"]) == 15 and "2450" in did["method"],
      "Fig2D uses 15 samples and 2,450 assignments")
check("no oxygen-condition effect within either genotype" in did["null"],
      "Fig2D sharp null is stated correctly")

fstats = {r["statistic"]: float(r["observed"]) for r in src_rows("Fig2_final_E_response_stats.csv")}
check(close(fstats["cosine_similarity"], -0.0696280991609931), "Fig2E cosine matches source")
check(close(fstats["euclidean_dist"], 5.950629545272408), "Fig2E Euclidean distance matches source")
check(close(fstats["perm_p_cosine"], 0.3650634936506349), "Fig2E cosine P matches source")
check(close(fstats["perm_p_euclidean"], 0.0007999200079992), "Fig2E Euclidean P matches source")

interval_stats = rows("interval_binding_matched_sensitivity.csv")
frow = unique_row(interval_stats, binding_definition="interval_454", metric="hypoxia_interaction_signed")
check(int(frow["n_pairs"]) == 406, "Fig2F has 406 matched pairs")
check(close(frow["effect"], 0.011348913769828347), "Fig2F paired effect matches source")
check(close(frow["circular_shift_p"], 0.650934906509349), "Fig2F circular-shift P matches source")
check(len(src_rows("Fig2_final_F_paired_interaction.csv")) == 406,
      "Fig2F per-pair source contains 406 rows")

# 7. Fig. 3 complete-case CV and recomputed matched effects.
models = src_rows("fig3_final_panelB_nested_cv.csv")
expected_models = {
    "M0_baseline": 0.6704972448218771,
    "M1_plus_RNA": 0.6707063843928396,
    "M2_plus_occupancy": 0.6708795896233134,
}
for model, value in expected_models.items():
    row = unique_row(models, model=model)
    check(close(row["cv_R2"], value), f"Fig3B {model} uses final complete-case CV")
    check(int(row["n_bins"]) == 848 and "none" in row["imputation"],
          f"Fig3B {model} uses 848 complete bins without imputation")

cv_cmp = src_rows("fig3_final_panelB_nested_cv_comparison.csv")
rna_inc = unique_row(cv_cmp, comparison="M1_plus_RNA - M0_baseline")
occ_inc = unique_row(cv_cmp, comparison="M2_plus_occupancy - M1_plus_RNA")
check(close(rna_inc["dR2"], 0.00020913957096246882), "Fig3B RNA increment matches source")
check(close(rna_inc["ci_lo"], -0.0006174379721674839) and close(rna_inc["ci_hi"], 0.002931003501848556),
      "Fig3B RNA increment interval matches source")
check(close(occ_inc["dR2"], 0.0001732052304738163), "Fig3B occupancy increment matches source")
check(close(occ_inc["ci_lo"], -0.0012704692484820312) and close(occ_inc["ci_hi"], 0.002399691199418227),
      "Fig3B occupancy increment interval matches source")

matched = src_rows("fig3_final_panelC_matched_summary.csv")
expected_matched = {
    ("gene", "|RNA log2FC|"): (825, 0.022951607436655825, -0.035560588869716386, 0.08139584529077666),
    ("gene", "P(DEG)"): (825, -0.013333333333333334, -0.05090909090909091, 0.024242424242424242),
    ("operon", "|RNA log2FC|"): (148, -0.009847638115355396, -0.13408937985810732, 0.11663648401790819),
    ("operon", "DEG fraction"): (148, -0.04480373230373231, -0.13006756756756757, 0.04247104247104247),
}
for (level, metric), (n, effect, lo, hi) in expected_matched.items():
    row = unique_row(matched, level=level, metric=metric)
    check(int(row["n_pairs"]) == n, f"Fig3C {level} {metric} has {n} finite pairs")
    check(close(row["observed"], effect), f"Fig3C {level} {metric} effect matches recomputation")
    check(close(row["ci_lo"], lo) and close(row["ci_hi"], hi),
          f"Fig3C {level} {metric} bootstrap interval matches recomputation")
    check(not row["p"].strip() and row["test"] == "not reported",
          f"Fig3C {level} {metric} has no retained spatial P")

gene_pairs = src_rows("fig3c_gene_pairs_recomputed.csv")
check(len(gene_pairs) == 825, "Fig3C gene detail contains 825 pairs")
gene_diff = [float(r["bound_abslog2FC"]) - float(r["matched_abslog2FC"]) for r in gene_pairs]
check(close(sum(gene_diff) / len(gene_diff), 0.022951607436655825),
      "Fig3C gene detail independently reproduces the |log2FC| effect")
check(len(src_rows("fig3c_operon_pairs_recomputed.csv")) == 148,
      "Fig3C operon detail contains 148 pairs")

classification = {r["category"]: r for r in src_rows("fig3_final_panelD_classification.csv")}
expected_counts = {"struct+RNA+": 144, "struct+RNA-": 3, "struct-RNA+": 308,
                   "struct-RNA-": 5, "struct+undet": 122, "struct-undet": 301}
check({k: int(v["n"]) for k, v in classification.items()} == expected_counts,
      "Fig3D six classification counts match the final rule")
check(classification["struct+undet"]["composition"] == "119 cannot_determine + 3 no_data",
      "Fig3D structure+ unresolved composition = 119 + 3")
check(classification["struct-undet"]["composition"] == "295 cannot_determine + 6 no_data",
      "Fig3D structure- unresolved composition = 295 + 6")

# 8. Fig. 4 spatial association and distinct classification rules.
spatial = unique_row(rows("fig4A_spatial_correlation_summary.csv"), analysis="primary_pairwise_median_contrasts")
check(close(spatial["estimator"], 0.5265262776248668), "Fig4A primary rho matches source")
check(close(spatial["ci_lo"], 0.4381324120246338) and close(spatial["ci_hi"], 0.6092263594456491),
      "Fig4A 100-kb block CI matches source")
check(close(spatial["p"], 0.0010998900109989002), "Fig4A random circular-shift P matches source")
rot = unique_row(rows("fig4A_spatial_correlation_summary.csv"), analysis="primary_exhaustive_rotation_sensitivity")
check(close(rot["p"], 0.0011325028312570782) and int(float(rot["n_permutations"])) == 883,
      "Fig4A exhaustive 883-rotation P matches source")
check(len(src_rows("Fig4_v2_A_baseline_vs_hypoxia.csv")) == 881,
      "Fig4A source has 881 finite bins")

overlap = {r["category"]: int(r["n_bins"]) for r in src_rows("Fig4_v2_B_response_class_counts.csv")}
check(overlap == {"neither": 35, "baseline only": 164, "hypoxia only": 90, "both (shared)": 594},
      "Fig4B overlap counts = 35/164/90/594")
check(overlap["both (shared)"] + overlap["hypoxia only"] == 684,
      "Fig4B hypoxia-responsive denominator = 684")

interaction = src_rows("Fig4_v2_C_interaction_per_bin.csv")
class_counts = Counter(r["interaction_class"] for r in interaction)
check(len(interaction) == 883, "Fig4C source contains all 883 bins")
check(class_counts == Counter({"detected_increase": 79, "detected_decrease": 236,
                               "supports_small_change": 455, "cannot_determine": 113}),
      "Fig4C interaction classes = 79/236/455/113")
fdr = [r for r in interaction
       if r["interaction_padj"].strip() and float(r["interaction_padj"]) < 0.05]
detected = [r for r in interaction if r["interaction_class"] in {"detected_increase", "detected_decrease"}]
fdr_not_detected = [r for r in fdr if r["interaction_class"] not in {"detected_increase", "detected_decrease"}]
check(len(detected) == 315, "Fig4C has 315 CI/effect-detected bins")
check(len(fdr) == 736, "Fig4C has 736 FDR-significant bins")
check(len(fdr_not_detected) == 422, "Fig4C has 422 FDR-significant but not CI/effect-detected bins")
values = sorted(abs(float(r["d_ins_interaction"])) for r in interaction
                if r["d_ins_interaction"].strip())
check(len(values) == 881, "Fig4C median uses 881 finite interaction values")
mid = len(values) // 2
median_value = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
check(close(median_value, 0.2003, tolerance=0.00005), "Fig4C median |interaction| = 0.2003")

# 9. Canonical documents contain the final values and distinguish the rules.
contains(legends, ["17/41", "10.897", "+0.05897", "2,450", "0.000408",
                   "154", "78", "187", "462", "5.95063", "0.670497", "0.670706",
                   "0.670880", "+0.02295", "825 finite matched pairs", "0.52653",
                   "594/684", "736", "422", "315"], "figure legends")
contains(guide, ["825 对基因", "148 对操纵子", "423 个 bins", "122／301",
                 "两类比较都未达到变化规则", "315 或 736", "Extended Data Fig. 3A fingerprint 热图"],
         "Chinese panel guide")

fig3d_legend = legends.split("**(D)** Joint classification", 1)[-1].split("## Rebuttal Figure 4", 1)[0]
fig4b_legend = legends.split("**(B)** Exploratory overlap", 1)[-1].split("**(C)**", 1)[0]
fig4c_legend = legends.split("**(C)** Distribution and classification", 1)[-1].split("**(D)**", 1)[0]
check("beyond +0.25 or below −0.25" in fig3d_legend,
      "Fig3D legend retains the ±0.25 structure-detected rule")
check("does not impose the ±0.25" in fig4b_legend and "5th–95th" in fig4b_legend,
      "Fig4B legend uses pairwise intervals without the ±0.25 rule")
check("Welch" in fig4c_legend and "1.645" in fig4c_legend and "+0.25" in fig4c_legend,
      "Fig4C legend retains Welch CI and effect-threshold rules")

# 10. Historical invalid values are confined to the audit's history section.
legacy_tokens = [
    "3.9e-33", "3.865589", "5.5e-64", "5.466368", "119/269", "17/18",
    "0.6712", "0.6713", "0.6711", "817 pairs", "n = 817",
    "−0.02938", "-0.0293847", "−0.03636", "-0.0363636",
    "p = 0.419", "p = 0.117",
]
for name, text in texts.items():
    if name == "analysis_audit.md":
        continue
    lowered = text.casefold()
    for token in legacy_tokens:
        check(token.casefold() not in lowered,
              f"historical token {token!r} is absent from {name}")
contains(audit, ["3.9e-33", "5.5e-64", "119/269", "17/18", "n = 817",
                 "0.6712 / 0.6713 / 0.6711"], "analysis audit history")

print("=" * 72)
if failures:
    print(f"VALIDATION FAILED: {len(failures)} of {checks} checks failed")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)
print(f"VALIDATION PASSED: {checks} checks passed")
print("Five documents, four single-page corrected figures, scripts and critical source values are consistent.")
