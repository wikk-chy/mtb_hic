#!/usr/bin/env python3
"""Validate the Reviewer 1 stage-1 audit and its editorial scope.

This script checks files and text only. It does not modify the manuscript,
figures, source tables, or any scientific result.
"""
from pathlib import Path
import re
import sys

from pypdf import PdfReader

DEL = Path(__file__).resolve().parents[2]
ROOT = DEL.parents[4]
OUT = DEL / "supplementary_analysis"
PROMPT_PATH = DEL / "R1_reanalysis_prompt_revised.md"
AUDIT_PATH = DEL / "R1_stage1_claim_evidence_audit.md"
MAIN_PATH = ROOT / "Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis" / "main.tex"
GUIDE_PATH = DEL / "figure_panel_guide_CN.md"
RESPONSE_PATH = DEL / "R1_response_draft.md"
ED_HEATMAP = DEL.parent / "figures" / "ED_Fig3A_fingerprint_heatmap.pdf"

checks = 0
failures = []


def require(label, condition):
    global checks
    checks += 1
    if condition:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        failures.append(label)


def contains(text, tokens, label):
    for token in tokens:
        require(f"{label}: {token}", token in text)


for path in (PROMPT_PATH, AUDIT_PATH, MAIN_PATH, GUIDE_PATH, RESPONSE_PATH, ED_HEATMAP):
    require(f"required file exists: {path.name}", path.is_file())

prompt = PROMPT_PATH.read_text(encoding="utf-8")
audit = AUDIT_PATH.read_text(encoding="utf-8")
main = MAIN_PATH.read_text(encoding="utf-8")
guide = GUIDE_PATH.read_text(encoding="utf-8")
response = RESPONSE_PATH.read_text(encoding="utf-8")

contains(prompt, [
    "正文保留现有 SVD KO-latent",
    "本轮 Response 不讨论、不评价、不新增这些推断",
    "不开展更深入的机制研究",
    "不进入本轮英文 Response",
    "本任务不新增 SVD 计算",
    "保留 Extended Data Fig. 3A",
    "共享 WT 参照的敏感性",
    "GO/KEGG 改为条件触发，默认不新增",
], "revised prompt scope")
for stale in ("不保留依赖该投影", "不再计算补出反事实结构", "删除 SVD KO-latent 反事实结构"):
    require(f"revised prompt excludes stale SVD instruction: {stale}", stale not in prompt)

contains(audit, [
    "## 四、主张—证据—限制—修改审计表",
    "## 六、`main.tex` 修订清单",
    "## 七、图件保留、移动与停止使用决策",
    "## 八、阶段 2 是否需要新增分析",
    "未修改 `main.tex`",
    "未修改四张 corrected PDF",
    "未重写完整英文 Response",
    "正文保留**现有 SVD KO-latent",
    "本轮 Response 不讨论、不评价、不新增这些推断",
    "保留结果”不等于自动保留所有过强因果措辞",
    "尚未同步的旧文档内容",
    "唯一优先项：Fig. 4A 共享 WT 参照敏感性",
    "尚未解决两个 contrast 共享 WT 参照的数学耦合",
    "Extended Data Fig. 3A 九指标 fingerprint 热图",
], "stage-1 audit required sections")

contains(audit, [
    "17／41，null 10.897，fold 1.56，P = 0.0328",
    "+0.05897，95% CI［−0.20292，+0.33452］，P = 0.685",
    "WT +0.0409758，KO −0.0670137，DID −0.1079895",
    "P = 0.000408163",
    "Euclidean 5.95063，P = 0.000800",
    "cosine −0.06963，P = 0.365",
    "+0.0113489，circular-shift P = 0.650935",
    "M0／M1／M2＝0.670497／0.670706／0.670880",
    "RNA 增量 +0.000209，CI［−0.000617，+0.002931］",
    "825 对基因、148 对操纵子",
    "144／3／308／5／122／301",
    "122＝119＋3，301＝295＋6",
    "ρ = 0.526526，CI［0.438132，0.609226］，shift P = 0.00109989",
    "883 rotations P = 0.00113250",
    "35／164／90／594；594／684＝86.8%",
    "79／236／455／113；detected 315；FDR 736",
    "442 个 feature-complete，406 个匹配成功，36 个未匹配，107 对超过诊断 caliper",
    "16／18 个普通 matched-pair 区间跨 0，不是 17／18",
], "stage-1 audit frozen values")

# The main audit table must have a header, a separator and 17 claim rows,
# each with eight columns. Escaped absolute-value bars do not split cells.
lines = audit.splitlines()
header = "| 论文主张 | 对应数据与源表 | 独立分析单位 | 当前统计是否合适 | 可支持表述 | 不可支持表述 | 保留／补充／停止 | 是否需新增分析 |"
require("main audit header is exact", header in lines)
if header in lines:
    start = lines.index(header)
    table_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        table_lines.append(line)
    require("main audit contains 17 claim rows", len(table_lines) == 19)
    for index, line in enumerate(table_lines, start=1):
        pipe_count = len(re.findall(r"(?<!\\)\|", line))
        require(f"main audit row {index} has eight columns", pipe_count == 9)

# Verify the manuscript locations quoted in the audit against the current file.
main_lines = main.splitlines()
quoted_lines = {
    63: ("essential for maintaining chromatin architectural plasticity", "dormancy establishment"),
    71: ("including CTCF and cohesin",),
    86: ("41 Chromosight-called loops", "30--995"),
    97: ("null mean of 10.66", "$P = 0.015$"),
    121: ("NapM is essential for chromosome architecture plasticity",),
    130: ("singular value decomposition (SVD)-based framework",),
    141: ("essential for maintaining this architectural plasticity", "dormancy establishment"),
    176: ("\\texttt{hiccups}", "Hidden Markov Model (HMM)"),
    333: ("counterfactual", "\\label{sec:svd}"),
    437: ("41 Chromosight-called loops", "5-kb resolution", "100-kb directional window"),
    451: ("41 validated loops",),
    452: ("10.66", "NapM-bridged"),
    477: ("NapM is required for chromosome architecture plasticity",),
    483: ("Conceptual model", "compatible with viability"),
    485: ("Counterfactual Hi-C contact matrix", "SVD"),
    486: ("RMSD", "per-bin displacement"),
}
for line_number, tokens in quoted_lines.items():
    text = main_lines[line_number - 1] if len(main_lines) >= line_number else ""
    for token in tokens:
        require(f"main.tex line {line_number} contains {token}", token in text)

for stale in ("3.9e-33", "5.5e-64", "0.6712", "0.6713", "0.6711", "17/18", "119/269", "showed exhibited"):
    require(f"main.tex excludes checked stale token: {stale}", stale not in main)

# Stage 1 explicitly records, rather than silently rewrites, old SVD advice.
require("old response still contains superseded SVD removal advice", "remove the SVD-derived KO-latent counterfactual map" in response)
require("old Chinese guide still contains superseded deletion advice", "整段删除，不只是移到补充材料" in guide)
contains(audit, [
    "`R1_response_draft.md`",
    "`figure_panel_guide_CN.md`",
    "`R1_major_concerns.md`",
    "`analysis_audit.md`",
    "`figure_legends.md` 与现有 corrected PDF",
    "均未在本阶段擅自改写",
], "pending editorial synchronization")

# Existing package validation logs must be the observed successful versions.
delivery_log = (OUT / "logs" / "delivery_validation.log").read_text(encoding="utf-8")
guide_log = (OUT / "logs" / "plain_guide_validation.log").read_text(encoding="utf-8")
require("delivery validation log reports 243 passes", "VALIDATION PASSED: 243 checks passed" in delivery_log)
require("delivery validation log has no failure", "[FAIL]" not in delivery_log and "VALIDATION FAILED" not in delivery_log)
require("plain guide validation log reports 113 passes", "PLAIN GUIDE VALIDATION PASSED: 113 checks passed" in guide_log)
require("plain guide validation log has no failure", "[FAIL]" not in guide_log)

# Confirm that stage 1 did not silently remove latency-related material from PDFs.
for number in range(1, 5):
    require(f"corrected PDF remains present: Fig{number}", (OUT / "figures" / f"Fig{number}_corrected.pdf").is_file())
fig4_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(OUT / "figures" / "Fig4_corrected.pdf")).pages)
require("existing Fig4 PDF still contains its latency boundary panel", "latency" in fig4_text.casefold())
require("existing Fig4 PDF still says KO-latent architecture cannot be inferred",
        "No KO-latent" in fig4_text and "or inferred." in fig4_text)
require("standalone ED Fig3A heatmap is retained", ED_HEATMAP.stat().st_size > 1000)

print("=" * 72)
if failures:
    print(f"STAGE1 VALIDATION FAILED: {len(failures)} of {checks} checks failed")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)
print(f"STAGE1 VALIDATION PASSED: {checks} checks passed")
print("Stage-1 scope, audit table, frozen values, manuscript locations and pending editorial synchronization are consistent.")
