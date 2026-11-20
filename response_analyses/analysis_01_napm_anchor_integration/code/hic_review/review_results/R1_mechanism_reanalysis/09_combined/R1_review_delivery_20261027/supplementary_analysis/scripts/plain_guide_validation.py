from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "figure_panel_guide_CN.md"
text = GUIDE.read_text(encoding="utf-8")
checks = 0


def require(label, condition):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


panels = [
    *[f"Fig. 1{x}" for x in "ABCDE"],
    *[f"Fig. 2{x}" for x in "ABCDEF"],
    *[f"Fig. 3{x}" for x in "ABCD"],
    *[f"Fig. 4{x}" for x in "ABCD"],
    *[f"Fig. 4{x}" for x in "EFGH"],
]
for panel in panels:
    require(f"plain guide covers {panel}", f"### {panel}" in text or f"#### {panel}" in text)

require("all 23 panels are covered", len(panels) == 23)
require("four-figure overview is present", "## 一、先用四句话理解四张图" in text)
require("invalid-results section is present", "## 七、哪些旧结果确实应停止使用" in text)
require("prioritization section is present", "## 八、当前修正图中，优先去掉或移入补充材料的内容" in text)
require("negative-results protection is present", "## 九、哪些结果不能因为“不显著”就当作无效结果删除" in text)
require("recommendations are marked as not yet applied", "下文“删除／移入补充”均是编辑建议，不表示已经执行" in text)
require("main.tex is not claimed to be modified", "`main.tex` 和四张 PDF 尚未因本文的删减建议而改动" in text)

required_text = {
    "independent sample size": "共 15 个",
    "independent factorial exact P": "P = 0.000408",
    "WT replicate consistency": "0.940",
    "KO replicate consistency": "0.977",
    "WT mean response": "+0.041",
    "KO mean response": "−0.067",
    "DID summary": "−0.108",
    "DID interval": "−0.114 至 −0.102",
    "multivariate magnitude P": "P = 0.0008",
    "cosine direction P": "P = 0.365",
    "Fig2F matched pairs": "406 对",
    "Fig2F spatial P": "P = 0.651",
    "Fig1D observed loops": "有 17 个的两个锚点都有 NapM",
    "Fig1D null mean": "10.9 个",
    "Fig1D enrichment": "1.56 倍",
    "Fig1D spatial P": "P 约为 0.033",
    "Fig1E effective loops": "39 个具有有效覆盖的 loop",
    "Fig1E effect": "+0.059",
    "Fig1E interval": "−0.203 至 +0.335",
    "Fig1E P": "P = 0.685",
    "Fig2C sign-reversal numerator": "462／881",
    "Fig2C sign-reversal percent": "52.4%",
    "Fig3A correlation": "−0.024",
    "Fig3B baseline CV": "0.670497",
    "Fig3B RNA CV": "0.670706",
    "Fig3B occupancy CV": "0.670880",
    "Fig3B RNA increment": "0.000209",
    "Fig3C corrected gene pairs": "825 对基因",
    "Fig3C corrected operon pairs": "148 对操纵子",
    "Fig3D resolved counts": "144 个",
    "Fig3D structure-positive RNA-negative": "3 个",
    "Fig3D structure-negative RNA-positive": "308 个",
    "Fig3D double-negative": "5 个",
    "Fig3D unresolved total": "423 个 bins",
    "Fig3D low-power group": "只有 8 个",
    "Fig3D Fisher P": "P = 0.714",
    "Fig4A correlation": "Spearman ρ = 0.527",
    "Fig4A block interval": "0.438–0.609",
    "Fig4A spatial P": "P 约为 0.0011",
    "Fig4B overlap": "594／684",
    "Fig4B percentage": "86.8%",
    "Fig4C increase": "79 个",
    "Fig4C decrease": "236 个",
    "Fig4C detected total": "合计 315 个",
    "Fig4C supports small": "455 个",
    "Fig4C undetermined": "113 个",
    "Fig4C FDR count": "736 个",
    "robustness corrected count": "16／18",
    "unresolved composition": "122／301",
}
for label, token in required_text.items():
    require(label, token in text)

for token in [
    "3.9e-33",
    "5.5e-64",
    "0.6712",
    "0.6713",
    "0.6711",
    "n = 817",
    "17/18",
    "119/269",
    "−0.02938",
    "-0.0293847",
]:
    require(f"historical invalid token {token} is not repeated", token not in text)

for phrase in [
    "不能证明 NapM 直接桥接 DNA",
    "不能排除远端调控、全局转录状态或其他间接机制",
    "区间跨 0 不是“证明没有效果”",
    "不能由其他条件投影出 KO-latent 结构",
    "不能把 315 或 736 当成独立生物样本数",
    "不能据此声称“52.4% 的区域发生了有意义的机制反转”",
    "不能解释为“86.8% 的区域由同一机制控制”",
]:
    require(f"evidence boundary: {phrase}", phrase in text)

require("ED Fig3A heatmap is explicitly retained", "| **Extended Data Fig. 3A fingerprint 热图** |" in text)
require("PCA compression does not delete fingerprint analysis", "保留九指标幅度分析及 Extended Data Fig. 3A 热图" in text)
require("standalone ED Fig3A PDF remains present", (ROOT.parent / "figures/ED_Fig3A_fingerprint_heatmap.pdf").is_file())

priority_panels = ["Fig. 4B", "Fig. 2C", "Fig. 3D", "Fig. 4C", "Fig. 2E 的 PCA 展示"]
positions = []
for rank, panel in enumerate(priority_panels, start=1):
    token = f"| **{rank}** | **{panel}** |"
    require(f"priority {rank} is {panel}", token in text)
    positions.append(text.index(token))
require("priority ordering is monotonic", positions == sorted(positions))

for filename in [
    "R1_major_concerns.md",
    "R1_response_draft.md",
    "figure_legends.md",
    "analysis_audit.md",
    "supplementary_analysis/figures/Fig1_corrected.pdf",
    "supplementary_analysis/figures/Fig2_corrected.pdf",
    "supplementary_analysis/figures/Fig3_corrected.pdf",
    "supplementary_analysis/figures/Fig4_corrected.pdf",
]:
    require(f"related deliverable remains present: {filename}", (ROOT / filename).is_file())

print("=" * 72)
print(f"PLAIN GUIDE VALIDATION PASSED: {checks} checks passed")
print("All 23 panels, key values, evidence boundaries and removal priorities are present.")
