#!/usr/bin/env python3
"""Test whether hypoxia reduces absolute WT/NapM-KO insulation differences.

Primary effect: |KO-WT|hypoxia - |KO-WT|normoxia. Both genotype contrasts
are computed directly from biological replicate arrays using all cross-replicate
pairwise differences and their median, rather than inferring the hypoxia gap
only from the interaction coefficient. Circular moving-block bootstrap across
5-kb bins accounts for local genomic dependence at the whole-genome level.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
BASE = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
OUT = BASE / 'insulation_convergence_20261030'
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/03_structure_metrics/bin_level_v2.csv'
DET = {'detected_increase', 'detected_decrease'}
N, N_BOOT, BLOCK = 883, 5000, 20
RNG = np.random.default_rng(20261030)
plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

b = pd.read_csv(RAW)
h = pd.read_csv(BASE / 'source_data/Hypoxia_bin_level.csv')
assert len(b) == len(h) == N
assert np.array_equal(b['bin'].to_numpy(), np.arange(N))
assert np.array_equal(b['bin'].to_numpy(), h['bin'].to_numpy())

rep = {}
for cond, prefix, nr in [('WT', 'WT', 4), ('KO', 'KO', 4),
                         ('HYP', 'hypoxia', 4), ('KOHYP', 'KO_hypoxia', 3)]:
    arr = b[[f'ins_{prefix}_rep{i}' for i in range(1, nr + 1)]].to_numpy(float).T
    rep[cond] = np.where(np.isfinite(arr), arr, np.nan)


def pair_stats(a, z):
    """Median and 5–95% range of all pairwise z-a replicate differences."""
    ds = (z[:, None, :] - a[None, :, :]).reshape(-1, a.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return (np.nanmedian(ds, axis=0), np.nanpercentile(ds, 5, axis=0),
                np.nanpercentile(ds, 95, axis=0))


def classify(est, lo, hi, bound=.25):
    return np.select([lo > bound, hi < -bound, (lo >= -bound) & (hi <= bound)],
                     ['detected_increase', 'detected_decrease', 'supports_small_change'],
                     default='cannot_determine')


d0, d0_lo, d0_hi = pair_stats(rep['WT'], rep['KO'])
d1, d1_lo, d1_hi = pair_stats(rep['HYP'], rep['KOHYP'])
assert np.allclose(d0, b.d_ins_KO_WT.to_numpy(), equal_nan=True)
assert np.allclose(d0_lo, b.d_ins_KO_WT_lo.to_numpy(), equal_nan=True)
assert np.allclose(d0_hi, b.d_ins_KO_WT_hi.to_numpy(), equal_nan=True)

z = pd.DataFrame({
    'bin': b['bin'], 'start': b.start, 'end': b.end,
    'NapM_bound': b.chip_bound.astype(bool),
    'hypoxia_response_evidence_class': h.hypoxia_response_evidence_class,
    'ins_WT_normoxia': b.ins_WT, 'ins_KO_normoxia': b.ins_KO,
    'ins_WT_hypoxia': b.ins_hypoxia, 'ins_KO_hypoxia': b.ins_KO_hypoxia,
    'gap_normoxia_KO_minus_WT': d0,
    'gap_normoxia_lo': d0_lo, 'gap_normoxia_hi': d0_hi,
    'gap_hypoxia_KO_minus_WT': d1,
    'gap_hypoxia_lo': d1_lo, 'gap_hypoxia_hi': d1_hi,
    'baseline_gap_class': b.ins_change_class,
    'hypoxia_gap_class': classify(d1, d1_lo, d1_hi),
    'WT_hypoxia_response': b.d_ins_hyp_WT,
    'KO_hypoxia_response': b.d_ins_hyp_KO,
    'interaction': b.d_ins_interaction,
    'interaction_class': b.interaction_class,
    'interaction_FDR': h.interaction_padj,
})
z['abs_gap_normoxia'] = z.gap_normoxia_KO_minus_WT.abs()
z['abs_gap_hypoxia'] = z.gap_hypoxia_KO_minus_WT.abs()
z['change_abs_gap_hypoxia_minus_normoxia'] = z.abs_gap_hypoxia - z.abs_gap_normoxia
z['convergence'] = z.change_abs_gap_hypoxia_minus_normoxia.lt(0)
z['gap_change_class'] = np.select(
    [z.convergence, z.change_abs_gap_hypoxia_minus_normoxia.gt(0)],
    ['convergence', 'divergence'], default='no_change')
z['opposite_interaction_to_baseline_gap'] = (
    np.sign(z.gap_normoxia_KO_minus_WT) != np.sign(z.interaction))
z['supported_convergence'] = (z.baseline_gap_class.isin(DET) &
    z.interaction_class.isin(DET) & z.opposite_interaction_to_baseline_gap & z.convergence)

# Locus labels used in the existing mechanism displays; overlapping labels are retained.
locus = pd.read_csv(BASE / 'source_data/Mechanism_locus_insulation_tracks.csv')
locus_map = locus.groupby('bin').locus.apply(lambda x: ';'.join(sorted(set(x))))
z['display_locus'] = z['bin'].map(locus_map).fillna('')
z.to_csv(OUT / 'bin_level_convergence.csv', index=False)

valid = np.isfinite(z.change_abs_gap_hypoxia_minus_normoxia.to_numpy())
base_masks = {
    'All_bins': np.ones(N, dtype=bool),
    'NapM_bound': z.NapM_bound.to_numpy(bool),
    'NapM_unbound': ~z.NapM_bound.to_numpy(bool),
    'Baseline_genotype_difference_detected': z.baseline_gap_class.isin(DET).to_numpy(),
    'Unresolved_NapM_structure_RNA_104': z.hypoxia_response_evidence_class.eq(
        'Mixed_or_unresolved_NapM_structure_RNA').to_numpy(),
    'Detected_genotype_x_hypoxia_interaction': z.interaction_class.isin(DET).to_numpy(),
    'Baseline_abs_gap_below_0.25': z.abs_gap_normoxia.lt(.25).to_numpy(),
    'Baseline_abs_gap_0.25_to_0.5': (z.abs_gap_normoxia.ge(.25) & z.abs_gap_normoxia.lt(.5)).to_numpy(),
    'Baseline_abs_gap_at_least_0.5': z.abs_gap_normoxia.ge(.5).to_numpy(),
}
masks = {name: mask & valid for name, mask in base_masks.items()}

# Circular moving-block bootstrap; each resample preserves the genome's bin order in blocks.
indices = np.empty((N_BOOT, N), dtype=np.int32)
offsets = np.arange(BLOCK)
for boot in range(N_BOOT):
    starts = RNG.integers(0, N, int(np.ceil(N / BLOCK)))
    indices[boot] = np.concatenate([(s + offsets) % N for s in starts])[:N]


def observed_stats(mask):
    x = z.loc[mask]
    delta = x.change_abs_gap_hypoxia_minus_normoxia.to_numpy()
    return dict(
        n_bins=len(x),
        median_abs_gap_normoxia=x.abs_gap_normoxia.median(),
        median_abs_gap_hypoxia=x.abs_gap_hypoxia.median(),
        median_change_abs_gap=np.median(delta),
        mean_change_abs_gap=np.mean(delta),
        percent_convergence=100 * np.mean(delta < 0),
        percent_divergence=100 * np.mean(delta > 0),
        wilcoxon_nominal_P=wilcoxon(delta).pvalue,
        n_hypoxia_gap_detected=int(x.hypoxia_gap_class.isin(DET).sum()),
        n_hypoxia_gap_supports_small=int(x.hypoxia_gap_class.eq('supports_small_change').sum()),
        n_supported_convergence=int(x.supported_convergence.sum()),
    )


rows = []
for name, mask in masks.items():
    row = dict(group=name, **observed_stats(mask))
    boot_delta = np.empty(N_BOOT)
    boot_convergence = np.empty(N_BOOT)
    delta = z.change_abs_gap_hypoxia_minus_normoxia.to_numpy()
    for boot, idx in enumerate(indices):
        vals = delta[idx[mask[idx]]]
        boot_delta[boot] = np.median(vals)
        boot_convergence[boot] = 100 * np.mean(vals < 0)
    row.update(
        median_change_block95_lo=np.percentile(boot_delta, 2.5),
        median_change_block95_hi=np.percentile(boot_delta, 97.5),
        percent_convergence_block95_lo=np.percentile(boot_convergence, 2.5),
        percent_convergence_block95_hi=np.percentile(boot_convergence, 97.5),
        bootstrap_blocks=N_BOOT, bootstrap_block_bins=BLOCK,
    )
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'convergence_summary.csv', index=False)

# Key displayed loci: descriptive summaries only, without small-n P values.
locus_rows = []
for name, x in z[z.display_locus.ne('') & valid].groupby('display_locus'):
    delta = x.change_abs_gap_hypoxia_minus_normoxia
    locus_rows.append(dict(
        locus=name, n_bins=len(x), bins=','.join(map(str, sorted(x['bin']))),
        median_abs_gap_normoxia=x.abs_gap_normoxia.median(),
        median_abs_gap_hypoxia=x.abs_gap_hypoxia.median(),
        median_change_abs_gap=delta.median(),
        percent_convergence=100 * delta.lt(0).mean(),
        n_hypoxia_gap_supports_small=x.hypoxia_gap_class.eq('supports_small_change').sum(),
        n_supported_convergence=x.supported_convergence.sum(),
    ))
locus_summary = pd.DataFrame(locus_rows).sort_values('locus')
locus_summary.to_csv(OUT / 'display_locus_convergence.csv', index=False)
z[z.display_locus.ne('')].to_csv(OUT / 'display_locus_bin_details.csv', index=False)

# ----------------------------- figure -------------------------------------
BLUE, ORANGE, GREY, DARK = '#2878B5', '#D97706', '#A7B0B7', '#263238'
fig = plt.figure(figsize=(11.2, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, .9, 1.25], wspace=.34)
ax = fig.add_subplot(gs[0, 0])
for bound, color, label in [(False, GREY, 'Other bins'), (True, ORANGE, 'NapM-bound')]:
    x = z[valid & z.NapM_bound.eq(bound)]
    ax.scatter(x.abs_gap_normoxia, x.abs_gap_hypoxia, s=8, color=color,
               alpha=.48, edgecolors='none', label=label)
lim = np.nanmax(z[['abs_gap_normoxia', 'abs_gap_hypoxia']].to_numpy()) * 1.03
ax.plot([0, lim], [0, lim], color=DARK, lw=.8, ls='--')
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='Normoxia |KO − WT|',
       ylabel='Hypoxia |KO − WT|', title='A  Direct genotype-gap comparison')
ax.legend(frameon=False, loc='upper left', handletextpad=.2)
ax.text(.97, .04, f'{summary.loc[summary.group.eq("All_bins"), "percent_convergence"].iloc[0]:.1f}% converge',
        transform=ax.transAxes, ha='right', color=DARK)

ax = fig.add_subplot(gs[0, 1])
delta = z.loc[valid, 'change_abs_gap_hypoxia_minus_normoxia']
ax.hist(delta, bins=45, color=BLUE, alpha=.85, edgecolor='none')
ax.axvline(0, color=DARK, lw=.8)
ax.axvline(delta.median(), color=ORANGE, lw=1.3)
ax.set(xlabel='Hypoxia − normoxia\nchange in |KO − WT|',
       ylabel='Number of 5-kb bins', title='B  Genome-wide gap change')
ax.text(.03, .95, f'Median = {delta.median():.3f}', transform=ax.transAxes,
        va='top', color=ORANGE)

ax = fig.add_subplot(gs[0, 2])
plot_groups = ['All_bins', 'NapM_bound', 'Baseline_genotype_difference_detected',
               'Unresolved_NapM_structure_RNA_104', 'Detected_genotype_x_hypoxia_interaction']
labels = ['All bins', 'NapM-bound', 'Baseline gap detected',
          'Unresolved 104', 'Interaction detected']
ps = summary.set_index('group').loc[plot_groups]
y = np.arange(len(ps))[::-1]
ax.axvline(0, color=DARK, lw=.8)
for yi, (name, row), label in zip(y, ps.iterrows(), labels):
    ax.plot([row.median_change_block95_lo, row.median_change_block95_hi], [yi, yi],
            color=BLUE, lw=1.5, solid_capstyle='round')
    ax.scatter(row.median_change_abs_gap, yi, s=28, color=ORANGE, zorder=3)
    ax.text(.98, yi, f'{row.percent_convergence:.1f}%', va='center', ha='right',
            transform=ax.get_yaxis_transform(), fontsize=7)
ax.set_yticks(y, labels)
ax.set(xlabel='Median change in |KO − WT|\n(circular block-bootstrap 95% CI)',
       title='C  Convergence across bin sets')
ax.text(.02, -.28, 'Percent at right = bins with reduced absolute gap',
        transform=ax.transAxes, fontsize=7, color=DARK)
fig.savefig(OUT / 'Insulation_convergence_validation.pdf', bbox_inches='tight')
fig.savefig(OUT / 'Insulation_convergence_validation.svg', bbox_inches='tight')
fig.savefig(OUT / 'Insulation_convergence_validation.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ----------------------------- report -------------------------------------
def md_table(df, columns, labels, decimals=None):
    decimals = decimals or {}
    safe_labels = [str(x).replace('|', '\\|') for x in labels]
    lines = ['| ' + ' | '.join(safe_labels) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        vals = []
        for c in columns:
            v = row[c]
            if c in decimals and pd.notna(v):
                v = f'{v:.{decimals[c]}f}'
            vals.append(str(v).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'

primary_groups = ['All_bins', 'NapM_bound', 'Baseline_genotype_difference_detected',
                  'Unresolved_NapM_structure_RNA_104', 'Detected_genotype_x_hypoxia_interaction']
ps = summary[summary.group.isin(primary_groups)].copy()
ps['group'] = ps.group.map({
    'All_bins': '全部 bin', 'NapM_bound': 'NapM 结合',
    'Baseline_genotype_difference_detected': '常氧基因型差值可检出',
    'Unresolved_NapM_structure_RNA_104': '104 个 unresolved',
    'Detected_genotype_x_hypoxia_interaction': '交互可检出'})
all_row = summary[summary.group.eq('All_bins')].iloc[0]
base = summary[summary.group.eq('Baseline_genotype_difference_detected')].iloc[0]
both_detected = z[z.baseline_gap_class.isin(DET) & z.interaction_class.isin(DET)]
report = [
    '# 缺氧后 WT／NapM-KO insulation 差值趋于一致的验证\n',
    '## 检验定义\n',
    '- 常氧差值：`D0 = KO − WT`；缺氧差值：`D1 = KO_hypoxia − WT_hypoxia`。两个差值均直接从相应条件的生物学重复值计算所有跨重复配对的差值，并取中位数。',
    '- 收敛指标：`Δ|D| = |D1| − |D0|`。`Δ|D| < 0` 表示缺氧后基因型绝对差值缩小；反之表示扩大。',
    '- 全局不确定性采用 20-bin（100 kb）环形区块 bootstrap，5,000 次重抽样；Wilcoxon P 仅列为未校正空间依赖的辅助结果。',
    '- 缺氧后差值是否处于小变化范围沿用既有规则：配对差值的 5–95% 范围完全位于 −0.25 至 +0.25 内时标记为 `supports_small_change`。\n',
    '## 全基因组与主要区域结果\n',
    md_table(ps, ['group', 'n_bins', 'median_abs_gap_normoxia', 'median_abs_gap_hypoxia',
                  'median_change_abs_gap', 'median_change_block95_lo', 'median_change_block95_hi',
                  'percent_convergence', 'percent_convergence_block95_lo',
                  'percent_convergence_block95_hi'],
             ['区域', 'n', '常氧中位 |D|', '缺氧中位 |D|', '中位 Δ|D|',
              '95% CI 下限', '95% CI 上限', '收敛 %', '收敛 % CI 下限', '收敛 % CI 上限'],
             {'median_abs_gap_normoxia': 3, 'median_abs_gap_hypoxia': 3,
              'median_change_abs_gap': 3, 'median_change_block95_lo': 3,
              'median_change_block95_hi': 3, 'percent_convergence': 1,
              'percent_convergence_block95_lo': 1, 'percent_convergence_block95_hi': 1}),
    f'全部 {int(all_row.n_bins)} 个可检验 bin 中，{all_row.percent_convergence:.1f}% 的绝对基因型差值缩小；中位 |KO−WT| 从 {all_row.median_abs_gap_normoxia:.3f} 降至 {all_row.median_abs_gap_hypoxia:.3f}，中位变化为 {all_row.median_change_abs_gap:.3f}（区块 bootstrap 95% CI {all_row.median_change_block95_lo:.3f} 至 {all_row.median_change_block95_hi:.3f}）。**因此，数据支持“缺氧后 WT 与 KO 的 insulation 总体更一致”这一描述。**\n',
    '## 与显著交互的一致性\n',
    f'常氧基因型差值和 genotype × hypoxia 交互均可检出的 bin 共 {len(both_detected)} 个，其中 {both_detected.opposite_interaction_to_baseline_gap.mean()*100:.1f}% 的交互方向与常氧差值相反，{both_detected.convergence.mean()*100:.1f}% 显示绝对差值缩小。该结果说明缺氧响应总体是在抵消既有基因型差值，而非统一拉大差值。\n',
    f'在常氧差值可检出的 {int(base.n_bins)} 个 bin 中，缺氧后有 {int(base.n_hypoxia_gap_supports_small)} 个达到 `supports_small_change`，仍有 {int(base.n_hypoxia_gap_detected)} 个保留可检出的基因型差值。因此“更一致”不等于所有位点都完全一致。\n',
    '## 已展示关键位点\n',
    md_table(locus_summary, ['locus', 'n_bins', 'median_abs_gap_normoxia',
                             'median_abs_gap_hypoxia', 'median_change_abs_gap',
                             'percent_convergence', 'n_hypoxia_gap_supports_small'],
             ['位点', 'bin 数', '常氧中位 |D|', '缺氧中位 |D|', '中位 Δ|D|',
              '收敛 %', '缺氧小差值 bin 数'],
             {'median_abs_gap_normoxia': 3, 'median_abs_gap_hypoxia': 3,
              'median_change_abs_gap': 3, 'percent_convergence': 1}),
    '\n## 解释边界\n',
    '- 该规律说明缺氧具有缩小 WT／KO insulation 基因型差异的总体趋势，可作为“应激使两种基因型的局部结构状态趋同”的假设性描述。',
    '- 它不能单独证明 NapM 的直接结构作用，也不能排除整体 insulation 动态范围变化、应激转录重塑或其他间接因素。常氧显著差异区域是结果导向选择，可能受回归均值影响，因此全基因组结果是主要依据。',
    '- “差值缩小”不是严格的统计等价检验；只有逐 bin 的小差值分类及区间可进一步说明差异是否已低于既有检测阈值。',
    '- 可复查数据见 `bin_level_convergence.csv`，关键位点逐 bin 结果见 `display_locus_bin_details.csv`。\n'
]
(OUT / '中文验证报告.md').write_text('\n'.join(report), encoding='utf-8')

metadata = dict(
    hypothesis='Hypoxia reduces absolute NapM-KO versus WT insulation differences',
    primary_statistic='abs(KO_hypoxia-WT_hypoxia)-abs(KO_normoxia-WT_normoxia)',
    normoxia_pairs=16, hypoxia_pairs=12,
    point_estimate='Median of all cross-replicate pairwise differences at each 5-kb bin',
    global_uncertainty='5000 circular moving-block bootstraps; block=20 bins=100 kb',
    small_gap_rule='5-95% pairwise difference range entirely within [-0.25,+0.25]',
    caution='Reduced absolute difference is not an equivalence test and does not establish direct NapM causality.'
)
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

print(summary[['group', 'n_bins', 'median_abs_gap_normoxia', 'median_abs_gap_hypoxia',
               'median_change_abs_gap', 'median_change_block95_lo', 'median_change_block95_hi',
               'percent_convergence', 'wilcoxon_nominal_P', 'n_hypoxia_gap_supports_small',
               'n_supported_convergence']].to_string(index=False))
print('\nDISPLAY LOCI')
print(locus_summary.to_string(index=False))
print(f'\nOutputs: {OUT}')
