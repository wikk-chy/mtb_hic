#!/usr/bin/env python3
"""Compare WT-latent, WT-hypoxia and baseline KO-WT insulation effects.

There is no KO-latent condition. KO latent-like behavior is therefore assessed
only by two non-equivalent proxies: (i) same-direction baseline KO-WT and WT
latent responses, and (ii) whether KO normoxia is closer to WT latent than WT
normoxia is. Neither proxy replaces a KO-latent experiment.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
BASE = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
OUT = BASE / 'latent_hypoxia_insulation_comparison_20261030'
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/03_structure_metrics/bin_level_v2.csv'
DET = {'detected_increase', 'detected_decrease'}
N, N_BOOT, BLOCK, N_PERM = 883, 5000, 20, 10000
RNG = np.random.default_rng(20261030)
plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

b = pd.read_csv(RAW)
h = pd.read_csv(BASE / 'source_data/Hypoxia_bin_level.csv')
l = pd.read_csv(BASE / 'source_data/Latency_bin_level.csv')
assert len(b) == len(h) == len(l) == N
assert np.array_equal(b['bin'], np.arange(N))


def arr(prefix, nr):
    a = b[[f'ins_{prefix}_rep{i}' for i in range(1, nr + 1)]].to_numpy(float).T
    return np.where(np.isfinite(a), a, np.nan)


def pair_median(a, z):
    ds = (z[:, None, :] - a[None, :, :]).reshape(-1, a.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(ds, axis=0)


W, K = arr('WT', 4), arr('KO', 4)
H, L = arr('hypoxia', 4), arr('latent', 4)
d0, dh, dl = pair_median(W, K), pair_median(W, H), pair_median(W, L)
dkl = pair_median(K, L)
assert np.allclose(d0, b.d_ins_KO_WT, equal_nan=True)
assert np.allclose(dh, b.d_ins_hyp_WT, equal_nan=True)
assert np.allclose(dl, l.d_ins_latent_WT, equal_nan=True)

z = pd.DataFrame({
    'bin': b['bin'], 'start': b.start, 'end': b.end,
    'NapM_bound': b.chip_bound.astype(bool),
    'hypoxia_response_evidence_class': h.hypoxia_response_evidence_class,
    'baseline_KO_minus_WT': d0,
    'WT_hypoxia_response': dh,
    'WT_latent_response': dl,
    'KO_to_WT_latent_difference': dkl,
    'baseline_class': b.ins_change_class,
    'hypoxia_response_class': h.WT_hypoxia_ins_class,
    'latent_response_class': l.latent_ins_class,
    'RNA_baseline_mean_log2FC': l.RNA_baseline_mean_log2FC,
    'RNA_latent_mean_log2FC': l.RNA_latent_mean_log2FC,
    'latency_candidate_region': l.latency_candidate_region,
})
for name, values in [('baseline', d0), ('hypoxia', dh), ('latent', dl), ('KO_to_latent', dkl)]:
    z[f'abs_{name}'] = np.abs(values)
z['latent_abs_minus_hypoxia_abs'] = z.abs_latent - z.abs_hypoxia
z['latent_abs_minus_baseline_KO_abs'] = z.abs_latent - z.abs_baseline
z['KO_to_latent_abs_minus_WT_to_latent_abs'] = z.abs_KO_to_latent - z.abs_latent
z['baseline_latent_same_direction'] = np.sign(d0) == np.sign(dl)
z['partial_latent_like'] = z.baseline_latent_same_direction & (z.abs_baseline < z.abs_latent)
z['baseline_latent_opposite_direction'] = np.sign(d0) != np.sign(dl)
z['dual_structure_detected'] = z.baseline_class.isin(DET) & z.latent_response_class.isin(DET)
z['KO_closer_to_latent_than_WT'] = z.KO_to_latent_abs_minus_WT_to_latent_abs < 0

locus = pd.read_csv(BASE / 'source_data/Mechanism_locus_insulation_tracks.csv')
locus_map = locus.groupby('bin').locus.apply(lambda x: ';'.join(sorted(set(x))))
z['display_locus'] = z['bin'].map(locus_map).fillna('')
z.to_csv(OUT / 'bin_level_comparison.csv', index=False)
z[z.dual_structure_detected].to_csv(OUT / 'dual_structure_detected_bins.csv', index=False)

masks = {
    'All_bins': np.ones(N, dtype=bool),
    'NapM_bound': z.NapM_bound.to_numpy(bool),
    'NapM_unbound': ~z.NapM_bound.to_numpy(bool),
    'Unresolved_NapM_structure_RNA_104': z.hypoxia_response_evidence_class.eq(
        'Mixed_or_unresolved_NapM_structure_RNA').to_numpy(),
    'Dual_baseline_and_latent_detected': z.dual_structure_detected.to_numpy(),
}
metrics = {
    'Latent_magnitude_minus_hypoxia_magnitude': z.latent_abs_minus_hypoxia_abs.to_numpy(),
    'Latent_magnitude_minus_baseline_KO_magnitude': z.latent_abs_minus_baseline_KO_abs.to_numpy(),
    'KO_to_latent_distance_minus_WT_to_latent_distance': z.KO_to_latent_abs_minus_WT_to_latent_abs.to_numpy(),
}

# Shared circular moving-block bootstrap indices.
indices = np.empty((N_BOOT, N), dtype=np.int32)
offsets = np.arange(BLOCK)
for boot in range(N_BOOT):
    starts = RNG.integers(0, N, int(np.ceil(N / BLOCK)))
    indices[boot] = np.concatenate([(s + offsets) % N for s in starts])[:N]

rows = []
for group, group_mask in masks.items():
    for comparison, values in metrics.items():
        mask = group_mask & np.isfinite(values)
        x = values[mask]
        boot_median = np.empty(N_BOOT)
        boot_positive = np.empty(N_BOOT)
        for boot, idx in enumerate(indices):
            vals = values[idx[mask[idx]]]
            boot_median[boot] = np.median(vals)
            boot_positive[boot] = 100 * np.mean(vals > 0)
        rows.append(dict(
            group=group, comparison=comparison, n_bins=len(x),
            median_difference=np.median(x), mean_difference=np.mean(x),
            median_block95_lo=np.percentile(boot_median, 2.5),
            median_block95_hi=np.percentile(boot_median, 97.5),
            percent_positive=100 * np.mean(x > 0),
            percent_positive_block95_lo=np.percentile(boot_positive, 2.5),
            percent_positive_block95_hi=np.percentile(boot_positive, 97.5),
            percent_negative=100 * np.mean(x < 0),
            wilcoxon_nominal_P=wilcoxon(x).pvalue,
        ))
summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'comparison_summary.csv', index=False)

# Condition-effect summaries for each region set.
effect_rows = []
for group, mask in masks.items():
    x = z[mask]
    effect_rows.append(dict(
        group=group, n_bins=len(x),
        median_abs_baseline_KO=x.abs_baseline.median(),
        median_abs_hypoxia_response=x.abs_hypoxia.median(),
        median_abs_latent_response=x.abs_latent.median(),
        median_abs_KO_to_latent=x.abs_KO_to_latent.median(),
        percent_baseline_latent_same_direction=100 * x.baseline_latent_same_direction.mean(),
        percent_partial_latent_like=100 * x.partial_latent_like.mean(),
        percent_KO_closer_to_latent=100 * x.KO_closer_to_latent_than_WT.mean(),
    ))
effects = pd.DataFrame(effect_rows)
effects.to_csv(OUT / 'group_effect_summary.csv', index=False)

# Weak-to-moderate response similarity, with a circular-shift null.
ok = np.isfinite(dl) & np.isfinite(dh)
rho = spearmanr(dl[ok], dh[ok]).statistic
null = np.empty(N_PERM)
for i in range(N_PERM):
    shifted = np.roll(dl, int(RNG.integers(1, N)))
    good = np.isfinite(shifted) & np.isfinite(dh)
    null[i] = spearmanr(shifted[good], dh[good]).statistic
rho_p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERM + 1)

# Displayed-locus summaries are descriptive and include overlapping locus labels.
locus_rows = []
for name, x in z[z.display_locus.ne('')].groupby('display_locus'):
    dual = x[x.dual_structure_detected]
    locus_rows.append(dict(
        locus=name, n_bins=len(x),
        median_abs_hypoxia_response=x.abs_hypoxia.median(),
        median_abs_latent_response=x.abs_latent.median(),
        median_abs_baseline_KO=x.abs_baseline.median(),
        median_latent_abs_minus_hypoxia_abs=x.latent_abs_minus_hypoxia_abs.median(),
        percent_latent_larger_than_hypoxia=100 * x.latent_abs_minus_hypoxia_abs.gt(0).mean(),
        percent_KO_closer_to_latent=100 * x.KO_closer_to_latent_than_WT.mean(),
        n_dual_detected=len(dual),
        percent_dual_opposite=100 * dual.baseline_latent_opposite_direction.mean() if len(dual) else np.nan,
    ))
locus_summary = pd.DataFrame(locus_rows).sort_values('locus')
locus_summary.to_csv(OUT / 'display_locus_comparison.csv', index=False)

# ----------------------------- figure -------------------------------------
BLUE, ORANGE, GREY, DARK = '#2878B5', '#D97706', '#A7B0B7', '#263238'
fig = plt.figure(figsize=(11.2, 3.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=.34)
ax = fig.add_subplot(gs[0, 0])
for bound, color, label in [(False, GREY, 'Other bins'), (True, ORANGE, 'NapM-bound')]:
    x = z[z.NapM_bound.eq(bound)]
    ax.scatter(x.abs_hypoxia, x.abs_latent, s=8, color=color, alpha=.45,
               edgecolors='none', label=label)
lim = np.nanmax(z[['abs_hypoxia', 'abs_latent']].to_numpy()) * 1.03
ax.plot([0, lim], [0, lim], ls='--', lw=.8, color=DARK)
all_lh = summary[(summary.group.eq('All_bins')) &
                 summary.comparison.eq('Latent_magnitude_minus_hypoxia_magnitude')].iloc[0]
ax.text(.97, .04, f'Latent larger: {all_lh.percent_positive:.1f}%', transform=ax.transAxes,
        ha='right')
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='|WT hypoxia response|',
       ylabel='|WT latent response|', title='A  Stress-response magnitude')
ax.legend(frameon=False, loc='upper left', handletextpad=.2)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(z.abs_baseline, z.abs_latent, s=8, color=GREY, alpha=.45, edgecolors='none')
dual = z[z.dual_structure_detected]
ax.scatter(dual.abs_baseline, dual.abs_latent, s=18, color=ORANGE, alpha=.85,
           edgecolors='none', label='Both detected')
lim = np.nanmax(z[['abs_baseline', 'abs_latent']].to_numpy()) * 1.03
ax.plot([0, lim], [0, lim], ls='--', lw=.8, color=DARK)
all_lb = summary[(summary.group.eq('All_bins')) &
                 summary.comparison.eq('Latent_magnitude_minus_baseline_KO_magnitude')].iloc[0]
ax.text(.97, .04, f'Latent larger: {all_lb.percent_positive:.1f}%', transform=ax.transAxes,
        ha='right')
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='|Normoxia KO − WT|',
       ylabel='|WT latent response|', title='B  Latent versus genotype effect')
ax.legend(frameon=False, loc='upper left', handletextpad=.2)

ax = fig.add_subplot(gs[0, 2])
comps = ['Latent_magnitude_minus_hypoxia_magnitude',
         'Latent_magnitude_minus_baseline_KO_magnitude',
         'KO_to_latent_distance_minus_WT_to_latent_distance']
labels = ['Latent > hypoxia', 'Latent > baseline KO effect', 'KO closer to latent']
ps = summary[summary.group.eq('All_bins')].set_index('comparison').loc[comps]
y = np.arange(3)[::-1]
for yi, (name, row), label in zip(y, ps.iterrows(), labels):
    # For the first two, positive is the labeled event; for distance, negative means KO closer.
    pct = row.percent_positive if not name.startswith('KO_to') else row.percent_negative
    if name.startswith('KO_to'):
        # Recompute the event-specific bootstrap interval from bin-level values in the same indices.
        values = metrics[name]
        boot_event = []
        for idx in indices:
            vals = values[idx]
            vals = vals[np.isfinite(vals)]
            boot_event.append(100 * np.mean(vals < 0))
        lo, hi = np.percentile(boot_event, [2.5, 97.5])
    else:
        lo, hi = row.percent_positive_block95_lo, row.percent_positive_block95_hi
    ax.barh(yi, pct, color=BLUE, alpha=.85, height=.52)
    ax.plot([lo, hi], [yi, yi], color=DARK, lw=1)
    ax.text(pct + 1.5, yi, f'{pct:.1f}%', va='center', fontsize=8)
ax.axvline(50, color=ORANGE, ls='--', lw=.9)
ax.set_yticks(y, labels)
ax.set(xlim=(0, 100), xlabel='Percent of bins (block-bootstrap 95% CI)',
       title='C  Three direct comparisons')
fig.savefig(OUT / 'Latent_hypoxia_insulation_comparison.pdf', bbox_inches='tight')
fig.savefig(OUT / 'Latent_hypoxia_insulation_comparison.svg', bbox_inches='tight')
fig.savefig(OUT / 'Latent_hypoxia_insulation_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ----------------------------- report -------------------------------------
def get(group, comparison):
    return summary[(summary.group.eq(group)) & (summary.comparison.eq(comparison))].iloc[0]


def md_table(df, columns, labels, decimals=None):
    decimals = decimals or {}
    safe = [str(x).replace('|', '\\|') for x in labels]
    lines = ['| ' + ' | '.join(safe) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        vals = []
        for c in columns:
            v = row[c]
            if c in decimals and pd.notna(v):
                v = f'{v:.{decimals[c]}f}'
            vals.append(str(v).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'


lh = get('All_bins', 'Latent_magnitude_minus_hypoxia_magnitude')
lb = get('All_bins', 'Latent_magnitude_minus_baseline_KO_magnitude')
kd = get('All_bins', 'KO_to_latent_distance_minus_WT_to_latent_distance')
un_lh = get('Unresolved_NapM_structure_RNA_104', 'Latent_magnitude_minus_hypoxia_magnitude')
dual_n = int(z.dual_structure_detected.sum())
dual_same = int((z.dual_structure_detected & z.baseline_latent_same_direction).sum())
dual_opposite = int((z.dual_structure_detected & z.baseline_latent_opposite_direction).sum())
dual_partial = int((z.dual_structure_detected & z.partial_latent_like).sum())
report = [
    '# WT latent、WT hypoxia 与 NapM-KO insulation 效应比较\n',
    '## 可检验问题与设计\n',
    '- `|latent−WT| − |hypoxia−WT|`：大于 0 表示 WT latent 响应幅度更大。',
    '- `|latent−WT| − |KO−WT|`：大于 0 表示 WT latent 响应幅度大于常氧基因型差异。',
    '- `|KO−latent| − |WT−latent|`：小于 0 才表示 KO 常氧状态比 WT 更接近 WT latent。该指标只是 latent-like 的替代检验。',
    '- 所有差值直接从对应条件的生物学重复计算跨重复配对中位数；全局区间采用 20-bin（100 kb）环形区块 bootstrap，5,000 次。',
    '- **现有数据没有 KO-latent 组，不能直接比较 KO 与 WT 的 latent 响应，也不能估计 genotype × latent 交互。**\n',
    '## 三个直接答案\n',
    f'1. **WT latent 响应仅略大于 hypoxia，不能称为普遍显著增强。** 中位绝对响应：latent {z.abs_latent.median():.3f}，hypoxia {z.abs_hypoxia.median():.3f}；latent 减 hypoxia 的中位差为 {lh.median_difference:.4f}（95% CI {lh.median_block95_lo:.4f} 至 {lh.median_block95_hi:.4f}），只有 {lh.percent_positive:.1f}% 的 bin 是 latent 更大。',
    f'2. **WT latent 响应整体不大于常氧 KO–WT 差异。** 中位绝对常氧基因型差为 {z.abs_baseline.median():.3f}，latent 减基线效应的中位差为 {lb.median_difference:.3f}（95% CI {lb.median_block95_lo:.3f} 至 {lb.median_block95_hi:.3f}）；只有 {lb.percent_positive:.1f}% 的 bin 是 latent 更大。',
    f'3. **KO 常氧状态整体并不比 WT 更接近 latent。** 只有 {kd.percent_negative:.1f}% 的 bin 满足 KO 更接近 latent；{kd.percent_positive:.1f}% 反而更远，距离差的中位数为 {kd.median_difference:.3f}（95% CI {kd.median_block95_lo:.3f} 至 {kd.median_block95_hi:.3f}）。',
    f'4. 在 104 个 unresolved 区域，latent 大于 hypoxia 的比例进一步降至 {un_lh.percent_positive:.1f}%，中位幅度差为 {un_lh.median_difference:.3f}，不支持这些区域中 latent 响应普遍更强。\n',
    '## 分区统计\n',
]
comp_labels = {
    'Latent_magnitude_minus_hypoxia_magnitude': 'latent − hypoxia 幅度',
    'Latent_magnitude_minus_baseline_KO_magnitude': 'latent − 常氧 KO 效应幅度',
    'KO_to_latent_distance_minus_WT_to_latent_distance': 'KO−latent − WT−latent 距离'}
group_labels = {'All_bins': '全部 bin', 'NapM_bound': 'NapM 结合',
                'Unresolved_NapM_structure_RNA_104': '104 个 unresolved',
                'Dual_baseline_and_latent_detected': '基线与 latent 均可检出'}
ps = summary[summary.group.isin(group_labels)].copy()
ps['group'] = ps.group.map(group_labels)
ps['comparison'] = ps.comparison.map(comp_labels)
report.append(md_table(ps, ['group', 'comparison', 'n_bins', 'median_difference',
                            'median_block95_lo', 'median_block95_hi', 'percent_positive',
                            'percent_negative'],
                       ['区域', '指标', 'n', '中位差', '95% CI 下限', '95% CI 上限',
                        '差值 > 0 %', '差值 < 0 %'],
                       {'median_difference': 3, 'median_block95_lo': 3,
                        'median_block95_hi': 3, 'percent_positive': 1, 'percent_negative': 1}))
ori_dual = z[z.display_locus.eq('oriC') & z.dual_structure_detected]
dos_dual = z[z.display_locus.eq('DosR_DosS') & z.dual_structure_detected]
report.extend([
    '\n## 最关键的结构性观察：KO 不是简单的 latent-like 状态\n',
    f'常氧 KO–WT 与 WT latent 结构变化均可检出的 bin 共 {dual_n} 个：{dual_opposite} 个（{dual_opposite/dual_n*100:.1f}%）方向相反，只有 {dual_same} 个（{dual_same/dual_n*100:.1f}%）方向相同；符合“同方向且 KO 幅度较小”的部分 latent-like 模式仅 {dual_partial} 个（{dual_partial/dual_n*100:.1f}%）。',
    f'WT latent 与 hypoxia 响应的全基因组 Spearman ρ = {rho:.3f}（环形位移 P = {rho_p:.4g}），同方向 bin 为 {100*np.mean(np.sign(dl[ok])==np.sign(dh[ok])):.1f}%。这表明两种应激只有有限共享性，不能视为同一结构程序的简单强弱差异。',
    f'**局部例外值得后续验证：**oriC 非重叠区域的 {len(ori_dual)} 个双检出 bin 均呈同方向且符合部分 latent-like 模式；DosR/DosS 的 {len(dos_dual)} 个双检出 bin 也呈同方向。这提示复制起始区及 DosR/DosS 周边可能存在位点特异的 NapM–latent 结构联系，但当前每处仅有极少数双检出 bin，不能据此确定机制。\n',
    '## 已展示关键位点\n',
    md_table(locus_summary, ['locus', 'n_bins', 'median_abs_hypoxia_response',
                             'median_abs_latent_response', 'median_abs_baseline_KO',
                             'percent_latent_larger_than_hypoxia', 'percent_KO_closer_to_latent',
                             'n_dual_detected', 'percent_dual_opposite'],
             ['位点', 'bin 数', '中位 |hypoxia|', '中位 |latent|', '中位 |KO−WT|',
              'latent 更大 %', 'KO 更接近 latent %', '双检出 n', '双检出反向 %'],
             {'median_abs_hypoxia_response': 3, 'median_abs_latent_response': 3,
              'median_abs_baseline_KO': 3, 'percent_latent_larger_than_hypoxia': 1,
              'percent_KO_closer_to_latent': 1, 'percent_dual_opposite': 1}),
    '\n## 可提出的机制假设及边界\n',
    '- 一个更合理且有数据支持的假设是：**hypoxia 与 latent 不是同一结构应激程序的连续加强版本。hypoxia 倾向于抵消常氧 WT／KO 差异，而 WT latent 重塑通常不复制 NapM-KO 的常氧结构偏移，并在双检出区域更多呈相反方向。**',
    '- 这提示“应激类型特异的染色体组织重塑”比“NapM 缺失使细胞预先进入休眠结构状态”更符合当前结果，可作为后续机制研究的切入点。',
    '- 但 insulation 本身不能确定分子介质，也不能证明 NapM 直接控制 latent 程序。只有加入 KO-latent 条件，才能直接检验 NapM 是否改变 latent 响应及其交互。',
    '- 关键位点的样本量很小，表中结果用于定位后续验证，不宜单独上升为通路机制结论。\n'
])
(OUT / '中文比较与机制解释报告.md').write_text('\n'.join(report), encoding='utf-8')
metadata = dict(
    KO_latent_available=False,
    direct_comparisons=list(metrics),
    bootstrap='5000 circular moving-block resamples; 20 bins=100 kb',
    latent_hypoxia_spearman_rho=float(rho),
    latent_hypoxia_circular_shift_P=float(rho_p),
    dual_detected_bins=dual_n, dual_same_direction=dual_same,
    dual_opposite_direction=dual_opposite, dual_partial_latent_like=dual_partial,
    interpretation='WT latent does not generally phenocopy baseline NapM-KO structure; no direct genotype-by-latent inference is possible without KO-latent data.'
)
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

print(summary.to_string(index=False))
print('\nDUAL:', metadata)
print('\nDISPLAY LOCI')
print(locus_summary.to_string(index=False))
print(f'\nOutputs: {OUT}')
