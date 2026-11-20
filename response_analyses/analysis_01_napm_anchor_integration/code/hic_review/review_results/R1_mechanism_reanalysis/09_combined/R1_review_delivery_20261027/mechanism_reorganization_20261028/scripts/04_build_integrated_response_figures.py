#!/usr/bin/env python3
"""Build four integrated Reviewer 1 response figures with panel-level source data.

The script assembles previously verified analyses and the new enrichment
sensitivity tests. It does not rerun upstream read processing or overwrite any
old response figure. All claims in the figures are deliberately limited to the
observed evidence; dashed model arrows denote untested hypotheses.
"""
from pathlib import Path
import json
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
DEL = RA / '09_combined/R1_review_delivery_20261027'
OLD = DEL / 'final_response_figure_20261027'
OSRC = OLD / 'source_data'
OTAB = OLD / 'tables'
STAB = DEL / 'supplementary_analysis/tables'
SSRC = STAB / 'corrected_figure_sources'
EXT4 = RA / '08_extended/fig4_fingerprint/tables'
OUT = DEL / 'mechanism_reorganization_20261028'
TAB = OUT / 'tables'
SD = OUT / 'source_data'
FIGD = OUT / 'figures'
for d in (TAB, SD, FIGD):
    d.mkdir(parents=True, exist_ok=True)

# Accessible, consistent scientific palette.
WT = '#3B6FB6'
KO = '#D97925'
HYP = '#2A9D8F'
KOHYP = '#C44E89'
CHIP = '#6F4E9C'
UP = '#D6604D'
DOWN = '#4393C3'
INK = '#263238'
MID = '#6F787D'
GREY = '#B8C0C4'
LIGHT = '#E8ECEE'
WHITE = '#FFFFFF'
RNG = np.random.default_rng(42)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.labelsize': 7.2, 'axes.titlesize': 8.0,
    'axes.titleweight': 'bold', 'xtick.labelsize': 6.4,
    'ytick.labelsize': 6.4, 'legend.fontsize': 6.2,
    'axes.linewidth': 0.65, 'xtick.major.width': 0.55,
    'ytick.major.width': 0.55, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})
MM = 1 / 25.4
manifest = []


def source(name, obj, panel, origin, description):
    """Freeze every panel's plotted data and record its provenance."""
    path = SD / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        pd.DataFrame(obj).to_csv(path, index=False)
    manifest.append(dict(panel=panel, source_file=str(path), original_input=str(origin),
                         description=description))
    return obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def lab(ax, letter, x=-0.15, y=1.12):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left', color=INK)


def title(fig, text, subtitle=None):
    fig.text(0.055, 0.982, text, ha='left', va='top', fontsize=11.5,
             fontweight='bold', color=INK)
    if subtitle:
        fig.text(0.055, 0.952, subtitle, ha='left', va='top', fontsize=7.2, color=MID)


def save_fig(fig, n):
    fig.savefig(FIGD / f'R1_Response_Figure{n}.pdf', metadata={
        'Title': f'Reviewer 1 integrated Response Figure {n}',
        'Subject': 'NapM chromosome architecture: evidence and limits',
    })
    fig.savefig(FIGD / f'R1_Response_Figure{n}.png', dpi=400)
    plt.close(fig)
    print('Wrote integrated Response Figure', n)


def p_fmt(x):
    return f'{x:.3f}' if x >= 0.001 else f'{x:.1e}'


# =========================== Load verified results ==========================
a0 = pd.read_csv(OSRC / 'A_loop_anchor_primary.csv')
a0 = a0[(a0.definition == 'interval') & (a0.metric == 'both')].iloc[0]
an = pd.read_csv(OSRC / 'A_loop_anchor_null.csv')
astr = pd.read_csv(OSRC / 'A_loop_anchor_stratified.csv')
loops = pd.read_csv(OSRC / 'B_loop_anchor_dose_per_loop.csv')
dose = pd.read_csv(OSRC / 'B_loop_anchor_dose_summary.csv')
dose = dose[dose.analysis_set == 'all_39_loops'].set_index('metric')
matched = pd.read_csv(OSRC / 'C_local_occupancy_matched_effects.csv')
rob = pd.read_csv(OSRC / 'C_local_occupancy_robustness.csv')
inc = pd.read_csv(OSRC / 'D_RNA_prediction_increments.csv')
rna_desc = pd.read_csv(OSRC / 'D_RNA_descriptive_summary.csv')
ind = pd.read_csv(OSRC / 'E_independent_sample_genome_mean_insulation.csv')
fact = pd.read_csv(OSRC / 'E_independent_sample_factorial.csv')
fp = pd.read_csv(EXT4 / 'fingerprint_per_sample.csv')
rv = pd.read_csv(EXT4 / 'fig4_response_vectors_contribution.csv')
fpstats = pd.read_csv(EXT4 / 'fig4_response_stats.csv')
sp0 = pd.read_csv(OSRC / 'G_original_spatial_association.csv').iloc[0]
splits = pd.read_csv(OSRC / 'G_disjoint_WT_reference_sensitivity.csv')
perbin = pd.read_csv(OSRC / 'G_original_per_bin_association.csv')
sets = pd.read_csv(TAB / 'multiomics_gene_set_summary.csv')
key_enrich = pd.read_csv(TAB / 'key_pathway_conditional_spatial_sensitivity.csv')
key_loci = pd.read_csv(TAB / 'key_loci_evidence.csv')
genes = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
peaks = pd.read_csv(RA / '01_chip_curated/NapM_binding_curated.tsv', sep='\t')
loops41 = pd.read_csv(RA / '02_loop_curated/loops_41_curated.tsv', sep='\t')
bins = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')

assert int(a0.observed) == 17 and len(an) == 10000
assert loops.groupby('n_bound_anchors').size().to_dict() == {0: 8, 1: 15, 2: 16}
assert len(matched) == 8 and int(((rob.ci_lo <= 0) & (rob.ci_hi >= 0)).sum()) == 16
assert len(ind) == 15 and len(splits) == 18 and len(perbin) == 881
assert not ((splits.ci_lo <= 0) & (splits.ci_hi >= 0)).any()
assert sets.set_index('gene_set').loc['Baseline_double_anchor_RNA', 'n_genes'] == 10
for db in ['GO_BP', 'KEGG']:
    sens = pd.read_csv(TAB / f'{db}_conditional_spatial_enrichment.csv')
    for col in ['q_conditional_within_gene_set', 'q_spatial_within_gene_set',
                'q_conditional_all_gene_sets', 'q_spatial_all_gene_sets']:
        assert not (sens[col] < 0.10).any(), (db, col)

# Initial enrichment rows matched to sensitivity rows.
screens = []
for db in ['GO_BP', 'KEGG']:
    screens.append(pd.read_csv(TAB / f'{db}_enrichment_screen.csv'))
screens = pd.concat(screens, ignore_index=True)
initial = key_enrich[['gene_set', 'database', 'term']].merge(
    screens, on=['gene_set', 'database', 'term'], how='left', validate='one_to_one')
assert initial.p_value.notna().all()
enrich = key_enrich.merge(
    initial[['gene_set', 'database', 'term', 'fold_enrichment', 'p_value', 'q_within_gene_set']],
    on=['gene_set', 'database', 'term'], suffixes=('_conditional', '_screen'))

# Freeze source tables for integrated Figures 1–3.
source('Fig1A_anchor_enrichment.csv', pd.DataFrame([a0]), 'Fig1A', OSRC / 'A_loop_anchor_primary.csv', 'Observed and circular-shift null for both-anchor NapM occupancy')
source('Fig1A_anchor_null.csv', an, 'Fig1A', OSRC / 'A_loop_anchor_null.csv', '10,000 circular shifts preserving genomic label runs')
source('Fig1A_stratified_sensitivity.csv', astr, 'Fig1A', OSRC / 'A_loop_anchor_stratified.csv', 'Loop-span, gene-density, expression and flanking-ChIP sensitivity analyses')
source('Fig1B_local_matched_effects.csv', matched, 'Fig1B', OSRC / 'C_local_occupancy_matched_effects.csv', 'Bound-minus-matched effects under two occupancy definitions')
source('Fig1B_robustness.csv', rob, 'Fig1B', OSRC / 'C_local_occupancy_robustness.csv', 'Eighteen baseline robustness variants')
source('Fig1C_loop_dose_per_loop.csv', loops, 'Fig1C', OSRC / 'B_loop_anchor_dose_per_loop.csv', 'Loop change by number of NapM-occupied anchors')
source('Fig1C_loop_dose_summary.csv', dose.reset_index(), 'Fig1C', OSRC / 'B_loop_anchor_dose_summary.csv', 'Group means, adjusted slope and hierarchical-bootstrap intervals')
source('Fig2A_independent_samples.csv', ind, 'Fig2A', OSRC / 'E_independent_sample_genome_mean_insulation.csv', 'Genome-mean insulation for 15 independent samples')
source('Fig2A_factorial_contrast.csv', fact, 'Fig2A', OSRC / 'E_independent_sample_factorial.csv', 'WT and KO hypoxia responses and genotype difference-in-differences')
source('Fig2B_fingerprint_per_sample.csv', fp, 'Fig2B', EXT4 / 'fingerprint_per_sample.csv', 'Nine structural metrics per sample')
source('Fig2B_response_vectors.csv', rv, 'Fig2B', EXT4 / 'fig4_response_vectors_contribution.csv', 'Standardized WT and KO hypoxia-response vectors')
source('Fig2B_response_statistics.csv', fpstats, 'Fig2B', EXT4 / 'fig4_response_stats.csv', 'Magnitude and direction permutation statistics')
source('Fig2C_spatial_per_bin.csv', perbin, 'Fig2C', OSRC / 'G_original_per_bin_association.csv', 'Baseline KO effect and WT hypoxia response across finite bins')
source('Fig2C_spatial_summary.csv', pd.DataFrame([sp0]), 'Fig2C', OSRC / 'G_original_spatial_association.csv', 'Primary spatial correlation, 100-kb block interval and rotation test')
source('Fig2D_disjoint_reference.csv', splits, 'Fig2D', OSRC / 'G_disjoint_WT_reference_sensitivity.csv', 'All 18 non-overlapping WT-reference sensitivity allocations')
source('Fig3A_gene_sets.csv', sets, 'Fig3A', TAB / 'multiomics_gene_set_summary.csv', 'Multi-omics gene-set sizes and NapM occupancy composition')
source('Fig3BC_pathway_sensitivity.csv', enrich, 'Fig3B,C', TAB / 'key_pathway_conditional_spatial_sensitivity.csv', 'Initial hypergeometric screen, RNA+ conditional test and gene-order spatial test')
source('Fig3E_local_RNA_increment.csv', inc, 'Fig3E', OSRC / 'D_RNA_prediction_increments.csv', 'Incremental cross-validated R2 from RNA and occupancy')
source('Fig3E_local_RNA_summary.csv', rna_desc, 'Fig3E', OSRC / 'D_RNA_descriptive_summary.csv', 'Same-bin RNA/structure descriptive association')

# ============================ Response Figure 1 =============================
fig = plt.figure(figsize=(210 * MM, 153 * MM))
title(fig, 'Response Figure 1 | NapM occupancy and local architectural tests',
      'A positional loop-anchor association is retained, while local and loop-dose analyses constrain a simple direct-action model.')
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[0.82, 1.40],
                      left=0.16, right=0.975, top=0.855, bottom=0.105,
                      hspace=0.66, wspace=0.52)

# A: observed against a compact null range, rather than a full-page histogram.
ax = clean(fig.add_subplot(gs[0, 0]))
null = an.null_count.to_numpy(float)
null_lo, null_hi = np.percentile(null, [2.5, 97.5])
ax.bar([0, 1], [a0.null_mean, a0.observed], width=0.58,
       color=[GREY, CHIP], edgecolor='none')
ax.errorbar(0, a0.null_mean, yerr=[[a0.null_mean - null_lo], [null_hi - a0.null_mean]],
            fmt='none', ecolor=INK, elinewidth=0.8, capsize=3)
for x, y, s in [(0, a0.null_mean, f'{a0.null_mean:.3f}'), (1, a0.observed, '17/41')]:
    ax.text(x, y + 0.65, s, ha='center', va='bottom', fontsize=7.2, fontweight='bold',
            color=CHIP if x == 1 else INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Circular-shift null\n(mean; 95% range)', 'Observed'])
ax.set_ylabel('Loops occupied at both anchors')
ax.set_ylim(0, 21)
ax.set_title('Positional enrichment at loop anchors', loc='left', pad=8)
ax.text(0.03, 0.96, f'fold = {a0.fold_enrichment:.2f}\none-sided P = {a0.p_empirical:.4f}',
        transform=ax.transAxes, va='top', fontsize=6.6, color=CHIP)
lab(ax, 'A', x=-0.30)

# C: loop-dose model.
ax = clean(fig.add_subplot(gs[0, 1]))
cs = {0: MID, 1: '#AA8DBB', 2: CHIP}
for group in [0, 1, 2]:
    v = loops.loc[loops.n_bound_anchors == group, 'd_log2OE'].to_numpy(float)
    x = group + RNG.uniform(-0.12, 0.12, len(v))
    ax.scatter(x, v, s=17, color=cs[group], alpha=0.68,
               edgecolor='white', linewidth=0.35, zorder=2)
    row = dose.loc[f'group{group}_mean']
    ax.errorbar(group + 0.25, row.effect,
                yerr=[[row.effect - row.ci_lo], [row.ci_hi - row.effect]],
                fmt='o', ms=4.5, color=INK, ecolor=INK, capsize=2.7, zorder=4)
ax.axhline(0, color=MID, lw=0.65, ls='--')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0\n(n = 8)', '1\n(n = 15)', '2\n(n = 16)'])
ax.set_xlabel('NapM-occupied anchors per loop')
ax.set_ylabel('Loop change, log$_2$(KO mean O/E / WT mean O/E)')
ax.set_xlim(-0.45, 2.55)
ax.set_title('No monotonic anchor-count effect on loop change', loc='left', pad=8)
s = dose.loc['adjusted_slope_per_anchor']
ax.text(0.985, 0.97,
        f'Adjusted slope per occupied anchor\nβ = {s.effect:+.4f} [{s.ci_lo:+.4f}, {s.ci_hi:+.4f}]\n'
        f'span-stratified permutation P = {s.p_span_stratified_permutation:.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=6.2,
        bbox=dict(facecolor='white', edgecolor=LIGHT, boxstyle='round,pad=0.35'))
lab(ax, 'C', x=-0.20, y=1.17)

# B: matched local effects across two occupancy definitions.
ax = clean(fig.add_subplot(gs[1, :]))
metric_order = ['baseline_abs_dins', 'local_contact_signed', 'local_contact_abs',
                'hypoxia_interaction_signed']
labels = {'baseline_abs_dins': 'Baseline |Δinsulation|',
          'local_contact_signed': 'Local contact (signed)',
          'local_contact_abs': '|Local contact|',
          'hypoxia_interaction_signed': 'G × hypoxia interaction'}
ypos = np.arange(4)[::-1] * 1.05
for y, metric in zip(ypos, metric_order):
    for definition, off, color, marker in [
        ('centre_433', 0.16, CHIP, 'o'), ('interval_454', -0.16, HYP, 's')]:
        r = matched[(matched.metric == metric) & (matched.binding_definition == definition)].iloc[0]
        ax.errorbar(r.effect, y + off,
                    xerr=[[r.effect - r.ci_lo], [r.ci_hi - r.effect]],
                    fmt=marker, ms=4.1, color=color, ecolor=color,
                    markerfacecolor=color if definition == 'centre_433' else 'white',
                    elinewidth=0.95, capsize=2.1)
ax.axvline(0, color=MID, lw=0.7, ls='--')
ax.set_yticks(ypos)
ax.set_yticklabels([labels[x] for x in metric_order])
ax.set_ylim(-0.75, 3.9)
ax.set_xlim(-0.055, 0.055)
ax.set_xlabel('NapM-bound − matched-locus effect (metric-specific units)')
ax.set_title('Local occupancy does not identify a consistent remodeling phenotype', loc='left', pad=8)
ax.legend(handles=[
    Line2D([], [], marker='o', color=CHIP, lw=0.9, label='Peak-centre definition'),
    Line2D([], [], marker='s', color=HYP, markerfacecolor='white', lw=0.9, label='Interval-overlap definition')],
    frameon=False, ncol=2, loc='upper right')
lab(ax, 'B', x=-0.15)
fig.text(0.055, 0.025,
         'Interpretation: NapM occupancy is positionally associated with loop anchors, but neither local effect sizes nor the anchor-count model establish direct loop maintenance.',
         fontsize=6.2, color=MID, ha='left')
save_fig(fig, 1)

# ============================ Response Figure 2 =============================
fig = plt.figure(figsize=(210 * MM, 192 * MM))
title(fig, 'Response Figure 2 | Genotype-dependent hypoxia remodeling',
      'Independent samples, nine structural metrics and disjoint-reference analyses test reproducibility without claiming a shared upstream pathway.')
gs = fig.add_gridspec(2, 3, height_ratios=[1.06, 0.92], width_ratios=[0.90, 1.05, 1.15],
                      left=0.09, right=0.965, top=0.86, bottom=0.115,
                      hspace=0.66, wspace=0.58)

# A: independent samples.
ax = clean(fig.add_subplot(gs[0, 0]))
conditions = [('WT', 0, 'WT\nnorm.', WT, 'o', False),
              ('HYP', 1, 'WT\nhyp.', HYP, 'o', True),
              ('KO', 3, 'KO\nnorm.', KO, 's', False),
              ('KO-HYP', 4, 'KO\nhyp.', KOHYP, 's', True)]
means = {}
for cond, x, label, color, marker, fill in conditions:
    v = ind.loc[ind.condition == cond, 'genome_mean_insulation'].to_numpy(float)
    means[cond] = v.mean()
    ax.scatter(x + RNG.uniform(-0.10, 0.10, len(v)), v, s=21, marker=marker,
               facecolors=color if fill else 'white', edgecolors=color, linewidth=0.85, zorder=3)
    ax.plot([x - 0.25, x + 0.25], [v.mean(), v.mean()], color=color, lw=1.5)
ax.plot([0, 1], [means['WT'], means['HYP']], color=HYP, lw=1.0)
ax.plot([3, 4], [means['KO'], means['KO-HYP']], color=KOHYP, lw=1.0)
ax.set_xticks([x[1] for x in conditions])
ax.set_xticklabels([x[2] for x in conditions], fontsize=5.9)
ax.set_ylabel('Genome-mean insulation\n(one value per biological sample)')
ax.set_title('Independent-sample response', loc='left', pad=8)
e_did = fact[fact.test == 'genotype contrast of genome-mean hypoxia responses'].iloc[0]
e_wt = fact[fact.test == 'WT genome-mean hypoxia response'].iloc[0]
e_ko = fact[fact.test == 'KO genome-mean hypoxia response'].iloc[0]
ax.text(0.02, 0.02,
        f'WT response = {e_wt.effect:+.4f}; KO response = {e_ko.effect:+.4f}\n'
        f'difference-in-differences = {e_did.effect:+.4f}\n'
        f'95% CI [{e_did.ci_lo:+.4f}, {e_did.ci_hi:+.4f}]\n'
        f'exact oxygen-label P = {e_did.p:.6f}; n = 4/4/4/3',
        transform=ax.transAxes, fontsize=5.7, va='bottom',
        bbox=dict(facecolor='white', edgecolor=LIGHT, boxstyle='round,pad=0.3'))
lab(ax, 'A', x=-0.26)

# C: original spatial association.
ax = clean(fig.add_subplot(gs[0, 1]))
hb = ax.hexbin(perbin.d_ins_hyp_WT, perbin.d_ins_KO_WT, gridsize=37,
               cmap='Blues', mincnt=1, linewidths=0)
ax.axhline(0, color=GREY, lw=0.5)
ax.axvline(0, color=GREY, lw=0.5)
lim = float(np.nanmax(np.abs(perbin[['d_ins_hyp_WT', 'd_ins_KO_WT']].to_numpy())))
ax.plot([-lim, lim], [-lim, lim], ls='--', lw=0.65, color=MID)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel('WT hypoxia response\nΔinsulation (hypoxia − WT)')
ax.set_ylabel('Baseline KO effect\nΔinsulation (KO − WT)')
ax.set_title('Shared spatial pattern', loc='left', pad=8)
ax.text(0.04, 0.96,
        f'Spearman ρ = {sp0.estimator:.4f}\n100-kb block 95% CI\n[{sp0.ci_lo:.4f}, {sp0.ci_hi:.4f}]\nn = 881 bins',
        transform=ax.transAxes, fontsize=5.9, va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.0))
cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
cb.set_label('bins', fontsize=5.8)
cb.ax.tick_params(labelsize=5.3)
cb.outline.set_visible(False)
lab(ax, 'C', x=-0.25)

# D: every disjoint-reference allocation, with original estimate for context.
ax = clean(fig.add_subplot(gs[0, 2]))
rows = [dict(label='Original shared reference', rho=sp0.estimator, ci_lo=sp0.ci_lo,
             ci_hi=sp0.ci_hi, design='original')]
for i, r in enumerate(splits.itertuples(), 1):
    design = '2+2' if r.reference_design == '2plus2' else '1+1'
    rows.append(dict(label=f'{design} allocation {i if design == "2+2" else i-6}',
                     rho=r.rho, ci_lo=r.ci_lo, ci_hi=r.ci_hi, design=r.reference_design))
dplot = pd.DataFrame(rows)
y = np.arange(len(dplot))[::-1]
for yi, r in zip(y, dplot.itertuples()):
    color = INK if r.design == 'original' else (WT if r.design == '2plus2' else HYP)
    ax.errorbar(r.rho, yi, xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]], fmt='o',
                ms=3.1 if r.design == 'original' else 2.4, color=color, ecolor=color,
                elinewidth=0.7, capsize=1.2)
ax.axvline(0, color=MID, lw=0.6, ls='--')
ax.set_yticks(y)
ax.set_yticklabels(dplot.label, fontsize=4.9)
ax.get_yticklabels()[0].set_fontweight('bold')
ax.set_xlim(0.28, 0.70)
ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_xlabel('Spearman ρ')
ax.set_title('18 non-overlapping WT references', loc='left', pad=8)
ax.text(0.02, 0.02, f'ρ range = {splits.rho.min():.3f}–{splits.rho.max():.3f}\n'
        'All 100-kb block intervals > 0\nSensitivity analyses, not new replicates',
        transform=ax.transAxes, fontsize=5.5, color=MID, va='bottom',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.8))
lab(ax, 'D', x=-0.39)

# B: nine-metric heatmap plus the two response vectors.
metrics = rv.metric.tolist()
metric_labels = [x.replace(' frac', '\nfrac').replace(' mean', '\nmean').replace(' depth', '\ndepth')
                 for x in rv.metric_label]
cond_order = ['WT', 'HYP', 'KO', 'KO-HYP']
# WT-latent samples remain available in the source table but are not part of this response.
fp = fp[fp.condition.isin(cond_order)].copy()
fp['condition'] = pd.Categorical(fp.condition, cond_order, ordered=True)
fp = fp.sort_values(['condition', 'replicate'])
assert len(fp) == 15
source('Fig2B_fingerprint_15_test_samples.csv', fp, 'Fig2B', EXT4 / 'fingerprint_per_sample.csv',
       'The exact 15 samples plotted in the main heatmap; WT-latent samples excluded')
mat = fp[metrics].to_numpy(float)
z = (mat - np.nanmean(mat, axis=0)) / np.nanstd(mat, axis=0, ddof=0)
response_mat = np.column_stack([rv.R_WT.to_numpy(float), rv.R_KO.to_numpy(float)])
combined = np.column_stack([z.T, np.full((z.shape[1], 1), np.nan), response_mat])
ax = fig.add_subplot(gs[1, :])
norm = mcolors.TwoSlopeNorm(vmin=-2.7, vcenter=0, vmax=2.7)
cmap = mcolors.LinearSegmentedColormap.from_list('fingerprint', [DOWN, '#F7F7F7', UP])
im = ax.imshow(combined, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
labels_x = [f'{c}{r}' for c, r in zip(fp.condition.astype(str), fp.replicate)] + ['', 'WT\nresponse', 'KO\nresponse']
ax.set_xticks(np.arange(len(labels_x)))
ax.set_xticklabels(labels_x, fontsize=5.5)
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metric_labels, fontsize=5.7)
ax.set_xticks(np.arange(-0.5, combined.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, combined.shape[0], 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.45)
ax.tick_params(which='minor', bottom=False, left=False)
for i in range(1, len(z)):
    if fp.condition.iloc[i] != fp.condition.iloc[i - 1]:
        ax.axvline(i - 0.5, color=INK, lw=1.1)
ax.axvline(len(z) + 0.5, color=INK, lw=1.1)
for x, text, color in [(1.5, 'WT normoxia', WT), (5.5, 'WT hypoxia', HYP),
                       (9.5, 'KO normoxia', KO), (13.5, 'KO hypoxia', KOHYP)]:
    ax.text(x, -0.83, text, color=color, fontsize=5.9, ha='center', va='bottom', fontweight='bold')
ax.set_title('Nine-metric per-sample fingerprints and standardized hypoxia-response vectors', loc='left', pad=16)
cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
cb.set_label('Sample z-score / standardized response', fontsize=5.8)
cb.ax.tick_params(labelsize=5.4)
cb.outline.set_visible(False)
lab(ax, 'B', x=-0.105, y=1.17)
fig.text(0.055, 0.058,
         'Response-vector Euclidean distance = 5.9506 (P = 0.00080); cosine similarity = −0.0696 (P = 0.365): magnitude differs, but global direction reversal is not established.',
         fontsize=5.8, color=MID)
fig.text(0.055, 0.025,
         'WT, wild type; KO, napM knockout. Heatmap columns are individual samples; the final two columns are response vectors, not additional samples.',
         fontsize=6.0, color=MID)
save_fig(fig, 2)

# ============================ Response Figure 3 =============================
fig = plt.figure(figsize=(210 * MM, 250 * MM))
title(fig, 'Response Figure 3 | Exploratory multi-omics gene programmes',
      'Functional screening generates biological hypotheses; RNA-background and gene-order tests define which enrichment claims are not robust.')
gs = fig.add_gridspec(3, 2, height_ratios=[0.72, 0.93, 1.18], width_ratios=[1.40, 1.00],
                      left=0.17, right=0.97, top=0.88, bottom=0.085,
                      hspace=0.62, wspace=0.58)

# A: gene sets and direct-occupancy composition.
ax = clean(fig.add_subplot(gs[0, :]))
set_order = ['Baseline_structure_RNA', 'Baseline_structure_RNA_direct',
             'Baseline_structure_RNA_unbound', 'Baseline_double_anchor_RNA',
             'Hypoxia_structure_RNA', 'Hypoxia_structure_RNA_direct',
             'Interaction_structure_hypoxia_RNA']
short = {'Baseline_structure_RNA': 'Baseline structure + RNA',
         'Baseline_structure_RNA_direct': 'Baseline; NapM-bound background',
         'Baseline_structure_RNA_unbound': 'Baseline; unbound background',
         'Baseline_double_anchor_RNA': 'Double-anchor genes + RNA + structure',
         'Hypoxia_structure_RNA': 'Hypoxia structure + RNA',
         'Hypoxia_structure_RNA_direct': 'Hypoxia; NapM-bound background',
         'Interaction_structure_hypoxia_RNA': 'Interaction + hypoxia RNA'}
sp = sets.set_index('gene_set').loc[set_order].reset_index()
y = np.arange(len(sp))[::-1]
ax.barh(y, sp.n_genes, color=LIGHT, edgecolor='none', height=0.64)
ax.barh(y, sp.n_direct, color=CHIP, edgecolor='none', height=0.64)
for yi, r in zip(y, sp.itertuples()):
    ax.text(r.n_genes + max(sp.n_genes) * 0.025, yi,
            f'{r.n_genes} ({r.n_direct} bound)', va='center', fontsize=5.7)
ax.set_yticks(y)
ax.set_yticklabels([textwrap.fill(short[x], 26) for x in sp.gene_set], fontsize=5.5)
ax.set_xlim(0, 1500)
ax.set_xlabel('Genes in set')
ax.set_title('Multi-omics gene sets', loc='left', pad=8)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=CHIP, label='NapM promoter/body occupied'),
                   Rectangle((0, 0), 1, 1, color=LIGHT, label='Total set')],
          frameon=False, loc='lower right', fontsize=5.3)
lab(ax, 'A', x=-0.20)

# Helper for initial-versus-conditional enrichment plots.
def enrichment_panel(ax, subset, panel_letter, panel_title):
    subset = subset.copy().iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(subset)) * 1.15
    for yi, r in zip(yy, subset.itertuples()):
        ax.plot([r.fold_enrichment_conditional, r.fold_enrichment_screen], [yi, yi],
                color=GREY, lw=1.0, zorder=1)
        ax.scatter(r.fold_enrichment_screen, yi, s=34, color=KO, zorder=3, label='Initial screen' if yi == 0 else None)
        ax.scatter(r.fold_enrichment_conditional, yi, s=30, color=WT, zorder=3, label='RNA+ conditional' if yi == 0 else None)
        ax.text(7.25, yi + 0.15, f'initial q = {r.q_within_gene_set:.3g}', fontsize=5.3, ha='right', va='center', color=KO)
        ax.text(7.25, yi - 0.20,
                f'RNA+ P = {r.p_conditional_RNAplus:.3g}; spatial P = {r.p_gene_order_circular_shift:.3g}',
                fontsize=5.1, ha='right', va='center', color=WT)
    ax.set_yticks(yy)
    long_labels = {
        'Biosynthesis of siderophore group nonribosomal peptides': 'Siderophore biosynthesis\n(mbtD/C/B/A)',
        'response to copper ion': 'Response to copper ion',
        'translation': 'Translation (GO)',
        'Ribosome': 'Ribosome (KEGG)',
        'ATP synthesis coupled electron transport': 'ATP synthesis-coupled\nelectron transport'}
    ax.set_yticklabels([long_labels[x] for x in subset.name], fontsize=5.6)
    ax.set_xlim(0, 7.35)
    ax.set_ylim(-0.90, max(yy) + 0.80)
    ax.set_xlabel('Fold enrichment')
    ax.set_title(panel_title, loc='left', pad=8)
    ax.axvline(1, color=MID, lw=0.6, ls='--')
    ax.text(0.02, 0.025, 'No pathway retains FDR < 0.10 in RNA+ or spatial tests',
            transform=ax.transAxes, fontsize=5.35, color=INK, va='bottom',
            bbox=dict(facecolor='#F5F7F8', edgecolor='none', boxstyle='round,pad=0.3'))
    lab(ax, panel_letter, x=-0.40)

bset = enrich[enrich.gene_set == 'Baseline_structure_RNA']
ax = clean(fig.add_subplot(gs[1, 0]))
enrichment_panel(ax, bset, 'B', 'Baseline programmes')
ax.legend(frameon=False, loc='upper right', fontsize=5.2)

cset = enrich[enrich.gene_set.isin(['Hypoxia_structure_RNA', 'Interaction_structure_hypoxia_RNA'])].copy()
cset['plot_order'] = cset.name.map({'response to copper ion': 0, 'Ribosome': 1,
                                   'ATP synthesis coupled electron transport': 2, 'translation': 3})
cset = cset.sort_values('plot_order')
ax = clean(fig.add_subplot(gs[1, 1]))
enrichment_panel(ax, cset, 'C', 'Hypoxia / interaction programmes')

# D: named genes, expression and local structure, with occupancy kept categorical.
selected = ['Rv0001', 'Rv2381c', 'Rv2382c', 'Rv2383c', 'Rv2384',
            'Rv0967', 'Rv0969', 'Rv1992c', 'Rv2200c',
            'Rv3155', 'Rv3156', 'Rv3157', 'Rv3158']
dgenes = genes[genes.gene_id.isin(selected)].copy()
dgenes['display'] = dgenes.gene_name.fillna(dgenes.gene_id) + ' (' + dgenes.gene_id + ')'
dgenes['order'] = dgenes.gene_id.map({g: i for i, g in enumerate(selected)})
dgenes = dgenes.sort_values('order')
source('Fig3D_named_gene_evidence.csv', dgenes, 'Fig3D', RA / '04_rna_integration/gene_level_rna_v2.csv',
       'Named genes: RNA responses, bin-level insulation change and local NapM occupancy')
ax = fig.add_subplot(gs[2, 0])
vals = dgenes[['bin_d_ins_KO_WT', 'log2FC_KOvsWT', 'log2FC_hypoxiavsWT']].to_numpy(float)
norm_rna = mcolors.TwoSlopeNorm(vmin=-3.0, vcenter=0, vmax=3.0)
cmap_rna = mcolors.LinearSegmentedColormap.from_list('rna', [DOWN, '#F7F7F7', UP])
im = ax.imshow(vals, aspect='auto', cmap=cmap_rna, norm=norm_rna, interpolation='nearest')
for i in range(len(dgenes)):
    for j in range(3):
        v = vals[i, j]
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=5.5,
                color='white' if abs(v) > 1.75 else INK)
    occupied = dgenes.binding_class.iloc[i] in ['promoter_bound', 'gene_body_bound']
    ax.add_patch(Rectangle((2.55, i - 0.43), 0.90, 0.86,
                           facecolor=CHIP if occupied else '#F0F2F3', edgecolor='white', lw=0.5))
    ax.text(3, i, 'bound' if occupied else 'unbound', ha='center', va='center', fontsize=5.2,
            color='white' if occupied else MID)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Δinsulation\nKO − WT', 'RNA log$_2$FC\nKO / WT', 'RNA log$_2$FC\nhypoxia / WT', 'Local NapM\noccupancy'], fontsize=5.8)
ax.set_yticks(np.arange(len(dgenes)))
ax.set_yticklabels(dgenes.display, fontsize=5.8, fontstyle='italic')
ax.set_xlim(-0.5, 3.5)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(dgenes), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.7)
ax.tick_params(which='minor', bottom=False, left=False)
ax.set_title('Named genes preserve the direct/indirect distinction', loc='left', pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.set_label('Δinsulation or RNA log$_2$FC', fontsize=5.8)
cb.ax.tick_params(labelsize=5.4)
cb.outline.set_visible(False)
lab(ax, 'D', x=-0.20)

# E: local linear RNA explanatory value, explicitly distinct from functional screens.
ax = clean(fig.add_subplot(gs[2, 1]))
y = [1, 0]
for yi, r, color in zip(y, inc.itertuples(), [WT, CHIP]):
    ax.errorbar(r.dR2, yi, xerr=[[r.dR2 - r.ci_lo], [r.ci_hi - r.dR2]], fmt='o',
                ms=5.0, color=color, ecolor=color, elinewidth=1.1, capsize=3)
    ax.text(0.0052, yi, f'{r.dR2:+.6f}\n[{r.ci_lo:+.6f}, {r.ci_hi:+.6f}]',
            ha='right', va='center', fontsize=5.8, color=color)
ax.axvline(0, color=MID, lw=0.65, ls='--')
ax.set_yticks(y)
ax.set_yticklabels(['Add local RNA', 'Then add NapM occupancy'])
ax.set_xlim(-0.0024, 0.0054)
ax.set_xticks([-0.002, 0, 0.002, 0.004])
ax.set_ylim(-0.85, 1.65)
ax.set_xlabel('Increment in contiguous-block\ncross-validated R$^2$')
ax.set_title('Weak local linear RNA coupling', loc='left', pad=8)
rho = rna_desc.loc[rna_desc.metric == 'same_bin_RNA_vs_delta_insulation_Spearman_rho'].iloc[0]
ax.text(0.03, 0.96,
        f'RNA–Δinsulation Spearman ρ = {rho.value:+.4f} (n = {int(rho.n)})\n'
        '848 complete bins; intervals include zero.\n'
        'This does not exclude distal or operon-level regulation.',
        transform=ax.transAxes, fontsize=5.8, va='top')
ax.text(0.03, 0.06,
        'Why both panels matter:\nD tests selected functional loci;\nE tests genome-wide same-bin linear prediction.',
        transform=ax.transAxes, fontsize=5.7, color=MID, va='bottom',
        bbox=dict(facecolor='#F5F7F8', edgecolor='none', boxstyle='round,pad=0.4'))
lab(ax, 'E', x=-0.30)
fig.text(0.055, 0.022,
         'Enrichment panels intentionally show attenuation after sensitivity testing. Pathways are hypothesis-generating priorities, not validated causal mechanisms.',
         fontsize=6.2, color=MID)
save_fig(fig, 3)

# ============================ Response Figure 4 =============================
# Browser code reuses the validated plotting O/E definition from the project.
sys.path.insert(0, str(RA / '09_combined/scripts'))
import fc9  # noqa: E402
import cooler  # noqa: E402
fc = fc9.fc
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.0,
                     'axes.labelsize': 7.0, 'axes.titlesize': 7.7,
                     'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0})
clrs = {c: cooler.Cooler(f'{fc.COOL}/{f}::/resolutions/5000')
        for c, f in [('WT', 'wt_5k.cool'), ('KO', 'napm_ko_5k.cool')]}
loci = [dict(panel='A', key='oriC', heading='oriC–dnaA | convergence with a published NapM–DnaA axis [1]',
             i0=0, i1=26, x0=0, x1=130000, label_genes=['dnaA', 'gyrB']),
        dict(panel='B', key='mbt1', heading='mbt-1 | local occupancy at mbtD, not the entire operon [2]',
             i0=528, i1=540, x0=2640000, x1=2700000, label_genes=['mbtD', 'mbtB', 'mbtA'])]
triangles = {}
allv = []
for loc in loci:
    for cond in ['WT', 'KO']:
        img, ext = fc.oe_triangle(clrs[cond], loc['i0'], loc['i1'])
        triangles[(loc['key'], cond)] = (img, ext)
        allv.append(img.compressed())
allv = np.concatenate(allv)
vmin = min(float(np.nanpercentile(allv, 2)), 0.98)
vmax = max(float(np.nanpercentile(allv, 98)), 1.02)
norm_oe = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
cmap_oe = mcolors.LinearSegmentedColormap.from_list('oe', [DOWN, '#F7F7F7', UP])
cmap_oe.set_bad('#D7D7D7')

fig = plt.figure(figsize=(210 * MM, 236 * MM))
title(fig, 'Response Figure 4 | Mechanistically informed loci and working model',
      'Published molecular evidence, current genomic associations and untested hypotheses are shown separately rather than merged into a causal claim.')
outer = fig.add_gridspec(2, 2, height_ratios=[1.66, 0.63],
                         left=0.075, right=0.965, top=0.885, bottom=0.065,
                         hspace=0.24, wspace=0.26)
browser_tracks = []
im0 = None
for k, loc in enumerate(loci):
    # Two O/E maps followed by delta-insulation, ChIP/loops, RNA and gene tracks.
    sub = outer[0, k].subgridspec(6, 1,
                                  height_ratios=[1.00, 1.00, 0.54, 0.55, 0.42, 0.48],
                                  hspace=0.13)
    i0, i1 = loc['i0'], loc['i1']
    x0m, x1m = loc['x0'] / 1e6, (loc['x1'] - 5000) / 1e6
    sub_bins = bins.iloc[i0:i1]
    sub_genes = genes[(genes.end >= loc['x0']) & (genes.start <= loc['x1'])].copy()
    sub_peaks = peaks[(peaks.center >= loc['x0']) & (peaks.center <= loc['x1'])].copy()
    sub_loops = loops41[(loops41.center1 >= loc['x0']) & (loops41.center2 <= loc['x1'])].copy()
    loc['genes'] = sub_genes
    loc['peaks'] = sub_peaks
    loc['loops'] = sub_loops
    loc['bins'] = sub_bins
    source(f'Fig4{loc["panel"]}_{loc["key"]}_bins.csv', sub_bins, f'Fig4{loc["panel"]}',
           RA / '03_structure_metrics/bin_level_v2.csv', f'{loc["key"]} structural tracks')
    source(f'Fig4{loc["panel"]}_{loc["key"]}_genes.csv', sub_genes, f'Fig4{loc["panel"]}',
           RA / '04_rna_integration/gene_level_rna_v2.csv', f'{loc["key"]} gene coordinates and RNA contrasts')
    source(f'Fig4{loc["panel"]}_{loc["key"]}_peaks.csv', sub_peaks, f'Fig4{loc["panel"]}',
           RA / '01_chip_curated/NapM_binding_curated.tsv', f'{loc["key"]} curated NapM ChIP peaks')
    source(f'Fig4{loc["panel"]}_{loc["key"]}_loops.csv', sub_loops, f'Fig4{loc["panel"]}',
           RA / '02_loop_curated/loops_41_curated.tsv', f'{loc["key"]} curated loops and occupancy labels')
    for j, cond in enumerate(['WT', 'KO']):
        ax = fig.add_subplot(sub[j, 0])
        img, ext = triangles[(loc['key'], cond)]
        arr = img.copy()
        arr[0, :] = np.nan
        im = ax.imshow(arr, cmap=cmap_oe, norm=norm_oe, extent=ext, origin='lower',
                       aspect='auto', interpolation='none')
        if im0 is None:
            im0 = im
        ax.text(0.02, 0.91, cond, transform=ax.transAxes, fontsize=6.2, fontweight='bold',
                color=WT if cond == 'WT' else KO, va='top')
        ax.set_xticks([])
        ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_visible(True)
            spn.set_linewidth(0.5)
        browser_tracks.append(ax)
        if j == 0:
            first_ax = ax
            lab(ax, loc['panel'], x=-0.14, y=1.18)
            ax.set_title(loc['heading'], loc='left', fontsize=6.6, pad=8)
        source(f'Fig4{loc["panel"]}_{loc["key"]}_{cond}_OE_triangle.csv',
               pd.DataFrame(np.ma.filled(img, np.nan)), f'Fig4{loc["panel"]}',
               f'{fc.COOL}/{"wt_5k.cool" if cond == "WT" else "napm_ko_5k.cool"}',
               f'{loc["key"]} {cond} balanced observed/expected triangle; extent={ext}')
    axd = clean(fig.add_subplot(sub[2, 0]))
    xx = (sub_bins.start.to_numpy(float) + 2500) / 1e6
    axd.axhline(0, color=MID, lw=0.5)
    axd.plot(xx, sub_bins.d_ins_KO_WT, color=INK, lw=0.9, label='KO − WT')
    axd.plot(xx, sub_bins.d_ins_hyp_WT, color=HYP, lw=0.8, label='WT hypoxia − WT')
    axd.set_ylabel('Δinsulation', fontsize=5.7)
    axd.set_xlim(x0m, x1m)
    axd.set_xticks([])
    axd.yaxis.set_major_locator(plt.MaxNLocator(3))
    axd.tick_params(labelsize=5.2)
    axd.legend(frameon=False, fontsize=4.8, loc='upper right', ncol=2, handlelength=1.2,
               borderaxespad=0.05, columnspacing=0.6)
    axp = clean(fig.add_subplot(sub[3, 0]))
    axp.vlines(sub_peaks.center / 1e6, 0, sub_peaks.score, color=CHIP, lw=0.9)
    for r in sub_loops.itertuples():
        c1, c2 = r.center1 / 1e6, r.center2 / 1e6
        t = np.linspace(0, np.pi, 80)
        xc = (c1 + c2) / 2 + (c2 - c1) / 2 * np.cos(t)
        yc = 1.12 + 0.30 * np.sin(t)
        col = CHIP if r.both_bound else GREY
        axp.plot(xc, yc, color=col, lw=0.9 if r.both_bound else 0.55)
        axp.text((c1 + c2) / 2, 1.47, r.loop_id, fontsize=4.7, color=col, ha='center')
    axp.set_ylim(0, 1.58)
    axp.set_xlim(x0m, x1m)
    axp.set_xticks([])
    axp.set_yticks([0, 1])
    axp.set_yticklabels(['0', '1'], fontsize=5.0)
    axp.set_ylabel('NapM ChIP\n+ loops', fontsize=5.4)
    axp.spines['left'].set_visible(False)
    axr = fig.add_subplot(sub[4, 0])
    for r in sub_genes.itertuples():
        start, end = max(r.start, loc['x0']) / 1e6, min(r.end, loc['x1']) / 1e6
        width = max(end - start, 0.0004)
        for row, value in [(1, r.log2FC_KOvsWT), (0, r.log2FC_hypoxiavsWT)]:
            col = cmap_rna(norm_rna(value)) if np.isfinite(value) else '#E5E5E5'
            axr.add_patch(Rectangle((start, row + 0.10), width, 0.78,
                                    facecolor=col, edgecolor='none'))
    axr.set_xlim(x0m, x1m)
    axr.set_ylim(-0.05, 2.05)
    axr.set_yticks([0.5, 1.5])
    axr.set_yticklabels(['hypoxia / WT', 'KO / WT'], fontsize=5.0)
    axr.set_xticks([])
    axr.set_ylabel('Gene RNA', fontsize=5.4)
    for spn in axr.spines.values():
        spn.set_visible(False)
    axg = clean(fig.add_subplot(sub[5, 0]))
    for r in sub_genes.itertuples():
        yy = 0.68 if r.strand == '+' else 0.30
        value = r.log2FC_KOvsWT
        col = cmap_rna(norm_rna(value)) if np.isfinite(value) else GREY
        axg.plot([max(r.start, loc['x0']) / 1e6, min(r.end, loc['x1']) / 1e6], [yy, yy],
                 color=col, lw=1.25, solid_capstyle='butt')
        if r.gene_name in loc['label_genes']:
            axg.text(r.tss / 1e6, 0.91 if r.strand == '+' else 0.06,
                     r.gene_name, fontsize=5.0, fontstyle='italic', rotation=45,
                     ha='left' if r.strand == '+' else 'right', va='center')
    axg.set_ylim(0, 1.05)
    axg.set_yticks([])
    axg.set_xlim(x0m, x1m)
    axg.set_xlabel('Genome position (Mb)', fontsize=5.7)
    axg.tick_params(labelsize=5.1)
    axg.xaxis.set_major_locator(plt.MaxNLocator(4))
    axg.spines['left'].set_visible(False)

cboe = fig.colorbar(im0, ax=browser_tracks, fraction=0.015, pad=0.012)
cboe.set_label('Balanced Hi-C O/E', fontsize=5.7)
cboe.ax.tick_params(labelsize=5.0)
cboe.outline.set_visible(False)

# C: two-lane working model with a solid/dashed evidence code.
axm = fig.add_subplot(outer[1, :])
axm.set_xlim(0, 1)
axm.set_ylim(0, 1)
axm.axis('off')
lab(axm, 'C', x=-0.02, y=1.10)
axm.text(0.02, 0.97, 'Working model: molecular evidence first, genomic convergence second, causal tests still required',
         fontsize=7.6, fontweight='bold', va='top')


def box(ax, x, y, w, h, heading, body, edge=GREY, face='#F8FAFA', heading_color=INK):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.012',
                           transform=ax.transAxes, facecolor=face, edgecolor=edge, lw=0.85)
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.035, heading, transform=ax.transAxes,
            fontsize=6.3, fontweight='bold', color=heading_color, va='top')
    ax.text(x + 0.012, y + h - 0.12, body, transform=ax.transAxes,
            fontsize=5.5, color=INK, va='top', linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                                arrowstyle='-|>', mutation_scale=8, lw=0.95,
                                linestyle='--' if dashed else '-', color=color))

# oriC lane.
box(axm, 0.03, 0.51, 0.26, 0.34, 'Published direct molecular evidence [1]',
    'NapM physically interacts with DnaA and inhibits\nDnaA oriC binding and ATPase activity.', edge=CHIP, face='#F5F1F8', heading_color=CHIP)
box(axm, 0.37, 0.51, 0.26, 0.34, 'Current multi-omics convergence (A)',
    'NapM occupies dnaA and both anchors of L00/L01.\nnapM deletion: dnaA RNA ↓; local Δinsulation ≠ 0.\nDnaA activity and replication initiation not measured.', edge=WT, face='#F3F7FB', heading_color=WT)
box(axm, 0.71, 0.51, 0.26, 0.34, 'Testable hypothesis — not demonstrated',
    'NapM–DnaA/replication-initiation state\ncontributes to oriC chromosome dynamics\nand the physiological hypoxia response.', edge=HYP, face='#F2F9F7', heading_color=HYP)
arrow(axm, 0.29, 0.68, 0.37, 0.68, dashed=False)
arrow(axm, 0.63, 0.68, 0.71, 0.68, dashed=True, color=HYP)
# mbt lane.
box(axm, 0.03, 0.08, 0.26, 0.32, 'Established mbt-1 regulation [2]',
    'IdeR–Fe²⁺ represses mbt genes under iron-replete\nconditions; HupB promotes mycobactin biosynthesis.\nThese are alternative or cooperating mechanisms.', edge=KO, face='#FCF7F2', heading_color=KO)
box(axm, 0.37, 0.08, 0.26, 0.32, 'Current locus-level evidence (B)',
    'NapM occupies mbtD, but not mbtC/B/A locally.\nThe four transcripts decrease in the KO.\nPathway enrichment is not sensitivity-robust.', edge=WT, face='#F3F7FB', heading_color=WT)
box(axm, 0.71, 0.08, 0.26, 0.32, 'Testable hypothesis — not demonstrated',
    'Local NapM action and established iron regulation\njointly shape mbt-1 output; direct promoter-level\nNapM control of the whole operon is not assumed.', edge=HYP, face='#F2F9F7', heading_color=HYP)
arrow(axm, 0.29, 0.24, 0.37, 0.24, dashed=False)
arrow(axm, 0.63, 0.24, 0.71, 0.24, dashed=True, color=HYP)
axm.text(0.50, 0.005,
         'Solid arrows connect established evidence to the present comparison; dashed arrows mark proposed causal links requiring perturbation and biochemical validation.',
         fontsize=5.4, color=MID, ha='center', va='bottom')
model = pd.DataFrame([
    dict(lane='oriC-dnaA', evidence='published', statement='NapM binds DnaA and inhibits DnaA oriC binding and ATPase activity', reference='[1]'),
    dict(lane='oriC-dnaA', evidence='current association', statement='NapM ChIP at dnaA; RNA and insulation change after napM deletion; L00/L01 both-anchor occupancy', reference='Response Fig.4A'),
    dict(lane='oriC-dnaA', evidence='untested hypothesis', statement='NapM-DnaA/replication-initiation state contributes to oriC dynamics and hypoxia physiology', reference='requires experimental test'),
    dict(lane='mbt-1', evidence='published', statement='IdeR-Fe2+ repression and HupB promotion regulate mycobactin biosynthesis', reference='[2]'),
    dict(lane='mbt-1', evidence='current association', statement='NapM ChIP at mbtD only; mbtD/C/B/A RNA decreases; enrichment fails sensitivity FDR', reference='Response Fig.4B'),
    dict(lane='mbt-1', evidence='untested hypothesis', statement='Local NapM action and established iron regulation jointly shape mbt-1 output', reference='requires experimental test'),
])
source('Fig4C_working_model_evidence.csv', model, 'Fig4C', TAB / 'key_loci_evidence.csv', 'Evidence-tier statements used in the working model')
save_fig(fig, 4)

pd.DataFrame(manifest).to_csv(SD / 'source_manifest.csv', index=False)
key_results = dict(
    figures=4,
    anchor_enrichment=dict(observed=17, total=41, null_mean=float(a0.null_mean), fold=float(a0.fold_enrichment), p=float(a0.p_empirical)),
    local_robustness_cross_zero=16,
    loop_dose=dict(beta=float(dose.loc['adjusted_slope_per_anchor', 'effect']),
                   ci=[float(dose.loc['adjusted_slope_per_anchor', 'ci_lo']), float(dose.loc['adjusted_slope_per_anchor', 'ci_hi'])],
                   p=float(dose.loc['adjusted_slope_per_anchor', 'p_span_stratified_permutation'])),
    independent_samples=15,
    did=float(e_did.effect),
    response_euclidean=5.95063,
    response_cosine=-0.069628,
    spatial_rho=float(sp0.estimator),
    disjoint_rho_range=[float(splits.rho.min()), float(splits.rho.max())],
    enrichment_sensitivity='No pathway has FDR < 0.10 in RNA+ conditional or gene-order spatial tests',
    key_loci=['oriC-dnaA', 'mbt-1'],
    OE_shared_scale=[vmin, vmax],
)
(TAB / 'integrated_figure_key_results.json').write_text(json.dumps(key_results, indent=2) + '\n')
print('Wrote', len(manifest), 'panel source tables and integrated figure key results.')
