#!/usr/bin/env python3
"""Redraw Response Figure 3 with direct-bound, hypothesis-generating gene sets.

Changes from the original Figure 3:
A keeps only the three parent gene sets and their NapM-bound composition.
B/C show one RNA+-conditional result per pathway, not initial-versus-final dots.
B uses the baseline direct-bound set; C uses the structural-interaction direct-
bound set (RNA criterion: WT hypoxia response, not RNA interaction).
D contains only NapM-bound genes from the displayed pathways; E is removed.
The original figure and all previously frozen result tables are left intact.
"""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = RA / '09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
TAB = OUT / 'tables'
SD = OUT / 'source_data'
FIGD = OUT / 'figures'
SD.mkdir(parents=True, exist_ok=True)
FIGD.mkdir(parents=True, exist_ok=True)

CHIP = '#6F4E9C'
KOHYP = '#C44E89'
UP = '#D6604D'
DOWN = '#4393C3'
INK = '#263238'
MID = '#6F787D'
LIGHT = '#E8ECEE'
MM = 1 / 25.4
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.labelsize': 7.0, 'axes.titlesize': 8.0,
    'axes.titleweight': 'bold', 'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3, 'legend.fontsize': 6.2,
    'axes.linewidth': 0.65, 'xtick.major.width': 0.55,
    'ytick.major.width': 0.55, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
})


def clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def lab(ax, letter, x=-0.15, y=1.13):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left', color=INK)


def pfmt(x):
    return f'{x:.3f}' if x >= 0.001 else f'{x:.1e}'


sets = pd.read_csv(TAB / 'multiomics_gene_set_summary.csv')
genes = pd.read_csv(RA / '04_rna_integration/gene_level_rna_v2.csv')
bins = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv').set_index('bin')
old = pd.concat([
    pd.read_csv(TAB / 'GO_BP_conditional_spatial_enrichment.csv'),
    pd.read_csv(TAB / 'KEGG_conditional_spatial_enrichment.csv'),
], ignore_index=True)
new = pd.read_csv(TAB / 'Interaction_direct_conditional_spatial_enrichment_audit.csv')
enrich = pd.concat([old, new], ignore_index=True, sort=False)

# A: exactly three parent sets; purple fractions define the direct-bound queries.
set_order = [
    'Baseline_structure_RNA',
    'Hypoxia_structure_RNA',
    'Interaction_structure_hypoxia_RNA',
]
short = {
    'Baseline_structure_RNA': 'Baseline structure + RNA',
    'Hypoxia_structure_RNA': 'Hypoxia structure + RNA',
    'Interaction_structure_hypoxia_RNA': 'Structural interaction +\nWT-hypoxia RNA',
}
sp = sets.set_index('gene_set').loc[set_order].reset_index()
sp['n_unbound'] = sp.n_genes - sp.n_direct
expected = {
    'Baseline_structure_RNA': (269, 45),
    'Hypoxia_structure_RNA': (1143, 241),
    'Interaction_structure_hypoxia_RNA': (550, 111),
}
assert {(r.gene_set, r.n_genes, r.n_direct) for r in sp.itertuples()} == {
    (k, *v) for k, v in expected.items()
}
sp.to_csv(SD / 'Fig3_redrawn_A_gene_sets.csv', index=False)

# B/C: predeclared, non-redundant candidates for the two direct-bound queries.
pathway_specs = [
    ('B', 'Baseline_structure_RNA_direct', 'Two-component system'),
    ('B', 'Baseline_structure_RNA_direct', 'proteolysis'),
    ('B', 'Baseline_structure_RNA_direct', 'fatty acid biosynthetic process'),
    ('C', 'Interaction_structure_hypoxia_RNA_direct', 'response to hypoxia'),
    ('C', 'Interaction_structure_hypoxia_RNA_direct', 'DNA repair'),
    ('C', 'Interaction_structure_hypoxia_RNA_direct', 'proton transmembrane transport'),
]
selected_pathways = []
for panel, gene_set, name in pathway_specs:
    hit = enrich[(enrich.gene_set == gene_set) & (enrich.name == name)]
    assert len(hit) == 1, (gene_set, name, len(hit))
    row = hit.iloc[0].copy()
    row['panel'] = panel
    selected_pathways.append(row)
pathways = pd.DataFrame(selected_pathways)
assert not (pathways.q_conditional_within_gene_set < 0.10).any()
assert not (pathways.q_spatial_within_gene_set < 0.10).any()
pathways.to_csv(SD / 'Fig3_redrawn_BC_pathway_results.csv', index=False)

# D: all genes from two representative pathways in B and two in C. Every gene is
# direct-bound by construction. Structural interaction is mapped from the 5-kb
# bin table; the gene-level `interaction` field is RNA interaction and is not
# reused as a structural measurement.
d_groups = [
    ('B', 'Two-component system', ['Rv0001', 'Rv0902c', 'Rv2332']),
    ('C', 'Response to hypoxia', ['Rv0569', 'Rv0573c', 'Rv0685', 'Rv2629', 'Rv3233c', 'Rv3234c']),
    ('C', 'DNA repair', ['Rv0003', 'Rv2973c', 'Rv3715c']),
]
drows = []
for panel, group, ids in d_groups:
    hit = pathways[(pathways.panel == panel) & (pathways.name.str.lower() == group.lower())]
    assert len(hit) == 1
    assert set(hit.iloc[0].genes.split(',')) == set(ids)
    for order, gene_id in enumerate(ids):
        row = genes.loc[genes.gene_id == gene_id].iloc[0].copy()
        row['panel'] = panel
        row['pathway_group'] = group
        row['within_group_order'] = order
        drows.append(row)
dgenes = pd.DataFrame(drows)
dgenes['structural_interaction'] = dgenes.bin_tss.map(bins.d_ins_interaction)
dgenes['structural_interaction_class'] = dgenes.bin_tss.map(bins.interaction_class)
dgenes['display'] = dgenes.gene_name.fillna(dgenes.gene_id) + ' (' + dgenes.gene_id + ')'
dgenes['occupancy'] = dgenes.binding_class.map({
    'promoter_bound': 'promoter', 'gene_body_bound': 'body'
})
assert dgenes.occupancy.notna().all()
assert len(dgenes) == 12 and dgenes.gene_id.nunique() == 12
assert np.isfinite(dgenes[['bin_d_ins_KO_WT', 'structural_interaction',
                           'log2FC_KOvsWT', 'log2FC_hypoxiavsWT']].to_numpy(float)).all()
# Genes in C meet the structural-interaction criterion, not an RNA-interaction criterion.
cgenes = dgenes[dgenes.panel == 'C']
assert cgenes.structural_interaction_class.isin(['detected_increase', 'detected_decrease']).all()
assert cgenes.class_hypoxiavsWT.isin(['detected_increase', 'detected_decrease']).all()
strict_rna_interaction = genes.padj_int.lt(0.05) & genes.interaction.abs().ge(1)
assert int(strict_rna_interaction.sum()) == 0

dcols = [
    'panel', 'pathway_group', 'within_group_order', 'gene_id', 'gene_name', 'display',
    'start', 'end', 'strand', 'tss', 'bin_tss', 'binding_class', 'occupancy',
    'n_peaks_promoter_up300', 'n_peaks_gene_body', 'bin_d_ins_KO_WT',
    'structural_interaction', 'structural_interaction_class', 'log2FC_KOvsWT',
    'padj_KOvsWT', 'log2FC_hypoxiavsWT', 'padj_hypoxiavsWT',
    'interaction', 'padj_int',
]
dgenes[dcols].to_csv(SD / 'Fig3_redrawn_D_direct_gene_evidence.csv', index=False)
pd.DataFrame([{
    'criterion': 'RNA padj_int < 0.05 and |RNA interaction| >= 1',
    'n_genes': int(strict_rna_interaction.sum()),
    'consequence': 'No true RNA-interaction gene set is used for enrichment; C is structural interaction + WT-hypoxia RNA.',
}]).to_csv(SD / 'Fig3_redrawn_RNA_interaction_audit.csv', index=False)

# ================================ Figure ====================================
fig = plt.figure(figsize=(210 * MM, 207 * MM))
fig.text(0.055, 0.982, 'Response Figure 3 | NapM-bound multi-omics candidates',
         ha='left', va='top', fontsize=11.5, fontweight='bold', color=INK)
fig.text(0.055, 0.953,
         'Direct-bound gene sets prioritize mechanistic follow-up; conditional and spatial tests delimit the strength of pathway evidence.',
         ha='left', va='top', fontsize=7.2, color=MID)
gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 0.83, 1.30],
                      left=0.19, right=0.97, top=0.885, bottom=0.105,
                      hspace=0.67, wspace=0.68)

# A
ax = clean(fig.add_subplot(gs[0, :]))
y = np.arange(len(sp))[::-1]
ax.barh(y, sp.n_direct, color=CHIP, edgecolor='none', height=0.58,
        label='NapM promoter/body occupied')
ax.barh(y, sp.n_unbound, left=sp.n_direct, color=LIGHT, edgecolor='none',
        height=0.58, label='Locally unbound')
for yi, r in zip(y, sp.itertuples()):
    ax.text(r.n_genes + 24, yi, f'{r.n_genes} genes  |  {r.n_direct} bound ({100*r.n_direct/r.n_genes:.1f}%)',
            va='center', fontsize=6.2, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([short[x] for x in sp.gene_set], fontsize=6.4)
ax.set_xlim(0, 1430)
ax.set_xticks([0, 300, 600, 900, 1200])
ax.set_xlabel('Genes in parent set')
ax.set_title('Three parent gene sets and their direct-bound composition', loc='left', pad=8)
ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=5.8,
          bbox_to_anchor=(1.0, 1.02))
ax.text(0.99, 1.10, 'B–D use only the NapM-bound subsets',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=5.9, color=CHIP)
lab(ax, 'A', x=-0.205, y=1.16)

# B/C helper: one final RNA+-conditional result per pathway, with spatial P.
def enrichment_panel(ax, sub, color, letter, panel_title):
    sub = sub.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(sub)) * 1.45
    ax.barh(yy, sub.fold_enrichment, height=0.56, color=color,
            alpha=0.88, edgecolor='none')
    for yi, r in zip(yy, sub.itertuples()):
        ax.text(r.fold_enrichment + 0.055, yi, f'{r.fold_enrichment:.2f}×',
                va='center', ha='left', fontsize=5.8, color=color, fontweight='bold')
        ax.text(0.02, yi - 0.43,
                f'k = {int(r.overlap)}  |  P = {pfmt(r.p_conditional_RNAplus)}  |  '
                f'q = {pfmt(r.q_conditional_within_gene_set)}  |  '
                f'gene-order spatial P = {pfmt(r.p_gene_order_circular_shift)}',
                va='center', ha='left', fontsize=5.1, color=MID)
    ax.set_yticks(yy)
    labels = [textwrap.fill(x, 27) for x in sub.name]
    ax.set_yticklabels(labels, fontsize=5.9)
    ax.set_xlim(0, 3.75)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(-0.85, max(yy) + 0.72)
    ax.set_xlabel('RNA$^+$-conditional fold enrichment')
    ax.axvline(1, color=MID, lw=0.6, ls='--', zorder=0)
    ax.set_title(panel_title, loc='left', pad=8)
    ax.text(0.0, -0.25, 'No pathway passes FDR < 0.10; candidates only.',
            transform=ax.transAxes, fontsize=5.5, color=MID, va='top')
    lab(ax, letter, x=-0.49, y=1.15)

bset = pathways[pathways.panel == 'B'].copy()
cset = pathways[pathways.panel == 'C'].copy()
ax = clean(fig.add_subplot(gs[1, 0]))
enrichment_panel(ax, bset, CHIP, 'B', 'Baseline direct-bound set (n = 45)')
ax = clean(fig.add_subplot(gs[1, 1]))
enrichment_panel(ax, cset, KOHYP, 'C', 'Structural-interaction direct set (n = 111)')

# D
ax = fig.add_subplot(gs[2, :])
vals = dgenes[['bin_d_ins_KO_WT', 'structural_interaction',
               'log2FC_KOvsWT', 'log2FC_hypoxiavsWT']].to_numpy(float)
norm = mcolors.TwoSlopeNorm(vmin=-3.0, vcenter=0, vmax=3.0)
cmap = mcolors.LinearSegmentedColormap.from_list('rna_structure', [DOWN, '#F7F7F7', UP])
im = ax.imshow(vals, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
for i in range(len(dgenes)):
    for j in range(4):
        v = vals[i, j]
        ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=5.55,
                color='white' if abs(v) > 1.75 else INK)
    ax.add_patch(Rectangle((3.55, i - 0.43), 0.90, 0.86,
                           facecolor=CHIP, edgecolor='white', lw=0.5))
    ax.text(4, i, dgenes.occupancy.iloc[i], ha='center', va='center',
            fontsize=5.25, color='white')
for boundary in [2.5, 8.5]:
    ax.axhline(boundary, color=INK, lw=0.75)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels([
    'Baseline Δinsulation\nKO − WT',
    'Structural interaction\nΔΔinsulation',
    'RNA log$_2$FC\nKO / WT',
    'RNA log$_2$FC\nWT hypoxia / WT',
    'NapM\noccupancy',
], fontsize=5.9)
ax.set_yticks(np.arange(len(dgenes)))
ax.set_yticklabels(dgenes.display, fontsize=5.9, fontstyle='italic')
for tick, panel in zip(ax.get_yticklabels(), dgenes.panel):
    tick.set_color(CHIP if panel == 'B' else KOHYP)
ax.set_xlim(-0.5, 4.5)
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(dgenes), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=0.7)
ax.tick_params(which='minor', bottom=False, left=False)
ax.set_title('Direct-bound genes underlying the displayed pathway candidates', loc='left', pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.017, pad=0.025)
cb.set_label('Δinsulation / ΔΔinsulation / RNA log$_2$FC', fontsize=5.7)
cb.ax.tick_params(labelsize=5.3)
cb.outline.set_visible(False)
lab(ax, 'D', x=-0.17, y=1.08)

fig.text(0.055, 0.061,
         'B,C: RNA$^+$-conditional tests with NapM-bound backgrounds; q = BH within gene set/database. '
         'Spatial P: all non-zero gene-order shifts.',
         fontsize=5.8, color=MID, ha='left')
fig.text(0.055, 0.038,
         'C = structural interaction + WT-hypoxia RNA, not dual-omics interaction. '
         'No gene has RNA-interaction FDR < 0.05 and |interaction| ≥ 1.',
         fontsize=5.8, color=MID, ha='left')
fig.text(0.055, 0.015,
         'D: all genes are NapM-bound. Examples illustrate candidates, not independent tests or proof of direct transcriptional regulation.',
         fontsize=5.8, color=MID, ha='left')

pdf = FIGD / 'R1_Response_Figure3_redrawn.pdf'
png = FIGD / 'R1_Response_Figure3_redrawn.png'
fig.savefig(pdf, metadata={
    'Title': 'Reviewer 1 Response Figure 3 redrawn: NapM-bound multi-omics candidates',
    'Subject': 'Direct-bound exploratory enrichment with conditional and spatial sensitivity tests',
})
fig.savefig(png, dpi=350)
plt.close(fig)
print('Wrote', pdf)
print('Wrote', png)
print('Strict RNA-interaction genes:', int(strict_rna_interaction.sum()))
print('D direct-bound genes:', len(dgenes), '; occupancy:', dgenes.occupancy.value_counts().to_dict())
print(pathways[['panel', 'gene_set', 'name', 'fold_enrichment',
                'p_conditional_RNAplus', 'q_conditional_within_gene_set',
                'p_gene_order_circular_shift']].to_string(index=False))
