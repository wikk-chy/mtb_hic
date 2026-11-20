#!/usr/bin/env python3
"""Build a one-page, data-constrained NapM hypoxia working-model figure."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
SRC = OUT / 'source_data'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

WT, KO = '#1F4E79', '#B35845'
PURPLE, GREEN, GREY, LIGHT = '#6B5B7A', '#3F7B70', '#646B70', '#D9DEE2'
mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 10,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

hyp_g = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
hyp_b = pd.read_csv(SRC / 'Hypoxia_bin_level.csv')
cors = pd.read_csv(TAB / 'Structure_transcription_correlations.csv')
regs = pd.read_csv(TAB / 'Structure_transcription_regressions.csv')
matched = pd.read_csv(TAB / 'Hypoxia_NapM_matched_tests.csv')

# Extract exact source rows rather than transcribing values into the figure.
gene_specs = [('devR', 'DosR/DevR'), ('devS', 'DosS/DevS'), ('sigE', 'SigE')]
gene_rows = []
for gene, label in gene_specs:
    row = hyp_g.loc[hyp_g.gene_name.eq(gene)].iloc[0]
    gene_rows.append({
        'panel': 'A', 'feature': label, 'gene_id': row.gene_id,
        'bin': int(row.bin_tss), 'WT_response': row.hyp_response_wt,
        'KO_response': row.hyp_response_ko, 'interaction': row.interaction,
        'interaction_FDR': row.padj_int,
        'local_RNA_evidence': 'gene-level response',
    })

bin_specs = [(272, 'sigE locus'), (0, 'oriC/dnaA'), (874, 'ESX-2')]
bin_rows = []
for bin_id, label in bin_specs:
    row = hyp_b.loc[hyp_b['bin'].eq(bin_id)].iloc[0]
    local_note = ('strong local RNA interaction' if bin_id == 272 else
                  'no local DEG detected')
    bin_rows.append({
        'panel': 'B', 'feature': label, 'gene_id': '',
        'bin': bin_id, 'WT_response': row.d_ins_hyp_WT,
        'KO_response': row.d_ins_hyp_KO, 'interaction': row.d_ins_interaction,
        'interaction_FDR': row.interaction_padj,
        'local_RNA_evidence': local_note,
    })
locus_source = pd.DataFrame(gene_rows + bin_rows)
locus_source.to_csv(SRC / 'Mechanism_model_locus_source.csv', index=False)

n_int = int(hyp_g.hypoxia_interaction_RNA_class.isin(
    ['detected_increase', 'detected_decrease']).sum())
rho = float(cors.loc[cors.contrast.eq('Hypoxia interaction'),
                     'spearman_rho'].iloc[0])
rho_p = float(cors.loc[cors.contrast.eq('Hypoxia interaction'),
                       'circular_shift_p'].iloc[0])
delta_r2 = float(regs.loc[regs.model.eq('Hypoxia interaction'),
                          'incremental_R2_RNA'].iloc[0])
bound_p = float(matched.loc[matched.comparison.eq(
    'Hypoxia genotype interaction (signed)'), 'circular_shift_p'].iloc[0])
global_source = pd.DataFrame([
    {'metric': 'RNA interaction DEGs', 'value': n_int,
     'definition': 'FDR < 0.05 and |log2FC| >= 1'},
    {'metric': 'RNA-insulation interaction Spearman rho', 'value': rho,
     'definition': f'circular-shift P = {rho_p:.6f}'},
    {'metric': 'RNA incremental R2 for insulation interaction', 'value': delta_r2,
     'definition': 'exploratory regression; no mediation model'},
    {'metric': 'NapM-bound structural interaction shift P', 'value': bound_p,
     'definition': '422 baseline-matched pairs; signed effects'},
])
global_source.to_csv(SRC / 'Mechanism_model_global_source.csv', index=False)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#E8E8E8', lw=.45, zorder=0)
    ax.set_axisbelow(True)


def panel_heading(ax, label, title):
    ax.text(-.16, 1.10, label, transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    ax.set_title(title, loc='left', fontweight='bold', pad=10)


def dumbbell(ax, rows, xlabel, interaction_x):
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color='#9B9B9B', lw=.6, zorder=1)
    for y, row in zip(ys, rows):
        ax.plot([row['WT_response'], row['KO_response']], [y, y],
                color='#B9C0C5', lw=1.5, zorder=2)
        ax.scatter(row['WT_response'], y, s=40, color=WT, zorder=3)
        ax.scatter(row['KO_response'], y, s=40, color=KO, zorder=3)
        ax.text(interaction_x, y,
                f"interaction {row['interaction']:+.3f}\nFDR {row['interaction_FDR']:.1e}",
                fontsize=6.2, va='center', color='#40464A')
    ax.set_yticks(ys)
    ax.set_yticklabels([r['feature'] for r in rows])
    ax.set_xlabel(xlabel)
    clean(ax)


def box(ax, xy, width, height, title, body, edge, fill):
    x, y = xy
    ax.add_patch(Rectangle((x, y), width, height, transform=ax.transAxes,
                           facecolor=fill, edgecolor=edge, lw=1.0))
    ax.text(x + width / 2, y + height * .70, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color=edge)
    ax.text(x + width / 2, y + height * .31, body, transform=ax.transAxes,
            ha='center', va='center', fontsize=7.2, color='#343A3E',
            linespacing=1.35)


def dashed_arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                                 arrowstyle='-|>', mutation_scale=11,
                                 lw=1.0, ls='--', color=color,
                                 shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(11.69, 8.27))
gs = fig.add_gridspec(2, 3, height_ratios=[1.04, .96], left=.065, right=.975,
                      top=.82, bottom=.105, hspace=.42, wspace=.52)
fig.suptitle('Hypoxia-associated transcriptional and structural responses to napM deletion',
             x=.055, y=.965, ha='left', fontsize=14.5, fontweight='bold')
fig.text(.055, .922,
         'A data-constrained working model: observed genotype × hypoxia effects are separated from unresolved causal paths.',
         ha='left', fontsize=9.2, color='#555C61')
fig.text(.055, .875,
         'WT', color=WT, fontsize=8, fontweight='bold')
fig.text(.083, .875,
         'hypoxia response', color='#444444', fontsize=8)
fig.text(.175, .875,
         'napM-KO', color=KO, fontsize=8, fontweight='bold')
fig.text(.238, .875,
         'hypoxia response', color='#444444', fontsize=8)

# A: regulator responses. The three genes are observations, not a pathway.
ax_a = fig.add_subplot(gs[0, 0])
dumbbell(ax_a, gene_rows, 'RNA hypoxia response (log2FC)', 3.45)
ax_a.set_xlim(-1.0, 5.25)
ax_a.set_xticks([-1, 0, 1, 2, 3])
panel_heading(ax_a, 'A', 'Stress-regulator responses are remodeled')
ax_a.text(0, -.27, 'No DosR/DosS → SigE regulatory relationship is inferred.',
          transform=ax_a.transAxes, fontsize=6.6, color=GREY)

# B: locus-level structural effects on a separate scale from RNA.
ax_b = fig.add_subplot(gs[0, 1])
dumbbell(ax_b, bin_rows, 'Insulation hypoxia response (Δ log2)', .83)
ax_b.set_xlim(-.80, 1.40)
ax_b.set_xticks([-.5, 0, .5])
panel_heading(ax_b, 'B', 'Transcription–structure dissociation')
ax_b.text(0, -.27,
          'oriC and ESX-2: no local DEG detected; this is not evidence of no transcription.',
          transform=ax_b.transAxes, fontsize=6.3, color=GREY)

# C: quantitative constraints on interpretation.
ax_c = fig.add_subplot(gs[0, 2])
ax_c.axis('off')
panel_heading(ax_c, 'C', 'Genome-wide constraints')
entries = [
    (f'{n_int:,}', 'RNA interaction DEGs',
     'genotype × hypoxia; FDR < 0.05, |log2FC| ≥ 1'),
    (f'ρ = {rho:.3f}', 'RNA–insulation interaction correlation',
     f'circular-shift P = {rho_p:.3f}; RNA incremental R² = {delta_r2:.6f}'),
    (f'P = {bound_p:.3f}', 'No detectable NapM-bound enrichment',
     'structural interaction; 422 baseline-matched pairs'),
]
for i, (value, label, detail) in enumerate(entries):
    y = .88 - i * .31
    ax_c.text(0, y, value, transform=ax_c.transAxes, fontsize=14,
              fontweight='bold', color=PURPLE, va='top')
    ax_c.text(0, y - .115, label, transform=ax_c.transAxes,
              fontsize=8, fontweight='bold', va='top')
    ax_c.text(0, y - .19, detail, transform=ax_c.transAxes,
              fontsize=6.6, color=GREY, va='top')
    if i < 2:
        ax_c.plot([0, 1], [y - .255, y - .255], transform=ax_c.transAxes,
                  color=LIGHT, lw=.7)

# D: hypotheses are dashed, and the two branches are not claimed independent.
ax_d = fig.add_subplot(gs[1, :])
ax_d.axis('off')
ax_d.text(-.035, 1.06, 'D', transform=ax_d.transAxes, fontsize=14,
          fontweight='bold', va='top')
ax_d.set_title('Working model: two evidence patterns, not two proven independent pathways',
               loc='left', fontweight='bold', pad=12)
box(ax_d, (.385, .72), .23, .22, 'napM deletion',
    'genotype × hypoxia effects', KO, '#FBF4F1')
box(ax_d, (.035, .40), .29, .24, 'Stress-regulator response pattern',
    'devR/devS induction attenuated\nsigE response reversed', WT, '#F1F5F8')
box(ax_d, (.675, .40), .29, .24, 'Locus-specific structural pattern',
    'oriC and ESX-2 insulation interactions\nwithout detected local DEGs', PURPLE, '#F5F2F6')
box(ax_d, (.035, .055), .29, .22, 'Transcriptional interpretation',
    'network-mediated or indirect effects\nare plausible candidates', GREEN, '#F1F7F5')
box(ax_d, (.675, .055), .29, .22, 'Structural interpretation',
    'direct, distal or global architectural\neffects remain possible', PURPLE, '#F5F2F6')
dashed_arrow(ax_d, (.43, .72), (.24, .645), WT)
dashed_arrow(ax_d, (.57, .72), (.76, .645), PURPLE)
dashed_arrow(ax_d, (.18, .40), (.18, .28), GREEN)
dashed_arrow(ax_d, (.82, .40), (.82, .28), PURPLE)
ax_d.text(.5, .405, 'Relationship between transcriptional and structural\nresponses is unresolved; independence is not assumed.',
          transform=ax_d.transAxes, ha='center', va='center', fontsize=7.2,
          color=GREY, linespacing=1.4)
ax_d.text(.5, .18,
          'Dashed arrows = hypotheses, not demonstrated causal links.\nBaseline WT ChIP-seq only; condition-specific occupancy\nand temporal order remain unknown.',
          transform=ax_d.transAxes, ha='center', va='center', fontsize=7,
          color=GREY, linespacing=1.45)

fig.text(.055, .035,
         'Observed contrasts are quantitative; the model is deliberately non-committal about direct causation, mediation and pathway independence.',
         fontsize=7.2, color='#555C61', ha='left')

png = FIG / 'NapM_hypoxia_working_model.png'
pdf = FIG / 'NapM_hypoxia_working_model.pdf'
fig.savefig(png, dpi=300)
plt.close(fig)

# Place the exact A4-landscape artwork on one PDF page.
page = landscape(A4)
c = canvas.Canvas(str(pdf), pagesize=page)
c.setTitle('NapM hypoxia: data-constrained working model')
c.setAuthor('Multi-omics integration analysis')
c.drawImage(str(png), 0, 0, width=page[0], height=page[1],
            preserveAspectRatio=True, anchor='c')
c.showPage()
c.save()
print(f'Wrote {png}')
print(f'Wrote {pdf}')
print(locus_source.to_string(index=False))
print(global_source.to_string(index=False))
