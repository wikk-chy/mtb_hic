#!/usr/bin/env python3
"""Plot all-79 anchor-DEG GO/KEGG enrichment with explicit FDR and background sensitivity."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis/09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028/anchor_centered_integration_20261029'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
res = pd.read_csv(TAB / 'anchor79_GO_KEGG_enrichment.csv')
primary = 'anchor79_vs_RNA_tested'
contexts = [primary, 'anchor79_vs_eligible_anchor_RNA_tested', 'anchor79_vs_eligible_anchor_DEGs']
colors = {'GO_BP': '#287D8E', 'GO_CC': '#B47846', 'GO_MF': '#6B4C9A', 'KEGG': '#C4514B'}
GREY = '#66717A'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white'})

fig = plt.figure(figsize=(16.0, 10.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], left=.23, right=.975,
                      top=.80, bottom=.20, wspace=.50)

# A: top two terms per database under the requested standard RNA-tested background.
ax = fig.add_subplot(gs[0, 0])
selected = []
for db in ['GO_BP', 'GO_CC', 'GO_MF', 'KEGG']:
    z = res[(res.gene_set.eq(primary)) & res.database.eq(db)].sort_values(['p_value', 'term']).head(2)
    selected.append(z)
z = pd.concat(selected, ignore_index=True)
y = np.arange(len(z))
vals = -np.log10(z.p_value)
ax.barh(y, vals, color=[colors[d] for d in z.database], height=.64, alpha=.90)
ax.axvline(-np.log10(.05), color='#626D74', lw=.9, ls=(0, (3, 3)), label='Nominal P=0.05')
labels = [f'{textwrap.fill(r.name, 36)}\n({r.overlap}/{r.query_size} annotated query genes)' for r in z.itertuples()]
ax.set_yticks(y, labels, fontsize=8.2)
ax.invert_yaxis()
for yi, row in zip(y, z.itertuples()):
    ax.text(4.32, yi, f'P={row.p_value:.2g}; q={row.q_BH_all_tested_terms:.3g}',
            ha='right', va='center', fontsize=8.2, color=GREY)
ax.set_xlim(0, 4.4)
ax.set_xlabel('−log₁₀(hypergeometric P)')
ax.grid(axis='x', color='#E8ECEF', lw=.7, zorder=0)
ax.legend(frameon=False, loc='lower right', fontsize=8.2)
ax.set_title('A  Standard RNA-tested background: no FDR-significant terms', loc='left', pad=10)
ax.text(0, -.10, 'Top two terms per database by nominal P; all q ≥ 0.05.',
        transform=ax.transAxes, fontsize=8.3, color=GREY)

# B: q-value sensitivity for representative terms across all three backgrounds.
ax = fig.add_subplot(gs[0, 1])
terms = ['GO:0015562', 'GO:0009039', 'GO:0006544', 'mtu01200', 'mtu02010']
row_labels = ['Efflux transmembrane\ntransporter activity', 'Urease activity',
              'Glycine metabolic process', 'Carbon metabolism', 'ABC transporters']
qmat = np.full((len(terms), len(contexts)), np.nan)
pmat = np.full_like(qmat, np.nan)
for i, term in enumerate(terms):
    for j, context in enumerate(contexts):
        hit = res[(res.term.eq(term)) & (res.gene_set.eq(context))]
        if len(hit):
            qmat[i, j] = hit.iloc[0].q_BH_all_tested_terms
            pmat[i, j] = hit.iloc[0].p_value
with np.errstate(divide='ignore'):
    score = -np.log10(qmat)
masked = np.ma.masked_invalid(score)
cmap = LinearSegmentedColormap.from_list('q', ['#F4F5F6', '#D9D0E5', '#8C6BB1', '#4B2E83'])
cmap.set_bad('#EEEEEE')
im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=1.65)
ax.set_xticks(range(3), ['RNA-tested\n(primary)', 'Anchor-matched\nRNA', 'Anchor-matched\nDEGs'], fontsize=8.5)
ax.set_yticks(range(len(terms)), row_labels, fontsize=8.5)
for i in range(len(terms)):
    for j in range(len(contexts)):
        if np.isnan(qmat[i, j]):
            text = 'not tested\n(term <3)'
            color = GREY
        else:
            text = f'q={qmat[i,j]:.3g}\nP={pmat[i,j]:.2g}'
            color = 'white' if score[i, j] > 1.0 else '#303A41'
        ax.text(j, i, text, ha='center', va='center', fontsize=8.0, color=color)
        if not np.isnan(qmat[i, j]) and qmat[i, j] < .05:
            ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False,
                                   edgecolor='#D69E2E', lw=2.2))
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(terms), 1), minor=True)
ax.grid(which='minor', color='white', lw=1.5)
ax.tick_params(which='both', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
cb.set_label('−log₁₀(BH q)')
cb.ax.axhline(1.301, color='#D69E2E', lw=1.5)
ax.set_title('B  Statistical significance is background-sensitive', loc='left', pad=10)

fig.suptitle('GO and KEGG enrichment of all 79 changed-anchor DEGs',
             x=.055, y=.960, ha='left', fontsize=16, fontweight='bold')
fig.text(.055, .913,
         'Primary query: 79 DEGs near changed-loop anchors (±10 kb); background: all 4,060 RNA-tested genes. '
         'GO-BP/CC/MF and KEGG were tested separately.',
         ha='left', fontsize=10.0, color=GREY)
fig.text(.055, .104,
         'Only efflux transmembrane transporter activity passes FDR in the anchor-matched RNA background (q=0.0294; gold outline). '
         'It is driven entirely by Rv2686c/Rv2687c/Rv2688c at L29_left and is not significant with the standard or DEG-conditioned background.',
         ha='left', fontsize=8.7, color=GREY)
fig.text(.055, .070,
         'The urease-related terms are driven by Rv1848/Rv1850; nominal pathways are hypotheses, not validated mechanisms. '
         'Direct mappings; terms with ≥3 background genes; “not tested” means term <3; BH includes zero-overlap terms.',
         ha='left', fontsize=8.7, color=GREY)
fig.text(.055, .036,
         'No enrichment result establishes direct NapM binding, contact-to-RNA causality, chromosomal plasticity, or dormancy establishment.',
         ha='left', fontsize=8.7, color='#5A3540', fontweight='bold')
fig.savefig(FIG / 'R1_anchor79_GO_KEGG_enrichment.pdf', bbox_inches='tight')
fig.savefig(FIG / 'R1_anchor79_GO_KEGG_enrichment.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Wrote', FIG / 'R1_anchor79_GO_KEGG_enrichment.pdf')
print(z[['database', 'term', 'name', 'overlap', 'p_value', 'q_BH_all_tested_terms']].to_string(index=False))
