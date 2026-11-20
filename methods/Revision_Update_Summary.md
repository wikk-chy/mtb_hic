# Final-Draft Package Addendum (Response Fig. 1–15 & Revised Manuscript)

This addendum documents the final submission numbering (**Response Fig. 1–15**,
November 20, 2026) and summarizes all newly added or updated analyses relative
to the earlier 11-figure `Response_Figures_Methods.md` baseline.

## Final Figure Numbering Map

| Final Figure | Reviewer | Legacy Number | Module Directory | Key Final-Draft Updates |
|---|---|---|---|---|
| **Response Fig. 1** | Reviewer #1 | Fig. 1 | `figure_01_napm_anchor_integration/` | Updated 41 Chromosight loops, 17 double-bound anchors, descriptive KO non-recovery (`11/17` vs `10/24`), updated editable `.ai` layout. |
| **Response Fig. 2** | Reviewer #1 | Fig. 2 | `figure_02_hypoxia_factorial_integration/` | Final 7-panel layout (`A–G`): `|Δinsulation interaction| >= 0.20` response-mode classification (`81` RNA-only, `329` structure-only, `108` joint, `365` other), explicit separation of merged-map bin means (`+0.034` / `−0.073`) vs independent-sample replicate genome-means (`+0.041` / `−0.067`, contrast `−0.108`, exact permutation `P = 4.08e-4`), and two-sided (`P = 0.0193`) + one-sided higher-than-null (`P = 0.0098`) circular-shift tests. |
| **Response Fig. 3** | Reviewer #2 | New | `figure_03_network_centrality_hubs/` | Integrated DNA–protein–PPI network centrality (`n = 3,870` nodes) comparing NapM (degree rank #6, 441; betweenness rank #5; 17/41 double-anchor loops) against the Top-10 hubs (`02_build_R2_fig3_network_hubs.py`). |
| **Response Fig. 4** | Reviewer #2 | Fig. 3 | `figure_04_napm_ko_go_kegg_enrichment/` | Separate one-sided hypergeometric GO/KEGG enrichment for `460` upregulated and `362` downregulated `napM`-KO DEGs with BH FDR `q < 0.05` (`03_build_R2_fig4_go_kegg.py`). |
| **Response Fig. 5** | Reviewer #2 | New | `figure_05_stress_spatial_polymer_evaluation/` | Standalone R2 composite integrating spatially non-uniform stress Hi-C metrics (`figure_14`) and OpenMM ring-polymer evaluation (`figure_15`) with continuous panels `A–G` (`04_build_R2_fig5_spatial_polymer.py`). |
| **Response Fig. 6** | Reviewer #2 | Fig. 4 | `figure_06_napm_expression_hypoxia_dependence/` | Adds 12 core mycobacterial NAP/SMC/ParAB/topoisomerase expression contrasts (`architectural_proteins_expression.csv`), nested 315/437 genotype-by-hypoxia insulation tiers, and recomputed 9-metric hypoxia response fingerprint (`01_recompute_fingerprint_fixed.py`, `05_build_R2_fig6_napm_dependence.py`). |
| **Response Fig. 7** | Reviewer #2 | Fig. 5 | `figure_07_napm_binding_mechanism_ppi/` | Self-contained 9-panel (`A–I`) NapM mechanism synthesis combining ChIP-seq occupancy, AT-rich 8-mers, DEG decoupling, global KO contact shift, local metaprofile, loop non-recovery, and exact 14-partner Wang et al. (2010) B2H ego network (`06_build_R2_fig7_napm_mechanism.py`, verified against `pr100808n_si_009.xls` / `si_011.xls`). |
| **Response Fig. 8** | Reviewer #2 | New | `figure_08_pilot_ter_fish_clumping/` | Standalone pilot *ter*-probe FISH overview across 4 biological replicates documenting cell clumping (`07_build_R2_fig8_pilot_fish.py`). |
| **Response Fig. 9** | Reviewer #2 | New | `figure_09_latency_like_stress_crossvalidation/` | Four-module pathway profiling (RecA/LexA SOS & mutasome, PafBC/DSB repair, ROS/redox stress, DosR dormancy) and genome-wide RNA/Hi-C cross-validation between acid/antibiotic-induced latency-like stress and antibiotic-free hypoxia (`08_build_R2_fig9_antibiotic_nrp.py`). |
| **Response Fig. 10** | Reviewer #4 | Fig. 6 | `figure_10_circular_eigenvector_saddle/` | Circular Fourier eigenvector & mean O/E saddle controls (`A–E`). |
| **Response Fig. 11** | Reviewer #4 | Fig. 7 | `figure_11_cid_directionality_validation/` | Updated to all **18 DI-defined CID boundaries** (`plot_response_fig7_all_boundaries.py`; within-CID/cross-boundary median ratio = `1.102`, `P = 0.001`; merged pileup O/E = `0.974`, replicates `0.967–0.975`); legacy title banner removed; retired whole-CID H–J evidence excluded. |
| **Response Fig. 12** | Reviewer #4 | Fig. 8 | `figure_12_chromosight_loop_validation/` | Chromosight loop validation (`41` loops, `35` APA-eligible, `n = 34` finite WT anchor-pixel O/E vs `n = 35` shifted null, one-sided Mann–Whitney `P = 2.3e-5`; `n = 33` WT–KO paired loops, two-sided paired Wilcoxon signed-rank `P = 0.020`). |
| **Response Fig. 13** | Reviewer #4 | Fig. 9 | `figure_13_wt_ko_compaction_metrics/` | Quantitative WT vs `napM`-KO compaction metrics (`A–H`) plus replicate QC tables (`compute_reproducibility_qc.py`). |
| **Response Fig. 14** | Reviewer #4 | Fig. 10 | `figure_14_stress_oriter_experiments/` | Experimental evaluation of alternative ori–ter mechanisms & pilot *ter*-probe FISH (`A–E`). |
| **Response Fig. 15** | Reviewer #4 | Fig. 11 | `figure_15_openmm_polymer_models/` | OpenMM ring-polymer scenario comparison with continuous panel numbering `A–H` (convergence diagnostics relabeled from `I` to `H`). |

## Final Revised Manuscript Materials (`manuscript/`)

- `manuscript/revised_manuscript/` contains the final LaTeX manuscript, bibliography, and compiled 39-page `Revised_Manuscript.pdf`, matching all 10 figures in `figures/manuscript/`.
- `manuscript/response_letter/` contains the final Overleaf LaTeX source and compiled 48-page `Response_to_Reviewers_EN.pdf`, matching all 15 figures in `figures/final/`.
- `manuscript/code/` and `manuscript/results/` preserve the 18-boundary Fig. 1f CID co-expression recalculation (`mean r = 0.302 vs 0.289`, two-sided empirical `P = 0.045`, 1,000 randomizations) and Fig. 2a–b network redraw scripts/outputs.
