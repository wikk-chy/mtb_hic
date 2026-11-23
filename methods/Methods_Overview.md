# Computational Methods Overview

This document summarizes the main computational workflows implemented in this repository.

## 1. Hi-C Contact Matrices & Circular-Genome Metrics

- **Contact maps:** Iterative-correction-balanced (ICE) contact matrices are stored as merged `.cool`/`.mcool` and `.hic` files as well as independent biological replicate matrices in [`data/`](../data/).
- **Circular genomic distance:** Pairwise genomic separations on the circular bacterial chromosome are computed using shorter-arc distance, $d_{ij} = \min(|i - j|, N - |i - j|)$.
- **Replicate reproducibility:** Biological replicate concordance is evaluated by pairwise contact-matrix correlation and contact-probability decay curves $P(s)$ across circular genomic separation (`analyses/03_wt_hic_reproducibility/`).

## 2. Chromosome-Scale Segregation, Domains & Loops

- **Circular eigenvector & saddle analysis:** Observed/expected (O/E) matrices normalized by circular $P(s)$ are eigendecomposed to extract the leading axis (`PC1`), evaluated against circular Fourier modes, synthetic $P(s)$-only negative controls, permutation nulls, and residual eigenvectors (`analyses/10_circular_eigenvector_saddle/`).
- **Chromosomal interaction domains (CIDs):** Directionality index (DI) tracks are computed across balanced matrices, and domain boundaries are validated via aggregate contact pileups, insulation metaplots, and matched random-partition permutation tests (`analyses/11_cid_directionality_validation/`).
- **Chromatin loops:** Focal chromatin loops are called with Chromosight and validated via aggregate peak analysis (APA), distance-matched random and shifted-position controls, and cross-condition comparison (`analyses/12_chromosight_loop_validation/`).

## 3. Multi-Omics Integration & Interaction Networks

- **Multi-layer interaction network:** Chromatin loops, protein–DNA binding associations, operon annotations, and protein–protein interactions (PPI) are integrated to quantify centrality and multi-layer enrichment across DNA-binding proteins (`analyses/04_napm_network_enrichment/`, `analyses/08_napm_binding_mechanism_ppi/`).
- **Differential expression & factorial modeling:** Bulk RNA-seq counts are normalized and modeled with `edgeR`/`limma` for pairwise contrasts, genotype-by-stress factorial interaction tests, and GO/KEGG pathway enrichment (`analyses/02_hypoxia_factorial_integration/`, `analyses/05_napm_ko_go_kegg_enrichment/`, `analyses/09_stress_transcriptome_crossvalidation/`).
- **Structure–function & multi-metric remodeling:** Local binding-to-architecture associations, independent-sample insulation trajectories, and multi-metric structural response fingerprints are evaluated using circular-shift nulls and exact permutation tests (`analyses/01_napm_anchor_integration/`, `analyses/06_stress_spatial_directionality/`, `analyses/07_napm_hypoxia_delta_remodeling/`, `analyses/13_wt_ko_compaction_metrics/`, `analyses/14_stress_oriter_signatures/`).

## 4. Coarse-Grained Polymer Modeling

- **Ring-polymer simulations:** Coarse-grained bead–spring ring-polymer models implemented in OpenMM compare candidate folding mechanisms against experimental contact maps, $P(s)$ curves, and held-out structural observables (`analyses/15_openmm_polymer_models/`).
