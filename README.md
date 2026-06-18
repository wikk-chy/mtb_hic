# Establishment and Remodeling of the Mycobacterium tuberculosis 3D Genome Architecture

## Overview

This repository contains code, analysis pipelines, and processed data for the study of three-dimensional (3D) genome organization in *Mycobacterium tuberculosis* (Mtb) under different physiological conditions. We integrated Hi-C, RNA-seq, and ChIP-seq data to investigate how chromatin architecture is established and remodeled in response to environmental stress, with a focus on the role of the nucleoid-associated protein NapM.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Data Description](#data-description)
- [Analysis Workflows](#analysis-workflows)
  - [1. Raw Data Processing](#1-raw-data-processing)
  - [2. Hi-C Contact Map Construction](#2-hi-c-contact-map-construction)
  - [3. Chromatin Domain Analysis](#3-chromatin-domain-analysis)
  - [4. 3D Genome Modeling](#4-3d-genome-modeling)
  - [5. Operon Interaction Network](#5-operon-interaction-network)
  - [6. Structural Plasticity Quantification](#6-structural-plasticity-quantification)
  - [7. Counterfactual Prediction](#7-counterfactual-prediction)
  - [8. Evolutionary Conservation](#8-evolutionary-conservation)
  - [9. Multi-omics Integration](#9-multi-omics-integration)
- [Statistical Analysis](#statistical-analysis)
- [Figures Reproduction](#figures-reproduction)
- [Citation](#citation)
- [Contact](#contact)

---

## Dependencies

### Python (≥3.7)
- **Hi-C analysis:** `cooler`, `cooltools`, `coolbox`
- **Data processing:** `pandas` (≥1.3.0), `numpy` (≥1.20.0)
- **Visualization:** `matplotlib` (≥3.3.0), `seaborn` (≥0.11.0)
- **Machine learning:** `scikit-learn` (≥0.24.0)
- **Network analysis:** `networkx` (≥2.5)
- **Genomics:** `bioframe`

### R (≥4.0)
- **Statistical analysis:** `ggplot2`, `dplyr`, `tidyr`
- **Differential analysis:** `DESeq2`, `edgeR`
- **Functional enrichment:** `clusterProfiler` (GO/KEGG analysis)
- **Visualization:** `ggrepel`, `pheatmap`, `VennDiagram`

### External Programs
- **Alignment:** [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) (v2.4+)
- **Hi-C pipeline:** [Juicer](https://github.com/aidenlab/juicer) (v1.6+)
- **Homology search:** [BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) (v2.12+)

**Installation example (conda):**
```bash
conda create -n mtb_hic python=3.8
conda activate mtb_hic
conda install -c bioconda cooler cooltools bowtie2 blast
pip install coolbox networkx bioframe
```

---

## Repository Structure

```
.
├── README.md                    # This file
├── script/                      # Analysis scripts
│   ├── operon-select_entropymoran's.ipynb  # Operon hub identification
│   ├── network_formation.ipynb  # Interaction network construction
│   ├── moran's_shannon.ipynb    # Structural order metrics
│   ├── ABcompartment.ipynb      # A/B compartment analysis
│   ├── napmkolatentpre_comparedrsmd.ipynb  # Counterfactual prediction
│   ├── hicdiff.ipynb            # Differential Hi-C analysis
│   ├── chipandrna.ipynb         # ChIP-seq and RNA-seq integration
│   ├── homologous.ipynb         # Homology analysis and visualization
│   ├── Rv0047c_statistics.R     # Statistical tests for NapM targets
│   ├── go_and_kegg.R            # Functional enrichment analysis
│   └── ...                      # Additional analysis scripts
├── 3d_model_pdbfile/            # 3D structure models (PDB format)
│   ├── wt_5k_*.pdb
│   ├── wt-hypoxia_5k_*.pdb
│   ├── wt-latent_5k_*.pdb
│   ├── napm_ko_5k_*.pdb
│   └── napm_ko-hypoxia_5k_*.pdb
├── DI_Score_results/            # Directionality Index scores
│   ├── wt.csv
│   ├── hypoxia.csv
│   ├── latent.csv
│   ├── napm-ko.csv
│   └── napm-ko-hypoxia.csv
├── hic_file/                    # Hi-C contact matrices (.hic format)
├── Interaction_network/         # Network data
│   ├── full_nodes.csv           # DNA and protein nodes
│   └── full_edges.csv           # Interaction edges
├── operon_metadata/             # Operon annotations
│   └── operon_list_updated_with_bin_coverage.csv
└── Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis.pdf  # Manuscript
```

---

## Data Description

### Hi-C Data
- **Conditions:** WT, WT-Hypoxia, WT-Latent, NapM-KO, NapM-KO-Hypoxia
- **Resolution:** 5 kb bins
- **Format:** `.hic` (Juicer format) and `.cool` (cooler format)
- **Normalization:** ICE (Iterative Correction and Eigenvector decomposition)

### 3D Models
- **Method:** Restraint-based modeling using Hi-C contact frequencies as spatial constraints
- **Resolution:** 5 kb beads per chromosome
- **Format:** PDB files with atomic coordinates
- **Reference frame:** Kabsch-aligned to WT structure

### RNA-seq and ChIP-seq
- **RNA-seq:** Bulk transcriptome across 5 conditions (biological replicates)
- **ChIP-seq:** NapM (Rv0047c) binding sites
- **Format:** BigWig (.bw) and BedGraph for genome browser visualization

### Network Data
- **Nodes:** 27 CID boundary bins + associated proteins (from ChIP-seq)
- **Edges:** 
  - DNA-DNA: Hi-C long-range interactions (>1% genome length)
  - DNA-Protein: ChIP-seq binding
  - Protein-Protein: Known physical interactions
- **Format:** CSV (nodes.csv, edges.csv)

---

## Analysis Workflows

### 1. Raw Data Processing

#### Hi-C Data Alignment
```bash
# Map reads to Mtb H37Rv genome using Juicer pipeline
juicer.sh -g H37Rv -d /path/to/fastq -s MboI -p references/H37Rv.chrom.sizes
```

#### Generate Contact Maps
```bash
# Convert to cooler format for downstream analysis
hic2cool convert input.hic output.cool -r 5000
```

**Script:** Data preprocessing details are documented in the Juicer pipeline output.

---

### 2. Hi-C Contact Map Construction

#### Visualization
Use `coolbox` to visualize contact maps and compare conditions:

```python
from coolbox.api import *

wt = DotHiC("wt.hic", style='triangular')
ko = DotHiC("napm_ko.hic", style='triangular')

frame = XAxis() + \
    HiCDiff(wt, ko, normalize='expect', diff_method='diff', 
            style='triangular', cmap='RdBu') + \
    MinValue(-2) + MaxValue(2)

frame.plot("Chromosome:1250000-1450000")
```

**Script:** [`hicdiff.ipynb`](script/hicdiff.ipynb)

**Output:** Differential contact maps showing regions with increased/decreased interactions

---

### 3. Chromatin Domain Analysis

#### Directionality Index (DI)
The Directionality Index quantifies the bias of chromatin interactions toward upstream or downstream regions, used to identify CID boundaries:

```python
# DI calculation is implemented in cooltools
import cooltools

di_track = cooltools.directionality(clr, window_bp=100000)
```

CID boundaries are identified as local DI minima with significant directional changes.

**Mathematical definition:**

$$
\text{DI}_i = \frac{(B - A)}{|B - A|} \cdot \left(\frac{(A - E)^2}{E} + \frac{(B - E)^2}{E}\right)
$$

Where $A$ and $B$ are the number of reads mapping upstream and downstream of bin $i$, and $E = (A + B)/2$.

**Script:** DI scores are pre-computed in [`DI_Score_results/`](DI_Score_results/)

---

### 4. 3D Genome Modeling

We constructed 3D models using restraint-based modeling where Hi-C contact frequencies were converted to spatial distances:

$$
d_{ij} = c \cdot f_{ij}^{-1/3}
$$

Where $d_{ij}$ is the spatial distance, $f_{ij}$ is the normalized contact frequency, and $c$ is a scaling constant.

**Tools used:** Custom pipeline with molecular dynamics simulation (not included in this repository; available upon request)

**Output:** PDB files in [`3d_model_pdbfile/`](3d_model_pdbfile/)

**Visualization:** Use PyMOL or Chimera to open PDB files

---

### 5. Operon Interaction Network

#### Network Construction
We integrated three types of interactions to build a comprehensive operon interaction network:

1. **DNA-DNA interactions:** Long-range Hi-C contacts between operons (>1% genome distance)
2. **DNA-Protein interactions:** ChIP-seq binding of regulatory proteins to operons
3. **Protein-Protein interactions:** Known physical interactions from databases

**Script:** [`network_formation.ipynb`](script/network_formation.ipynb)

#### Hub Identification
Cooperative operon hubs were identified using dual criteria:
- **High expression:** ≥90th percentile across conditions
- **High Hi-C interaction:** ≥95th percentile contact frequency

**Script:** [`operon-select_entropymoran's.ipynb`](script/operon-select_entropymoran's.ipynb)

**Key parameters:**
```python
high_expr_thresh = np.percentile(expr_values, 90)
high_hic_thresh = np.percentile(hic_matrix_vals, 95)
distance_threshold = 0.01 * genome_length  # 1% genome
```

**Output:**
- Network visualization (spring layout)
- List of hub operons with interaction strengths
- Subnetworks for DNA-DNA, DNA-Protein, and Protein-Protein interactions

---

### 6. Structural Plasticity Quantification

We quantified chromatin order and structural plasticity using two complementary metrics:

#### Shannon Entropy
Measures the disorder in DI score distributions (lower entropy = higher structural organization):

$$
H = -\sum_i p_i \log_2(p_i)
$$

Where $p_i$ is the frequency of discretized DI scores.

#### Moran's I
Measures spatial autocorrelation of DI scores along the genome (higher Moran's I = more clustered/organized):

$$
I = \frac{n}{\sum_{i,j} w_{ij}} \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}
$$

Where $x_i$ is the DI score at bin $i$, $\bar{x}$ is the mean DI, $n$ is the number of bins, and $w_{ij} = 1$ if bins are adjacent, otherwise 0.

**Script:** [`moran's_shannon.ipynb`](script/moran's_shannon.ipynb)

**Implementation:**
```python
def shannon_entropy(data):
    unique_vals, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    return -np.sum(probs * np.log2(probs))

def morans_i_1d(data):
    n = len(data)
    mean_data = np.mean(data)
    # Build adjacency matrix (neighbors only)
    w = np.zeros((n, n))
    for i in range(n):
        if i > 0: w[i, i-1] = 1
        if i < n-1: w[i, i+1] = 1
    
    numerator = sum(w[i,j] * (data[i]-mean_data) * (data[j]-mean_data) 
                    for i in range(n) for j in range(n))
    denominator = np.sum((data - mean_data)**2)
    return (n * numerator) / (denominator * np.sum(w))
```

**Results:** Both metrics reveal decreased structural order in NapM-KO conditions.

---

### 7. Counterfactual Prediction

#### NapM-KO Latent Prediction
We developed a counterfactual learning framework to predict the Hi-C contact matrix for the NapM-KO latent condition (not experimentally available) based on observed conditions.

**Method:** SVD-based feature imputation with genotype and environmental contrasts

**Mathematical framework:**
1. Log-transform and vectorize Hi-C matrices: $Y \in \mathbb{R}^{G \times N}$ (genes × conditions)
2. Encode experimental design: $F \in \mathbb{R}^{N \times P}$ (conditions × features)
   - Genotype contrast: WT vs KO
   - Environmental contrast: Normal, Hypoxia, Latent
3. Perform rank-$k$ SVD: $Y_c = U_k S_k V_k^T$
4. Learn feature mapping: $Z = V_k S_k$, solve $W = (F^T F + \alpha I)^{-1} F^T Z$
5. Predict counterfactual: $Y^* = U_k S_k (F^* W)^T$

**Model selection:** Leave-one-condition-out cross-validation (LOCO-CV) to select optimal rank

**Script:** [`napmkolatentpre_comparedrsmd.ipynb`](script/napmkolatentpre_comparedrsmd.ipynb)

**Key code:**
```python
def svd_feature_impute(Y, F, F_target, rank=2):
    col_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - col_mean
    U, s, VT = np.linalg.svd(Yc, full_matrices=False)
    U_k, S_k, V_k = U[:, :rank], np.diag(s[:rank]), VT[:rank, :].T
    Z = V_k @ S_k
    alpha = 1.0
    W = np.linalg.solve(F.T @ F + alpha*np.eye(F.shape[1]), F.T @ Z)
    Z_star = F_target @ W
    Y_star = U_k @ S_k @ Z_star.T
    return (Y_star + col_mean.mean(axis=1, keepdims=True)).ravel()
```

**Output:** Predicted contact matrix, model selection plots, volcano plots for differential genes

---

### 8. Evolutionary Conservation

#### Homology Analysis
We analyzed the conservation of Rv0047c (NapM) across bacterial species using BLASTP:

```bash
blastp -query rv0047c.faa \
       -db bacterial_proteins.fa \
       -out rv0047c_homologs.txt \
       -outfmt 6 \
       -evalue 1e-5 \
       -num_threads 8 \
       -max_target_seqs 10000
```

**Visualization:** Manhattan-style plot showing -log10(p-value) colored by genus

**Script:** [`homologous.ipynb`](script/homologous.ipynb)

**Key features:**
- Parse FASTA headers to extract organism taxonomy
- Group by genus and assign colors
- Identify conserved domains and functional motifs
- Statistical analysis across phylogenetic groups

---

### 9. Multi-omics Integration

#### ChIP-seq and RNA-seq Integration
We integrated NapM ChIP-seq binding sites with RNA-seq expression data to identify direct regulatory targets:

**Script:** [`chipandrna.ipynb`](script/chipandrna.ipynb)

**Analysis:**
```python
# Overlay ChIP-seq peaks with RNA-seq tracks
frame = XAxis() + \
    BigWig(chip, style='fill', max_value=10000) + Color("#1F928B") + Title("ChIP") + \
    BedGraph(wt_rna, style='fill') + Color("#BC3E03") + Title("WT RNA") + \
    BedGraph(ko_rna, style='fill') + Color("#BC3E03") + Title("NapM-KO RNA")
```

#### A/B Compartment Analysis
Compartment strength was calculated using the first eigenvector of the correlation matrix:

**Script:** [`ABcompartment.ipynb`](script/ABcompartment.ipynb)

**Method:**
```python
import cooltools

# Calculate eigenvectors with GC correction
gc_track = bioframe.frac_gc(bins, genome)
cis_eigs = cooltools.eigs_cis(clr, gc_track, n_eigs=3)
eigenvector_track = cis_eigs[1][['chrom','start','end','E1']]

# Compute compartment strength
cvd = cooltools.expected_cis(clr=clr, view_df=view_df)
```

**Statistical comparison:** Bootstrap resampling (1000 iterations) to compute 95% confidence intervals for compartment strength across conditions.

---

## Statistical Analysis

All statistical tests were performed in R or Python with significance threshold α = 0.05.

### Primary Tests
- **Two-sample comparisons:** Welch's t-test (unequal variances)
- **Multiple testing correction:** Benjamini-Hochberg FDR
- **RMSD comparisons:** Paired t-tests after Kabsch alignment
- **Compartment strength:** Bootstrap 95% CI with permutation tests

### Example: Structural Displacement Analysis
```python
from scipy.stats import ttest_ind

# Compare WT vs KO in hypoxia condition
wt_hyp_disp = displacement_df[displacement_df['condition']=='WT-Hypoxia']['displacement']
ko_hyp_disp = displacement_df[displacement_df['condition']=='KO-Hypoxia']['displacement']

t_stat, p_value = ttest_ind(wt_hyp_disp, ko_hyp_disp, equal_var=False)
print(f"Hypoxia WT vs KO: t={t_stat:.3f}, p={p_value:.2e}")
```

**Script:** [`napmkolatentpre_comparedrsmd.ipynb`](script/napmkolatentpre_comparedrsmd.ipynb) (RMSD analysis)

### GO/KEGG Enrichment
Functional enrichment of differentially interacting operons:

**Script:** [`go_and_kegg.R`](script/go_and_kegg.R)

```R
library(clusterProfiler)
library(org.EcK12.eg.db)  # Adjust for Mtb annotation

ego <- enrichGO(gene = gene_list,
                OrgDb = org.EcK12.eg.db,
                keyType = "SYMBOL",
                ont = "BP",
                pAdjustMethod = "BH",
                qvalueCutoff = 0.05)
```

---

## Figures Reproduction

### Main Figures

| Figure | Description | Script | Data |
|--------|-------------|--------|------|
| Fig. 1 | Hi-C contact maps across conditions | `hicdiff.ipynb` | `hic_file/*.hic` |
| Fig. 2 | DI tracks and CID boundaries | `moran's_shannon.ipynb` | `DI_Score_results/*.csv` |
| Fig. 3 | Operon interaction network | `network_formation.ipynb` | `Interaction_network/*.csv` |
| Fig. 4 | 3D structure and RMSD analysis | `napmkolatentpre_comparedrsmd.ipynb` | `3d_model_pdbfile/*.pdb` |
| Fig. 5 | NapM conservation across bacteria | `homologous.ipynb` | BLASTP output |
| Fig. 6 | ChIP-RNA integration | `chipandrna.ipynb` | BigWig/BedGraph files |

### Supplementary Figures
- **Saddle plots (A/B compartments):** `ABcompartment.ipynb`
- **Entropy and Moran's I bar charts:** `moran's_shannon.ipynb`
- **Volcano plots (differential genes):** `napmkolatentpre_comparedrsmd.ipynb`
- **Functional enrichment:** `go_and_kegg.R`

**Reproduction:**
```bash
# Run all notebooks in order
jupyter nbconvert --execute --to notebook script/*.ipynb

# Or run individually
jupyter notebook script/network_formation.ipynb
```

---

## Citation

If you use this code or data, please cite:

> [Author et al.] Establishment and remodeling of the *Mycobacterium tuberculosis* 3D genome architecture. [Journal Name] (Year). DOI: XXX

**Preprint:** See [`Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis.pdf`](Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis.pdf)

---

## Contact

For questions or issues, please:
- Open an issue on GitHub: https://github.com/wikk-chy/mtb_hic/issues
- Email: [your-email@institution.edu]

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- Hi-C data processing: Juicer pipeline (Aiden Lab)
- 3D modeling: Custom restraint-based framework
- Statistical analysis: R/Bioconductor community
- Visualization: coolbox, matplotlib, seaborn

**Funding:** [Add funding sources]

---

## Change Log

### v1.0 (2025)
- Initial release with complete analysis pipeline
- All scripts for main and supplementary figures
- Processed Hi-C, RNA-seq, and ChIP-seq data
- 3D structure models for 5 conditions
