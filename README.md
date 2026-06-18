# Codes and Analysis for Mycobacterium tuberculosis Hi-C Study

## Overview

This repository contains the code and data for investigating three-dimensional (3D) genome organization in *Mycobacterium tuberculosis* (Mtb). The analyses integrate Hi-C, RNA-seq, and ChIP-seq data to examine chromatin architecture remodeling under different physiological conditions (WT, Hypoxia, Latent, NapM-KO, NapM-KO-Hypoxia).

---

## Table of Contents

- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Analysis Workflows](#analysis-workflows)
  - [Hi-C Contact Map Visualization](#hi-c-contact-map-visualization)
  - [Directionality Index and CID Boundaries](#directionality-index-and-cid-boundaries)
  - [3D Genome Models](#3d-genome-models)
  - [Operon Interaction Network](#operon-interaction-network)
  - [Structural Plasticity Quantification](#structural-plasticity-quantification)
  - [Counterfactual Prediction](#counterfactual-prediction)
  - [Homology Analysis](#homology-analysis)
  - [A/B Compartment Analysis](#ab-compartment-analysis)
- [Figures Reproduction](#figures-reproduction)

---

## Dependencies

### Python (≥3.7)
- `cooler`, `cooltools`, `coolbox` - Hi-C analysis
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `networkx` - Network analysis
- `bioframe` - Genomics utilities

### R (≥4.0)
- `ggplot2`, `dplyr` - Data visualization and manipulation
- `clusterProfiler` - Functional enrichment analysis

### External Programs
- [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml) - Read alignment
- [Juicer](https://github.com/aidenlab/juicer) - Hi-C pipeline
- [BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) - Homology search

---

## Repository Structure

```
.
├── script/                      # Analysis scripts
│   ├── hicdiff.ipynb            # Differential Hi-C visualization
│   ├── moran's_shannon.ipynb    # Structural order metrics (Shannon entropy, Moran's I)
│   ├── operon-select_entropymoran's.ipynb  # Operon hub identification
│   ├── network_formation.ipynb  # Interaction network construction
│   ├── napmkolatentpre_comparedrsmd.ipynb  # Counterfactual prediction
│   ├── ABcompartment.ipynb      # A/B compartment analysis
│   ├── chipandrna.ipynb         # ChIP-seq and RNA-seq integration
│   ├── homologous.ipynb         # Homology analysis
│   ├── Rv0047c_statistics.R     # Statistical analysis
│   └── go_and_kegg.R            # GO/KEGG enrichment
├── 3d_model_pdbfile/            # 3D structure models (PDB format)
├── DI_Score_results/            # Directionality Index scores (5 conditions)
├── hic_file/                    # Hi-C contact matrices (.hic format)
├── Interaction_network/         # Network nodes and edges (CSV)
└── operon_metadata/             # Operon annotations
```

---

## Analysis Workflows

### Hi-C Contact Map Visualization

Visualize and compare Hi-C contact maps across conditions using differential analysis.

**Script:** [`hicdiff.ipynb`](script/hicdiff.ipynb)

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

---

### Directionality Index and CID Boundaries

The Directionality Index (DI) quantifies the bias of chromatin interactions toward upstream or downstream regions. It is used to identify chromatin interaction domain (CID) boundaries.

$$
\text{DI}_i = \frac{(B - A)}{|B - A|} \cdot \left(\frac{(A - E)^2}{E} + \frac{(B - E)^2}{E}\right)
$$

Where A and B are upstream and downstream interaction counts, and E = (A + B)/2.

**Data:** Pre-computed DI scores available in [`DI_Score_results/`](DI_Score_results/)

---

### 3D Genome Models

3D chromosome structures were modeled using [LorDG](https://github.com/TaoYang-dev/LorDG) (Low-rank Distance Geometry) with Hi-C contact frequencies as distance constraints.

**Output:** PDB files in [`3d_model_pdbfile/`](3d_model_pdbfile/)  
**Visualization:** Use PyMOL, Chimera, or other molecular visualization tools

---

### Operon Interaction Network

Construct multi-layer interaction networks integrating:
- **DNA-DNA:** Hi-C loops
- **DNA-Protein:** ChIP-seq binding
- **Protein-Protein:** Known interactions

**Scripts:**
- [`network_formation.ipynb`](script/network_formation.ipynb) - Network construction and visualization
- [`operon-select_entropymoran's.ipynb`](script/operon-select_entropymoran's.ipynb) - Hub identification

---

### Structural Plasticity Quantification

Quantify chromatin structural organization using Shannon entropy and Moran's I on DI score distributions.

**Script:** [`moran's_shannon.ipynb`](script/moran's_shannon.ipynb)

**Shannon Entropy** (lower = more organized):

$$
H = -\sum_i p_i \log_2(p_i)
$$

**Moran's I** (higher = more structured):

$$
I = \frac{n}{\sum_{i,j} w_{ij}} \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}
$$

Where $w_{ij} = 1$ for adjacent bins, 0 otherwise.

---

### Counterfactual Prediction

Predict the NapM-KO latent Hi-C contact matrix using SVD-based counterfactual learning on observed conditions.

**Script:** [`napmkolatentpre_comparedrsmd.ipynb`](script/napmkolatentpre_comparedrsmd.ipynb)

**Method:**
- Feature encoding: genotype (WT/KO) and environment (normal/hypoxia/latent)
- SVD-based imputation with ridge regularization

---

### Homology Analysis

Analyze conservation of Rv0047c (NapM) across bacterial species.

**Script:** [`homologous.ipynb`](script/homologous.ipynb)

```bash
blastp -query rv0047c.faa \
       -db bacterial_proteins.fa \
       -out rv0047c_homologs.txt \
       -outfmt 6 \
       -evalue 1e-5 \
       -max_target_seqs 10000
```

**Visualization:** Manhattan plot of -log10(p-value) colored by genus

---

### A/B Compartment Analysis

Identify A/B compartments using the first eigenvector of the Hi-C correlation matrix with GC bias correction.

**Script:** [`ABcompartment.ipynb`](script/ABcompartment.ipynb)

```python
import cooltools
import bioframe

gc_track = bioframe.frac_gc(bins, genome)
cis_eigs = cooltools.eigs_cis(clr, gc_track, n_eigs=3)
eigenvector_track = cis_eigs[1][['chrom','start','end','E1']]
```

**Statistical analysis:** Bootstrap 95% confidence intervals for compartment strength comparison.

---

## Figures Reproduction

Key analysis scripts for main figures:

| Analysis | Script | Input Data |
|----------|--------|------------|
| Hi-C differential maps | [`hicdiff.ipynb`](script/hicdiff.ipynb) | `hic_file/*.hic` |
| DI and structural metrics | [`moran's_shannon.ipynb`](script/moran's_shannon.ipynb) | `DI_Score_results/*.csv` |
| Operon interaction network | [`network_formation.ipynb`](script/network_formation.ipynb) | `Interaction_network/*.csv` |
| 3D structure comparison | [`napmkolatentpre_comparedrsmd.ipynb`](script/napmkolatentpre_comparedrsmd.ipynb) | `3d_model_pdbfile/*.pdb` |
| NapM homology | [`homologous.ipynb`](script/homologous.ipynb) | BLASTP output |
| A/B compartments | [`ABcompartment.ipynb`](script/ABcompartment.ipynb) | `.cool` files |
| ChIP-RNA integration | [`chipandrna.ipynb`](script/chipandrna.ipynb) | BigWig/BedGraph |
| GO/KEGG enrichment | [`go_and_kegg.R`](script/go_and_kegg.R) | Gene lists |

**Statistical Analysis:** All significance tests use Benjamini-Hochberg FDR correction (α = 0.05). Bootstrap resampling (n=1000) for confidence intervals. See individual scripts for details.

---

## Citation

If you use this code or data, please cite:

> Establishment and remodeling of the *Mycobacterium tuberculosis* 3D genome architecture. (Manuscript in preparation)

---

## Contact

For questions, please open an issue on GitHub: https://github.com/wikk-chy/mtb_hic/issues

---

## License

MIT License

Copyright (c) 2025 wikk-chy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
