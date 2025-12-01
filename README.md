## Codes and functions for the mycobacterium tuberculosis Hi-C data analysis ##
This repository contains the code and data used in the paper: The goal of this project is to  analysis mycobacterium tuberculosis Hi-C data.The codes presented here should allow to reproduce the different graphs and figures from the main text and the supplementary data.
### Table of contents
* [Dependencies](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Dependencies)
* [Raw data extraction and alignment](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Raw-data-extraction-and-alignment)
* [Building contacts map](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Building-contacts-map)
* [Construct the 3D genome model](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Construct-the-3D-genome-model)
* [Find cooperative operon hubs](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Find-cooperative-operon-hubs)
* [Search for homologous genes](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Search-for-homologous-genes)
* [Measure the chromatin order and the structural plasticity](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Measure-the-chromatin-order-and-the-structural-plasticity)
* [Predict KO-latent Hi-C contact matrix](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Predict-KO-latent-Hi-C-contact-matrix)
* [Kabsch alignment and RMSD calculation](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Kabsch-alignment-and-RMSD-calculation)
* [Statistical Analysis](https://github.com/wikk-chy/mtb_hic/edit/main/README.md#Statistical-Analysis)
### Dependencies
Scripts and codes can be run on OS X and other Unix-based systems, and necessitate:
#### *Python (>=3.1)*
* Coolbox
* Pandas
* Numpy
* Matplotlib
* Scipy
* Seaborn
* Sklearn
#### External programs
* `R` / [R](https://cran.r-project.org/)
* `Bowtie2 ` / [bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
* `blast ` / [blast](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
* `jucier ` / [jucier](https://github.com/aidenlab/juicer)
## Raw data extraction and alignment
#### Data extraction
#### Alignment
We use the H37Rv reference genome ( NCBI:txid83332, total length 4411532).
## Building contacts map
To build the contact map , we use
### Scalogram
The scale map tool can visualize the dispersion of contact signals along spatial scales.This function is implemented by the code.
### Directionality index
The Directionality Index (DI) quantifies the bias of chromatin interactions toward upstream or downstream regions. It is widely used to identify TAD boundaries in 3D genome organization.we use the code to calculate.
## Construct the 3D genome model
To study the three-dimensional structure of chromosomes, we use [method] to directly simulate the positional relationships between nucleotides.
## Find cooperative operon hubs
## Search for homologous genes
We obtained all homologous genes of Rv0047c at the bacterial level from [orthoDB](https://www.orthodb.org/)then we use ncbi-blast to calculate homology relationship.
```bash
blastp -query protein.faa -out rv0047c.txt -db fasta.fa -outfmt 6 -evalue 1e-5 -num_threads 2 -max_target_seqs 10000
```
## Measure the chromatin order and the structural plasticity
To Measure the chromatin order and the structural plasticity,we selected Shannon entropy and Moran’s I as the evaluation metrics.
### Shannon entropy
Shannon entropy was applied to measure the level of orderliness in DI distributions.Higher structural organization, indicated by clearer CID boundaries, is characterized by lower Shannon entropy.Shannon entropy was calculated as follows:

$$
H = -\sum_i p_i \log_2(p_i)
$$

Where $p_i$ represents the frequency of discretized DI scores.
### Moran’s I
The one-dimensional Moran’s I was computed to measure spatial autocorrelation among adjacent genomic bins:

$$
I = \frac{n}{\sum_{i,j} w_{ij}}
    \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}
         {\sum_i (x_i - \bar{x})^2}
$$

Where $x_i$ denotes the DI score at genomic bin $i$, $\bar{x}$ is the mean DI score across bins, $n$ is the total number of bins, and $w_{ij}$ represents the adjacency-based weighting (with a value of 1 if bins $i$ and $j$ are neighbors, otherwise 0).
## Predict NapM-KO Latent Hi-C contact matrix
We modeled the chromatin contact landscape of Mycobacterium tuberculosis under the NapM-KO latent condition using an SVD-based counterfactual framework. Log-transformed Hi-C matrices from five conditions were vectorized and encoded with genotype and environmental contrasts. The optimal latent rank was selected by cross-validation, and the trained model was used to reconstruct the NapM-KO latent contact matrix via latent projection and inverse transformation.
## Kabsch alignment and RMSD calculation
We quantified structural differences between conditions using the Kabsch algorithm. For each pairwise comparison, structures were centered, optimally rotated via SVD, and aligned to the WT reference. After alignment, the global RMSD was computed:

$$
\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^N \lVert r_i^{\text{ref}} - r_i^{\text{aln}} \rVert^2 }
$$

We also calculated per-atom displacement vectors,

$$
d_i = \lVert r_i^{\text{ref}} - r_i^{\text{aln}} \rVert
$$

to quantify local positional changes.
## Statistical Analysis

All statistical analyses in this study can be reproduced using the scripts provided in the `script` directory, with the required raw data stored in the `data` directory. Unless otherwise specified, all significance tests were performed using *t*-tests.
