#!/bin/bash
# R4C5 polymer_openmm - full pipeline. Run from any directory.
# Requires: hic_review/.venv_cooltools/bin/python (openmm 8.1.1, numpy,
# scipy, pandas, matplotlib). macOS arm64: the venv is x86_64 (Rosetta);
# OpenMM CPU platform, 2 threads per process, 4 processes in parallel.
# Total simulation time ~5 h on an 8-core M1 Pro.
set -e
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm
mkdir -p $POLY/runs $POLY/logs $POLY/tables $POLY/figures

# 0. experimental reference metrics
$PY $POLY/code/01_experiment_metrics.py

# 1. baseline grid (confinement scale)
bash $POLY/code/driver_baseline_grid.sh

# 2. pick baseline scale (P(s) shape RMSE vs WT); calibrate contact cutoff
$PY $POLY/code/03_analyze.py --mode grid
BEST_TAG=$(python3 -c "
import pandas as pd
df = pd.read_csv('$POLY/tables/baseline_grid.csv')
print(df.iloc[df.ps_rmse_WT.argmin()]['tag'])")
BEST_SCALE=$(python3 -c "
import pandas as pd
df = pd.read_csv('$POLY/tables/baseline_grid.csv')
print(df.iloc[df.ps_rmse_WT.argmin()]['scale'])")
echo "best baseline: $BEST_TAG (scale $BEST_SCALE)"
$PY $POLY/code/03_analyze.py --mode cutoff --run $BEST_TAG

# 3. S1/S2/S3 sweeps
bash $POLY/code/driver_scenarios.sh $BEST_SCALE

# 4. interim analysis -> S3 optimum -> S3app sweep + seed-777 replication
$PY $POLY/code/03_analyze.py --mode all
LAM=$(python3 -c "
import pandas as pd
df = pd.read_csv('$POLY/tables/sim_metrics.csv')
sub = df[(df.scenario=='S3') & (df.seed==42)]
print(sub.iloc[sub.ps_rmse_latent.argmin()]['strength'])")
echo "S3 optimum lambda = $LAM"
bash $POLY/code/driver_s3app.sh $BEST_SCALE $LAM

# 5. final analysis, model comparison, figures
$PY $POLY/code/03_analyze.py --mode all
$PY $POLY/code/04_model_comparison.py
$PY $POLY/code/05_figures.py
echo "PIPELINE DONE"
