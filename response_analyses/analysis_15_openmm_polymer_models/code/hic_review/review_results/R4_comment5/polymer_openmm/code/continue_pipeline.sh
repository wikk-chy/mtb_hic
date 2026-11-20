#!/bin/bash
# Continuation after the baseline grid: waits for the grid to finish,
# then runs scale selection, cutoff calibration, scenario sweeps,
# S3app + seed-777, and the final analysis/figures.
set -e
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm

# wait for grid driver to finish (marker line in driver_baseline.out)
while ! grep -q "BASELINE GRID DONE" $POLY/logs/driver_baseline.out 2>/dev/null; do
  sleep 60
done
echo "grid finished: $(date)"

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

bash $POLY/code/driver_scenarios.sh $BEST_SCALE
echo "scenarios done: $(date)"

$PY $POLY/code/03_analyze.py --mode all
LAM=$(python3 -c "
import pandas as pd
df = pd.read_csv('$POLY/tables/sim_metrics.csv')
sub = df[(df.scenario=='S3') & (df.seed==42)]
print(sub.iloc[sub.ps_rmse_latent.argmin()]['strength'])")
echo "S3 optimum lambda = $LAM"
bash $POLY/code/driver_s3app.sh $BEST_SCALE $LAM
echo "s3app+seeds done: $(date)"

$PY $POLY/code/03_analyze.py --mode all
$PY $POLY/code/04_model_comparison.py
$PY $POLY/code/05_figures.py
echo "PIPELINE DONE $(date)"
