#!/bin/bash
# S3+apposition sweep + seed-777 replication of optima.
# Usage: driver_s3app.sh <best_scale> <lambda_S3_best>  (k_app swept 1..30)
set -e
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm
SCALE=$1
LAM=$2
i=0
run() {  # scenario strength strength2 seed
  $PY $POLY/code/02_simulate.py --scenario $1 --strength $2 --strength2 $3 \
    --scale $SCALE --seed $4 --equil 250000 --nconf 700 \
    --sample-every 3000 --threads 2 --outdir $POLY/runs \
    > $POLY/logs/$1_str$2_str2$3_seed$4.log 2>&1 &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
}
for K in 1 3 10 30; do run S3app $LAM $K 42; done
wait
# seed-777 replication: baseline + optima of S1/S2/S3/S3app (seed-42 tables
# must already exist: run 03_analyze.py --mode all first)
$PY $POLY/code/03_analyze.py --mode all
best() {  # scenario column
  python3 -c "
import pandas as pd
df = pd.read_csv('$POLY/tables/sim_metrics.csv')
sub = df[(df.scenario=='$1') & (df.seed==42)]
print(sub.iloc[sub.ps_rmse_latent.argmin()]['$2'])"
}
run baseline 0 0 777
run S1 $(best S1 strength) 0 777
run S2 $(best S2 strength) 0 777
run S3 $(best S3 strength) 0 777
run S3app $LAM $(best S3app strength2) 777
wait
echo "S3APP + SEED777 DONE"
