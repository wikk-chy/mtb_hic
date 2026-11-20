#!/bin/bash
# Scenario sweeps at the best baseline scale (passed as $1).
# S1/S2/S3 run first (12 runs); S3app needs the S3 optimum and is launched
# separately by driver_s3app.sh after an interim analysis.
set -e
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm
SCALE=$1
i=0
run() {
  $PY $POLY/code/02_simulate.py --scenario $1 --strength $2 --strength2 $3 \
    --scale $SCALE --seed 42 --equil 250000 --nconf 700 \
    --sample-every 3000 --threads 2 --outdir $POLY/runs \
    > $POLY/logs/$1_str$2_str2$3.log 2>&1 &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
}
for S in 0.25 0.5 1.0 2.0; do run S1 $S 0; done
for S in 1 3 10 30;     do run S2 $S 0; done
for S in 0.25 0.5 1.0 2.0; do run S3 $S 0; done
wait
echo "SCENARIO SWEEPS (S1/S2/S3) DONE"
