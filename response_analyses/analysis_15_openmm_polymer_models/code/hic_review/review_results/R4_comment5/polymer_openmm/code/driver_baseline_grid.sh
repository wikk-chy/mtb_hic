#!/bin/bash
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm
mkdir -p $POLY/runs $POLY/logs
i=0
for SCALE in 0.7 0.85 1.0 1.2 1.4; do
  $PY $POLY/code/02_simulate.py --scenario baseline --scale $SCALE --seed 42 \
    --equil 250000 --nconf 700 --sample-every 3000 --threads 2 \
    --outdir $POLY/runs > $POLY/logs/baseline_scale${SCALE}.log 2>&1 &
  i=$((i+1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "BASELINE GRID DONE"
ls $POLY/runs/
