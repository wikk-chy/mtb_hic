#!/bin/bash
# Post-pipeline patch runs:
#  1) S2 k_t=10 (crashed under the pre-ramp tether implementation, NaN)
#  2) S3 lambda=1.0 with extended burn-in (1e6 steps): the standard 250k
#     burn-in run underwent the toroid-folding transition MID-PRODUCTION
#     (block analysis), so its production sample is a non-equilibrium
#     mixture. Saved under seed 142 to keep the original file for the
#     record; analysis prefers this extended run (see 03_analyze.py).
set -e
PY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/.venv_cooltools/bin/python
POLY=/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R4_comment5/polymer_openmm
$PY $POLY/code/02_simulate.py --scenario S2 --strength 10 --scale 0.7 \
  --seed 42 --equil 250000 --nconf 700 --sample-every 3000 --threads 2 \
  --outdir $POLY/runs > $POLY/logs/S2_str10_rerun.log 2>&1 &
$PY $POLY/code/02_simulate.py --scenario S3 --strength 1.0 --scale 0.7 \
  --seed 142 --equil 1000000 --nconf 700 --sample-every 3000 --threads 2 \
  --outdir $POLY/runs > $POLY/logs/S3_str1_longburnin.log 2>&1 &
wait
echo "PATCH RUNS DONE"
