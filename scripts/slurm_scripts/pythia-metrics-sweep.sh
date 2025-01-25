#!/bin/bash
USE_SLURM=0

MODEL_NAME="Pythia"
MODEL_SIZES=('14m' '70m' '160m' '410m' '1b')
REVISION="main"
PURPOSE="run_entropy_metrics"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME $size"
    python MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $size \
        --revision $REVISION \
        --purpose run_entropy_metrics
done
