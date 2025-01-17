#!/bin/bash

USE_SLURM=0

MODEL_NAME="bert"
MODEL_SIZES=('base' 'large')
REVISION="main"
PURPOSE="run_entropy_metrics"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME $size"
    python MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $size \
        --revision $REVISION \
        --purpose run_entropy_metrics \

done
