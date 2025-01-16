#!/bin/bash
USE_SLURM=1

MODEL_NAME="mamba"
MODEL_SIZES=('130m' '370m' '790m')
MAX_LAYER=50
REVISION="main"
PURPOSE="run_tasks"
for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        if [ $USE_SLURM -eq 1 ]; then
            sbatch slurm_submit.sh \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $REVISION \
                --evaluation_layer $layer \
                --purpose $PURPOSE
        else
            echo "Running evaluation for $MODEL_NAME $size layer $layer"
            python MTEB-Harness.py \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $REVISION \
                --evaluation_layer $layer \
                --purpose $PURPOSE
        fi
    done
done
