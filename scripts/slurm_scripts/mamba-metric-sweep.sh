#!/bin/bash
USE_SLURM=0
MODEL_NAME="mamba"
MODEL_SIZES=('370m')
REVISION="main"
PURPOSE="run_wikitext_metrics"

for size in ${MODEL_SIZES[@]}; do
    if [ $USE_SLURM -eq 1 ]; then
        sbatch slurm_submit.sh \
            --model_family $MODEL_NAME \
            --model_size $size \
            --revision $REVISION \
            --evaluation_layer -1 \
            --purpose $PURPOSE
    else
        python MTEB-Harness.py \
            --model_family $MODEL_NAME \
            --model_size $size \
            --revision $REVISION \
            --evaluation_layer -1 \
            --purpose $PURPOSE
    fi
done
