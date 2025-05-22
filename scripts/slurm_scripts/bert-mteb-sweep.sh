#!/bin/bash


USE_SLURM=0

MODEL_NAME=("bert" "roberta")
MODEL_SIZES=('base')
MAX_LAYER=24
REVISION="main"

for model in ${MODEL_NAME[@]}; do
    for size in ${MODEL_SIZES[@]}; do
        for layer in $(seq 0 $MAX_LAYER); do
            if [ $USE_SLURM -eq 1 ]; then
                sbatch slurm_submit.sh $model $size $REVISION $layer
            else
                echo "Running evaluation for $model $size layer $layer"
                python MTEB-Harness.py --model_family $model --model_size $size --revision $REVISION --evaluation_layer $layer
            fi
        done
    done
done
