#!/bin/bash
USE_SLURM=0

MODEL_NAME="LLM2Vec-mntp-unsup-simcse"
#MODEL_NAME="LLM2Vec-mntp"
MAX_LAYER=9
REVISION="main"
SIZE="8B"

for layer in $(seq 9 $MAX_LAYER); do
    if [ $USE_SLURM -eq 1 ]; then
        JOBNAME="llm2vec-$layer"
        sbatch -J $JOBNAME slurm_submit.sh --model_family $MODEL_NAME --model_size $SIZE --revision $REVISION --evaluation_layer $layer --purpose run_tasks
    else
        python MTEB-Harness.py \
            --model_family $MODEL_NAME \
            --model_size $SIZE \
            --revision $REVISION \
            --evaluation_layer -1 \
            --purpose run_tasks \
            --raise_error True
    fi
done
