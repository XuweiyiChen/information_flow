#!/bin/bash

MODEL_NAME="Pythia"
SIZE="410m"
MAX_LAYER=50  # Set high to be safe
REVISION="main"
PURPOSE="run_tasks"
RESULTS_PATH="experiments/results_V2"

for layer in $(seq 0 $MAX_LAYER); do
    echo "=== Running layer $layer ==="
    python MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $SIZE \
        --revision $REVISION \
        --base_results_path $RESULTS_PATH \
        --purpose $PURPOSE \
        --evaluation_layer $layer
done
