#!/bin/bash

# Get number of available GPUs
NUM_GPUS=2 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Track running jobs with PIDs and GPUs
declare -A GPU_JOB_MAP

# Function to find a free GPU
find_free_gpu() {
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        if [[ -z "${GPU_JOB_MAP[$gpu]}" ]]; then
            echo $gpu
            return
        fi
    done
    echo -1  # No free GPU found
}

# Function to clean up finished jobs
cleanup_finished_jobs() {
    for gpu in "${!GPU_JOB_MAP[@]}"; do
        pid=${GPU_JOB_MAP[$gpu]}
        if ! kill -0 $pid 2>/dev/null; then
            echo "Job on GPU $gpu (PID $pid) finished. Freeing up GPU..."
            unset GPU_JOB_MAP[$gpu]
        fi
    done
}

# Define the ranges for eval_layer and model-selection-idx
MAX_LAYER=24  # Adjust this based on your model's maximum layer
MAX_MODEL_IDX=6  # Adjust this based on the number of models in models_to_try
LAYER_WINDOW=0

for model_idx in $(seq $MAX_MODEL_IDX -1 0); do
    for layer in $(seq $MAX_LAYER -1 1); do
        while true; do
            cleanup_finished_jobs  # Remove finished jobs
            GPU_ID=$(find_free_gpu)  # Find an available GPU

            if [[ $GPU_ID -ge 0 ]]; then
                echo "Launching job on GPU $GPU_ID for model index $model_idx on layer $layer"

                CUDA_VISIBLE_DEVICES=$GPU_ID python Vision-Probing-Harness.py \
                    --eval_layer $layer \
                    --model-selection-idx $model_idx \
                    --layer_window $LAYER_WINDOW > logs/model${model_idx}_layer${layer}.log 2>&1 &

                JOB_PID=$!
                GPU_JOB_MAP[$GPU_ID]=$JOB_PID  # Store the job's PID for tracking
                break  # Move to the next combination
            else
                echo "No free GPU available. Checking again in 5 seconds..."
                sleep 5
            fi
        done
    done
done

# Final wait to ensure all jobs complete before script exits
wait
echo "All jobs finished!"