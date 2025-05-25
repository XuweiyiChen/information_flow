# List of labels

MODEL_SIZES=('410m')
MAX_LAYER=32
TASK='toxigen'
LENS_TYPES=('tuned')
# Number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

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

for lens_type in ${LENS_TYPES[@]}; do
    for size in ${MODEL_SIZES[@]}; do
        for layer in $(seq 0 $MAX_LAYER); do
            while true; do
            cleanup_finished_jobs  # Remove finished jobs
            GPU_ID=$(find_free_gpu)  # Find an available GPU

            if [[ $GPU_ID -ge 0 ]]; then
                echo "Launching job on $GPU_ID for Pythia-$size on layer $layer"

                CUDA_VISIBLE_DEVICES=$GPU_ID python MMLU-Harness.py \
                    --model_size $size \
                    --evaluation_layer $layer \
                    --lens-type $lens_type \
                    --task $TASK > logs/${size}_${layer}_${TASK}_${lens_type}.log 2>&1 &

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
done

# Final wait to ensure all jobs complete before script exits
wait
echo "All jobs finished!"