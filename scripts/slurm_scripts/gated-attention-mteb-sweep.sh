#!/bin/bash
USE_SLURM=1

MODEL_FAMILIES=('gated_attention_baseline' 'gated_attention_elementwise' 'gated_attention_headwise')
MODEL_SIZE="1B"
MAX_LAYER=28
REVISION="main"
PURPOSE="run_tasks"

WORKDIR="/anvil/scratch/x-xchen8/information_flow"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for model_family in "${MODEL_FAMILIES[@]}"; do
    for layer in $(seq 0 $MAX_LAYER); do
        log_dir="${WORKDIR}/anvil_logs/${model_family}/${MODEL_SIZE}/${REVISION}/layer-${layer}"
        mkdir -p "$log_dir"

        if [ $USE_SLURM -eq 1 ]; then
            echo "Submitting SLURM job: $model_family $MODEL_SIZE layer $layer"
            sbatch \
                --job-name="mteb_${model_family}_L${layer}" \
                --output="${log_dir}/slurm_%j.out" \
                --error="${log_dir}/slurm_%j.err" \
                "${SCRIPT_DIR}/anvil_slurm_submit.sh" \
                --model_family "$model_family" \
                --model_size "$MODEL_SIZE" \
                --revision "$REVISION" \
                --layer "$layer" \
                --purpose "$PURPOSE"
        else
            echo "Running evaluation for $model_family $MODEL_SIZE layer $layer"
            cd "$WORKDIR"
            python3 -u MTEB-Harness.py \
                --model_family "$model_family" \
                --model_size "$MODEL_SIZE" \
                --revision "$REVISION" \
                --evaluation_layer "$layer" \
                --base_results_path experiments/results \
                --purpose "$PURPOSE"
        fi
    done
done
