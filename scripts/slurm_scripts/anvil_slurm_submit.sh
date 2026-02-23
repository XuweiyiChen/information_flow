#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH --account=nairr250073-ai
#SBATCH --mail-user=xuweic@email.virginia.edu
#SBATCH --mail-type=ALL

model_family=""
model_size=""
revision=""
layer=""
purpose=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_family)
            model_family="$2"
            shift 2
            ;;
        --model_size)
            model_size="$2"
            shift 2
            ;;
        --revision)
            revision="$2"
            shift 2
            ;;
        --layer)
            layer="$2"
            shift 2
            ;;
        --purpose)
            purpose="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$model_family" || -z "$model_size" || -z "$revision" || -z "$layer" ]]; then
    echo "Usage: $0 --model_family <model_family> --model_size <model_size> --revision <revision> --layer <layer> --purpose <purpose>"
    exit 1
fi

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: ${model_family} ${model_size} rev=${revision}"
echo "Layer: ${layer}"
echo "Purpose: ${purpose}"
echo "Started at: $(date)"

WORKDIR="/anvil/scratch/x-xchen8/information_flow"
cd "$WORKDIR"

mkdir -p experiments/results

python3 -u MTEB-Harness.py \
    --model_family "$model_family" \
    --model_size "$model_size" \
    --revision "$revision" \
    --evaluation_layer "$layer" \
    --base_results_path experiments/results \
    --purpose "$purpose"

echo "Completed at: $(date)"
