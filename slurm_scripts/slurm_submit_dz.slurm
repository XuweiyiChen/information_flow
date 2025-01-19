#!/bin/bash

#SBATCH --job-name=llm2vec-mteb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --open-mode=append
#SBATCH --mail-type=END
#SBATCH --mail-user dz1158@nyu.edu
#SBATCH --requeue

echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST "

# CONTAINER="$PROJECT/lsa273_uksr/containers/information_plane/information_plane.sif"

# # Check if ~/.cache is a symlink
# if [ ! -L ~/.cache ]; then
#     echo "Error: ~/.cache is not a symlink. Disable this check at your own risk."
#     exit 1
# fi

# Initialize variables with default empty values
model_family=""
model_size=""
revision=""
layer=""
purpose=""

# Parse command line arguments
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
    echo "Usage: $0 --model_family <model_family> --model_size <model_size> --revision <revision> --evaluation_layer <evaluation_layer> --purpose <purpose>"
    exit 1
fi

# Create the directory for logs
log_dir="./lcc_logs/${model_family}/${model_size}/${revision}/layer-${layer}"
mkdir -p "$log_dir"
output_file="${log_dir}/logs.out"
error_file="${log_dir}/logs.err"

SCRIPT="/home/ofsk222/projects/information_flow/MTEB-Harness.py"
FULL_SCRIPT="python3 -u $SCRIPT --model_family $model_family --model_size $model_size --revision $revision --evaluation_layer $layer --base_results_path experiments/large_results --purpose $purpose"
srun --output="$output_file" --error="$error_file" singularity run --app pytorch222 --nv $CONTAINER $FULL_SCRIPT
