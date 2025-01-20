#!/bin/bash
#SBATCH --job-name=llm2vec-mntp-mteb-lastlay
#SBATCH --time=32:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --open-mode=append
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user dz1158@nyu.edu
#SBATCH --requeue


OVERLAY_PATH="/scratch/dz1158/model_merge/overlay-50G-10M.ext3"
SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"

MODEL_NAME="LLM2Vec-mntp"
MAX_LAYER=32
MIN_LAYER=16
REVISION="main"
SIZE="8B"

singularity exec --bind /scratch --nv --overlay ${OVERLAY_PATH}:ro ${SINGULARITY_IMAGE} /bin/bash -c "
source /ext3/env.sh
conda activate infoflow

for layer in \$(seq $MAX_LAYER -1 $MIN_LAYER); do
    echo \"processing layer \$layer\"
    python /scratch/dz1158/info_metric/information_flow/MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $SIZE \
        --revision $REVISION \
        --evaluation_layer \$layer \
        --purpose run_tasks \
        --raise_error False
done
"