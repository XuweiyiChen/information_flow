
---

# 📄 Layer by Layer: Uncovering Hidden Representations in Language Models

**📌 [Read the Paper on arXiv](https://arxiv.org/abs/2502.02013)**

**📚 Citation:**

```bibtex
@article{skean2025layer,
  title={Layer by {L}ayer: {U}ncovering hidden representations in language models},
  author={Skean, Oscar and Arefin, Md Rifat and Zhao, Dan and Patel, Niket and Naghiyev, Jalal and LeCun, Yann and Shwartz-Ziv, Ravid},
  journal={ICML},
  year={2025}
}
```

---

## 🛠 Installation

Create a new conda environment and install the dependencies:

```bash
pip install -r requirements.txt
```

If you encounter issues with `representation-itl` (`repitl`), try installing it manually:

```bash
pip3 install -e git+https://github.com/uk-cliplab/representation-itl.git#egg=representation-itl
```

---

## 🧪 MTEB Results

### 🔧 The MTEB Harness

The `MTEB-Harness.py` script handles interactions between models and the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard). Key arguments:

* `--model_family`: Type of model (e.g. `'Pythia'`).
* `--model_size`: Model size variant (e.g. `'14m'`).
* `--revision`: Model version to use (default: `'main'`).
* `--evaluation_layer`: Which layer to evaluate (e.g. `-1` for final).
* `--base_results_path`: Where results will be saved.
* `--purpose`: Task type: `'run_tasks'`, `'run_entropy_metrics'`, or `'download_datasets'`.
* `--raise_error`: Raise and stop on errors (`True/False`).

---

### ▶️ Running MTEB Benchmarks

**Single Model, Single Layer**

To run benchmarks on the **final layer** (`--evaluation_layer -1`) of a model:

```bash
python3 -u MTEB-Harness.py \
    --model_family "Pythia" \
    --model_size "14m" \
    --revision "main" \
    --evaluation_layer -1 \
    --base_results_path "experiments/results" \
    --purpose run_tasks
```

---

**Single Model, All Layers**

To benchmark **all layers** in a model:

```bash
MODEL_NAME="Pythia"
SIZE="410m"
MAX_LAYER=50  # Set high to be safe
REVISION="main"
PURPOSE="run_tasks"
RESULTS_PATH="experiments/results"

for layer in $(seq 0 $MAX_LAYER); do
    python MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $SIZE \
        --revision $REVISION \
        --base_results_path $RESULTS_PATH \
        --purpose $PURPOSE \
        --evaluation_layer $layer
done
```

---

**Multiple Models, All Layers**

```bash
MODEL_NAME="Pythia"
MODEL_SIZES=("14m" "70m" "410m")
MAX_LAYER=50
REVISION="main"
PURPOSE="run_tasks"
RESULTS_PATH="experiments/results"

for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        python MTEB-Harness.py \
            --model_family $MODEL_NAME \
            --model_size $size \
            --revision $REVISION \
            --base_results_path $RESULTS_PATH \
            --purpose $PURPOSE \
            --evaluation_layer $layer
    done
done
```

More scripts, including SLURM examples, are available in the `slurm_scripts/` folder.

---

### Gated Attention Models (Qwen3 variants)

Three custom Qwen3 models with gating mechanisms are supported:

* `gated_attention_baseline` — standard Qwen3, no gating (1B parameters, 28 layers)
* `gated_attention_elementwise` — per-element gating on attention output (1B parameters, 28 layers)
* `gated_attention_headwise` — per-head scalar gating on attention output (1B parameters, 28 layers)

Checkpoints are located at `gated_attention/checkpoints/gated_attention/`.

**Single model, single layer:**

```bash
python3 -u MTEB-Harness.py \
    --model_family "gated_attention_baseline" \
    --model_size "1B" \
    --revision "main" \
    --evaluation_layer -1 \
    --base_results_path "experiments/results" \
    --purpose run_tasks
```

**All three models, all layers (SLURM on Anvil):**

The SLURM script is pre-configured for Anvil (`--partition=ai`, `--account=nairr250073-ai`). To launch:

```bash
cd scripts/slurm_scripts
bash gated-attention-mteb-sweep.sh
```

This submits 87 SLURM jobs (3 models x 29 layers). Set `USE_SLURM=0` in the sweep script to run locally instead.

---

## 📊 Calculating Representation Metrics

To compute **metrics like prompt entropy** across all layers:

```bash
MODEL_NAME="Pythia"
MODEL_SIZES=("14m" "70m" "160m" "410m")
REVISION="main"
PURPOSE="run_entropy_metrics"

for size in ${MODEL_SIZES[@]}; do
    echo "Evaluating $MODEL_NAME-$size"
    python MTEB-Harness.py \
        --model_family $MODEL_NAME \
        --model_size $size \
        --revision $REVISION \
        --base_results_path "experiments/results" \
        --purpose run_entropy_metrics
done
```
