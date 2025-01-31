To install required libraries, make a new conda environment and then run 'pip install -r requirements.txt'
If there are issues with the representation-itl (aka repitl), then try running manually 'pip3 install -e git+https://github.com/uk-cliplab/representation-itl.git#egg=representation-itl'
'

# MTEB Results

## The MTEB Harness

The `MTEB-Harness.py` is the script that handlesinteractions between models and the MTEB tasks/datasets. It uses the below arguments. 

`--model_family`: The family/type of model to evaluate (default: 'Pythia'). See `text_automodel_wrapper.py` for the full list of choices.

`--model_size`: Size variant of the model (default: '14m'). Valid choices depend on the model family. See `text_automodel_wrapper.py` for the full list of choices. 

`--revision`: Model revision/version to use (default: 'main'). You probably want to use 'main' here, unless for instance you are evaluating Pythia training checkpoints.

`--evaluation_layer`: Which layer to use for evaluation, 0-indexed. -1 means final layer (default: -1). If the choice is invalid for the specified model/size/revision, an error will be thrown and no benchmarks will be ran. This parameter is ignored unless `--purpose = run_tasks`.

`--base_results_path`: Base path for saving results (default: 'experiments/results')

`--purpose`: What should the script do (choices: 'run_tasks', 'run_entropy_metrics', 'download_datasets', default: 'run_tasks')

`--raise_error`: Whether to raise errors during evaluation and stop execution (default: False)

## Running MTEB Benchmarks

To run the 32 English benchmark tasks for the last model layer, use the below command with `--purpose run_tasks`. Notice how `--evaluation_layer == -1` whihc will use the last layer representations for the benchmarks as is typically done for the leaderboards. 

Example usage:
```
# script to run benchmarks for one layer for one model

python3 -u MTEB-Harness.py
    --model_family "Pythia"
    --model_size "14m"
    --revision "main"
    --evaluation_layer "-1" 
    --base_results_path "experiments/results"
    --purpose run_tasks
```

To run the benchmarks for ALL layers in a model, we can make a shell script to call the MTEB Harness multiple times. An example is like below where we iterate over the first 50 layers in Pythia-410m. Note that this model has only has 24 layers, so the remaining 26 calls will fail quickly. If you are unsure how many layers your model has, setting MAX_LAYER to be high is a good trick.

```
# script to run benchmarks for all layers for one model

MODEL_NAME="Pythia"
SIZE="410m"
MAX_LAYER=50
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
        --evaluation_layer $layer \
done
```

To run the benchmarks for ALL layers in MANY models, we can add another loop in the shell script. More shell script examples, including parallelization with SLURM, can be found in the `slurm_scripts` folder.

```
# script to run benchmarks for all layers for many models

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
            --evaluation_layer $layer \
    done
done
```

## Calculating Metrics on MTEB datasets

The other purpose of the MTEB harness is to calculate metrics like prompt entropy on the task datasets. To do this, set `purpose run_entropy_metrics`. In this setting, the `evaluation_layer` argument is ignored and metrics are computed for all layers.

Here is an example where we calculate metrics for different Pythia sizes on the 32 task datasets.
```
MODEL_SIZES=('14m' '70m' '160m' '410m')
REVISION="main"
PURPOSE="run_entropy_metrics"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME-$size"
    python MTEB-Harness.py \
        --model_family $MODEL_NAME 
        --model_size $size 
        --revision $REVISION 
        --base_results_path "experiments/results" 
        --purpose run_entropy_metrics
done
```