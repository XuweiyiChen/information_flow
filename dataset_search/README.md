This folder implements text search for the large PILE dataset. We did this to investigate how Pythia test-time prompts were similar to train-time prompts.

## Step 1 - Download PILE
Download the PILE dataset with `python3 download_pile_dataset.py`. This downloads a subset of the original PILE, which is no longer fully available.

This step downloads 926G of data, and the outputs of step2 are about 271G. So you should put this on an HDD with plenty of space.


## Step 2 - Index PILE
Create corpus indices with `./make_index_shards.sh`. 

This may take several hours to run, but the end result is well worth it. The dataset indices allow for constant-time string search in the entire PILE dataset.

The shell script chunks it into 10 sections to avoid OOM errors.

## Step 3 - Search PILE
In `search_pile_dataset.py`, a medical chatbot dataloader is implemented. The role of this script is to take prompts from the dataloader and to find similar prompts in PILE. There are many overlapping prompts since the medical dataset and PILE both have data from PubMed.