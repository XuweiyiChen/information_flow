#! /bin/bash

for shard_idx in {0..9}
do 
    python create_dataset_indices.py --shard_idx $shard_idx
done

