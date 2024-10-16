from datasets import load_dataset

data_path = "/Data/Disk2/cliplab/ofsk222/datasets"
ds = load_dataset("monology/pile-uncopyrighted", cache_dir=data_path, split="train", num_proc=16)