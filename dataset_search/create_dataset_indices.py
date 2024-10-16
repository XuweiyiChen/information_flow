import bm25s
from datasets import load_dataset
import Stemmer
import tqdm
from multiprocessing import Pool, cpu_count, current_process
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_idx", type=int, required=True)
    return parser.parse_args()

def format_medical_sample(sample):
    sample["text"] = f"Question: {sample['question']}\nContext: {sample['context']}"
    return sample

def tokenize_fn(dataset_idx_range):
    process_id = current_process().name
    print(f"Processing {process_id} with range {dataset_idx_range}")
    texts = ds.select(range(*dataset_idx_range))["text"]
    print(f"Tokenizing {len(texts)} texts")
    stemmer = Stemmer.Stemmer("english")
    return bm25s.tokenize(texts=texts, return_ids=False, show_progress=False, stemmer=stemmer)

args = parse_args()
SHARD_IDX = args.shard_idx
USE_PILE = True
PILE_DATAPATH = "/dataldr/cliplab/ofsk222/"
MEDICAL_PILESETS = ["PubMed Abstracts", "PubMed Central"]
NUM_PROCESSES = 64

timer = time.time()

if USE_PILE:
    INDEX_NAME = "pile_index"
    PERCENT_PER_SHARD = 10
    shard_to_percent = PERCENT_PER_SHARD*SHARD_IDX
    ds = load_dataset("monology/pile-uncopyrighted", cache_dir=PILE_DATAPATH, split="train[%d%%:%d%%]" % (shard_to_percent, shard_to_percent+PERCENT_PER_SHARD), num_proc=NUM_PROCESSES, )
    ds = ds.filter(lambda x: x["meta"]["pile_set_name"] in MEDICAL_PILESETS, num_proc=NUM_PROCESSES)
else:
    INDEX_NAME = "medical_dataset_index"
    ds = load_dataset("ruslanmv/ai-medical-dataset")["train"]

print(f"Loaded dataset with {len(ds)} samples")

# tokenize the corpus
with Pool(NUM_PROCESSES) as pool:
    corpus_tokens = []
    dataset_idx_starts = [x for x in range(0, len(ds), len(ds)//NUM_PROCESSES)]
    dataset_idx_ends = dataset_idx_starts[1:] + [len(ds)]
    dataset_idx_ranges = list(zip(dataset_idx_starts, dataset_idx_ends))
    for result in pool.map(tokenize_fn, dataset_idx_ranges):
        corpus_tokens.extend(result)

# create the retriever and index the corpus
retriever = bm25s.BM25(corpus=ds["text"])
retriever.index(corpus_tokens)
retriever.save(f"datasets/indices/{INDEX_NAME}_{SHARD_IDX}")

print(f"Time taken: {time.time() - timer} seconds")

