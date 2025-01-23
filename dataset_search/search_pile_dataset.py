import bm25s
import Stemmer
from datasets import load_dataset, load_from_disk
import numpy as np
import multiprocessing as mp
from functools import partial
import tqdm
import pickle

SHOULD_LOAD_CORPUS = True
RETRIEVER_LOCATIONS = [f"datasets/indices/pile_index_{idx}" for idx in range(10)]

def longestCommonString(s1, s2):
    m = len(s1)
    n = len(s2)

    LCSuf = [[0] * (n + 1) for _ in range(m + 1)]
    res = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                LCSuf[i][j] = LCSuf[i - 1][j - 1] + 1
                res = max(res, LCSuf[i][j])
            else:
                LCSuf[i][j] = 0

    return res

def longestCommonSubsequence(text1: str, text2: str) -> int: 
    m, n = len(text1), len(text2)
    l = []

    for i in range(m+1):
        v = [0]*(n+1)
        l.append(v)

    for i in range(1, m+1):
        for j in range(1, n+1):

            if text1[i-1] == text2[j-1]:
                l[i][j] = l[i-1][j-1] + 1

            else:
                l[i][j] = max(l[i-1][j], l[i][j-1])

    return l[m][n]

def process_longest_common_string(args):
    doc, query = args
    query_length = len(query)
    longest_common_substring = longestCommonString(doc, query)
    percent_overlap = longest_common_substring / query_length
    print(f"\tLongest common substring: {longest_common_substring}")
    print(f"\tPercent overlap: {percent_overlap}")
    return longest_common_substring, percent_overlap
    
def search_dataset(queries):
    stemmer = Stemmer.Stemmer("english")
    query_idx_to_best_result = {idx: None for idx in range(len(queries))}


    for retriever_location in RETRIEVER_LOCATIONS:
        print(f"\n\nLoading Retriever: {retriever_location}")
        retriever = bm25s.BM25.load(retriever_location, load_corpus=SHOULD_LOAD_CORPUS)
        print(f"Retriever loaded")
 

        # get retriever results for each query
        tokenized_queries = bm25s.tokenize(queries, stemmer=stemmer)
        documents, scores = retriever.retrieve(tokenized_queries, k=20)
        query_results = []
        for query, document, score in zip(queries, documents, scores):
            score_ratio =  score[0] / score[9]
            query_result = {
                "scores": score,
                "score_ratio": score_ratio,
                "retriever": retriever_location,
                "query_length": len(query),
                "query": query,
                "top_doc": document[0]['text'] if SHOULD_LOAD_CORPUS else document[0]
            }
            query_results.append(query_result)

        if SHOULD_LOAD_CORPUS:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                function_inputs = [(query_result["top_doc"], query_result["query"]) for query_result in query_results]
                
                lcs_outputs = list(tqdm.tqdm(
                    pool.imap(process_longest_common_string, function_inputs), 
                    total=len(function_inputs),
                    desc="Calculating LCS"
                ))
                for idx, query_result in enumerate(query_results):
                    query_result["longest_common_substring"] = lcs_outputs[idx][0]
                    query_result["longest_common_substring_percent_overlap"] = lcs_outputs[idx][1]

        # update best results
        for idx, query_result in enumerate(query_results):
            if query_idx_to_best_result[idx] is None or query_result["score_ratio"] > query_idx_to_best_result[idx]["score_ratio"]:
                query_idx_to_best_result[idx] = query_result

    return query_idx_to_best_result

def medical_tokenize_function(examples):
    medical_prompt = """You are an AI Medical Assistant Chatbot, trained to answer medical questions. Below is an instruction that describes a task, paired with an response context. Write a response that appropriately completes the request.

        ### Instruction:
        {}


        ### Response:
        {}"""
    
    print(examples)
    instructions = [examples["question"]]
    outputs      = [examples["context"]]
    texts = []
    for instruction, output in zip(instructions,  outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = medical_prompt.format(instruction,  output)
        texts.append(text)
    return texts


ds = load_dataset("ruslanmv/ai-medical-dataset", split="train[0:1]")
ds = ds.shuffle(seed=42)
samples_to_keep = [sample for sample in ds if len(sample["context"]) > 5]

queries = [sample["context"] for sample in samples_to_keep][0:1050]
questions_metadata = [sample["question"] for sample in samples_to_keep][0:1050]

best_results = search_dataset(queries)
for idx in range(len(best_results)):
    best_results[idx]["question"] = questions_metadata[idx]

with open("best_results.pkl", "wb") as f:
    pickle.dump(best_results, f)