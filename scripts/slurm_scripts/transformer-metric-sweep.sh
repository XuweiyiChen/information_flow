#!/bin/bash

python MTEB-Harness.py \
    --model_family "LLM2Vec-mntp-unsup-simcse" \
    --model_size "8B" \
    --revision "main" \
    --purpose "run_wikitext_metrics"

python MTEB-Harness.py \
    --model_family "LLM2Vec-mntp-supervised" \
    --model_size "8B" \
    --revision "main" \
    --purpose "run_wikitext_metrics"

# python MTEB-Harness.py \
#     --model_family "LLM2Vec-mntp" \
#     --model_size "8B" \
#     --revision "main" \
#     --purpose "run_wikitext_metrics"


# python MTEB-Harness.py \
#     --model_family "Llama3" \
#     --model_size "8B" \
#     --revision "main" \
#     --purpose "run_wikitext_metrics"

# python MTEB-Harness.py \
#     --model_family "Pythia" \
#     --model_size "6.9b" \
#     --revision "main" \
#     --purpose "run_wikitext_metrics"