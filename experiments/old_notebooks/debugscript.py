import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications

from utils.metrics.metric_calling import (
    compute_per_forward_pass,
    compute_on_concatenated_passes,
    metric_name_to_function,
    EvaluationMetricSpecifications,
    calculate_and_save_layerwise_metrics
)
from utils.dataloaders.text_dataloader import (
    get_dataloader,
    get_augmentation_collated_dataloader
)

def compute_sentence_entropies(model, dataloader, granularity='sentence', alpha=1):
    """
    Compute the entropy of each sentence in the dataloader.
    """
    compute_func_kwargs = {
        'alpha': alpha,
        'normalizations': ['maxEntropy', 'raw', 'length']
    }
    compute_function =  metric_name_to_function['entropy']
    forward_pass_func = compute_per_forward_pass if granularity == 'sentence' else compute_on_concatenated_passes

    results = forward_pass_func(model, dataloader, compute_function, should_average_over_layers=False, **compute_func_kwargs)
    return results

model_specs = TextModelSpecifications(
    model_family="Pythia",
    model_size="410m",
    revision="main"
)
model = TextLayerwiseAutoModelWrapper(model_specs, device_map="auto")
print('heel')

# evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric="dataset-entropy", alpha=2) # alpha=2 for faster computation, 1 used for paper

# dataloader_wikitext = get_dataloader(
#     model.tokenizer,
#       "wikitext", 
#       split="train", 
#       num_samples=10000,
#       batch_size=16,
#       num_workers=16,
#       filter_text_columns=True
# )

# dataloader_amazon_counterfactual = get_dataloader(
#     model.tokenizer,
#     "mteb/amazon_counterfactual", 
#     split="train", 
#     num_samples=1000, 
#     batch_size=2,
#     num_workers=4,
#     filter_text_columns=True
# )

# wikitext_dataset_entropy = calculate_and_save_layerwise_metrics(
#     model, 
#     dataloader_wikitext, 
#     model_specs, 
#     evaluation_metric_specs, 
#     dataloader_kwargs={"num_samples": 1000, "dataset_name": "wikitext"},
#     should_save_results=False
# )

# mteb_amazon_counterfactual_dataset_entropy = calculate_and_save_layerwise_metrics(
#     model, 
#     dataloader_amazon_counterfactual, 
#     model_specs, 
#     evaluation_metric_specs, 
#     dataloader_kwargs={"num_samples": 1000, "dataset_name": "mteb/amazon_counterfactual"},
#     should_save_results=False
# )

evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric="infonce")
augmented_wikitext = get_augmentation_collated_dataloader(
    model.tokenizer,
    "wikitext", 
    split="train", 
    num_samples=1000, 
    batch_size=32,
    num_workers=4,
    filter_text_columns=True,
    num_augmentations_per_sample=2
)

wikitext_dime = calculate_and_save_layerwise_metrics(
    model, 
    augmented_wikitext, 
    model_specs, 
    evaluation_metric_specs, 
    dataloader_kwargs={"num_samples": 1000, "dataset_name": "wikitext"},
    should_save_results=False
)